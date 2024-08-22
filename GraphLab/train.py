import copy
import logging
import time

import pandas as pd
import torch
import torch.nn.functional as F
from lifelines import CoxPHFitter
from sklearn.metrics import accuracy_score

from GraphLab.checkpoint import clean_ckpt, load_ckpt, save_ckpt
from GraphLab.config import cfg
from GraphLab.loss import compute_loss
from GraphLab.utils.epoch import is_eval_epoch
from lifelines.utils import concordance_index

from GraphLab.utils.utils import seed_anything

criterion = torch.nn.CrossEntropyLoss()
mse_loss = torch.nn.MSELoss(reduction=cfg.model.size_average)
seed_anything(cfg.seed)
def _get_pred_int(pred_score):
    if len(pred_score.shape) == 1 or pred_score.shape[1] == 1:
        return (pred_score > cfg.model.thresh).long()
    else:
        return pred_score.max(dim=1)[1]


def convert_to_one_hot(x, num_classes):
    # 创建一个全0张量
    one_hot = torch.zeros((len(x), num_classes)).to(torch.device(cfg.device))

    # 遍历张量，将对应位置置为1
    for i in range(len(x)):
        one_hot[i, int(x[i]) - 1] = 1

    return one_hot


def train_epoch(logger, loader, model, optimizer, scheduler):
    model.train()
    time_start = time.time()
    for batch in loader:
        optimizer.zero_grad()
        batch.to(torch.device(cfg.device))
        if cfg.model.attention:
            pred, true, score = model(batch)
        else:
            pred, true = model(batch)
        loss, pred_score = compute_loss(model, pred, true)
        subtask_loss = torch.tensor(0., dtype=loss.dtype).to(torch.device(cfg.device))
        if cfg.dataset.multitasking:
            # 定义损失函数
            # 调用损失函数
            target = F.one_hot(batch.node_label.view(-1).long() - 1, cfg.dataset.subtaskdim)
            subtask_loss = criterion(batch.node_level_feature, target.float())
            pred_int = _get_pred_int(batch.node_level_feature)
            # print(batch.node_label)
            subtask_metric = {
                'accuracy': round(accuracy_score(batch.node_label.detach().cpu() - 1, pred_int.detach().cpu()),
                                  cfg.round),
            }
            print("subtask_loss: ", subtask_loss.item(), end=",")
            print(subtask_metric, end=",")
            print()
        loss += subtask_loss
        loss.backward()
        optimizer.step()
        logger.update_stats(true=true.detach(),
                            pred=pred.detach(),
                            loss=loss.item(),
                            lr=scheduler.get_last_lr()[0],
                            time_used=time.time() - time_start,
                            params=cfg.params)
        time_start = time.time()
    scheduler.step()


from lifelines.statistics import logrank_test
import numpy as np


def cox_log_rank(hazardsdata, labels, survtime_all):
        median = np.median(hazardsdata)
        hazards_dichotomize = np.zeros([len(hazardsdata)], dtype=int)
        hazardsdata = hazardsdata.detach().numpy()
        hazards_dichotomize[np.where(hazardsdata > median)[0]] = 1
        idx = hazards_dichotomize == 0
        T1 = survtime_all[idx]
        T2 = survtime_all[~idx]
        E1 = labels[idx]
        E2 = labels[~idx]
        results = logrank_test(T1, T2, event_observed_A=E1, event_observed_B=E2)
        pvalue_pred = results.p_value
        return (pvalue_pred)


@torch.no_grad()
def eval_epoch(logger, loader, model):
    model.eval()
    time_start = time.time()
    # all_test_preds = []
    # all_test_labels = []
    for batch in loader:
        batch.to(torch.device(cfg.device))
        if cfg.model.attention:
            pred, true, score = model(batch)
        else:
            pred, true = model(batch)
        if cfg.dataset.multitasking:
            pred_int = _get_pred_int(batch.node_level_feature)
            subtask_metric = {
                'accuracy': round(accuracy_score(batch.node_label.detach().cpu() - 1, pred_int.detach().cpu()),
                                  cfg.round),
            }
            print(subtask_metric, end=",")
            print()
        # for item in pred:
        #     all_test_preds.append(item)
        # for item in true:
        #     all_test_labels.append(item)
        # loss=compute_DSLoss(pred, true[:, 0], true[:, 1], model)
        loss, pred_score = compute_loss(model, pred, true)
        logger.update_stats(true=true.detach(),
                            pred=pred.detach(),
                            loss=loss.item(),
                            lr=0,
                            time_used=time.time() - time_start,
                            params=cfg.params,
                            )
        time_start = time.time()
    # all_test_labels=torch.stack(all_test_labels,dim=0).cpu()
    # all_test_preds=torch.stack(all_test_preds,dim=0).cpu()
    # cindex_test = concordance_index(all_test_labels[:, 0], -all_test_preds, all_test_labels[:, 1])
    # pvalue_test = cox_log_rank(all_test_preds, all_test_labels[:, 1], all_test_labels[:, 0])
    # print('Test c-index {}'.format(cindex_test))
    # print('Test p-value {}'.format(pvalue_test))


def train(loggers, loaders, model, optimizer, scheduler):
    r"""
    The core training pipeline

    Args:
        loggers: List of loggers
        loaders: List of loaders
        model: GNN model
        optimizer: PyTorch optimizer
        scheduler: PyTorch learning rate scheduler

    """
    start_epoch = 0
    if cfg.train.auto_resume:
        start_epoch = load_ckpt(model, None, None)
        start_epoch = 1
    if start_epoch == cfg.optim.max_epoch:
        logging.info('Checkpoint found, Task already done')
    else:
        logging.info('Start from epoch {}'.format(start_epoch))

    num_splits = len(loggers)
    for cur_epoch in range(start_epoch, cfg.optim.max_epoch):
        # save_ckpt(model, optimizer, scheduler, -1)
        train_epoch(loggers[0], loaders[0], model, optimizer, scheduler)
        loggers[0].write_epoch(cur_epoch)
        if is_eval_epoch(cur_epoch):

            if cfg.dataset.task_type == 'classification2regression':

                train_data = torch.tensor([]).to(torch.device(cfg.device))
                val_data = torch.tensor([]).to(torch.device(cfg.device))
                test_data = torch.tensor([]).to(torch.device(cfg.device))
                final_data = [train_data, val_data, test_data]
                model.eval()
                for idx, loader in enumerate(loaders):
                    for data in loader:
                        data.to(torch.device(cfg.device))
                        pred, true = model(data)
                        result = torch.cat((pred, true), dim=1)
                        final_data[idx] = torch.cat((final_data[idx], result), dim=0)
                    final_data[idx] = final_data[idx].detach().cpu().numpy()
                    final_data[idx] = pd.DataFrame(final_data[idx])
                    final_data[idx].columns = ['0', '1', '2', '3', '4', 'survtime', 'event']

                train_d = pd.concat((final_data[0], final_data[1]), axis=0)
                test_d = final_data[2].drop(['survtime', 'event'], axis=1)
                cph = CoxPHFitter(penalizer=0.01)
                cph.fit(train_d, duration_col='survtime', event_col='event')
                cph.print_summary()
                # cph.predict_survival_function(final_data[2])
                # cph.predict_median(final_data[2])
                predicted_risk = cph.predict_partial_hazard(test_d)
                c_index = concordance_index(final_data[2]['survtime'], -predicted_risk, final_data[2]['event'])
                print(f"final c-index: {c_index}")

            for i in range(1, num_splits):
                eval_epoch(loggers[i], loaders[i], model)
                loggers[i].write_epoch(cur_epoch)
                # model1 = copy.deepcopy(model)
                # load_ckpt(model1, optimizer, scheduler, cur_epoch)
                # model1.eval()
                # eval_epoch(loggers[i], loaders[i], model1)
            save_ckpt(model, optimizer, scheduler, cur_epoch)
        # if is_ckpt_epoch(cur_epoch):
        #     save_ckpt(model, optimizer, scheduler, cur_epoch)
    for logger in loggers:
        logger.close()
    if cfg.train.ckpt_clean:
        clean_ckpt()

    if cfg.dataset.task_type == 'classification2regression':

        train_data = torch.tensor([]).to(torch.device(cfg.device))
        val_data = torch.tensor([]).to(torch.device(cfg.device))
        test_data = torch.tensor([]).to(torch.device(cfg.device))
        final_data = [train_data, val_data, test_data]
        model.eval()
        for idx, loader in enumerate(loaders):
            for data in loader:
                data.to(torch.device(cfg.device))
                pred, true = model(data)
                result = torch.cat((pred, true), dim=1)
                final_data[idx] = torch.cat((final_data[idx], result), dim=0)
            final_data[idx] = final_data[idx].detach().cpu().numpy()
            final_data[idx] = pd.DataFrame(final_data[idx])
            final_data[idx].columns = ['0', '1', '2', '3', '4', 'survtime', 'event']

        train_d = pd.concat((final_data[0], final_data[1]), axis=0)
        test_d = final_data[2].drop(['survtime', 'event'], axis=1)
        cph = CoxPHFitter(penalizer=0.01)
        cph.fit(train_d, duration_col='survtime', event_col='event')
        cph.print_summary()
        # cph.predict_survival_function(final_data[2])
        # cph.predict_median(final_data[2])
        predicted_risk = cph.predict_partial_hazard(test_d)
        c_index = concordance_index(final_data[2]['survtime'], -predicted_risk, final_data[2]['event'])
        print(f"final c-index: {c_index}")
    logging.info('Task done, results saved in {}'.format(cfg.out_dir))
