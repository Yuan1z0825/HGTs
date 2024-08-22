
import logging
import os
import sys


sys.path.append('../')
import torch
from GraphLab.utils.utils import seed_anything
from GraphLab.cmd_args import parse_args
from GraphLab.config import cfg, dump_cfg, load_cfg, set_run_dir, set_out_dir
seed_anything(cfg.seed)
# Load cmd line args
args = parse_args()
# Load configs file
load_cfg(cfg, args)

set_out_dir(cfg.out_dir, args.cfg_file)
# argument_dataset()
# print("hello")
# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)
dump_cfg(cfg)
# Repeat for different random seeds
test_data = None
# 设置seed
from GraphLab.loader import create_dataset, create_loader
from GraphLab.logger import create_logger, setup_printing
from GraphLab.model_builder import create_model
from GraphLab.optimizer import create_optimizer, create_scheduler
from GraphLab.register import train_dict
from GraphLab.train import train
from GraphLab.utils.agg_runs import agg_runs
from GraphLab.utils.comp_budget import params_count
from GraphLab.utils.device import auto_select_device

if __name__ == '__main__':


    datasets = create_dataset()
    print("创建数据集完成，开始创建loader")
    for i in range(args.repeat):
        set_run_dir(cfg.out_dir)
        setup_printing()
        # Set configurations for each run
        auto_select_device()
        # Set machine learning pipeline

        loaders = create_loader(datasets, cfg.train.batch_size)
        loggers = create_logger()
        model = create_model()
        optimizer = create_optimizer(model.parameters())
        scheduler = create_scheduler(optimizer)
        # Print model info
        logging.info(model)
        logging.info(cfg)
        cfg.params = params_count(model)
        logging.info('Num parameters: %s', cfg.params)
        # Start training
        if cfg.train.mode == 'standard':
            train(loggers, loaders, model, optimizer, scheduler)
        else:
            train_dict[cfg.train.mode](loggers, loaders, model, optimizer,
                                       scheduler)

    # Aggregate results from different seeds
    agg_runs(cfg.out_dir, cfg.metric_best)
    # When being launched in batch mode, mark a yaml as done

    if args.mark_done:
        os.rename(args.cfg_file, f'{args.cfg_file}_done')
