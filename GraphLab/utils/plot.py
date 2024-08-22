import os

import matplotlib.pyplot as plt
import numpy as np

from GraphLab.config import cfg
from GraphLab.utils.io import (json_to_dict_list,
                               string_to_python)

try:
    from tensorboardX import SummaryWriter
except ImportError:
    SummaryWriter = None


def is_seed(s):
    try:
        int(s)
        return True
    except Exception:
        return False


def is_split(s):
    if s in ['train', 'val', 'test']:
        return True
    else:
        return False


def join_list(l1, l2):
    assert len(l1) == len(l2), \
        'Results with different seeds must have the save format'
    for i in range(len(l1)):
        l1[i] += l2[i]
    return l1


def agg_dict_list(dict_list):
    """
    Aggregate a list of dictionaries: mean + std
    Args:
        dict_list: list of dictionaries

    """
    dict_agg = {'epoch': dict_list[0]['epoch']}
    for key in dict_list[0]:
        if key != 'epoch':
            value = np.array([dict[key] for dict in dict_list])
            dict_agg[key] = np.mean(value).round(cfg.round)
            dict_agg['{}_std'.format(key)] = np.std(value).round(cfg.round)
    return dict_agg


def name_to_dict(run):
    cols = run.split('-')[1:]
    keys, vals = [], []
    for col in cols:
        try:
            key, val = col.split('=')
        except Exception:
            print(col)
        keys.append(key)
        vals.append(string_to_python(val))
    return dict(zip(keys, vals))


def rm_keys(dict, keys):
    for key in keys:
        dict.pop(key, None)


def agg_runs(dir):
    r'''
    Aggregate over different random seeds of a single experiment

    Args:
        dir (str): Directory of the results, containing 1 experiment
        metric_best (str, optional): The metric for selecting the best
        validation performance. Options: auto, accuracy, auc.

    '''
    results = {'train': None, 'val': None, 'test': None}
    results_best = {'train': None, 'val': None, 'test': None}
    for seed in os.listdir(dir):
        if is_seed(seed):
            dir_seed = os.path.join(dir, seed)
            for split in os.listdir(dir_seed):
                if is_split(split):
                    dir_split = os.path.join(dir_seed, split)
                    fname_stats = os.path.join(dir_split, 'stats.json')
                    stats_list = json_to_dict_list(fname_stats)
                    loss = stats_list['loss']
                    epoch = np.arange(0, len(stats_list), 1)

                    plt.plot(epoch, loss, linestyle="-.", color='#b8860b', marker='o')
                    plt.xlabel('epoch')
                    plt.ylabel('loss')
                    plt.legend()
                    plt.show()
                    plt.savefig(fname_stats + 'loss.png')


if __name__ == '__main__':
    agg_runs(cfg.out_dir)
