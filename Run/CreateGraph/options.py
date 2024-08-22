import argparse
import os

import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--label_data_path',
                        type=str,
                        default='/data0/pathology/all_patients',
                        help="all labeled path")

    parser.add_argument('--graph_out_path',
                        type=str,
                        default='/data/yuanyz/datasets/SLIC')

    parser.add_argument('--WSI_data_path',
                        type=str,
                        default='/data0/pathology/all_patients',
                        help='original WSI path')

    parser.add_argument('--follow_up_data',
                        type=str,
                        default='/data0/pathology/follow_up_data.txt',
                        help='follow up data path')

    parser.add_argument('--checkpoints_dir',
                        type=str,
                        default='../results',
                        help='models are saved here')

    parser.add_argument('--method',
                        type=str,
                        default='slic',
                        help='slic fz quick watershed')

    parser.add_argument('--gpu_ids',
                        type=str,
                        default='0',
                        help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

    opt = parser.parse_known_args()[0]
    print_options(parser, opt)
    # opt = parse_gpuids(opt)
    return opt


def print_options(parser, opt):
    """Print and save options

    It will print both current options and default values(if different).
    It will save options into a text file / [checkpoints_dir] / opt.txt
    """
    message = ''
    message += '----------------- Options ---------------\n'
    for k, v in sorted(vars(opt).items()):
        comment = ''
        default = parser.get_default(k)
        if v != default:
            comment = '\t[default: %s]' % str(default)
        message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
    message += '----------------- End -------------------'
    print(message)

    # save to the disk
    # expr_dir = opt.checkpoints_dir
    # mkdirs(expr_dir)
    # file_name = os.path.join(expr_dir, '{}_opt.txt'.format('train'))
    # with open(file_name, 'wt') as opt_file:
    #     opt_file.write(message)
    #     opt_file.write('\n')


def parse_gpuids(opt):
    # set gpu ids
    str_ids = opt.gpu_ids.split(',')
    opt.gpu_ids = []
    for str_id in str_ids:
        id = int(str_id)
        if id >= 0:
            opt.gpu_ids.append(id)
    if len(opt.gpu_ids) > 0:
        torch.cuda.set_device(opt.gpu_ids[0])

    return opt


def mkdirs(paths):
    """create empty directories if they don't exist

    Parameters:
        paths (str list) -- a list of directory paths
    """
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            mkdir(path)
    else:
        mkdir(paths)


def mkdir(path):
    """create a single empty directory if it didn't exist

    Parameters:
        path (str) -- a single directory path
    """
    if not os.path.exists(path):
        os.makedirs(path)
