import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()

def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def get_mean_var_info(data_dir):
    print('loading dataset...')
    data = fvecs_read(data_dir)
    mean = data.flatten().mean()
    sigma = data.flatten().std()
    return mean, sigma


if __name__ == '__main__':
    # data_dir = '/home/disk/data/ANNS/GIST1M/gist_base.fvecs'
    # data_name = 'GIST1M'

    data_dir = '/home/disk/data/ANNS/Deep1M/deep1M_base.fvecs'
    data_name = 'Deep1M'

    mean, sigma = get_mean_var_info(data_dir)
    print('{}|| mean:{} || sigma:{}'.format(data_name, mean, sigma))


# GIST1M|| mean:0.06963662058115005 || sigma:0.046855390071868896
# Deep1M|| mean:-0.0008674514247104526 || sigma:0.0621178112924099



