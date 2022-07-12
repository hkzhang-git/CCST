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


def euclidean_dist(x, y):
    dist = torch.cdist(x, y, p=2)
    return dist


def get_distrib_info(data_dir, sample_num, is_compress, data_name):
    print('loading dataset...')
    if is_compress:
        data=np.load(data_dir)
    else:
        data = fvecs_read(data_dir)

    if data_name == 'GIST1M':
        data = (data-0.0696)/0.0469
    elif data_name == 'Deep1M':
        data = (data + 0.0008) / 0.0621

    shuffle_indices = np.random.permutation(len(data))
    x = data[shuffle_indices[:sample_num]]
    y = data[shuffle_indices[sample_num:sample_num*2]]
    dist =euclidean_dist(torch.from_numpy(x), torch.from_numpy(y))
    dist_mean = dist.mean()
    dist_sigma = dist.std()
    return float(dist_mean), float(dist_sigma), dist.numpy()


def info_plot(data, mean, sigma, data_name, save_dir, is_compress):
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.hist(data, bins=256*2, color='blue')
    plt.xticks(np.linspace(0, data.max(), 15), fontsize=5)
    if is_compress:
        plt.xlabel('{}_c || euclidean_dist|| mean={}, sigma={}'.format(data_name, round(mean, 4), round(sigma, 4)))
    else:
        plt.xlabel('{} || euclidean_dist|| mean={}, sigma={}'.format(data_name, round(mean, 4), round(sigma, 4)))
    plt.ylabel('P')
    if is_compress:
        plt.savefig(os.path.join(save_dir, '{}_c.jpg'.format(data_name)))
    else:
        plt.savefig(os.path.join(save_dir, '{}.jpg'.format(data_name)))


if __name__ == '__main__':
    data_name = 'Deep1M'
    if data_name == 'GIST1M':
        data_dir = '/home/disk/data/ANNS/GIST1M/gist_base.fvecs'
    elif data_name == 'Deep1M':
        data_dir = '/home/disk/data/ANNS/Deep1M/deep1M_base.fvecs'

    result_save_dir = './distrib_info'
    sample_num = 1e+04
    if not os.path.exists(result_save_dir):
        os.makedirs(result_save_dir)
    mean, sigma, disk = get_distrib_info(data_dir, int(sample_num), is_compress=False, data_name=data_name)
    print('drawing the final result...')
    info_plot(disk.flatten(), mean, sigma, data_name, result_save_dir, is_compress=False)



