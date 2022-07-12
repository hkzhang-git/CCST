import os
import torch
import numpy as np
from .datasets import dataset_dict


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def build_dataset(cfg, mode='train'):

    if cfg.DATASET.DATA_NAME == 'GIST1M':
        base = 'gist'
    elif cfg.DATASET.DATA_NAME == 'Deep1M':
        base = 'deep1M'

    if mode == 'train':
        shuffle_ids = np.random.permutation(int(cfg.DATASET.FEAT_NUM))
        inter_index = int(cfg.DATASET.FEAT_NUM * (1-cfg.SOLVER.VAL_PORTION))
        train_indices = shuffle_ids[: inter_index]
        val_indices = shuffle_ids[inter_index:]



        print('loading dataset...')
        data = fvecs_read(os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.DATA_NAME, '{}_base.fvecs'.format(base)))
        print('dataset has been loaded...')

        dataset_train = dataset_dict[cfg.DATASET.DATA_NAME](data[train_indices])
        dataset_val = dataset_dict[cfg.DATASET.DATA_NAME](data[val_indices])

        train_loader = torch.utils.data.DataLoader(
            dataset_train,
            shuffle=True,
            batch_size=cfg.DATALOADER.BATCH_SIZE_TRAIN,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            dataset_val,
            shuffle=False,
            batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True)

        return train_loader, val_loader
    elif mode == 'infe':

        print('loading base data...')
        base_data = fvecs_read(os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.DATA_NAME, '{}_base.fvecs'.format(base)))

        print('loading query data...')
        query_data = fvecs_read(os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.DATA_NAME, '{}_query.fvecs'.format(base)))

        print('dataset has been loaded...')

        dataset_base = dataset_dict[cfg.DATASET.DATA_NAME](base_data)
        dataset_query = dataset_dict[cfg.DATASET.DATA_NAME](query_data)

        base_data_loader = torch.utils.data.DataLoader(
            dataset_base,
            shuffle=False,
            batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True)

        query_data_loader = torch.utils.data.DataLoader(
            dataset_query,
            shuffle=False,
            batch_size=cfg.DATALOADER.BATCH_SIZE_TEST,
            num_workers=cfg.DATALOADER.NUM_WORKERS,
            pin_memory=True)

        return base_data_loader, query_data_loader




