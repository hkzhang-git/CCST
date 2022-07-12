import os
import torch
import argparse
from config import cfg
from model import build_model
from data import build_dataset
from utils.losses import create_criterion
from utils.utils import make_if_not_exist
from utils import trainer, val
from torch.utils.tensorboard import SummaryWriter
from utils import Checkpointer, create_optimizer, create_scheduler
from glob import glob
import numpy as np
import time


def state_dict_load(cfg):
    final_model = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATA_NAME,
                             '{}_x{}_{}'.format(cfg.STRUC.MODEL, cfg.STRUC.C_FACTOR, cfg.SOLVER.LOSS),
                             'Train/model_final.pth')
    if os.path.exists(final_model):
        state_dict = torch.load(final_model).pop("model")
        return state_dict

    else:
        model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATA_NAME,
                                 '{}_x{}_{}'.format(cfg.STRUC.MODEL, cfg.STRUC.C_FACTOR, cfg.SOLVER.LOSS),
                                 'Train/models')
        model_list = glob(model_dir + '/*.pth')
        list_arr = [int(item.split('/')[-1].split('.')[0]) for item in model_list]
        list_arr.sort()
        last_model = model_dir + '/{}.pth'.format(list_arr[-1])
        state_dict = torch.load(last_model).pop("model")

        return state_dict


def restore_c_pointer(cfg):
    model_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATA_NAME,
                             '{}_x{}_{}'.format(cfg.STRUC.MODEL, cfg.STRUC.C_FACTOR, cfg.SOLVER.LOSS), 'Train/models')
    model_list = glob(model_dir + '/*.pth')
    if not model_list.__len__() == 0:
        list_arr = [int(item.split('/')[-1].split('.')[0]) for item in model_list]
        list_arr.sort()
        last_file = model_dir + '/{}.pth'.format(list_arr[-1])
        print('restoring model from {}'.format(last_file))
        checkpoint_file = torch.load(last_file)
        model_state_dict = checkpoint_file.pop("model")
        init_lr = checkpoint_file['optimizer']['param_groups'][0]['lr']
        epoch = list_arr[-1]

        return model_state_dict, init_lr, epoch+1
    else:
        return None, None, 1


def train(cfg, output, restore):
    model, params_n = build_model(cfg.STRUC)
    model = torch.nn.DataParallel(model).cuda()
    print('model {} contains {}M parameters'.format(cfg.STRUC.MODEL, params_n))

    start_epoch = 1
    if restore:
        model_state_dict, init_lr, start_epoch = restore_c_pointer(cfg)
        if start_epoch>1:
            try:
                model.load_state_dict(model_state_dict)
            except:
                model.module.load_state_dict(model_state_dict)

            optimizer = create_optimizer(cfg, model, re_lr=init_lr)
            lr_scheduler = create_scheduler(cfg, optimizer)
        else:
            optimizer = create_optimizer(cfg, model)
            lr_scheduler = create_scheduler(cfg, optimizer)
    else:
        optimizer = create_optimizer(cfg, model)
        lr_scheduler = create_scheduler(cfg, optimizer)

    criterion = create_criterion(cfg)

    checkpointer = Checkpointer(model, optimizer, lr_scheduler, save_dir=output + '/models')
    train_loader, val_loader = build_dataset(cfg, mode='train')
    writer = SummaryWriter(output + '/log' )

    print('training ...')

    for epoch in range(start_epoch, cfg.SOLVER.MAX_EPOCH+1):
        if cfg.SOLVER.VISUAL:
            train_loss_w, train_loss, heat_map, heat_map_x, heat_map_y = trainer(model, optimizer, train_loader,
                                                                                 criterion, epoch, visual=True)
            if epoch % cfg.SOLVER.VALIDATE_PERIOD == 0:
                val_loss_w, val_loss = val(model, val_loader, criterion)
                writer.add_scalars('loss/loss_w', {'train_loss': train_loss_w, 'val_loss': val_loss_w}, epoch)
                writer.add_scalars('loss/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
                writer.add_scalars('learning_rate', {'learning_rate': optimizer.param_groups[0]['lr']}, epoch)
                writer.add_image('histogram_2d/xy', heat_map, epoch)
                writer.add_image('histogram_2d/xx', heat_map_x, epoch)
                writer.add_image('histogram_2d/yy', heat_map_y, epoch)
                checkpointer.save(epoch)

        else:
            train_loss_w, train_loss = trainer(model, optimizer, train_loader, criterion, epoch, visual=True)
            if epoch % cfg.SOLVER.VALIDATE_PERIOD == 0:
                val_loss_w, val_loss = val(model, val_loader, criterion)
                writer.add_scalars('loss/loss_w', {'train_loss': train_loss_w, 'val_loss': val_loss_w}, epoch)
                writer.add_scalars('loss/loss', {'train_loss': train_loss, 'val_loss': val_loss}, epoch)
                writer.add_scalars('learning_rate', {'learning_rate': optimizer.param_groups[0]['lr']}, epoch)

        lr_scheduler.step()
        if epoch == cfg.SOLVER.MAX_EPOCH:
            final_model_dir = os.path.join(output, "model_final.pth")
            data = {}
            data["model"] = model.state_dict()
            torch.save(data, final_model_dir)


def infe(cfg):
    from utils.losses import CR_error

    model, _ = build_model(cfg.STRUC)
    model = torch.nn.DataParallel(model).cuda()

    state_dict = state_dict_load(cfg)
    try:
        model.load_state_dict(state_dict)
    except:
        model.module.load_state_dict(state_dict)

    model.eval()

    base_data_loader, query_data_loader = build_dataset(cfg, mode='infe')
    criterion = CR_error()

    # compress base data
    compress_save_dir = os.path.join(cfg.DATASET.DATA_ROOT, cfg.DATASET.DATA_NAME, cfg.STRUC.MODEL)
    if not os.path.exists(compress_save_dir):
        os.makedirs(compress_save_dir)
    compress_error_info = os.path.join(compress_save_dir, 'compress_error.txt')

    if cfg.DATASET.DATA_NAME == 'GIST1M':
        base = 'gist'
    elif cfg.DATASET.DATA_NAME == 'Deep1M':
        base = 'deep1M'

    cr_error_avg = 0
    iter_num = 0
    base_data = []
    with torch.no_grad():
        t_s = 0
        for feats in base_data_loader:
            feats = feats.cuda()
            t_b = time.time()
            c_feats = model(feats)
            torch.cuda.synchronize()
            t_e = time.time()
            t_s += (t_e-t_b)
            cr_error = criterion(x=feats, y=c_feats)

            cr_error_avg += float(cr_error)
            iter_num += 1

            base_data.append(c_feats.cpu().numpy())
    print('time cost:{}'.format(t_s))

    np.save(compress_save_dir + '/{}_base_{}_{}_c_x{}.npy'.format(base, cfg.STRUC.MODEL, cfg.SOLVER.LOSS, cfg.STRUC.C_FACTOR), np.vstack(base_data))
    with open(compress_error_info, 'a') as f:
        f.write('{} || base_data || compression error:{} || time cost:{}s\n'.format(cfg.SOLVER.LOSS, cr_error_avg / iter_num, t_s))

    # compress query data
    cr_error_avg = 0
    iter_num = 0
    query_data = []
    with torch.no_grad():
        for feats in query_data_loader:
            feats = feats.cuda()
            c_feats = model(feats)
            cr_error = criterion(x=feats, y=c_feats)
            cr_error_avg += float(cr_error)
            iter_num += 1

            query_data.append(c_feats.cpu().numpy())

    np.save(compress_save_dir + '/{}_query_{}_{}_c_x{}.npy'.format(base, cfg.STRUC.MODEL, cfg.SOLVER.LOSS, cfg.STRUC.C_FACTOR),  np.vstack(query_data))
    with open(compress_error_info, 'a') as f:
        f.write('{} || query_data || compression error:{}\n'.format(cfg.SOLVER.LOSS, cr_error / iter_num))


def main():
    parser = argparse.ArgumentParser(description="FeCT training and evaluation")
    parser.add_argument("--config-file", type=str, default='./config/Deep1M/ccst_x4.yaml')
    parser.add_argument("--device", type=str, default='5')
    parser.add_argument("--mode", type=str, default='train')
    parser.add_argument("--restore", type=bool, default=True)
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    cfg.merge_from_file(args.config_file)
    cfg.freeze()

    if args.mode == 'train':
        output_dir = os.path.join(cfg.OUTPUT_DIR, cfg.DATASET.DATA_NAME, '{}_x{}_{}'.format(cfg.STRUC.MODEL, cfg.STRUC.C_FACTOR, cfg.SOLVER.LOSS), 'Train')
        make_if_not_exist(output_dir)
        train(cfg, output_dir, args.restore)
    elif args.mode == 'infe':
        infe(cfg)


if __name__ == '__main__':
    main()




