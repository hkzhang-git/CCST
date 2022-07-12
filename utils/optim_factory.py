import torch.optim as optim


def create_optimizer(cfg, model, re_lr=0):
    opt_lower = cfg.SOLVER.OPTIM
    parameters = model.parameters()

    if re_lr == 0:
        init_lr = cfg.SOLVER.INIT_LR
    else:
        init_lr = re_lr

    if opt_lower == 'sgd' or opt_lower == 'nesterov':
        optimizer = optim.SGD(parameters, lr=init_lr)
    elif opt_lower == 'adam':
        optimizer = optim.Adam(parameters, lr=init_lr)
    elif opt_lower == 'adamw':
        optimizer = optim.AdamW(parameters, lr=init_lr)
    elif opt_lower == 'adadelta':
        optimizer = optim.Adadelta(parameters)
    else:
        assert False and "Invalid optimizer"
        raise ValueError

    return optimizer


def create_scheduler(cfg, optimizer):
    if cfg.SOLVER.SCHEDULER == 'poly':
        lr_scheduler = PolynormialLR(optimizer, cfg.SOLVER.MAX_EPOCH)
    elif cfg.SOLVER.SCHEDULER == 'multistep':
        lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * cfg.SOLVER.MAX_EPOCH),
                                                                             int(0.8 * cfg.SOLVER.MAX_EPOCH)])
    elif cfg.SOLVER.SCHEDULER == 'none':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.998)
    else:
        print('undefined lr_scheduler, please select from [poly, multistep, none]')

    return lr_scheduler


class PolynormialLR(optim.lr_scheduler._LRScheduler):
    def __init__(
            self,
            optimizer,
            max_epoch,
            power=0.9,
            last_epoch=-1,
    ):
        self.max_epoch = max_epoch
        self.power = power
        super(PolynormialLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return [
            base_lr
            * (1 - self.last_epoch / self.max_epoch) ** self.power
            for base_lr in self.base_lrs
        ]            