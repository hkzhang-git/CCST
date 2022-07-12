import torch
import torch.nn as nn
import numpy as np


def euclidean_dist(x, y):
    dist = torch.cdist(x, y, p=2)
    return dist


# compression error
class CR_error(nn.Module):
    def __init__(self):
        super(CR_error, self).__init__()

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        return loss.mean()

    # try to keep distance info of all data pairs


class CR_loss_l1(nn.Module):
    def __init__(self):
        super(CR_loss_l1, self).__init__()

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)

        loss = torch.abs(xx - yy).clamp(min=1e-12)

        return loss.mean(), loss.mean()

    # try to keep distance info of all data pairs


class CR_loss_l2(nn.Module):
    def __init__(self):
        super(CR_loss_l2, self).__init__()

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        return loss.mean(), loss.mean()


# inhomogeneous neighborhood relationship preserving (INRP) loss
class CR_loss_l2_s(nn.Module):
    def __init__(self, alpha=0.1, beta=0.9, range_a=0, range_b=1):
        super(CR_loss_l2_s, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.range_a = range_a
        self.range_b = range_b

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        wx = torch.zeros_like(xx) + self.beta
        wx[xx < self.range_a] = self.alpha
        wx[xx > self.range_b] = self.alpha

        loss_w = loss * wx

        return loss_w.mean(), loss.mean()

    # INRP loss, linear


class CR_loss_l2_s_li1(nn.Module):
    def __init__(self, mean, sigma):
        super(CR_loss_l2_s_li1, self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.a = 1.0 / (3 * sigma - mean)
        self.b = 1.0

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        xxw = xx.clone().detach()
        xxw = self.a * xxw + self.b
        xxw[xxw < 0] = 0
        loss_w = loss * xxw

        return loss_w.mean(), loss.mean()


# INRP loss, linear
class CR_loss_l2_s_li2(nn.Module):
    def __init__(self, mean, sigma):
        super(CR_loss_l2_s_li2, self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.a = -1.0 / mean
        self.b = 1.0

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        xxw = xx.clone().detach()
        xxw = self.a * xxw + self.b
        xxw[xxw < 0] = 0
        loss_w = loss * xxw

        return loss_w.mean(), loss.mean()

    # INRP loss, linear


class CR_loss_l2_s_li3(nn.Module):
    def __init__(self, mean, sigma, c_factor):
        super(CR_loss_l2_s_li3, self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.scale = np.sqrt(c_factor)

        # w = ax +b
        self.a = -1.0 / mean
        self.b = 1.0

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx / self.scale - yy), 2).clamp(min=1e-12))

        xxw = xx.clone().detach()
        xxw = self.a * xxw + self.b
        xxw[xxw < 0] = 0

        loss_w = loss * xxw

        return loss_w.mean(), loss.mean()


# INRP loss, log
class CR_loss_l2_s_log1(nn.Module):
    def __init__(self, mean, sigma):
        super(CR_loss_l2_s_log1, self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.boundary = self.mean

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        xxw = xx.clone().detach()

        xxw = -torch.log(xxw / self.boundary)

        xxw[xxw > 2.0] = 2.0
        xxw[xxw < 0.01] = 0.01

        loss_w = loss * xxw

        return loss_w.mean(), loss.mean()


# INRP loss, log
class CR_loss_l2_s_log2(nn.Module):
    def __init__(self, mean, sigma):
        super(CR_loss_l2_s_log2, self).__init__()
        self.mean = mean
        self.sigma = sigma
        self.boundary = self.mean

    def forward(self, x, y):
        xx = euclidean_dist(x, x)
        yy = euclidean_dist(y, y)
        loss = torch.sqrt(torch.pow((xx - yy), 2).clamp(min=1e-12))

        xxw = xx.clone().detach()

        xxw = -torch.log(xxw / self.boundary)

        xxw[xxw > 2.0] = 2.0
        xxw[xxw < 0.0] = 0.0

        loss_w = loss * xxw

        return loss_w.mean(), loss.mean()


def create_criterion(cfg):
    loss = cfg.SOLVER.LOSS

    if loss == 'CR_loss_l1':
        criterion = CR_loss_l1()
    elif loss == 'CR_loss_l2':
        criterion = CR_loss_l2()
    elif loss == 'CR_loss_l2_s':
        criterion = CR_loss_l2_s(alpha=0.1, beta=0.9, range_a=0, range_b=cfg.DATASET.MEAN)
    elif loss == 'CR_loss_l2_s_li1':
        criterion = CR_loss_l2_s_li1(cfg.DATASET.MEAN, cfg.DATASET.SIGMA)
    elif loss == 'CR_loss_l2_s_li2':
        criterion = CR_loss_l2_s_li2(cfg.DATASET.MEAN, cfg.DATASET.SIGMA)
    elif loss == 'CR_loss_l2_s_li3':
        criterion = CR_loss_l2_s_li3(cfg.DATASET.MEAN, cfg.DATASET.SIGMA, cfg.STRUC.C_FACTOR)
    elif loss == 'CR_loss_l2_s_log1':
        criterion = CR_loss_l2_s_log1(cfg.DATASET.MEAN, cfg.DATASET.SIGMA)
    elif loss == 'CR_loss_l2_s_log2':
        criterion = CR_loss_l2_s_log2(cfg.DATASET.MEAN, cfg.DATASET.SIGMA)

    return criterion                                                 