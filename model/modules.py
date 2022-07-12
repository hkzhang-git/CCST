import torch
import torch.nn as nn


class Residual(torch.nn.Module):
    def __init__(self, func, drop=0):
        super().__init__()
        self.func = func
        self.drop = drop

    def forward(self, x):
        if self.training and self.drop > 0:
            return x + self.func(x) * torch.rand(x.size(0), 1, 1,
                                              device=x.device).ge_(self.drop).div(1 - self.drop).detach()
        else:
            return x + self.func(x)


class Pre_proj(torch.nn.Sequential):
    def __init__(self, Dim, Token_n=16, Token_dim=480,):
        super().__init__()
        self.Dim_in = Dim
        self.Token_n = Token_n
        self.Dim_out = Token_n*Token_dim
        self.proj = nn.Linear(self.Dim_in, self.Dim_out, bias=True)
        self.norm = nn.BatchNorm1d(self.Dim_out)
    def forward(self, input):
        out = self.proj(input)
        out = self.norm(out)
        B, D = out.shape
        tokens = out.reshape(B, self.Token_n, D//self.Token_n)
        return tokens


class Linear_BN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)


    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        if len(x.size())==3:
            return bn(x.flatten(0, 1)).reshape_as(x)
        elif len(x.size())==2:
            return bn(x)

class Linear_ABN(torch.nn.Sequential):
    def __init__(self, a, b, bn_weight_init=1, a_f = nn.Hardswish):
        super().__init__()
        self.add_module('c', torch.nn.Linear(a, b, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('a_f', a_f())
        self.add_module('bn', bn)


    def forward(self, x):
        l, a_f, bn = self._modules.values()
        x = l(x)
        x = a_f(x)
        if len(x.size())==3:
            return bn(x.flatten(0, 1)).reshape_as(x)
        elif len(x.size())==2:
            return bn(x)


class Glinear_BN(torch.nn.Sequential):
    def __init__(self, a, b, group_num=1, bn_weight_init=1):
        super().__init__()
        self.add_module('c', torch.nn.Conv1d(a, b, kernel_size=1, groups=group_num, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('bn', bn)

    def forward(self, x):
        l, bn = self._modules.values()
        x = l(x)
        return bn(x.squeeze(dim=2)).reshape_as(x)


class Glinear_ABN(torch.nn.Sequential):
    def __init__(self, a, b, group_num=1, bn_weight_init=1, a_f = nn.Hardswish):
        super().__init__()
        self.add_module('c', torch.nn.Conv1d(a, b, kernel_size=1, groups=group_num, bias=False))
        bn = torch.nn.BatchNorm1d(b)
        torch.nn.init.constant_(bn.weight, bn_weight_init)
        torch.nn.init.constant_(bn.bias, 0)
        self.add_module('a_f', a_f())
        self.add_module('bn', bn)

    def forward(self, x):
        l, a_f, bn = self._modules.values()
        x = l(x)
        x = a_f(x)
        return bn(x.squeeze(dim=2)).reshape_as(x)



class Attention_module(torch.nn.Module):
    def __init__(self, token_num = 16, dim=64,  num_heads=8, attn_ratio=4, activation=nn.Hardtanh):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.randn(num_heads, token_num, token_num))
        self.num_heads = num_heads
        kq_dim = dim//attn_ratio
        v_dim = dim
        self.scale = kq_dim ** -0.5
        self.kq_dim = kq_dim
        self.v_dim = v_dim
        self.nh_kq = kq_dim * num_heads
        self.nh_v = v_dim * num_heads
        h = self.nh_kq*2 + self.nh_v
        self.qkv = Linear_BN(dim, h)
        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.nh_v, dim, bn_weight_init=0))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.kq_dim, self.kq_dim, self.v_dim], dim=3)
        q = q.permute(0, 2, 1, 3)
        k = k.permute(0, 2, 1, 3)
        v = v.permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.pos_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.nh_v)
        x = self.proj(x)
        return x



class Compression_module(torch.nn.Module):
    def __init__(self, token_num, dim_in, dim_out, num_heads=8, attn_ratio=2, activation=nn.Hardtanh):
        super().__init__()
        self.pos_bias = nn.Parameter(torch.randn(num_heads, token_num, token_num))
        self.num_heads = num_heads
        self.scale = dim_out ** -0.5
        self.kq_dim = dim_out // attn_ratio
        self.nh_kq = self.kq_dim * num_heads
        self.v_dim = dim_out
        self.nh_v = self.v_dim * num_heads
        h = self.nh_kq * 2 + self.nh_v
        self.qkv = Linear_BN(dim_in, h)

        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.nh_v, dim_out))

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.view(B, N, self.num_heads, -
                           1).split([self.kq_dim, self.kq_dim, self.v_dim], dim=3)

        q = q.permute(0, 2, 1, 3)  # BHNC
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC

        attn = (q @ k.transpose(-2, -1)) * self.scale + self.pos_bias
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, self.nh_v)
        x = self.proj(x)
        return x


# compress both dim and number of input tokens in a fator of 2
class Compression_module_t(torch.nn.Module):
    def __init__(self, in_dim, out_dim, key_dim, num_heads=8, attn_ratio=2, activation=None):
        super().__init__()
        self.num_heads = num_heads
        self.scale = key_dim ** -0.5
        self.key_dim = key_dim
        self.nh_kd = nh_kd = key_dim * num_heads
        self.d = int(attn_ratio * key_dim)
        self.dh = int(attn_ratio * key_dim) * self.num_heads
        self.attn_ratio = attn_ratio
        h = self.dh + nh_kd
        self.qkv = Linear_BN(in_dim, h)

        self.proj = torch.nn.Sequential(activation(), Linear_BN(
            self.dh, out_dim))

    def forward(self, x):
        B, N, C = x.shape
        k, v = self.kv(x).view(B, N, self.num_heads, -
        1).split([self.key_dim, self.d], dim=3)
        k = k.permute(0, 2, 1, 3)  # BHNC
        v = v.permute(0, 2, 1, 3)  # BHNC
        q = self.q(x).view(B, self.resolution_2, self.num_heads,
                           self.key_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale + \
               (self.attention_biases[:, self.attention_bias_idxs]
                if self.training else self.ab)
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, -1, self.dh)
        x = self.proj(x)
        return x