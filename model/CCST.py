import os
import torch
import torch.nn as nn
from model.modules import Linear_BN, Linear_ABN, Glinear_BN, Glinear_ABN, Attention_module, Compression_module, Residual
import torch.distributions as tdist

def compute_params(model):
    n_params = 0
    for m in model.parameters():
        n_params += m.numel()
    return round(n_params / 1e6, 4)


# Connecting Compression Spaces with Transformer (CCST)
class CCST(torch.nn.Module):
    def __init__(self,
                 Dim=960,
                 expand = 3,
                 token_num = 12,
                 Depths=[1, 1],
                 heads = [4, 4],
                 attn_ratio=4,
                 mlp_ratio=2,
                ):
        super().__init__()

        mlp_activation = nn.Hardswish

        self.trans = nn.ModuleList()
        self.token_num = token_num
        self.token_dim = Dim * expand // token_num

        # projection part **********************************************************************************************
        # mapping input feature into multiple subspaces with multiple projections, which are initialized following
        # equations proposed in (Li, Hastie and Church 2006)

        self.MRP_tokens = nn.Sequential(
            nn.Linear(Dim, Dim * expand, bias=True),
            nn.BatchNorm1d(Dim * expand)
        )

        # s=1/density, here density is set to 1/sqrt(n_features), the minimum density recommended by Ping Li et al.
        s = torch.sqrt(torch.tensor(Dim))
        neg_v, pos_v = -torch.sqrt(s/self.token_dim), torch.sqrt(s/self.token_dim)

        proj_matrices = torch.zeros((Dim*expand, Dim))
        matrix_g = tdist.Uniform(torch.tensor([0.0]), torch.tensor([1.0]))
        prob_matrices = matrix_g.sample((Dim*expand, Dim)).squeeze(dim=2)
        proj_matrices[prob_matrices<=(1/2/s)]=neg_v
        proj_matrices[prob_matrices >= 1- (1 / 2 / s)] = pos_v

        self.MRP_tokens[0].weight.data = proj_matrices


        # global optimization part**************************************************************************************
        for i, (dep, hn) in enumerate(zip(Depths, heads)):
            stage_i = []
            for i in range(dep):
                    # attention
                    stage_i.append(
                        Residual(
                            Attention_module(token_num=token_num + 1, dim=self.token_dim, num_heads=hn,
                                             attn_ratio=attn_ratio)
                        )
                    )
                    # MLP
                    stage_i.append(
                        Residual(
                            nn.Sequential(
                                Linear_BN(self.token_dim, self.token_dim * mlp_ratio),
                                Linear_ABN(self.token_dim * mlp_ratio, self.token_dim)
                            )
                        )
                    )
            self.trans.append(nn.Sequential(*stage_i))

        # Compression part**********************************************************************************************
        # compression module
        self.cf_token = Linear_ABN(Dim, self.token_dim, a_f=mlp_activation)

        # Linear projection A
        self.skip = torch.nn.Linear(Dim, self.token_dim, bias=False)

        # Linear projection B
        self.to_out = torch.nn.Linear(self.token_dim, self.token_dim, bias=False)

        # randomly initialized compression token, which is discarded
        self.c_token = nn.Parameter(torch.randn(1, 1, self.token_dim))


    def forward(self, x):
        skip = self.skip(x)
        tokens = self.MRP_tokens(x)
        tokens = torch.stack(tokens.split(self.token_dim , dim=1), dim=1)
        cf_tokens = self.cf_token(x).unsqueeze(dim=1)
        x_tokens = torch.cat((tokens, cf_tokens), dim=1)

        for stage_i in self.trans:
            x_tokens = stage_i(x_tokens)
            x_tokens[:, -1] = x_tokens[:, -1]+skip
        token = x_tokens[:, -1]
        out = self.to_out(token+skip)

        return out


if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    batch_size=1

    input_feat = torch.randn(2, 128).cuda()
    label = torch.randn(2, 32).cuda()
    model = CCST(Dim=128,
                expand=30,
                token_num=120,
                Depths=[1, 1],
                heads=[4, 4],
                attn_ratio=4,
                mlp_ratio=2)
    model = model.cuda()
    model.eval()
    model.requires_grad_(False)
    repeat = 1000
    criterion = torch.nn.L1Loss()
    for i in range(repeat):
        output = model(input_feat)
        loss = criterion(output, label)
        # loss.backward()



