from .CCST import CCST


def compute_params(model):
    n_params = 0
    for m in model.parameters():
        n_params += m.numel()
    return round(n_params / 1e6, 4)


def build_model(STRUC):
    model = CCST(
        Dim=STRUC.FEAT_DIM,
        expand=STRUC.EXPAND,
        token_num=STRUC.TOKEN_NUM,
        Depths=STRUC.DEPTH,
        heads=STRUC.HEADS,
        attn_ratio=STRUC.ATTN_RATIO,
        mlp_ratio=STRUC.MLP_RATIO
    )

    return model, compute_params(model)





