from yacs.config import CfgNode as CN
_C = CN()

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.DATA_ROOT = "/home/disk/data/ANNS"
_C.DATASET.DATA_NAME = 'GIST1M'
_C.DATASET.FEAT_NUM = 1e+06
_C.DATASET.MEAN = 41.5257
_C.DATASET.SIGMA = 10.2938


# -----------------------------------------------------------------------------
# STRUCTURE
# -----------------------------------------------------------------------------
_C.STRUC = CN()

# basic
_C.STRUC.MODEL = 'CCST'
_C.STRUC.FEAT_DIM = 960
_C.STRUC.WIDTHS = [512, 256, 128]
_C.STRUC.DEPTH = [2, 3, 3, 1]

# transformer
_C.STRUC.EXPAND = 3
_C.STRUC.TOKEN_NUM = 12
_C.STRUC.HEADS = [4, 6]
_C.STRUC.ATTN_RATIO = 4
_C.STRUC.MLP_RATIO = 2
_C.STRUC.C_FACTOR = 4


# -----------------------------------------------------------------------------
# DataLoader
# -----------------------------------------------------------------------------
_C.DATALOADER = CN()
_C.DATALOADER.NUM_WORKERS = 4
_C.DATALOADER.BATCH_SIZE_TRAIN = 2
_C.DATALOADER.BATCH_SIZE_TEST = 2
_C.DATALOADER.DATA_AUG = ['rand_drop']

# -----------------------------------------------------------------------------
# Solver
# -----------------------------------------------------------------------------
_C.SOLVER = CN()
_C.SOLVER.OPTIM = 'adamw'
_C.SOLVER.VISUAL = True
_C.SOLVER.INIT_LR = 0.05
_C.SOLVER.POWER = 0.9
_C.SOLVER.MAX_EPOCH = 160
_C.SOLVER.VAL_PORTION = 0.05
_C.SOLVER.SCHEDULER = 'poly' # 'poly', 'multistep', 'none'
_C.SOLVER.VALIDATE_PERIOD = 10
_C.SOLVER.ALPHA = 0.1
_C.SOLVER.BETA = 0.8
_C.SOLVER.LOSS = 'CR_loss_l1'



# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
_C.OUTPUT_DIR = "."

