"""Configuration file (powered by YACS).
Adapted from CoTTA config file.
"""

import argparse
import os
import sys
import logging
import random
import torch
import numpy as np
from datetime import datetime
from iopath.common.file_io import g_pathmgr
from yacs.config import CfgNode as CfgNode


# Global config object (example usage: from core.config import cfg)
_C = CfgNode()
cfg = _C

# ----------------------------- Victim options ------------------------------- #
_C.VICTIM = CfgNode()

# Victim dataset
_C.VICTIM.DATASET = 'CIFAR10'

# Victim dataset root directory
_C.VICTIM.DATA_ROOT = 'data'

# Victim model architecture
_C.VICTIM.ARCH = 'cnn32'

# Path to victim's model weights
_C.VICTIM.PATH = 'ckpts/cifar10-cnn32/ckpt.pt'


# ----------------------------- Thief options ------------------------------- #
_C.THIEF = CfgNode()

# Thief dataset
_C.THIEF.DATASET = 'imagenet32'

# Thief model architecture
_C.THIEF.ARCH = ['cnn32']

# Thief dataset root directory
_C.THIEF.DATA_ROOT = ''

# Training data size
_C.THIEF.NUM_TRAIN = 1281167

_C.THIEF.SUBSET    = 1281167

# ----------------------------- Stealing Method options ------------------------------- #
_C.ACTIVE = CfgNode()

# Whether to use a pretrained model
_C.ACTIVE.USE_PRETRAINED = False

# Pretrained model path
_C.ACTIVE.PRETRAINED_PATH = None
#_C.ACTIVE.PRETRAINED_PATH = 'results/cifar10_cnn32/imagenet32_cnn32/pretrained/2batches.pth'

# Active learning method
_C.ACTIVE.METHOD = 'random'

# Active learning budget
_C.ACTIVE.BUDGET = 30000

_C.ACTIVE.STDAUG = False

_C.ACTIVE.AUGMENT = None

_C.ACTIVE.CUTMIX_PROB = 0.

_C.ACTIVE.BETA = 1.0

_C.ACTIVE.VAL = 0
_C.ACTIVE.INITIAL  = 0
_C.ACTIVE.CYCLES = 1
_C.ACTIVE.ADDENDUM = 0

_C.ACTIVE.TEMP = 1.
_C.ACTIVE.ALPHA = 0.

# ----------------------------- Training options ------------------------------- #
_C.TRAIN = CfgNode()

# Batch size
_C.TRAIN.BATCH = 256

# Optimizer
_C.TRAIN.OPTIMIZER = 'Adam'

# Learning rate
_C.TRAIN.LR = 0.001 # use this for adam

# Momentum
_C.TRAIN.MOMENTUM = 0.9

# Weight decay
_C.TRAIN.WDECAY = 5e-4

# Training epochs
_C.TRAIN.EPOCH = 500

# Milestones: epochs at which learning rate is reduced
_C.TRAIN.MILESTONES = [120]

# Factor by which learning rate is reduced at milestones
_C.TRAIN.GAMMA = 0.1


# ----------------------------- Learning Loss Options ------------------------------- #
_C.LL = CfgNode()

# Margin
_C.LL.MARGIN = 1.0 # xi

# Weight of learning prediction loss wrt usual training loss
_C.LL.WEIGHT = 1.0 # lambdas

# What exactly is this?
_C.LL.EPOCHL = 120


# ----------------------------- Miscellaneous Options ------------------------------- #

# Experiment trials to be conducted
_C.TRIALS = 5

# Experiment name
_C.METHOD_NAME = 'None'

# Output directory
_C.OUT_DIR = ""

# Note that non-determinism is still present due to non-deterministic GPU ops
_C.RNG_SEED = 1

 # uses this seed when splitting datasets
_C.DS_SEED = 123 

# Log destination (in SAVE_DIR)
_C.LOG_DEST = "log.txt"

# Log datetime
_C.LOG_TIME = ''

# Benchmark to select fastest CUDNN algorithms (best for fixed input sizes)
_C.CUDNN_BENCHMARK = True

_C.CORR_METHOD = ""

_C.FOLDER_NAME = ""

# --------------------------------- Default config -------------------------- #
_CFG_DEFAULT = _C.clone()
_CFG_DEFAULT.freeze()



def merge_from_file(cfg_file):
    with g_pathmgr.open(cfg_file, "r") as f:
        cfg = _C.load_cfg(f)
    _C.merge_from_other_cfg(cfg)


def dump_cfg():
    """Dumps the config to the output directory."""
    cfg_file = os.path.join(_C.SAVE_DIR, _C.CFG_DEST)
    with g_pathmgr.open(cfg_file, "w") as f:
        _C.dump(stream=f)


def load_cfg(out_dir, cfg_dest="config.yaml"):
    """Loads config from specified output directory."""
    cfg_file = os.path.join(out_dir, cfg_dest)
    merge_from_file(cfg_file)


def reset_cfg():
    """Reset config to initial state."""
    cfg.merge_from_other_cfg(_CFG_DEFAULT)


def budget_breakup(method, budget):
    if method=='random':
        _C.ACTIVE.CYCLES = 1 #cannot change this in main code as cfg is immutable

    cycles = _C.ACTIVE.CYCLES
    if method in ('random', 'energy', 'cutmix_entropy', 'cnc_entropy'):
        val = budget // 10
        initial = budget - val
        addendum = 0
    else:        
        val = budget // 10
        rest = budget - val
        addendum = rest // (cycles)
        initial = addendum

    _C.ACTIVE.VAL = val
    _C.ACTIVE.INITIAL  = initial
    _C.ACTIVE.ADDENDUM = addendum


def load_cfg_fom_args(description="Config options."):
    """Load config from command line args and set any specified options."""
    current_time = datetime.now().strftime("%y%m%d_%H%M%S")
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--cfg", dest="cfg_file", type=str, required=True,
                        help="Config file location")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                        help="See conf.py for all options")
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    
    merge_from_file(args.cfg_file)
    cfg.merge_from_list(args.opts)
    budget_breakup(_C.ACTIVE.METHOD, _C.ACTIVE.BUDGET)

    log_dest = os.path.basename(args.cfg_file)
    log_dest = log_dest.replace('.yaml', '_{}.txt'.format(current_time))

    cfg.SAVE_DIR = f'{_C.OUT_DIR}/{_C.VICTIM.DATASET}_{_C.VICTIM.ARCH}/{_C.THIEF.DATASET}_{"aot"}/{_C.TRAIN.OPTIMIZER}/{_C.ACTIVE.BUDGET}_val{_C.ACTIVE.VAL}/{_C.ACTIVE.METHOD}_{_C.METHOD_NAME}'
    g_pathmgr.mkdirs(cfg.SAVE_DIR)
    cfg.FOLDER_NAME = _C.FOLDER_NAME
    cfg.LOG_TIME, cfg.LOG_DEST = current_time, log_dest
    cfg.freeze()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s: %(lineno)4d]: %(message)s",
        datefmt="%y/%m/%d %H:%M:%S",
        handlers=[
            logging.FileHandler(os.path.join(cfg.SAVE_DIR, cfg.LOG_DEST)),
            logging.StreamHandler()
        ])

    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)
    random.seed(cfg.RNG_SEED)
    torch.backends.cudnn.benchmark = cfg.CUDNN_BENCHMARK

    logger = logging.getLogger(__name__)
    version = [torch.__version__, torch.version.cuda,
               torch.backends.cudnn.version()]
    logger.info(
        "PyTorch Version: torch={}, cuda={}, cudnn={}".format(*version))
    logger.info(cfg)
