import _init_paths
import argparse
import torch
import torch.nn as nn
from torch.utils import data, model_zoo
import numpy as np
import pickle
from torch.autograd import Variable
import torch.optim as optim
import scipy.misc
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import sys
import os
import os.path as osp
import random

from model.deeplab_multi import DeeplabMulti

import time
import datetime
import itertools
import pandas as pd
import _init_paths
from evaluate import evaluate_Single, evaluate_DiceJac
import matplotlib.pyplot as plt
plt.switch_backend('agg')

MODEL = 'DeepLab'
BATCH_SIZE = 8
ITER_SIZE = 1
NUM_WORKERS = 4
DATA_DIRECTORY = '/home/xiaoqiguo2/scratch/UDA_Natural/GTA5'
DATA_LIST_PATH = '../dataset/gta5_list/train.txt'
IGNORE_LABEL = 255
INPUT_SIZE = '256,256'
DATA_DIRECTORY_TARGET = '/home/xiaoqiguo2/Class2affinity/polyp/ETIS/'
DATA_LIST_PATH_TARGET = '/home/xiaoqiguo2/Class2affinity/dataset/polyp_list/ETIS.txt'
INPUT_SIZE_TARGET = '256,256'
LEARNING_RATE = 1e-4
LEARNING_RATE_T = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 4
OPEN_CLASSES = 5
NUM_STEPS = 100
NUM_STEPS_STOP = 100  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
RESTORE_FROM = '../snapshots/Noise_ellipse/Instrument_epoch157_mIoU58.452.pth'
# RESTORE_FROM = '../snapshots/Pseudo_Adaptsegnet.pth'
# RESTORE_FROM = '../snapshots/resnet_pretrain.pth'
# RESTORE_FROM = '../snapshots/DSP_best.pth'
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = '../snapshots/IGNet/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log/'
Threshold_high = 0.8
Threshold_low = 0.2
lambda_Place = 0.1
lambda_Convex = 0.5
lambda_Volume = 0.1
lambda_Anchor = 0.5

LAMBDA_SEG = 0.1
TARGET = 'cityscapes'
SET = 'train'

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="available options : DeepLab")
    parser.add_argument("--target", type=str, default=TARGET,
                        help="available options : cityscapes")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--iter-size", type=int, default=ITER_SIZE,
                        help="Accumulate gradients for ITER_SIZE iterations.")
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the source dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the source dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--input-size", type=str, default=INPUT_SIZE,
                        help="Comma-separated string with height and width of source images.")
    parser.add_argument("--data-dir-target", type=str, default=DATA_DIRECTORY_TARGET,
                        help="Path to the directory containing the target dataset.")
    parser.add_argument("--data-list-target", type=str, default=DATA_LIST_PATH_TARGET,
                        help="Path to the file listing the images in the target dataset.")
    parser.add_argument("--input-size-target", type=str, default=INPUT_SIZE_TARGET,
                        help="Comma-separated string with height and width of target images.")
    parser.add_argument("--is-training", action="store_true",
                        help="Whether to updates the running means and variances during the training.")
    parser.add_argument("--learning-rate", type=float, default=LEARNING_RATE,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--learning-rate-T", type=float, default=LEARNING_RATE_T,
                        help="Base learning rate for discriminator.")
    parser.add_argument("--lambda-seg", type=float, default=LAMBDA_SEG,
                        help="lambda_seg.")
    parser.add_argument("--Threshold-high", type=float, default=Threshold_high,
                        help="Threshold_high")
    parser.add_argument("--Threshold-low", type=float, default=Threshold_low,
                        help="Threshold_low")
    parser.add_argument("--lambda-Place", type=float, default=lambda_Place,
                        help="lambda_Place")
    parser.add_argument("--lambda-Convex", type=float, default=lambda_Convex,
                        help="lambda_Convex")
    parser.add_argument("--lambda-Volume", type=float, default=lambda_Volume,
                        help="lambda_Volume")
    parser.add_argument("--lambda-Anchor", type=float, default=lambda_Anchor,
                        help="lambda_Anchor")
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--open-classes", type=int, default=OPEN_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-steps", type=int, default=NUM_STEPS,
                        help="Number of training steps.")
    parser.add_argument("--num-steps-stop", type=int, default=NUM_STEPS_STOP,
                        help="Number of training steps for early stopping.")
    parser.add_argument("--power", type=float, default=POWER,
                        help="Decay parameter to compute the learning rate.")
    parser.add_argument("--random-mirror", action="store_true",
                        help="Whether to randomly mirror the inputs during the training.")
    parser.add_argument("--random-scale", action="store_true",
                        help="Whether to randomly scale the inputs during the training.")
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED,
                        help="Random seed to have reproducible results.")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--save-pred-every", type=int, default=SAVE_PRED_EVERY,
                        help="Save summaries and checkpoint every often.")
    parser.add_argument("--snapshot-dir", type=str, default=SNAPSHOT_DIR,
                        help="Where to save snapshots of the model.")
    parser.add_argument("--weight-decay", type=float, default=WEIGHT_DECAY,
                        help="Regularisation parameter for L2-loss.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose adaptation set.")
    parser.add_argument("--log-dir", type=str, default=LOG_DIR,
                        help="Path to the directory of log.")
    return parser.parse_args()

args = get_arguments()
# if not os.path.exists(args.log_dir):
#     os.makedirs(args.log_dir)
print('Leanring_rate: ', args.learning_rate)
print('Leanring_rate_T: ', args.learning_rate_T)
print('Open-set class: ', args.open_classes)
print('Threshold_high: ', args.Threshold_high)
print('Threshold_low: ', args.Threshold_low)
print('lambda_Place: ', args.lambda_Place)
print('lambda_Convex: ', args.lambda_Convex)
print('lambda_Volume: ', args.lambda_Volume)
print('lambda_Anchor: ', args.lambda_Anchor)
print('restore_from: ', args.restore_from)

def main():
    cudnn.enabled = True
    gpu = args.gpu

    # Create network
    pretrained_dict = torch.load(args.restore_from)
    model = DeeplabMulti(num_classes=args.num_classes, affinity=True).cuda()
    net_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)

    now = datetime.datetime.now()
    print(now.strftime("%Y-%m-%d %H:%M:%S"))
    # mIoU = evaluate_Single(model)
    mIoU = evaluate_DiceJac(model, refine=True)
    print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))

if __name__ == '__main__':
    main()
