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

from model.deeplab_multi import DeeplabMulti, sig_NTM, NTM
from utils.loss import *
from dataset.endovis_dataset import EndovisDataSet

import time
import datetime
import itertools
import pandas as pd
import _init_paths
from evaluate import evaluate_Single, evaluate_DiceJac
import matplotlib.pyplot as plt
plt.switch_backend('agg')

MODEL = 'DeepLab'
BATCH_SIZE = 3
NUM_WORKERS = 2
IGNORE_LABEL = 255
INPUT_SIZE = '320,256'
DATA_DIRECTORY_TARGET = '/home/xiaoqiguo2/OpensetNTM_instru/instrument/Endovis18/'
DATA_LIST_PATH_TARGET = '/home/xiaoqiguo2/Class2affinity/dataset/endocv_list/train.txt'
INPUT_SIZE_TARGET = '320,256'
LEARNING_RATE = 1e-4
LEARNING_RATE_T = 1e-3
MOMENTUM = 0.9
NUM_CLASSES = 4
NUM_STEPS = 200
NUM_STEPS_STOP = 200  # early stopping
POWER = 0.9
RANDOM_SEED = 1234
# RESTORE_FROM = '/home/xiaoqiguo2/Class2affinity/snapshots/Noise_ellipse/Instrument_warmup_epoch197_mIoU45.951.pth'
RESTORE_FROM = '/home/xiaoqiguo2/OpensetNTM/snapshots/resnet_pretrain.pth'
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = '../snapshots/SparseReg/'
WEIGHT_DECAY = 0.0005
LOG_DIR = './log/'

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
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS,
                        help="number of workers for multithread dataloading.")
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
    parser.add_argument("--momentum", type=float, default=MOMENTUM,
                        help="Momentum component of the optimiser.")
    parser.add_argument("--not-restore-last", action="store_true",
                        help="Whether to not restore last (FC) layers.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
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

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr * ((1 - float(iter) / max_iter) ** (power))

def adjust_learning_rate(optimizer, epoch):
    lr = lr_poly(args.learning_rate, epoch, args.num_steps, args.power)
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) > 1:
        optimizer.param_groups[1]['lr'] = lr * 10

class pNorm(nn.Module):
    def __init__(self, p=0.5):
        super(pNorm, self).__init__()
        self.p = p

    def forward(self, pred, p=None):
        if p:
            self.p = p
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1)
        norm = torch.sum(pred ** self.p, dim=1)
        return norm.mean()

def main():
    """Create the model and start the training."""
    print('Start: '+time.asctime(time.localtime(time.time())))
    best_iter = 0
    best_mIoU = 0
    mIoU = 0

    w, h = map(int, args.input_size_target.split(','))
    input_size_target = (w, h)

    cudnn.enabled = True
    gpu = args.gpu

    pretrained_dict = torch.load(args.restore_from)
    model = DeeplabMulti(num_classes=args.num_classes, affinity=True).cuda()
    net_dict = model.state_dict()
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if (k in net_dict)}
    pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in net_dict) and (v.shape==net_dict[k[6:]].shape)}
    net_dict.update(pretrained_dict)
    model.load_state_dict(net_dict)
    model.train()

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    target_dataset = EndovisDataSet(root=args.data_dir_target,
                                list_path=args.data_list_target,
                                mirror_prob=0.5,
                                crop_size=input_size_target,
                                ignore_label=255,
                                pseudo_label=True #False means true label
                                )
    targetloader = data.DataLoader(target_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                shuffle=True,
                                pin_memory=True)

    optimizer = optim.SGD(model.optim_parameters(args),
                          lr=args.learning_rate, momentum=args.momentum, weight_decay=args.weight_decay)

    Class_T = NTM(gpu, args.num_classes, init=2)
    optimizer_Class_T = optim.SGD(Class_T.parameters(), lr=args.learning_rate_T, weight_decay=0, momentum=0.9)

    seg_loss = torch.nn.CrossEntropyLoss(ignore_index=255)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)
    for epoch in range(args.num_steps):
        for i, batch in enumerate(targetloader):
            model.train()

            adjust_learning_rate(optimizer, epoch)
            adjust_learning_rate(optimizer_Class_T, epoch)

            image_target, label_target, _, name = batch
            image_target = image_target.cuda()
            label_target = label_target.long().cuda()

            _, _, pred1, pred2 = model(image_target)
            pred1 = interp_target(pred1)
            pred2 = interp_target(pred2)

            tau = 0.5
            lamb = 1
            pred1 = F.normalize(pred1, dim=1)
            pred2 = F.normalize(pred2, dim=1)
            loss_seg1 = seg_loss(pred1 / tau, label_target) + lamb * pNorm(p=0.01)(pred1 / tau)
            loss_seg2 = seg_loss(pred2 / tau, label_target) + lamb * pNorm(p=0.01)(pred2 / tau)

            loss = loss_seg2 + args.lambda_seg * loss_seg1

            optimizer.zero_grad()
            optimizer_Class_T.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_Class_T.step()

            if (i) % 20 == 0:
                print(
                'epoch = {0:3d}/{1:3d}, iter = {2:3d}, loss_seg1 = {3:.3f} loss_seg2 = {4:.3f}'.format(
                    epoch, args.num_steps, i, loss_seg1.data.cpu().numpy(), loss_seg2.data.cpu().numpy()))

        if epoch % args.save_pred_every == 0:# and epoch != 0:
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on epoch {0:8d}/{1:8d}  '.format(epoch, args.num_steps))
            mIoU = evaluate_DiceJac(model, refine=True)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                old_file = osp.join(args.snapshot_dir, 'Instrument_epoch' + str(best_iter) + '_mIoU' + str(best_mIoU) + '.pth')
                if os.path.exists(old_file) is True:
                    os.remove(old_file) 
                print('Saving model with mIoU: ', mIoU)
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Instrument_epoch' + str(epoch) + '_mIoU' + str(mIoU) + '.pth'))
                best_mIoU = mIoU
                best_iter = epoch


if __name__ == '__main__':
    main()