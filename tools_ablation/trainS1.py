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
BATCH_SIZE = 4
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
# RESTORE_FROM = '/home/xiaoqiguo2/Class2affinity/snapshots/Instrument_warmup_epoch80_mIoU54.514.pth'
RESTORE_FROM = '/home/xiaoqiguo2/OpensetNTM/snapshots/resnet_pretrain.pth'
SAVE_PRED_EVERY = 1
SNAPSHOT_DIR = '../snapshots/Noise_symmetric50/'
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

def Cosine_similarity(x):
    # x: bs * (hw) * d
    # try two different ways
    x_relu = nn.ReLU(inplace=True)(x)
    x_norm = x_relu/(torch.norm(x_relu, dim=2, keepdim=True) + 10e-10)
    dist = x_norm.bmm(x_norm.permute(0, 2, 1))
    return dist

def class2affinity(transition_matrix):
    v00 = v01 = v10 = v11 = 0
    t = transition_matrix
    num_classes = transition_matrix.shape[0]
    CD = np.load('/home/xiaoqiguo2/Class2affinity/ClassDist/ClassDist_symmetric50.npy')
    for i in range(num_classes):
        for j in range(num_classes):
            a = t[i][j]
            ni = CD[i]
            for m in range(num_classes):
                for n in range(num_classes):
                    b = t[m][n]
                    nm = CD[m]
                    if i == m and j == n:
                        v11 += a * b * ni * nm
                    if i == m and j != n:
                        v10 += a * b * ni * nm
                    if i != m and j == n:
                        v01 += a * b * ni * nm
                    if i != m and j != n:
                        v00 += a * b * ni * nm
    aff_T = torch.zeros([2, 2]).cuda()
    aff_T[0][0] = v11 / (v11 + v10)
    aff_T[0][1] = v10 / (v11 + v10)
    aff_T[1][0] = v01 / (v01 + v00)
    aff_T[1][1] = v00 / (v01 + v00)
    return aff_T

def Affinity_branch(feat, label_target, AT, input_size_target):
    bs, d, h, w = feat.size()
    interp_feat = nn.Upsample(size=(int(input_size_target[1]/4), int(input_size_target[0]/5)), mode='bilinear', align_corners=True)
    interp_lab = nn.Upsample(size=(int(input_size_target[1]/4), int(input_size_target[0]/5)), mode='nearest')
    feat = interp_feat(feat).permute(0, 2, 3, 1).contiguous().view(bs, -1, d)
    affinity = Cosine_similarity(feat) ## bs * hw * hw
    affinity = affinity.clamp(min=1e-10, max=1.0)

    labels = torch.where(label_target == 255*torch.ones_like(label_target), args.num_classes*torch.ones_like(label_target), label_target)
    labels = torch.eye(args.num_classes + 1)[labels.long()].float().cuda().permute(0, 3, 1, 2)
    labels = interp_lab(labels).permute(0, 2, 3, 1).contiguous().view(bs, -1, args.num_classes+1)
    affinity_map_label = Cosine_similarity(labels) ## bs * hw * hw

    loss_aff = F.binary_cross_entropy(affinity, affinity_map_label)
    return affinity, loss_aff

def Class_branch(pred, label_target, affinity_map, CT, input_size_target):
    seg_loss_woSoftmax = CrossEntropy2d(ignore_label=255, is_softmax=False).cuda()
    interp_feat = nn.Upsample(size=(int(input_size_target[1]/4), int(input_size_target[0]/5)), mode='bilinear', align_corners=True)
    interp_target = nn.Upsample(size=(input_size_target[1], input_size_target[0]), mode='bilinear', align_corners=True)

    affinity_map = F.normalize(affinity_map, p=1, dim=2)
    affinity_map_inv = F.normalize(torch.ones_like(affinity_map) - affinity_map, p=1, dim=2)

    pred = interp_feat(pred)
    # pred = F.softmax(pred, dim=1)
    bs, c, h, w = pred.size()
    predict = pred.permute(0, 2, 3, 1).contiguous().view(bs, -1, c)
    predict1 = pred + affinity_map.bmm(predict).view(bs, h, w, c).permute(0, 3, 1, 2)
    predict2 = pred - affinity_map_inv.bmm(predict).view(bs, h, w, c).permute(0, 3, 1, 2)
    predict = (predict1 + predict2) / 2.
    predict = interp_target(predict)
    predict = F.softmax(predict, dim=1)#F.normalize(predict, p=1, dim=1)
    loss_seg = seg_loss_woSoftmax(predict, label_target) 

    pred = interp_target(pred)
    bs, _, h, w = pred.size()
    pred = F.softmax(pred, dim=1)
    loss_seg += seg_loss_woSoftmax(pred, label_target) 
    
    return loss_seg

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

    Affinity_T = NTM(gpu, 2, init=2)
    optimizer_Affinity_T = optim.SGD(Affinity_T.parameters(), lr=args.learning_rate_T, weight_decay=0, momentum=0.9)
    for epoch in range(args.num_steps):
        for i, batch in enumerate(targetloader):
            model.train()

            adjust_learning_rate(optimizer, epoch)
            adjust_learning_rate(optimizer_Class_T, epoch)
            adjust_learning_rate(optimizer_Affinity_T, epoch)

            image_target, label_target, _, name = batch
            image_target = image_target.cuda()
            label_target = label_target.long().cuda()

            feat1, feat2, pred1, pred2 = model(image_target)

            #### Affinity level learning ####
            AT = Affinity_T()
            affinity1, loss_aff1 = Affinity_branch(feat1, label_target, AT, input_size_target)
            affinity2, loss_aff2 = Affinity_branch(feat2, label_target, AT, input_size_target)

            #### Class level learning ####
            CT = Class_T()
            loss_seg1 = Class_branch(pred1, label_target, affinity1, CT, input_size_target)
            loss_seg2 = Class_branch(pred2, label_target, affinity2, CT, input_size_target)

            #### joint loss ####
            vol_loss = CT.slogdet().logabsdet + AT.slogdet().logabsdet

            C2A_T = class2affinity(CT)
            class_aff_simi = torch.nn.MSELoss(reduction='mean')(C2A_T, AT)

            loss = loss_seg2 + args.lambda_seg * loss_seg1 + loss_aff1  + args.lambda_seg * loss_aff2 + 0.1 * vol_loss #+ 0.001 * class_aff_simi

            optimizer.zero_grad()
            optimizer_Class_T.zero_grad()
            optimizer_Affinity_T.zero_grad()

            loss.backward()

            optimizer.step()
            optimizer_Class_T.step()
            optimizer_Affinity_T.step()

            if (i) % 20 == 0:
                print(
                'epoch = {0:3d}/{1:3d}, iter = {2:3d}, loss_seg1 = {3:.3f} loss_seg2 = {4:.3f} loss_aff1 = {5:.3f}  loss_aff2 = {6:.3f} loss_vol = {7:.3f} loss_similar = {8:.3f}'.format(
                    epoch, args.num_steps, i, loss_seg1.data.cpu().numpy(), loss_seg2.data.cpu().numpy(), loss_aff1.data.cpu().numpy(), loss_aff2.data.cpu().numpy(), vol_loss.data.cpu().numpy(), class_aff_simi.data.cpu().numpy()))

        if epoch % args.save_pred_every == 0:# and epoch != 0:
            now = datetime.datetime.now()
            print(now.strftime("%Y-%m-%d %H:%M:%S"), '  Begin evaluation on epoch {0:8d}/{1:8d}  '.format(epoch, args.num_steps))
            mIoU = evaluate_DiceJac(model, refine=True)
            print('Finish Evaluation: '+time.asctime(time.localtime(time.time())))
            if mIoU > best_mIoU:
                old_file = osp.join(args.snapshot_dir, 'Instrument_warmup_epoch' + str(best_iter) + '_mIoU' + str(best_mIoU) + '.pth')
                if os.path.exists(old_file) is True:
                    os.remove(old_file) 
                print('Saving model with mIoU: ', mIoU)
                torch.save(model.state_dict(), osp.join(args.snapshot_dir, 'Instrument_warmup_epoch' + str(epoch) + '_mIoU' + str(mIoU) + '.pth'))
                best_mIoU = mIoU
                best_iter = epoch


if __name__ == '__main__':
    main()