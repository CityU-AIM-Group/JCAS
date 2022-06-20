import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys

import torch
from torch.autograd import Variable
import torchvision.models as models
import torch.nn.functional as F
from torch.utils import data, model_zoo
from model.deeplab_multi import DeeplabMulti
from dataset.endovis_dataset import EndovisDataSet
from collections import OrderedDict
import os
from PIL import Image
import json
from os.path import join
import matplotlib.pyplot as plt
import torch.nn as nn

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/xiaoqiguo2/OpensetNTM_instru/instrument/Endovis18/'
DATA_LIST_PATH = '/home/xiaoqiguo2/Class2affinity/dataset/endocv_list/val.txt'
SAVE_PATH = '../result/cityscapes'

IGNORE_LABEL = 255
NUM_CLASSES = 4
NUM_STEPS = 500 # Number of images in the validation set.
RESTORE_FROM = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_multi-ed35151c.pth'
RESTORE_FROM_VGG = 'http://vllab.ucmerced.edu/ytsai/CVPR18/GTA2Cityscapes_vgg-ac4ac9f6.pth'
RESTORE_FROM_ORC = 'http://vllab1.ucmerced.edu/~whung/adaptSeg/cityscapes_oracle-b7b9934.pth'
SET = 'val'

MODEL = 'DeeplabMulti'

palette = [0, 0, 0, 0, 137, 255, 255, 165, 0, 255, 156, 201]
zero_pad = 256 * 3 - len(palette)
for i in range(zero_pad):
    palette.append(0)


def colorize_mask(mask):
    # mask: numpy array of the mask
    new_mask = Image.fromarray(mask.astype(np.uint8)).convert('P')
    new_mask.putpalette(palette)

    return new_mask

def get_arguments():
    """Parse all the arguments provided from the CLI.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="DeepLab-ResNet Network")
    parser.add_argument("--model", type=str, default=MODEL,
                        help="Model Choice (DeeplabMulti/DeeplabVGG/Oracle).")
    parser.add_argument("--data-dir", type=str, default=DATA_DIRECTORY,
                        help="Path to the directory containing the Cityscapes dataset.")
    parser.add_argument("--data-list", type=str, default=DATA_LIST_PATH,
                        help="Path to the file listing the images in the dataset.")
    parser.add_argument("--ignore-label", type=int, default=IGNORE_LABEL,
                        help="The index of the label to ignore during the training.")
    parser.add_argument("--num-classes", type=int, default=NUM_CLASSES,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--restore-from", type=str, default=RESTORE_FROM,
                        help="Where restore model parameters from.")
    parser.add_argument("--gpu", type=int, default=0,
                        help="choose gpu device.")
    parser.add_argument("--set", type=str, default=SET,
                        help="choose evaluation set.")
    parser.add_argument("--save", type=str, default=SAVE_PATH,
                        help="Path to save result.")
    return parser.parse_args()

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k], minlength=n ** 2).reshape(n, n)


def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))


def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def evaluate_Single(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/OpensetNTM_instru/dataset/endocv_list', post=False):
    """Create the model and start the evaluation process."""

    # if not os.path.exists(pred_dir):
    #     os.makedirs(pred_dir)
    device = torch.device("cuda")

    eval_dataset = EndovisDataSet(root=DATA_DIRECTORY,
                                    list_path=DATA_LIST_PATH,
                                    mirror_prob=0,
                                    crop_size=(320, 256),
                                    mean=np.array([93.97403134, 88.57638238, 119.21115404], dtype=np.float32),
                                    std=np.array([52.62953975, 50.02263679, 53.63186511], dtype=np.float32),
                                    ignore_label=255
                                    )
    eval_loader = data.DataLoader(eval_dataset,
                                    batch_size=1,
                                    num_workers=1,
                                    shuffle=False,
                                    pin_memory=True)

    interp = nn.Upsample(size=(256, 320), mode='bilinear', align_corners=True)
    print('Evaluate for testing data')

    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    hist = np.zeros((num_classes, num_classes))

    seg_model.eval()
    with torch.no_grad():
        for index, batch in enumerate(eval_loader):
            image, label, _, name = batch
            image = image.to(device)

            output1, output2 = seg_model(image)
            output = interp(output2).cpu().data[0].numpy()
            del output1
            del output2

            output = np.asarray(np.argmax(output, axis=0))
            label = torch.squeeze(label).cpu().numpy()
            # print(output.shape, label.shape)

            if len(label.flatten()) != len(output.flatten()):
                print('Skipping: len(gt) = {:d}, len(pred) = {:d}, {:s}, {:s}'.format(len(label.flatten()), len(pred.flatten()), gt_imgs[ind], pred_imgs[ind]))
                continue
            hist += fast_hist(label.flatten(), output.flatten(), num_classes)

            # output_col = colorize_mask(np.uint8(output))
            # # label_col = colorize_mask(np.uint8(label))
            # output = Image.fromarray(np.uint8(output))
            # name = name[0].split('/')[-1]
            # pred_dir = '/home/xiaoqiguo2/OpensetNTM_instru/results_SimT'
            # # output.save('%s/%s' % (pred_dir, name))
            # output_col.save('%s/%s_color.png' % (pred_dir, name.split('.')[0]))

    mIoUs = per_class_iu(hist)
    for ind_class in range(num_classes):
        print('===>' + name_classes[ind_class] + ':\t' + str(round(mIoUs[ind_class] * 100, 2)))
    print('===> mIoU: ' + str(round(np.nanmean(mIoUs[1:]) * 100, 2)))
    return round(np.nanmean(mIoUs[1:]) * 100, 2)

def Cosine_similarity(x):
    # x: bs * (hw) * d
    x_relu = nn.ReLU(inplace=True)(x)
    x_norm = x_relu/(torch.norm(x_relu, dim=2, keepdim=True) + 10e-10)
    dist = x_norm.bmm(x_norm.permute(0, 2, 1))
    return dist

def evaluate_DiceJac(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/OpensetNTM_instru/dataset/endocv_list', post=False, refine=False):
    """Create the model and start the evaluation process."""

    # if not os.path.exists(pred_dir):
    #     os.makedirs(pred_dir)
    device = torch.device("cuda")

    eval_dataset = EndovisDataSet(root=DATA_DIRECTORY,
                                    list_path=DATA_LIST_PATH,
                                    mirror_prob=0,
                                    crop_size=(320, 256),
                                    mean=np.array([93.97403134, 88.57638238, 119.21115404], dtype=np.float32),
                                    std=np.array([52.62953975, 50.02263679, 53.63186511], dtype=np.float32),
                                    ignore_label=255
                                    )
    eval_loader = data.DataLoader(eval_dataset,
                                    batch_size=1,
                                    num_workers=1,
                                    shuffle=False,
                                    pin_memory=True)

    interp = nn.Upsample(size=(256, 320), mode='bilinear', align_corners=True)
    print('Evaluate for testing data')

    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    hist = np.zeros((num_classes, num_classes))

    Shaft_dice_array = 0.
    Shaft_jac_array = 0.
    Shaft_sample_num = 0.
    Wrist_dice_array = 0.
    Wrist_jac_array = 0.
    Wrist_sample_num = 0.
    Clasper_dice_array = 0.
    Clasper_jac_array = 0.
    Clasper_sample_num = 0.
    seg_model.eval()
    with torch.no_grad():
        for index, batch in enumerate(eval_loader):
            image, label, _, name = batch
            image = image.to(device)

            pred = seg_model(image)
            if len(pred) == 2:
                output = interp(pred[1]).cpu().data.numpy()
                del pred
            else:
                if refine:
                    interp_feat = nn.Upsample(size=(32, 32), mode='bilinear', align_corners=True)
                    bs, d, h, w = pred[1].size()
                    feat = interp_feat(pred[1]).permute(0, 2, 3, 1).contiguous().view(bs, -1, d)
                    affinity_map = Cosine_similarity(feat) ## bs * hw * hw
                    affinity_map = affinity_map.clamp(min=1e-10, max=1.0)
                    affinity_map = F.normalize(affinity_map, p=1, dim=2)
                    affinity_map_inv = F.normalize(torch.ones_like(affinity_map) - affinity_map, p=1, dim=2)
                    p = interp_feat(pred[3])
                    # p = F.softmax(p, dim=1)
                    bs, c, h, w = p.size()
                    predict = p.permute(0, 2, 3, 1).contiguous().view(bs, -1, c)
                    predict1 = p + affinity_map.bmm(predict).view(bs, h, w, c).permute(0, 3, 1, 2)
                    predict2 = p - affinity_map_inv.bmm(predict).view(bs, h, w, c).permute(0, 3, 1, 2)
                    predict = (predict1 + predict2) / 2.
                    output = interp(F.softmax(predict, dim=1)).cpu().data.numpy()
                    output += interp(F.softmax(p, dim=1)).cpu().data.numpy()

                    interp_lab = nn.Upsample(size=(32, 32), mode='nearest')
                    labels = torch.where(label == 255*torch.ones_like(label), 4*torch.ones_like(label), label)
                    labels = torch.eye(4 + 1)[labels.long()].float().cuda().permute(0, 3, 1, 2)
                    labels = interp_lab(labels).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4 + 1)
                    affinity_map_label = Cosine_similarity(labels) ## bs * hw * hw
                    feat = interp_feat(pred[1])
                    feat = nn.ReLU(inplace=True)(feat)
                    feat = torch.mean(feat, dim=1)
                    feat = feat.cpu().data.numpy()
                    affinity_map1 = affinity_map.cpu().data.numpy()
                    affinity_map_inv1 = affinity_map_inv.cpu().data.numpy()
                    affinity_map_label1 = affinity_map_label.cpu().data.numpy()
                    del affinity_map_label

                    del pred
                    del p
                    del predict
                    del predict1
                    del predict2
                    del affinity_map
                    del affinity_map_inv
                else:
                    output = interp(pred[3]).cpu().data.numpy()
                    del pred

            # del output1
            # del output2

            output = np.asarray(np.argmax(output, axis=1))
            label = label.cpu().numpy()

            for i in range(output.shape[0]):
                p = output[i,:,:]           
                l = label[i,:,:]
                predict = np.int32(p)
                gt = np.int32(l)
                cal = gt<255
                mask = (predict==gt) * cal  

                # output_col = colorize_mask(np.uint8(p))
                # name = name[i].split('/')[-1]
                # pred_dir = '/home/xiaoqiguo2/Class2affinity/results'
                # output_col.save('%s/%s' % (pred_dir, name))

                P = np.sum((predict==1)).astype(np.float64)
                T = np.sum((gt==1)).astype(np.float64)
                TP = np.sum((gt==1)*(predict==1)).astype(np.float64)
                TN = np.sum((gt!=1)*(predict!=1)).astype(np.float64)
                if T!= 0: 
                    DICE = 2*TP/(T+P)
                    IoU = TP/(T+P-TP)
                    Shaft_dice_array += DICE
                    Shaft_jac_array += IoU
                    Shaft_sample_num += 1
 
                P = np.sum((predict==2)).astype(np.float64)
                T = np.sum((gt==2)).astype(np.float64)
                TP = np.sum((gt==2)*(predict==2)).astype(np.float64)
                TN = np.sum((gt!=2)*(predict!=2)).astype(np.float64)
                if T!= 0: 
                    DICE = 2*TP/(T+P)
                    IoU = TP/(T+P-TP)
                    Wrist_dice_array += DICE
                    Wrist_jac_array += IoU
                    Wrist_sample_num += 1

                P = np.sum((predict==3)).astype(np.float64)
                T = np.sum((gt==3)).astype(np.float64)
                TP = np.sum((gt==3)*(predict==3)).astype(np.float64)
                TN = np.sum((gt!=3)*(predict!=3)).astype(np.float64)
                if T!= 0:
                    DICE = 2*TP/(T+P)
                    IoU = TP/(T+P-TP)
                    Clasper_dice_array += DICE
                    Clasper_jac_array += IoU
                    Clasper_sample_num += 1

        Shaft_dice_array = Shaft_dice_array*100/Shaft_sample_num
        Shaft_jac_array = Shaft_jac_array*100/Shaft_sample_num
        Wrist_dice_array = Wrist_dice_array*100/Wrist_sample_num
        Wrist_jac_array = Wrist_jac_array*100/Wrist_sample_num
        Clasper_dice_array = Clasper_dice_array*100/Clasper_sample_num
        Clasper_jac_array = Clasper_jac_array*100/Clasper_sample_num
        mDice = (Shaft_dice_array + Wrist_dice_array + Clasper_dice_array)/ 3.
        mJac = (Shaft_jac_array + Wrist_jac_array + Clasper_jac_array)/ 3.
        print('%10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%   %10s:%7.3f%%'%('Shaft_dice',
            Shaft_dice_array,'Shaft_jac',Shaft_jac_array,'Wrist_dice',Wrist_dice_array,'Wrist_jac',Wrist_jac_array,'Clasper_dice',Clasper_dice_array,'Clasper_jac',Clasper_jac_array))
        print('%10s:%7.3f%%   %10s:%7.3f%%\n'%('mDice',round(mDice, 3),'mJac',round(mJac, 3)))
    return round(mJac, 3)
           

def evaluate_DiceJac_vis(seg_model, pred_dir=None, devkit_dir='/home/xiaoqiguo2/OpensetNTM_instru/dataset/endocv_list', post=False, refine=False):
    """Create the model and start the evaluation process."""

    # if not os.path.exists(pred_dir):
    #     os.makedirs(pred_dir)
    device = torch.device("cuda")

    eval_dataset = EndovisDataSet(root=DATA_DIRECTORY,
                                    list_path=DATA_LIST_PATH,
                                    mirror_prob=0,
                                    crop_size=(320, 256),
                                    mean=np.array([93.97403134, 88.57638238, 119.21115404], dtype=np.float32),
                                    std=np.array([52.62953975, 50.02263679, 53.63186511], dtype=np.float32),
                                    ignore_label=255
                                    )
    eval_loader = data.DataLoader(eval_dataset,
                                    batch_size=1,
                                    num_workers=1,
                                    shuffle=False,
                                    pin_memory=True)

    interp = nn.Upsample(size=(256, 320), mode='bilinear', align_corners=True)
    print('Evaluate for testing data')

    with open(join(devkit_dir, 'info.json'), 'r') as fp:
        info = json.load(fp)
    num_classes = np.int(info['classes'])
    name_classes = np.array(info['label'], dtype=np.str)
    hist = np.zeros((num_classes, num_classes))
    seg_model.eval()
    with torch.no_grad():
        for index, batch in enumerate(eval_loader):
            image, label, _, name = batch
            image = image.to(device)

            pred = seg_model(image)
            if len(pred) == 2:
                output = interp(pred[1]).cpu().data.numpy()
                del pred
            else:
                if refine:
                    interp_lab = nn.Upsample(size=(64, 64), mode='nearest')
                    labels = torch.squeeze(interp_lab(torch.unsqueeze(label, 1)), 1)
                    bs, h, w = labels.size()
                    labels = torch.where(labels == 255*torch.ones_like(labels), 4*torch.ones_like(labels), labels).contiguous().view(bs, -1)
                    labels, indices = torch.sort(labels, dim=1, descending=False)
                    labels = labels.contiguous().view(bs, h, w)
                    labels = torch.eye(4 + 1)[labels.long()].float().cuda().permute(0, 3, 1, 2)
                    labels = interp_lab(labels).permute(0, 2, 3, 1).contiguous().view(bs, -1, 4 + 1)
                    # labels, indices = torch.sort(labels, descending=True)  
                    affinity_map_label = Cosine_similarity(labels) ## bs * hw * hw

                    interp_feat = nn.Upsample(size=(64, 64), mode='bilinear', align_corners=True)
                    bs, d, h, w = pred[1].size()
                    feat = interp_feat(pred[1]).permute(0, 2, 3, 1).contiguous().view(bs, -1, d)
                    feat = torch.index_select(feat, 1, indices[0].cuda()) 

                    affinity_map = Cosine_similarity(feat) ## bs * hw * hw
                    affinity_map = affinity_map.clamp(min=1e-10, max=1.0)
                    affinity_map = F.normalize(affinity_map, p=1, dim=2)
                    affinity_map_inv = F.normalize(torch.ones_like(affinity_map) - affinity_map, p=1, dim=2)
                    p = interp_feat(pred[3])
                    # p = F.softmax(p, dim=1)
                    bs, c, h, w = p.size()
                    predict = p.permute(0, 2, 3, 1).contiguous().view(bs, -1, c)
                    predict1 = p + affinity_map.bmm(predict).view(bs, h, w, c).permute(0, 3, 1, 2)
                    predict2 = p - affinity_map_inv.bmm(predict).view(bs, h, w, c).permute(0, 3, 1, 2)
                    predict = (predict1 + predict2) / 2.
                    output = interp(F.softmax(predict, dim=1)).cpu().data.numpy()
                    output += interp(F.softmax(p, dim=1)).cpu().data.numpy()

                    feat = interp_feat(pred[1])
                    feat = nn.ReLU(inplace=True)(feat)
                    feat = torch.mean(feat, dim=1)
                    feat = feat.cpu().data.numpy()
                    affinity_map1 = affinity_map.cpu().data.numpy()
                    affinity_map_inv1 = affinity_map_inv.cpu().data.numpy()
                    affinity_map_label1 = affinity_map_label.cpu().data.numpy()
                    del affinity_map_label

                    del pred
                    del p
                    del predict
                    del predict1
                    del predict2
                    del affinity_map
                    del affinity_map_inv
                else:
                    output = interp(pred[3]).cpu().data.numpy()
                    del pred

            # del output1
            # del output2

            output = np.asarray(np.argmax(output, axis=1))
            label = label.cpu().numpy()

            for i in range(output.shape[0]):
                p = output[i,:,:]           
                l = label[i,:,:]
                predict = np.int32(p)
                gt = np.int32(l)
                cal = gt<255
                mask = (predict==gt) * cal  

                name = name[i].split('/')[-1]
                import cv2
                if 'seq_2_frame025' in name:
                    feature = feat[i,:,:]
                    affinity = affinity_map1[i,:,:]
                    affinity = (affinity-affinity.min()) / (affinity.max()-affinity.min())
                    affinity_inv = affinity_map_inv1[i,:,:]
                    affinity_inv = (affinity_inv-affinity_inv.min()) / (affinity_inv.max()-affinity_inv.min())
                    affinity_label = affinity_map_label1[i,:,:]
                    print(np.unique(feature), np.unique(affinity), np.unique(affinity_inv), np.unique(affinity_label))
                    print(feature.shape, affinity.shape, affinity_inv.shape)
                    cv2.imwrite('/home/xiaoqiguo2/Class2affinity/results_feat/feat.png', cv2.applyColorMap(np.uint8(255. - 255. * feature), cv2.COLORMAP_JET))
                    cv2.imwrite('/home/xiaoqiguo2/Class2affinity/results_feat/affinity.png', cv2.applyColorMap(np.uint8(155.*affinity + 50), cv2.COLORMAP_SUMMER))
                    cv2.imwrite('/home/xiaoqiguo2/Class2affinity/results_feat/affinity_inv.png', cv2.applyColorMap(np.uint8(155.*affinity_inv + 50), cv2.COLORMAP_SUMMER))
                    cv2.imwrite('/home/xiaoqiguo2/Class2affinity/results_feat/affinity_lbl.png', cv2.applyColorMap(np.uint8(95.*affinity_label + 80), cv2.COLORMAP_SUMMER))
                    xiaoqing