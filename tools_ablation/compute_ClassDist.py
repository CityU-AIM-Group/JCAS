import argparse
import scipy
from scipy import ndimage
import numpy as np
import sys
import _init_paths
from collections import OrderedDict
import os
import time
from PIL import Image
import json
from os.path import join
import torch.nn as nn

import itertools
import pandas as pd
import matplotlib.pyplot as plt
plt.switch_backend('agg')

IMG_MEAN = np.array((104.00698793,116.66876762,122.67891434), dtype=np.float32)

DATA_DIRECTORY = '/home/xiaoqiguo2/OpensetNTM_instru/instrument/Endovis18/part/Noise_symmetric50/Noise_symmetric50'
DATA_LIST_PATH = '/home/xiaoqiguo2/Class2affinity/dataset/endocv_list/train.txt'
SAVE_PATH = '/home/xiaoqiguo2/Class2affinity/ClassDist'

IGNORE_LABEL = 255
NUM_CLASSES = 4

def fast_hist(a, n):
    ka = (a >= 0) & (a < n)
    return np.bincount(a[ka], minlength=n)

def per_class_iu(hist):
    return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))

def label_mapping(input, mapping):
    output = np.copy(input)
    for ind in range(len(mapping)):
        output[input == mapping[ind][0]] = mapping[ind][1]
    return np.array(output, dtype=np.int64)


def compute_CD(gt_dir, pred_dir, devkit_dir='/home/xiaoqiguo2/Class2affinity/dataset/endocv_list'):
    """
    Compute IoU given the predicted colorized images and 
    """
    num_classes = 4
    print('Num classes', num_classes)

    image_path_list = join(devkit_dir, 'train.txt')
    pred_imgs = open(image_path_list, 'r').read().splitlines()
    pred_imgs = [join(pred_dir, x.split('/')[-1]) for x in pred_imgs]

    CM = np.zeros(4)
    for ind in range(len(pred_imgs)):
        pred = np.array(Image.open(pred_imgs[ind]))
        CM += fast_hist(pred.flatten(), 4)
    return CM

if __name__ == '__main__':
    gt_dir = None
    pred_dir = '/home/xiaoqiguo2/OpensetNTM_instru/instrument/Endovis18/part/Noise_SFDA/Noise_SFDA'
    Class_dist = compute_CD(gt_dir, pred_dir)
    Class_dist_norm = Class_dist/(np.sum(Class_dist)+10e-10)
    np.save("/home/xiaoqiguo2/Class2affinity/ClassDist/ClassDist_sfda.npy", Class_dist_norm)
    print(Class_dist, Class_dist_norm)