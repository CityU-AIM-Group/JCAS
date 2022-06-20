import os
import os.path as osp
import numpy as np
import random
import matplotlib.pyplot as plt
import torch
import torchvision
from torch.utils import data
from PIL import Image

class EndovisDataSet(data.Dataset):
    def __init__(self, root, list_path, mirror_prob=0, max_iters=None, crop_size=(321, 321), mean=(0, 0, 0),
            std=(1, 1, 1), ignore_label=255, mapping=None, pseudo_label=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.mirror_prob = mirror_prob
        self.mapping = mapping
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s" % name)
            if pseudo_label:
                label_file = osp.join(self.root, "part/Noise_ellipse/Noise_ellipse/%s" % name) #Noise_ellipse, Noise_symmetric50, Noise_SFDA
            else:
                label_file = osp.join(self.root, "part/part_lbl/%s" % name)
            self.files.append({
                "img": img_file,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        # flip
        if np.random.rand(1) < self.mirror_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.asarray(image, np.float32)
        label = np.asarray(label, np.float32)

        # if self.mapping:
        #     for src, trg in enumerate(self.mapping):
        #         label[label == src] = trg

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= 128
        image = image / 128
        image = image.transpose((2, 0, 1))

        return image.copy(), label.copy(), np.array(size), name

class EndovisDataSet1(data.Dataset):
    def __init__(self, root, list_path, mirror_prob=0, max_iters=None, crop_size=(321, 321), mean=(0, 0, 0),
            std=(1, 1, 1), ignore_label=255, mapping=None, pseudo_label=False):
        self.root = root
        self.list_path = list_path
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.mirror_prob = mirror_prob
        self.mapping = mapping
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.img_ids = [i_id.strip() for i_id in open(list_path)]
        if not max_iters==None:
            self.img_ids = self.img_ids * int(np.ceil(float(max_iters) / len(self.img_ids)))
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name in self.img_ids:
            img_file = osp.join(self.root, "img/%s" % name)
            img_file1 = osp.join(self.root, "source_like/%s" % name)
            if pseudo_label:
                # label_file = osp.join(self.root, "Pseudo4SimT/%s" % name)
                label_file = osp.join('/home/xiaoqiguo2/OpensetNTM_instru/instrument/Endovis18_pseudo/pseudo_shot/%s' % name)
            else:
                label_file = osp.join(self.root, "type_lbl/%s" % name)
            self.files.append({
                "img": img_file,
                "img1": img_file1,
                "label": label_file,
                "name": name
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        image = Image.open(datafiles["img"]).convert('RGB')
        image_source = Image.open(datafiles["img1"]).convert('RGB')
        label = Image.open(datafiles["label"])
        name = datafiles["name"]

        # resize
        image = image.resize(self.crop_size, Image.BICUBIC)
        image_source = image_source.resize(self.crop_size, Image.BICUBIC)
        label = label.resize(self.crop_size, Image.NEAREST)

        # flip
        if np.random.rand(1) < self.mirror_prob:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            image_source = image_source.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)

        image = np.asarray(image, np.float32)
        image_source = np.asarray(image_source, np.float32)
        label = np.asarray(label, np.float32)

        # if self.mapping:
        #     for src, trg in enumerate(self.mapping):
        #         label[label == src] = trg

        size = image.shape
        image = image[:, :, ::-1]  # change to BGR
        image -= 128
        image = image / 128
        image = image.transpose((2, 0, 1))

        image_source = image_source[:, :, ::-1]  # change to BGR
        image_source -= 128
        image_source = image_source / 128
        image_source = image_source.transpose((2, 0, 1))

        return image.copy(), image_source.copy(), label.copy(), name


class Split_Polyp(data.Dataset):
    def __init__(self, root=None, mirror_prob=0, max_iters=None, crop_size=(321, 321), mean=(0, 0, 0),
            std=(1, 1, 1), ignore_label=255, mapping=None, pseudo_label=False):
        self.root = '/home/xiaoqiguo2/OpensetNTM_instru/instrument/Endovis18/'
        self.crop_size = crop_size
        self.ignore_label = ignore_label
        self.mean = mean
        self.std = std
        self.mirror_prob = mirror_prob
        self.mapping = mapping
        self.easy_list_path = '/home/xiaoqiguo2/OpensetNTM_instru/dataset/endocv_list/easy_split.txt'
        self.hard_list_path = '/home/xiaoqiguo2/OpensetNTM_instru/dataset/endocv_list/hard_split.txt'
        # self.mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
        self.easy_img_ids = [i_id.strip() for i_id in open(self.easy_list_path)]
        self.hard_img_ids = [i_id.strip() for i_id in open(self.hard_list_path)]
        if not max_iters==None:
            self.easy_img_ids = self.easy_img_ids * int(np.ceil(float(max_iters) / len(self.easy_img_ids)))
            self.hard_img_ids = self.hard_img_ids * int(np.ceil(float(max_iters) / len(self.hard_img_ids)))
        self.files = []

        # for split in ["train", "trainval", "val"]:
        for name, name1 in zip(self.easy_img_ids, self.hard_img_ids):
            easy_img_file = osp.join(self.root, "img/%s" % name)
            hard_img_file = osp.join(self.root, "img/%s" % name1)
            self.files.append({
                "source_image": easy_img_file,
                "target_image": hard_img_file,
                "name": name,
                "name1": name1
            })

    def __len__(self):
        return len(self.files)


    def __getitem__(self, index):
        datafiles = self.files[index]

        source_image = Image.open(datafiles["source_image"]).convert('RGB')
        target_image = Image.open(datafiles["target_image"]).convert('RGB')
        name = datafiles["name"]
        name1 = datafiles["name1"]

        # resize
        source_image = source_image.resize(self.crop_size, Image.BICUBIC)
        target_image = target_image.resize(self.crop_size, Image.BICUBIC)

        # flip
        if np.random.rand(1) < self.mirror_prob:
            source_image = source_image.transpose(Image.FLIP_LEFT_RIGHT)
            target_image = target_image.transpose(Image.FLIP_LEFT_RIGHT)

        source_image = np.asarray(source_image, np.float32)
        target_image = np.asarray(target_image, np.float32)

        size = source_image.shape
        source_image = source_image[:, :, ::-1]  # change to BGR
        source_image -= 128
        source_image = source_image / 128
        source_image = source_image.transpose((2, 0, 1))

        size = target_image.shape
        target_image = target_image[:, :, ::-1]  # change to BGR
        target_image -= 128
        target_image = target_image / 128
        target_image = target_image.transpose((2, 0, 1))

        return source_image.copy(), target_image.copy(), name, name1


