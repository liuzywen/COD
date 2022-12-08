import os
from PIL import Image
import cv2
import torch
from torch.utils import data
from torchvision import transforms
from torchvision.transforms import functional as F
import numbers
import numpy as np
import random

img_size = 256

class ImageDataTrain(data.Dataset):
    def __init__(self, data_root, data_list):
        self.sal_root = data_root
        self.sal_source = data_list

        with open(self.sal_source, 'r') as f:
            self.sal_list = [x.strip() for x in f.readlines()]

        self.sal_num = len(self.sal_list)


    def __getitem__(self, item):
        # sal data loading
        im_name = self.sal_list[item % self.sal_num].split()[0]
        gt_name = self.sal_list[item % self.sal_num].split()[1]
        dep_name = self.sal_list[item % self.sal_num].split()[2]
        sal_image = load_image(os.path.join(self.sal_root, im_name))
        sal_label = load_sal_label(os.path.join(self.sal_root, gt_name))
        sal_depth = load_depth(os.path.join(self.sal_root, dep_name))

        sal_image, sal_label, sal_depth = cv_random_flip(sal_image, sal_label, sal_depth)
        sal_image = torch.Tensor(sal_image)
        sal_label = torch.Tensor(sal_label)
        sal_depth = torch.Tensor(sal_depth)

        sample = {'sal_image': sal_image, 'sal_label': sal_label, 'sal_depth':sal_depth}
        return sample

    def __len__(self):
        return self.sal_num

class ImageDataTest(data.Dataset):
    def __init__(self, data_root, data_list):
        self.data_root = data_root
        self.data_list = data_list
        with open(self.data_list, 'r') as f:
            self.image_list = [x.strip() for x in f.readlines()]

        self.image_num = len(self.image_list)

    def __getitem__(self, item):
        image, im_size = load_image_test(os.path.join(self.data_root, self.image_list[item % self.image_num].split()[0]))
        dep_name = self.image_list[item % self.image_num].split()[1]
        depth = load_depth(os.path.join(self.data_root, dep_name))
        image = torch.Tensor(image)
        depth = torch.Tensor(depth)

        return {'image': image, 'name': self.image_list[item % self.image_num].split()[0], 'size': im_size, 'depth': depth}

    def __len__(self):
        return self.image_num


def get_loader(config, mode='train', pin=False):
    shuffle = False
    if mode == 'train':
        shuffle = True
        dataset = ImageDataTrain(config.train_root, config.train_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    else:
        dataset = ImageDataTest(config.test_root, config.test_list)
        data_loader = data.DataLoader(dataset=dataset, batch_size=config.batch_size, shuffle=shuffle, num_workers=config.num_thread, pin_memory=pin)
    return data_loader

def load_image(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    im = cv2.resize(im, (img_size, img_size))
    in_ = np.array(im, dtype=np.float32)
    # in_ -= np.array((104.00699, 116.66877, 122.67892))
    in_ = in_.transpose((2,0,1))
    return in_

def load_depth(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path)
    im = im.resize((img_size, img_size), Image.ANTIALIAS)
    in_ = np.array(im, np.float32) / 255.
    in_ = in_[np.newaxis, ...]
    return in_

def load_image_test(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = cv2.imread(path)
    in_ = np.array(im, dtype=np.float32)
    im_size = tuple(in_.shape[:2])
    im = cv2.resize(im, (img_size, img_size))
    in_ = np.array(im, dtype=np.float32)
    in_ = in_.transpose((2,0,1))
    return in_, im_size

def load_sal_label(path):
    if not os.path.exists(path):
        print('File {} not exists'.format(path))
    im = Image.open(path).convert('L')
    im = im.resize((img_size, img_size), Image.ANTIALIAS)
    label = np.array(im, dtype=np.float32)
    if len(label.shape) == 3:
        label = label[:,:,0]
    label = label / 255.
    label = label[np.newaxis, ...]
    return label

def cv_random_flip(img, label, depth):
    flip_flag = random.randint(0, 1)
    if flip_flag == 1:
        img = img[:,:,::-1].copy()
        label = label[:,:,::-1].copy()
        depth = depth[:,:,::-1].copy()
    return img, label, depth
