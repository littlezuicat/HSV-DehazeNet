import torch.utils.data as data
import torchvision.transforms as tfs
from torchvision.transforms import functional as FF
import os,sys
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import random
from PIL import Image
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
from torchvision.utils import make_grid
from metrics import *
from option import opt
BS=opt.bs
print(BS)
crop_size='whole_img'
if opt.crop:
    crop_size=opt.crop_size

class HSVDataset(data.Dataset):
    def __init__(self, path, train, size=crop_size, format='.jpg'):
        super().__init__()
        self.size = size
        print('crop size', size)
        self.train = train
        self.format = format
        
        # 预加载数据集到内存
        haze_imgs_dir = sorted(os.listdir(os.path.join(path, 'hazy')))
        self.haze_imgs = [
            Image.open(os.path.join(path, 'hazy', img)).convert("RGB")
            for img in haze_imgs_dir if img.endswith(('.jpg', '.png', 'JPG'))
        ]
        
        clear_imgs_dir = sorted(os.listdir(os.path.join(path, 'GT')))
        self.clear_imgs = [
            Image.open(os.path.join(path, 'GT', img)).convert("RGB")
            for img in clear_imgs_dir if img.endswith(('.jpg', '.png', 'JPG'))
        ]
        
        print(f"Loaded {len(self.haze_imgs)} hazy images and {len(self.clear_imgs)} clear images into memory.")

    def __getitem__(self, index):
        haze = self.haze_imgs[index]  # 直接从内存获取
        clear = self.clear_imgs[index]  # 直接从内存获取
        # print(haze)
        if not isinstance(self.size, str):
            i, j, h, w = tfs.RandomCrop.get_params(haze, output_size=(self.size, self.size))
            haze = FF.crop(haze, i, j, h, w)
            clear = FF.crop(clear, i, j, h, w)
        
        haze, clear= self.augData(haze, clear)
        return haze, clear
    
    def augData(self, data, target):
        if self.train:
            rand_hor = random.randint(0, 1)
            rand_rot = random.randint(0, 3)
            
            data = tfs.RandomHorizontalFlip(rand_hor)(data)
            target = tfs.RandomHorizontalFlip(rand_hor)(target)
            
            if rand_rot:
                data = FF.rotate(data, 90 * rand_rot)
                target = FF.rotate(target, 90 * rand_rot)
        
        # 归一化
        if opt.trainset == 'Haze4K_train':
            mean = [0.58131991, 0.55842627, 0.54895015]
            std = [0.20261328, 0.20865026, 0.21723576]
        elif opt.trainset == "Ohaze_train":
            mean = [0.47562887, 0.50872567, 0.5683684]
            std = [0.10191456, 0.10476941, 0.1142297]
        elif opt.trainset == "Ihaze_train":
            mean = [0.52328509, 0.52063797, 0.51897897]
            std = [0.13663095, 0.13907871, 0.13727794]
        elif opt.trainset == "DenseHaze_train":
            mean = [0.61347476, 0.64410202, 0.68819155]
            std = [0.07049278, 0.06893078, 0.06532876]
        elif opt.trainset == "NHhaze_train":
            mean = [0.50673132, 0.53998279, 0.59829398]
            std = [0.12409935, 0.12749854, 0.13855921]
        else:
            raise ValueError("数据集的均值和标准差不存在，请在data_utils.py中手动添加")
        
        data = tfs.ToTensor()(data)
        # data = tfs.Normalize(mean=mean, std=std)(data)
        target = tfs.ToTensor()(target)
        
        return data, target

    def __len__(self):
        return len(self.haze_imgs)

import os
pwd=os.getcwd()
print(pwd)
path=r'D:\Desktop\renyi\Dataset' #path to your 'data' folder

# ITS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+r'\RESIDE\ITS',train=True,size=crop_size, format='.png'),batch_size=BS,shuffle=True)
# ITS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+r'\RESIDE\SOTS\indoor',train=False,size='whole img', format='.png'),batch_size=1,shuffle=False)

# OTS_train_loader=DataLoader(dataset=RESIDE_Dataset(path+r'\RESIDE\OTS',train=True, format='.png'),batch_size=BS,shuffle=True)
# OTS_test_loader=DataLoader(dataset=RESIDE_Dataset(path+r'\RESIDE\SOTS\outdoor',train=False,size='whole img',format='.png'),batch_size=1,shuffle=False)

Ohaze_train_loader=DataLoader(dataset=HSVDataset(path+r'\O-Haze\train',train=True,size=crop_size, format='.jpg'),batch_size=BS,shuffle=True, num_workers=opt.workers)
Ohaze_test_loader=DataLoader(dataset=HSVDataset(path+r'\O-Haze\test',train=False,size='whole image', format='.jpg'),batch_size=1,shuffle=False, num_workers=opt.workers)

# Ihaze_train_loader=DataLoader(dataset=HSVDataset(path+r'/I-Haze/train',train=True,size=crop_size, format='.jpg'),batch_size=BS,shuffle=True, num_workers=opt.workers)
# Ihaze_test_loader=DataLoader(dataset=HSVDataset(path+r'/I-Haze/test',train=False,size=crop_size, format='.jpg'),batch_size=1,shuffle=False, num_workers=opt.workers)

# Haze4K_train_loader=DataLoader(dataset=HSVDataset(path+r'/Haze4K/train',train=True,size=crop_size, format='.png'),batch_size=BS,shuffle=True, num_workers=opt.workers)
# Haze4K_test_loader=DataLoader(dataset=HSVDataset(path+r'/Haze4K/test',train=False,size=crop_size, format='.png'),batch_size=1,shuffle=False, num_workers=opt.workers)

# ITS_train_loader=DataLoader(dataset=HSVDataset(path+r'\ITS',train=True,size=crop_size, format='.png'),batch_size=BS,shuffle=True, num_workers=opt.workers)
# ITS_test_loader=DataLoader(dataset=HSVDataset(path+r'\SOTS\indoor',train=False,size='whole image', format='.png'),batch_size=1,shuffle=False, num_workers=opt.workers)

if __name__ == "__main__":
    pass
