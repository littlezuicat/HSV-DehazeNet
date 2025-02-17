import os, argparse
import numpy as np
from PIL import Image
from models.HSVNet import HSVNet
import torch
import torch.nn as nn
import torchvision.transforms as tfs
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from option import opt
from metrics import ssim, psnr
from os.path import join as join


abs = os.getcwd() + '/'


img_dir = r'D:\Desktop\renyi\Dataset\SOTS\indoor\hazy'
dataset = 'ITS'
zip_size = 1024  #原图太大存不下



#gps = opt.gps
#blocks = opt.blocks

output_dir = abs + f'pred_HSVNet_{dataset}/'
print("pred_dir:", output_dir)
if not os.path.exists(output_dir):
    os.mkdir(output_dir)
model_dir = f"trained_models/{dataset}_train_HSVNet_0.5.pk"
print('test_model_dir', model_dir)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
ckp = torch.load(model_dir, map_location=device)
net = HSVNet(dim=64)
net = nn.DataParallel(net)
net.load_state_dict(ckp['model'])
net.eval()
if opt.trainset == 'Haze4K_train':
    mean=[0.58131991,0.55842627,0.54895015]
    std=[0.20261328,0.20865026,0.21723576]
elif opt.trainset == "Ohaze_train":
    mean = [0.4757, 0.5087, 0.5682]
    std = [0.1034, 0.1062, 0.1155]
elif opt.trainset == "Ihaze_train":
    mean = [0.52328509,0.52063797,0.51897897]
    std = [0.13663095,0.13907871,0.13727794]
elif opt.trainset == "ITS_train":
    mean = [0.63633537, 0.59616035, 0.58574339]
    std = [0.14598589, 0.14998051, 0.15494896]
else:
    raise "数据集的均值和标准差不存在，请在data_utils.py中手动添加"
for im in os.listdir(img_dir):
    
    haze = Image.open(join(img_dir, im))
    
    h, w = haze.size
    haze = tfs.CenterCrop(min(h ,w)//8*8)(haze)
    # haze = haze.resize((zip_size, zip_size), Image.Resampling.LANCZOS)  #直接使用原图，不去压缩
    haze_HSV = haze.convert("HSV")
    haze_tensor = tfs.Compose([
        tfs.ToTensor(),
        tfs.Normalize(mean,std)
    ])(haze)[None, ::]
    haze_HSV_tensor = tfs.Compose([
        tfs.ToTensor()
    ])(haze_HSV)[None, ::]
    
    with torch.no_grad():
        out1x_RGB, predict, hsv = net(haze_tensor, haze_HSV_tensor)


    ts = torch.squeeze(predict.clamp(0, 1).cpu())
    
    vutils.save_image(ts, output_dir + im.split('.')[0] + '_HSV.png')
print('successful\npredict images are saved at:',output_dir)
