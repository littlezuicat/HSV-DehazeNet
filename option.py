import torch,os,sys,torchvision,argparse
import torchvision.transforms as tfs
import time,math
import numpy as np
from torch.backends import cudnn
from torch import optim
import torch,warnings
from torch import nn
import torchvision.utils as vutils
warnings.filterwarnings('ignore')

parser=argparse.ArgumentParser()
parser.add_argument('--steps',type=int,default=100000)
parser.add_argument('--device',type=str,default='Automatic detection')
parser.add_argument('--resume',type=bool,default=False)
parser.add_argument('--eval_step',type=int,default=500)
parser.add_argument('--lr', default=0.0001, type=float, help='learning rate')
parser.add_argument('--model_dir',type=str,default='./trained_models/')
parser.add_argument('--trainset',type=str,default='ITS_train')
parser.add_argument('--testset',type=str,default='ITS_test')
parser.add_argument('--net',type=str,default='RYNet')
parser.add_argument('--bs',type=int,default=2,help='batch size')
parser.add_argument('--workers',type=int,default=0,help='num of workers')
parser.add_argument('--crop',action='store_true')
parser.add_argument('--crop_size',type=int,default=256,help='Takes effect when using --crop ')
parser.add_argument('--no_lr_sche',action='store_true',help='no lr cos schedule')
parser.add_argument('--perloss',action='store_true',help='perceptual loss')
parser.add_argument('--ablation',default='NO',help='ablation study')

ablation_list = ["conv_3", "conv_3_5", "att_series"]

opt=parser.parse_args()
opt.device='cuda' if torch.cuda.is_available() else 'cpu'
model_name=opt.trainset+'_'+opt.net.split('.')[0]
opt.model_dir=opt.model_dir+model_name+'.pk'
log_dir='logs/'+model_name

print(opt)
print('model_dir:',opt.model_dir)


if not os.path.exists('trained_models'):
	os.mkdir('trained_models')
if not os.path.exists('logs'):
	os.mkdir('logs')
if not os.path.exists('samples'):
	os.mkdir('samples')
if not os.path.exists(f"samples/{model_name}"):
	os.mkdir(f'samples/{model_name}')
if not os.path.exists(log_dir):
	os.mkdir(log_dir)
