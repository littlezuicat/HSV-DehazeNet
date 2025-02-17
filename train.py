import torch, os, sys, torchvision, argparse
import torchvision.transforms as tfs
from metrics import psnr, ssim
from models.HSVNet import HSVNet
import time, math
import numpy as np
from torch.backends import cudnn
from torch import optim
import warnings
from torch import nn
import torchvision.utils as vutils
from option import opt, model_name, log_dir
from data_utils import *
from torchvision.models import vgg16
from tqdm import tqdm
from datetime import datetime
from loss import loss_fn
from option import model_name
from torch.nn import functional as FF
#导入tensorboard


warnings.filterwarnings('ignore')

print('log_dir :', log_dir)
print('model_name:', model_name)
print('ablation', opt.ablation)

models_ = {
    'HSVNet': HSVNet(dim=64),
}

loaders_ = {
    'Ohaze_train': Ohaze_train_loader,
    'Ohaze_test': Ohaze_test_loader,
    # 'Ihaze_train': Ihaze_train_loader,
    # 'Ihaze_test': Ihaze_test_loader,
    # 'Haze4K_train': Haze4K_train_loader,
    # 'Haze4K_test': Haze4K_test_loader,
    # 'ITS_train': ITS_train_loader,
    # 'ITS_test': ITS_test_loader,
}

start_time = time.time()
T = opt.steps

def lr_schedule_cosdecay(t, T, init_lr=opt.lr):
    lr = 0.5 * (1 + math.cos(t * math.pi / T)) * init_lr
    return lr

def train(net, loader_train, loader_test, optim):
    losses = []
    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_.txt')

    # 检查是否从已有模型继续训练
    if opt.resume and os.path.exists(opt.model_dir):
        print(f'resume from {opt.model_dir}')
        ckp = torch.load(opt.model_dir)
        losses = ckp['losses']
        net.load_state_dict(ckp['model'])
        start_step = ckp['step']
        max_ssim = ckp['max_ssim']
        max_psnr = ckp['max_psnr']
        psnrs = ckp['psnrs']
        ssims = ckp['ssims']
        print(f'start_step: {start_step} start training ---')
    else:
        print('train from scratch *** ')
    with tqdm(range(start_step + 1, opt.steps + 1), unit='step') as steps:
        with open(log_file, 'a') as f:
            for step in steps:
                net.train()
                
                lr = opt.lr
                steps.set_description(f"model:{model_name} step:[{step}]/{opt.steps} dataset->{opt.trainset}")
                
                if not opt.no_lr_sche:
                    lr = lr_schedule_cosdecay(step, T)
                    for param_group in optim.param_groups:
                        param_group["lr"] = lr  

                x, y= next(iter(loader_train))
                x = x.to(opt.device)
                y = y.to(opt.device)
                
                predict = net(x)
                #损失函数
                loss = FF.l1_loss(predict, y)
                
                loss.backward()
               

                optim.step()
                optim.zero_grad()
                losses.append(loss.item())
                if step % 50 == 0:
                    log_message = f'model:{opt.net} step: {step}, loss: {loss.item():.6f}, psnr: {max_psnr}, ssim: {max_ssim}, minute:{(time.time()-start_time)/60:.2f}'
                    f.write(log_message + '\n')

                steps.set_postfix(dataset=opt.trainset, loss=np.mean(losses), lr=lr, bs=opt.bs)

                # 测试模型并保存最好结果
                if step % opt.eval_step == 0:
                    with torch.no_grad():
                        ssim_eval, psnr_eval = test(net, loader_test, max_psnr, max_ssim, step)
                    print(f'\nstep :{step} | ssim:{ssim_eval:.4f} | psnr:{psnr_eval:.4f}')
                    ssims.append(ssim_eval)
                    psnrs.append(psnr_eval)

                    if ssim_eval > max_ssim and psnr_eval > max_psnr:
                        max_ssim = max(max_ssim, ssim_eval)
                        max_psnr = max(max_psnr, psnr_eval)
                        torch.save({
                            'step': step,
                            'max_psnr': max_psnr,
                            'max_ssim': max_ssim,
                            'ssims': ssims,
                            'psnrs': psnrs,
                            'losses': losses,
                            'model': net.state_dict()
                        }, opt.model_dir)
                    print(f'\n model saved at {opt.model_dir}, step: {step} | max_psnr: {max_psnr:.4f} | max_ssim: {max_ssim:.4f}')
                    
                    

def test(net, loader_test, max_psnr, max_ssim, step):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets) in enumerate(loader_test):
        # if i < 200:
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        pred= net(inputs)
        ssim1 = ssim(pred, targets).item()
        psnr1 = psnr(pred, targets)
        ssims.append(ssim1)
        psnrs.append(psnr1)

    return np.mean(ssims), np.mean(psnrs)

if __name__ == "__main__":
    loader_train = loaders_[opt.trainset]
    loader_test = loaders_[opt.testset]
    net = models_[opt.net]
    net = net.to(opt.device)

    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))


    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.lr, betas=(0.9, 0.999), eps=1e-08)
    optimizer.zero_grad()

    train(net, loader_train, loader_test, optimizer)
