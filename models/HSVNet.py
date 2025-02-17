import torch
import numpy
import os
import pandas
import torch.nn as nn
import cv2
from models.Transformer import Transformer
import torch.nn.functional as FF
from option import opt
from metrics import *
import kornia


class HSVNet(nn.Module):
    def __init__(self, dim):
        """return out1x_RGB, out1x_HSV, H, S, V"""
        super(HSVNet, self).__init__()
        print('dim:', dim)
        self.ConvIn = nn.Conv2d(3, dim, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[Block(dim) for _ in range(5)])  # 5个Block
        self.ConvOut = nn.Conv2d(dim, 3, kernel_size=3, padding=1)
    def forward(self, x):
        x = kornia.color.rgb_to_hsv(x)
        x = self.ConvIn(x)
        x = self.blocks(x)
        x = self.ConvOut(x)

        # 创建张量的副本并进行修改
        # x_clamped = x.clone()
        # x_clamped[0:1, :, :] = torch.clamp(x_clamped[0:1, :, :], 0, 2*np.pi)
        # x_clamped[1:2, :, :] = torch.clamp(x_clamped[1:2, :, :], 0, 1)
        # x_clamped[2:3, :, :] = torch.clamp(x_clamped[2:3, :, :], 0, 1)

        x = kornia.color.hsv_to_rgb(x)
        return  x
    

class Block(nn.Module):
    def __init__(self, dim):
        super(Block, self).__init__()
        self.norm = nn.BatchNorm2d(num_features=dim)
        self.conv1 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.CALayer = CA(dim=dim)
        self.PALayer = PA(dim=dim)
        self.conv3 = nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        self.fusion = AFF(dim=dim)
    def forward(self, x):
        res = FF.relu(self.conv1(self.norm(x)))
        res = FF.relu(self.conv2(res))
        res_ca = self.CALayer(res)
        res_pa = self.PALayer(res)
        
        res = self.conv3(self.fusion(res_ca, res_pa))
        
        return res + x


class CA(nn.Module):
    def __init__(self, dim):
        super(CA, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv1 = nn.Conv2d(dim, dim//8, kernel_size=3, padding=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim//8, dim, kernel_size=3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        res = self.conv1(self.gap(x))
        res = self.conv2(self.act(res))
        weight = self.sigmoid(res)
        return x * weight
    



class PA(nn.Module):
    def __init__(self, dim):
        super(PA, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim//8, kernel_size=1)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(dim//8, dim, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        res = self.conv1(x)
        res = self.conv2(self.act(res))
        weight = self.sigmoid(res)
        return x * weight
    
class AFF(nn.Module):
    def __init__(self, dim, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.conv_in = nn.ModuleList([nn.Conv2d(dim, dim // 8, kernel_size=1) for i in range(2)])
        self.conv_out = nn.ModuleList([nn.Conv2d(dim // 8, dim, kernel_size=1) for i in range(2)])
        self.batch_norm64 = nn.BatchNorm2d(dim)
        self.batch_norm8 = nn.BatchNorm2d(dim // 8)
    def forward(self, x, y):
        f = x + y
        f1 = FF.gelu(self.batch_norm8(self.conv_in[0](self.GAP(f))))
        f1 = self.batch_norm64(self.conv_out[0](f1))
        
        f2 = FF.gelu(self.batch_norm8(self.conv_in[1](f)))
        f2 = self.batch_norm64(self.conv_out[1](f2))
        
        sum = FF.sigmoid(f1 + f2)
        
        out = x*sum + y * (1-sum)
        return out

class HueCalibrationModule(nn.Module):  # Hue纠正模块
    def __init__(self, channels=32):
        super(HueCalibrationModule, self).__init__()
        
        # 局部特征提取（卷积）
        self.conv1 = nn.Conv2d(1, channels, kernel_size=3, padding=1)  
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        
        # 扩展感受野（膨胀卷积）
        self.dilated_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=2, dilation=2)  

        # 自适应校正层
        self.adaptive_conv = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

        # 归一化
        self.norm = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU()

    def forward(self, H_dehazed):
        """
        输入: H_dehazed (去雾后的色相通道, [B, 1, H, W])
        输出: H_corrected (校正后的色相通道, [B, 1, H, W])
        """
        x = self.relu(self.conv1(H_dehazed))
        x = self.relu(self.conv2(x))
        x = self.relu(self.dilated_conv(x))  # 让网络感知大范围的色相偏移
        
        delta_H = self.adaptive_conv(x)  # 预测色相修正值
        H_corrected = H_dehazed + delta_H  # 进行色相校正
        H_corrected = torch.clamp(H_corrected, 0, 1)  # 确保色相范围合法

        return H_corrected

class SKFF(nn.Module):
    def __init__(self, dim, reduction=8):
        super(SKFF, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)  # 全局平均池化
        self.fc1 = nn.Linear(dim, dim // reduction, bias=False)  # MLP 降维
        self.fc2 = nn.Linear(dim // reduction, dim * 2, bias=False)  # 计算两个分支的权重
        self.softmax = nn.Softmax(dim=1)  # 归一化权重

    def forward(self, feat1, feat2):
        b, c, h, w = feat1.size()

        # Step 1: 计算全局信息
        fused = feat1 + feat2  # 直接相加，获取全局信息
        gap = self.gap(fused).view(b, c)  # 全局平均池化

        # Step 2: 计算注意力权重
        attn = FF.relu(self.fc1(gap))  # MLP 降维 & ReLU
        attn = self.fc2(attn)  # 恢复通道维度
        attn = attn.view(b, 2, c, 1, 1)  # 变形为 (batch, 2, dim, 1, 1)
        alpha, beta = self.softmax(attn).split(1, dim=1)  # 分割两个权重分支

        # Step 3: 加权融合
        output = alpha.squeeze(1) * feat1 + beta.squeeze(1) * feat2
        return output