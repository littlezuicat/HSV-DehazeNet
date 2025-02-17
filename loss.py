import torch
import numpy
from metrics import *
import torch.nn.functional as FF


def loss_fn(predict, GT, hsv, GT_HSV):
    # 本代码的所有hsv都是H, S, S-V
    S_GT, V_GT = GT_HSV[:, 1:2, :, :], GT_HSV[:, 2:3, :, :]
    S, V, haze = hsv[:, 0:1, :, :], hsv[:, 1:2, :, :], hsv[:, 2:3, :, :]

    RGB_loss = FF.l1_loss(predict, GT) # 两种方法都尝试一下，看看直接predict和GT做loss更好还是RGB和GT做loss更好
    HSV_loss =  (FF.smooth_l1_loss(S_GT, S) + FF.smooth_l1_loss(V_GT, V)) + FF.smooth_l1_loss(S_GT-V_GT, haze)
    return RGB_loss + 0.5*HSV_loss

