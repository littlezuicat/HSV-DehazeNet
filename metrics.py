from math import exp
import math
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from math import exp
import math
import numpy as np

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from  torchvision.transforms import ToPILImage
from option import opt



def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def ssim(img1, img2, window_size=11, size_average=True):
    img1=torch.clamp(img1,min=0,max=1)
    img2=torch.clamp(img2,min=0,max=1)
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)
def psnr(pred, gt):
    pred=pred.clamp(0,1).cpu().numpy()
    gt=gt.clamp(0,1).cpu().numpy()
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10( 1.0 / rmse)



def denormalize_tensor(tensor):
    if opt.trainset == 'haze4k_train':
        mean=[0.5755, 0.5443, 0.5320]
        std=[0.2299, 0.2393, 0.2500]
    elif opt.trainset == "Ohaze_train":
        mean = [0.4757, 0.5087, 0.5682]
        std = [0.1034, 0.1062, 0.1155]
    else:
        raise "数据集的均值和标准差不存在，请在data_utils.py中手动添加"
    """
    反标准化 tensor 数据
    :param tensor: 输入的图像 Tensor，形状为 (B, C, H, W)
    :param mean: RGB 均值 (3,)
    :param std: RGB 标准差 (3,)
    :return: 反标准化后的 tensor
    """
    new_tensor = tensor.clone()
    for i in range(3):  # 对每个通道进行反标准化
        new_tensor[:, i, :, :] = new_tensor[:, i, :, :] * std[i] + mean[i]
    return new_tensor

def rgb_to_hsv_fn(rgb_tensor):
    """
    将形状为 (B, C, H, W) 的 RGB Tensor 转换为 HSV Tensor
    :param rgb_tensor: 输入的 RGB 图像 Tensor, 形状为 (B, C, H, W)
    :return: 返回的 HSV 图像 Tensor, 形状为 (B, C, H, W) 范围是0-1 注意已经不是正态分布了
    """
    "注意此函数将会把已经正态分布的数据变回原始值！！！"
    # 将 PyTorch Tensor 转换为 Numpy 数组，注意需要转换为 (B * H * W, 3) 的形状
    denormalize_tensor(rgb_tensor)  #此时范围是0-1
    b, c, h, w = rgb_tensor.shape
    rgb_tensor = rgb_tensor.permute(0, 2, 3, 1)  # 转换为 (B, H, W, C)
    rgb_tensor = rgb_tensor.clone()
    
    rgb_np = rgb_tensor.cpu().detach().numpy().astype(np.uint8)  # 此时已经转化为0-255范围
    # 转换为 Numpy 数组

    # 用 OpenCV 将 RGB 转换为 HSV
    hsv_np = np.zeros_like(rgb_np, dtype=np.float32)
    for i in range(b):
        hsv_np[i] = cv2.cvtColor(rgb_np[i], cv2.COLOR_RGB2HSV)  # 转换每张图片

    # 将 HSV 图像的各个通道进行归一化处理
    # 色调 H 归一化: 0 - 179 -> 0 - 1
    hsv_np[:, :, :, 0] = hsv_np[:, :, :, 0] / 179.0  # H 归一化
    # 饱和度 S 和亮度 V 归一化: 0 - 255 -> 0 - 1
    hsv_np[:, :, :, 1] = hsv_np[:, :, :, 1] / 255.0  # S 归一化
    hsv_np[:, :, :, 2] = hsv_np[:, :, :, 2] / 255.0  # V 归一化

    # 将 Numpy 数组转换回 PyTorch Tensor，并返回
    hsv_tensor = torch.from_numpy(hsv_np).permute(0, 3, 1, 2)  # 转换回 (B, C, H, W)
    if torch.cuda.is_available():
       hsv_tensor = hsv_tensor.cuda()
    return hsv_tensor

def hsv_to_rgb_fn(hsv_tensor):
    """
    将形状为 (B, C, H, W) 的 HSV Tensor 转换为 RGB Tensor
    :param hsv_tensor: 输入的 HSV 图像 Tensor, 形状为 (B, C, H, W)，H、S、V 范围在 [0, 1]
    :return: 返回的 RGB 图像 Tensor, 形状为 (B, C, H, W)，RGB 范围在 [0, 1]
    """
    # 获取张量的形状 (B, C, H, W)
    b, c, h, w = hsv_tensor.shape
    hsv_tensor = hsv_tensor.permute(0, 2, 3, 1)  # 转换为 (B, H, W, C)
    hsv_tensor = hsv_tensor.clone()
    hsv_np = hsv_tensor.cpu().detach().numpy()

    # 将 HSV 张量的每个通道反归一化
    # H 反归一化: [0, 1] -> [0, 179]
    hsv_np[:, :, :, 0] = hsv_np[:, :, :, 0] * 179.0
    # S 和 V 反归一化: [0, 1] -> [0, 255]
    hsv_np[:, :, :, 1] = hsv_np[:, :, :, 1] * 255.0
    hsv_np[:, :, :, 2] = hsv_np[:, :, :, 2] * 255.0

    # 将 HSV 图像的类型转换为 uint8
    hsv_np[:, :, :, 0] = np.clip(hsv_np[:, :, :, 0], 0, 179)  # H 只裁剪到 [0, 179]
    hsv_np = np.clip(hsv_np, 0, 255).astype(np.uint8)

    # 使用 OpenCV 将 HSV 转换为 RGB
    rgb_np = np.zeros_like(hsv_np, dtype=np.uint8)
    for i in range(b):
        rgb_np[i] = cv2.cvtColor(hsv_np[i], cv2.COLOR_HSV2RGB)  # 转换每张图片

    # 将 Numpy 数组转换回 PyTorch Tensor，并返回
    rgb_tensor = torch.from_numpy(rgb_np).permute(0, 3, 1, 2) / 255  # 转换回 (B, C, H, W)
    if torch.cuda.is_available():
        rgb_tensor = rgb_tensor.cuda()
    return rgb_tensor



def rgb_to_hsv(tensor):
    """
    将BCHW格式的RGB张量转换为HSV格式
    
    参数：
        tensor (torch.Tensor): 输入的RGB张量，格式为BCHW（B: batch size, C: 3, H: 高度, W: 宽度）
    
    返回：
        torch.Tensor: 转换后的HSV张量，格式仍为BCHW
    """
    tensor = denormalize_tensor(tensor)
    # print(111111111111111111111111111111,torch.max(tensor), torch.min(tensor))
    # 提取RGB三个通道
    r, g, b = tensor[:, 0:1, :, :], tensor[:, 1:2, :, :], tensor[:, 2:3, :, :]
    
    # 计算最大值、最小值及它们的差值
    max_val, _ = torch.max(tensor, dim=1, keepdim=True)  # 最大值
    min_val, _ = torch.min(tensor, dim=1, keepdim=True)  # 最小值
    delta = max_val - min_val  # 最大值与最小值之差
    
    # 初始化色调、饱和度和明度
    h = torch.zeros_like(max_val)  # 初始化色调通道
    s = torch.zeros_like(max_val)  # 初始化饱和度通道
    v = max_val  # 明度就是最大值
    
    # 避免除零错误，只有当最大值和最小值不同（即有颜色）时才计算色调
    mask = delta != 0
    # print(v == r, v == g, v== b)
    # 计算色调（H）
    # 如果 R 是最大值
    if(r == max_val).any():
        # print('r is the best')
        h[mask] = 60 * ((g[mask] - b[mask]) / delta[mask])
    # 如果 G 是最大值
    elif(g == max_val).any():
        # print('g is the best')
        h[mask] = 60 * ((b[mask] - r[mask]) / delta[mask] + 2) 
    elif(b == max_val).any():
        # print('b is the best')
    # 如果 B 是最大值
        h[mask] = 60 * ((r[mask] - g[mask]) / delta[mask] + 4)
    # 计算饱和度（S）
    s[mask] = delta[mask] / max_val[mask]  # 饱和度为差值与最大值的比值
    # print(22222222222222222222, torch.max(h), torch.min(h))
    # 将H、S、V通道合并成一个HSV张量
    hsv_tensor = torch.cat([h, s, v], dim=1)  # 拼接为BCHW格式的HSV张量

    return hsv_tensor


def hsv_to_rgb1(hsv_tensor):
    """
    将BCHW格式的HSV张量转换为RGB格式
    
    参数：
        hsv_tensor (torch.Tensor): 输入的HSV张量，格式为BCHW（B: batch size, C: 3, H: 高度, W: 宽度）
            - h (色调)：范围 [0, 360)
            - s (饱和度)：范围 [0, 1]
            - v (明度)：范围 [0, 1]
    
    返回：
        torch.Tensor: 转换后的RGB张量，格式仍为BCHW
            - r, g, b (RGB通道)：范围 [0, 1]
    """
    
    # 提取HSV通道
    h, s, v = hsv_tensor[:, 0:1, :, :], hsv_tensor[:, 1:2, :, :], hsv_tensor[:, 2:3, :, :]

    # 计算色调对应的区间（0 <= H < 360），并转换为 [0, 1] 之间的比例
    h = h / 60.0  # 将色调的度数转换为 [0, 6) 的比例
    i = torch.floor(h)  # 色相区间 [0, 6)，表示色轮的哪个部分
    f = h - i  # 色相的余数部分 [0, 1) 之间的小数

    p = v * (1 - s)  # 计算亮度和饱和度的乘积，得到低饱和度的颜色分量
    q = v * (1 - f * s)  # 计算红色和绿色区间之间的过渡色值
    t = v * (1 - (1 - f) * s)  # 计算蓝色和绿色区间之间的过渡色值
    
    # 初始化RGB通道
    r, g, b = torch.zeros_like(v), torch.zeros_like(v), torch.zeros_like(v)

    # 根据色调的区间 i 来选择不同的 RGB 值
    # 这里根据i的不同值对颜色分量进行赋值
    r[i == 0] = v[i == 0]
    g[i == 0] = t[i == 0]
    b[i == 0] = p[i == 0]

    r[i == 1] = q[i == 1]
    g[i == 1] = v[i == 1]
    b[i == 1] = p[i == 1]

    r[i == 2] = p[i == 2]
    g[i == 2] = v[i == 2]
    b[i == 2] = t[i == 2]

    r[i == 3] = p[i == 3]
    g[i == 3] = q[i == 3]
    b[i == 3] = v[i == 3]

    r[i == 4] = t[i == 4]
    g[i == 4] = p[i == 4]
    b[i == 4] = v[i == 4]

    r[i == 5] = v[i == 5]
    g[i == 5] = p[i == 5]
    b[i == 5] = q[i == 5]

    # 合并RGB通道
    rgb_tensor = torch.cat([r, g, b], dim=1)
    
    return rgb_tensor

if __name__ == "__main__":
    pass
