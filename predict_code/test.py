import torch
import argparse
import torch.backends.cudnn as cudnn
import os
from architecture import *
from utils import save_matv73
import cv2
import numpy as np
import itertools
# 创建一个参数解析器
parser = argparse.ArgumentParser(description="SSR")
# 添加各种命令行参数选项
# 方法名称
parser.add_argument('--method', type=str, default='mst_plus_plus')
# 预训练模型路径
parser.add_argument('--pretrained_model_path', type=str, default='./model_zoo/mst_plus_plus.pth')
# RGB 图像路径
parser.add_argument('--rgb_path', type=str, default='./demo/ARAD_1K_0912.jpg')
# 输出目录路径输出目录路径
parser.add_argument('--outf', type=str, default='./exp/mst_plus_plus/')
# 集成模式
parser.add_argument('--ensemble_mode', type=str, default='mean')
# CUDA GPU ID
parser.add_argument("--gpu_id", type=str, default='0')
# 解析命令行参数
opt = parser.parse_args()
# 设置 CUDA 设备
os.environ["CUDA_DEVICE_ORDER"] = 'PCI_BUS_ID'
os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
# 如果输出目录不存在，则创建它
if not os.path.exists(opt.outf):
    os.makedirs(opt.outf)

def main():
    # 启用 CuDNN 加速
    cudnn.benchmark = True
    # 获取预训练模型路径和方法名称
    pretrained_model_path = opt.pretrained_model_path
    method = opt.method
    # 生成指定方法的模型并将其移动到 CUDA 设备上
    model = model_generator(method, pretrained_model_path).cuda()
    # 对模型进行测试并保存输出
    test(model, opt.rgb_path, opt.outf)

def test(model, rgb_path, save_path):
    var_name = 'cube'
    # 读取RGB图像并进行预处理
    bgr = cv2.imread(rgb_path)
    # 将BGR图像转换为RGB图像
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    # 将图像的数据类型转换为np.float32
    rgb = np.float32(rgb)
    # 将图像归一化到 [0, 1] 范围
    rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())
    # 对图像进行维度调整和转置，使其适合模型输入，将RGB图像的维度从行（height）、列（width）、通道（channels）的顺序，转换为通道、行、列的顺序，这是因为在深度学习中通常将通道维度放在最前面。
    # 将图像的通道维度从最后一个位置移动到第一个位置，然后在第一个位置添加一个新的维度
    # 结果将是一个4维数组：[batch_size=1, channels=3, height, width]
    rgb = np.expand_dims(np.transpose(rgb, [2, 0, 1]), axis=0).copy()
    # 将图像数据转换为CUDA张量（如果有可用的CUDA设备）
    rgb = torch.from_numpy(rgb).float().cuda()
    print(f'Reconstructing {rgb_path}')
    # 使用模型进行前向推理
    with torch.no_grad():
        result = forward_ensemble(rgb, model, opt.ensemble_mode)
    # 后处理结果
    result = result.cpu().numpy() * 1.0
    result = np.transpose(np.squeeze(result), [1, 2, 0])
    result = np.minimum(result, 1.0)
    result = np.maximum(result, 0)
    # 生成保存的.mat文件名和路径
    mat_name = rgb_path.split('/')[-1][:-4] + '.mat'
    mat_dir = os.path.join(save_path, mat_name)
    # 使用之前的函数保存结果到.mat文件
    save_matv73(mat_dir, var_name, result)
    # f-string（格式化字符串），它将一个变量（在这里是mat_dir，即保存的.mat文件的完整路径）插入到字符串中，以便在输出中显示该变量的值
    print(f'The reconstructed hyper spectral image are saved as {mat_dir}.')

def forward_ensemble(x, forward_func, ensemble_mode = 'mean'):
    # 定义变换函数，xflip, yflip, transpose,都是布尔值
    def _transform(data, xflip, yflip, transpose, reverse=False):

        if not reverse:  # 正向变换
            if xflip:
                # 水平翻转
                data = torch.flip(data, [3])
            if yflip:
                # 垂直翻转
                data = torch.flip(data, [2])
            if transpose:
                data = torch.transpose(data, 2, 3)
        else:  # 反向变换
            if transpose:
                # 转置
                data = torch.transpose(data, 2, 3)
            if yflip:
                # 垂直翻转
                data = torch.flip(data, [2])
            if xflip:
                # 水平翻转
                data = torch.flip(data, [3])
        return data

    # 存储所有的推理结果
    outputs = []
    opts = itertools.product((False, True), (False, True), (False, True))
    # 通过 itertools.product 生成所有可能的变换方式的组合
    for xflip, yflip, transpose in opts:
        # 克隆输入数据，以便多次使用
        data = x.clone()
        # 对输入数据进行变换
        data = _transform(data, xflip, yflip, transpose)
        # 使用模型进行前向传播
        data = forward_func(data)
        # 对模型输出的结果进行反向变换，以还原到原始 状态
        outputs.append(
            _transform(data, xflip, yflip, transpose, reverse=True))
    # 根据集成模式计算最终结果
    if ensemble_mode == 'mean':
        return torch.stack(outputs, 0).mean(0)
    elif ensemble_mode == 'median':
        return torch.stack(outputs, 0).median(0)[0]


if __name__ == '__main__':
    main()