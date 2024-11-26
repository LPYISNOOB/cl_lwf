import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, functional

# 使用torch.no_grad()上下文管理器来禁用梯度计算，这通常在推理或者测试模型时使用，以减少内存消耗和计算量。
with torch.no_grad():
    T = 4  # 时间步长或序列长度
    N = 2  # 批次大小
    C = 4  # 通道数
    H = 8  # 图像高度
    W = 8  # 图像宽度
    x_seq = torch.rand([T, N, C, H, W])  # 创建一个随机的五维张量，模拟输入序列

    # 定义一个序列模型，使用默认的step-by-step（逐时间步）模式
    net = nn.Sequential(
        layer.Conv2d(C, C, kernel_size=3, padding=1, bias=False),  # 2D卷积层，无偏置项
        layer.BatchNorm2d(C),  # 批量归一化层
        neuron.IFNode()  # 积分-发放（Integrate-and-Fire）神经元节点
    )
    y_seq = functional.multi_step_forward(x_seq, net)  # 使用多步前向传播函数
    # y_seq.shape = [T, N, C, H, W]  # 输出张量的形状
    functional.reset_net(net)  # 重置网络，准备下一次前向传播

    # 将网络设置为layer-by-layer（逐层）模式
    functional.set_step_mode(net, step_mode='m')  # 设置step_mode为'm'，表示逐层模式
    y_seq = net(x_seq)  # 直接使用net调用进行前向传播
    # y_seq.shape = [T, N, C, H, W]  # 输出张量的形状
    functional.reset_net(net)  # 重置网络，准备下一次前向传播