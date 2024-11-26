# from brian2 import *
#
# # 定义时间常数和模拟时间
# defaultclock.dt = 0.1*ms
# start_scope()
#
# # 神经元模型：使用简单的LIF（Leaky Integrate-and-Fire）模型
# eqs = '''
# dv/dt = (ge - v + I) / tau : volt (unless refractory)
# dge/dt = -ge / tau_m : 1
# I : volt
# '''
#
# # 神经元组
# N = 100  # 神经元数量
# tau_m = 10*ms  # 膜电位时间常数
# tau_syn = 5*ms  # 突触电导时间常数
# G = NeuronGroup(N, eqs, threshold='v > 1', reset='v = 0', refractory=5*ms, method='euler')
# G.v = 'rand() * 2 * mV - 1 * mV'  # 初始膜电位，使用mV作为单位
# G.ge = 0  # 初始突触电导
# G.I = 0  # 初始输入电流
# G.tau = tau_m  # 膜电位时间常数
# G.tau_m = tau_syn  # 突触电导时间常数
#
# # 连接模式：随机连接
# S = Synapses(G, G, on_pre='ge += 0.5')  # 突触前事件
# S.connect('i != j', p=0.1)  # 随机连接10%的神经元
#
# # 监视器
# M = SpikeMonitor(G)
# V = StateMonitor(G, 'v', record=True)
#
# # 运行模拟
# run(1*second)
#
# # 绘制结果
# from brian2tools import plot_raster
# plot_raster(M)
#
# # 绘制膜电位变化
# from matplotlib import pyplot as plt
# plt.figure()
# for i in range(N):
#     plt.plot(V.t / ms, V.v[i], label=f'Neuron {i}')
# plt.xlabel('Time (ms)')
# plt.ylabel('Membrane Potential (mV)')
# plt.title('Membrane Potential Over Time')
# plt.legend()
# plt.show()
# import torch
# from spikingjelly.activation_based import neuron
#
# net_s = neuron.IFNode(step_mode='s')
# x = torch.rand([4])
# print(net_s)
# print(f'the initial v={net_s.v}')
# y = net_s(x)
# print(f'x={x}')
# print(f'y={y}')
# print(f'v={net_s.v}')
import torch
import torch.nn as nn

# 定义一个二维卷积层
# 输入通道数为1（例如灰度图像），输出通道数为32，卷积核大小为3x3
conv_layer = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=1, padding=1)

# 创建一个随机输入数据，假设是28x28的灰度图像，批量大小为4
input_data = torch.randn(4, 1, 28, 28)

# 前向传播，得到卷积层的输出
output_data = conv_layer(input_data)

print(output_data.shape)