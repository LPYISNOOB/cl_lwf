import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt

# if_layer = neuron.IFNode()
# print("if层的电压："+str(if_layer.v))
# 单个神经元的v 和 s 输出图像
# if_layer.reset()
# x = torch.as_tensor([0.02])
# T = 150
# s_list = []
# v_list = []
# for t in range(T):
#     s_list.append(if_layer(x))
#     v_list.append(if_layer.v)
#
# dpi = 300
# figsize = (9, 6)
# visualizing.plot_one_neuron_v_s(torch.cat(v_list).numpy(), torch.cat(s_list).numpy(), v_threshold=if_layer.v_threshold,
#                                 v_reset=if_layer.v_reset,
#                                 figsize=figsize, dpi=dpi)
#plt.show()

#32个神经元的模拟过程
#
# if_layer.reset()
# T = 50
# x = torch.rand([32]) / 8.
# s_list = []
# v_list = []
# for t in range(T):
#     s_list.append(if_layer(x).unsqueeze(0))
#     v_list.append(if_layer.v.unsqueeze(0))
#
#
# s_list = torch.cat(s_list)
# v_list = torch.cat(v_list)
#
# figsize = (12, 8)
# dpi = 200
# visualizing.plot_2d_heatmap(array=v_list.numpy(), title='membrane potentials', xlabel='simulating step',
#                             ylabel='neuron index', int_x_ticks=True, x_max=T, figsize=figsize, dpi=dpi)
#
#
# visualizing.plot_1d_spikes(spikes=s_list.numpy(), title='membrane sotentials', xlabel='simulating step',
#                         ylabel='neuron index', figsize=figsize, dpi=dpi)
#
# plt.show()

# 两种多步步进模式的表示方法
# import torch
# from spikingjelly.activation_based import neuron, functional
# if_layer = neuron.IFNode(step_mode='s')
# T = 8
# N = 2
# x_seq = torch.rand([T, N])
# y_seq = functional.multi_step_forward(x_seq, if_layer)
# print("if层的电压："+str(if_layer.v))
# if_layer.reset()
#
# if_layer.step_mode = 'm'
# y_seq = if_layer(x_seq)
# print("if层的电压："+str(if_layer.v))
# if_layer.reset()

# import torch
# from spikingjelly.activation_based import neuron
#
# class SquareIFNode(neuron.BaseNode):
#
#     def neuronal_charge(self, x: torch.Tensor):
#         self.v = self.v + x ** 2
#
# sif_layer = SquareIFNode()
#
# T = 4
# N = 1
# x_seq = torch.rand([T, N])
# print(f'x_seq={x_seq}')
#
# for t in range(T):
#     yt = sif_layer(x_seq[t])
#     print(f'sif_layer.v[{t}]={sif_layer.v}')
#
# sif_layer.reset()
# sif_layer.step_mode = 'm'
# y_seq = sif_layer(x_seq)
# print(f'y_seq={y_seq}')
# sif_layer.reset()



#
# import torch
# from spikingjelly.activation_based import neuron
# if_layer = neuron.IFNode()
# print(f'if_layer.backend={if_layer.backend}')
# # if_layer.backend=torch
#
# print(f'step_mode={if_layer.step_mode}, supported_backends={if_layer.supported_backends}')
# # step_mode=s, supported_backends=('torch',)
#
#
# if_layer.step_mode = 'm'
# print(f'step_mode={if_layer.step_mode}, supported_backends={if_layer.supported_backends}')
# # step_mode=m, supported_backends=('torch', 'cupy')
#
# device = 'cuda:0'
# if_layer.to(device)
# if_layer.backend = 'cupy'  # switch to the cupy backend
# print(f'if_layer.backend={if_layer.backend}')
# # if_layer.backend=cupy
#
# x_seq = torch.rand([8, 4], device=device)
# y_seq = if_layer(x_seq)
# if_layer.reset()
import torch
from spikingjelly.activation_based import surrogate

sg = surrogate.Sigmoid(alpha=4.)

x = torch.rand([8]) - 0.5
x.requires_grad = True
y = sg(x)
y.sum().backward()
print(f'x={x}')
print(f'y={y}')
print(f'x.grad={x.grad}')