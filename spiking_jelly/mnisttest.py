# import torch
# import torch.nn as nn
# from spikingjelly.clock_driven import neuron
#
#
# class Net(nn.Module):
#     def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28 * 28, 14 * 14, bias=False),
#             neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
#             nn.Linear(14 * 14, 10, bias=False),
#             neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
#         )
#
#     def forward(self, x):
#         return self.fc(x)
#
#
#
import spikingjelly.clock_driven.examples.lif_fc_mnist as lif_fc_mnist
lif_fc_mnist.main()
# import torch
# import torch.nn as nn
# from spikingjelly.activation_based import neuron
#
# # 确保spikingjelly已经安装
# # pip install spikingjelly
#
# # 定义网络结构
# class Net(nn.Module):
#     def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
#         super().__init__()
#         self.fc = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28 * 28, 14 * 14, bias=False),
#             neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
#             nn.Linear(14 * 14, 10, bias=False),
#             neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
#         )
#
#     def forward(self, x):
#         # 注意：x应该是一个包含时间步长的张量，例如[time_steps, batch_size, input_features]
#         # 这里我们假设x是二维的，即[batch_size, input_features]，并且只有一个时间步长
#         # 我们需要将其扩展为三维的，以匹配LIFNode的期望输入
#         x = x.unsqueeze(0)  # 增加一个时间步长的维度
#         return self.fc(x).squeeze()  # 移除时间步长的维度并返回结果
#
# # 创建网络实例，使用默认参数
# net = Net()
#
# # 创建一个随机输入数据，假设批次大小为1
# # 输入数据应该是一个28x28像素的图像，展平为784个特征
# input_data = torch.rand(1, 28 * 28)
#
# # 测试网络
# output = net(input_data)
#
# print("Output of the network:", output)