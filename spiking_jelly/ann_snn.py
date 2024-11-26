import torch
from spikingjelly.activation_based import neuron
from spikingjelly import visualizing
from matplotlib import pyplot as plt
import numpy as np

plt.rcParams['figure.dpi'] = 200
if_node = neuron.IFNode(v_reset=None)
T = 128
x = torch.arange(-0.2, 1.2, 0.04)
# plt.scatter(torch.arange(x.shape[0]), x)
# plt.title('Input $x_{i}$ to IF neurons')
# plt.xlabel('Neuron index $i$')
# plt.ylabel('Input $x_{i}$')
# plt.grid(linestyle='-.')
# plt.show()

s_list = []
for t in range(T):
    s_list.append(if_node(x).unsqueeze(0))

out_spikes = np.asarray(torch.cat(s_list))
visualizing.plot_1d_spikes(out_spikes, 'IF neurons\' spikes and firing rates', 't', 'Neuron index $i$')
plt.show()