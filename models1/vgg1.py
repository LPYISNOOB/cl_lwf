import torch
from torch import nn
import torchvision
from avalanche.models import MultiTaskModule, MultiHeadClassifier

# 定义一个多任务学习模型，继承自 Avalanche 库的 MultiTaskModule
class MultiHeadVGG(MultiTaskModule):
    def __init__(self, n_classes=20):
        super().__init__()
        # 使用预训练的 VGG11 网络作为特征提取器
        self.vgg = torchvision.models.vgg11()
        # 初始化一个多头分类器，用于处理不同的任务
        self.classifier = MultiHeadClassifier(in_features=1000, initial_out_features=n_classes)

    def forward(self, x, task_labels):
        # 前向传播方法，接收输入 x 和任务标签 task_labels
        x = self.vgg(x)  # 通过 VGG 网络提取特征
        x = torch.flatten(x, 1)  # 展平特征
        # 返回分类头的输出结果
        return self.classifier(x, task_labels)

"""
Small VGG net adapted from https://github.com/Mattdl/CLsurvey/
"""

# 定义 VGG 网络的配置参数
cfg = [64, 'M', 64, 'M', 64, 64, 'M', 128, 128, 'M']
# 卷积核大小
conv_kernel_size = 3
# 输入图像的通道数
img_input_channels = 3





# 定义一个简化版的 VGG 网络，用于特征提取
class VGGSmall(torchvision.models.VGG):
    """
    Creates VGG feature extractor from config and custom classifier.
    """

    def __init__(self):
        in_channels = img_input_channels  # 输入通道数
        layers = []  # 存储网络层
        for v in cfg:
            if v == 'M':
                # 最大池化层
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                # 卷积层和激活层
                conv2d = nn.Conv2d(in_channels, v, kernel_size=conv_kernel_size, padding=1)
                layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v  # 更新输入通道数

        # 初始化 VGG 网络
        super(VGGSmall, self).__init__(nn.Sequential(*layers), init_weights=True)

        # 兼容 PyTorch > 1.0.0 的版本
        if hasattr(self, 'avgpool'):
            self.avgpool = torch.nn.Identity()

        # 删除原始 VGG 的分类器
        del self.classifier

    def forward(self, x):
        # 前向传播方法，只返回特征提取部分的输出
        x = self.features(x)
        return x



# 定义一个多任务学习模型，使用简化版的 VGG 网络
class MultiHeadVGGSmall(MultiTaskModule):
    def __init__(self, n_classes=200, hidden_size=128):
        super().__init__()
        self.vgg = VGGSmall()  # 初始化 VGGSmall 特征提取器
        # 定义一个前馈网络，用于进一步处理特征
        self.feedforward = nn.Sequential(
            nn.Linear(128*4*4, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
        )
        # 初始化一个多头分类器，用于处理不同的任务
        self.classifier = MultiHeadClassifier(in_features=128, initial_out_features=n_classes)


    def forward(self, x, task_labels):
        # 前向传播方法，接收输入 x 和任务标签 task_labels
        x = self.vgg(x)  # 通过 VGGSmall 提取特征
        x = torch.flatten(x, 1)  # 展平特征
        x = self.feedforward(x)  # 通过前馈网络处理特征
        # 返回分类头的输出结果
        return self.classifier(x, task_labels)



# 定义一个单任务学习模型，使用简化版的 VGG 网络
class SingleHeadVGGSmall(nn.Module):
    def __init__(self, n_classes=200, hidden_size=128):
        super().__init__()
        self.vgg = VGGSmall()  # 初始化 VGGSmall 特征提取器
        # 定义一个前馈网络，用于进一步处理特征
        self.feedforward = nn.Sequential(
            nn.Linear(128 * 4 * 4, hidden_size),
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(True),
        )
        # 定义一个单一的分类器
        self.classifier = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        # 前向传播方法，只接收输入 x
        x = self.vgg(x)  # 通过 VGGSmall 提取特征
        x = torch.flatten(x, 1)  # 展平特征
        x = self.feedforward(x)  # 通过前馈网络处理特征
        # 返回分类器的输出结果
        return self.classifier(x)