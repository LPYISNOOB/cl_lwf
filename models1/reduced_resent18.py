# 导入必要的库和模块
import torch
from avalanche.models import MultiHeadClassifier, MultiTaskModule
from torch import nn
from torch.nn.functional import avg_pool2d

# 注释说明了代码的起始部分是从GEM（Gradient Episodic Memory）项目代码中来的，
# 并且替换了其中的分类器为Avalanche的多头分类器。
"""
START: FROM GEM CODE https://github.com/facebookresearch/GradientEpisodicMemory/
CLASSIFIER REMOVED AND SUBSTITUTED WITH AVALANCHE MULTI-HEAD CLASSIFIER
"""


# 定义一个3x3的卷积层
def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=True
    )

# 定义ResNet的基本块
class BasicBlock(nn.Module):
    expansion = 1  # 基本块的扩展系数

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(planes)  # 第一个批归一化层
        self.conv2 = conv3x3(planes, planes)  # 第二个卷积层
        self.bn2 = nn.BatchNorm2d(planes)  # 第二个批归一化层

        self.shortcut = nn.Sequential()  # 快捷连接
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=True,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))  # 通过第一个卷积和批归一化层，然后激活
        out = self.bn2(self.conv2(out))  # 通过第二个卷积和批归一化层
        out += self.shortcut(x)  # 加上快捷连接
        out = nn.functional.relu(out)  # 再次激活
        return out

# 定义ResNet模型
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, nf):
        super(ResNet, self).__init__()
        self.in_planes = nf  # 输入通道数

        self.conv1 = conv3x3(3, nf * 1)  # 第一个卷积层
        self.bn1 = nn.BatchNorm2d(nf * 1)  # 第一个批归一化层
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)  # 第一层
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)  # 第二层
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)  # 第三层
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)  # 第四层

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)  # 创建步长列表
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))  # 添加基本块
            self.in_planes = planes * block.expansion  # 更新输入通道数
        return nn.Sequential(*layers)

    def forward(self, x):
        bsz = x.size(0)  # 获取批次大小
        out = nn.functional.relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))  # 通过第一个卷积和批归一化层，然后激活
        out = self.layer1(out)  # 通过第一层
        out = self.layer2(out)  # 通过第二层
        out = self.layer3(out)  # 通过第三层
        out = self.layer4(out)  # 通过第四层
        out = avg_pool2d(out, 4)  # 平均池化
        return out

# 结束注释说明了代码是从GEM项目代码中来的。

# 定义一个多头简化版的ResNet18模型
class MultiHeadReducedResNet18(MultiTaskModule):
    """
    根据GEM论文，这是一个更小的ResNet18版本，所有层中的特征图数量是原来的三分之一。
    它使用多头输出层。
    """

    def __init__(self, size_before_classifier=160):
        super().__init__()  # 调用父类的构造函数
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)  # 创建ResNet模型
        self.classifier = MultiHeadClassifier(size_before_classifier)  # 创建多头分类器

    def forward(self, x, task_labels):
        out = self.resnet(x)  # 通过ResNet模型
        out = out.view(out.size(0), -1)  # 重塑输出
        return self.classifier(out, task_labels)  # 通过多头分类器

# 定义一个单头简化版的ResNet18模型
class SingleHeadReducedResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()  # 调用父类的构造函数
        self.resnet = ResNet(BasicBlock, [2, 2, 2, 2], 20)  # 创建ResNet模型
        self.classifier = nn.Linear(160, num_classes)  # 创建线性分类器

    def feature_extractor(self, x):
        out = self.resnet(x)  # 通过ResNet模型
        return out.view(out.size(0), -1)  # 重塑输出

    def forward(self, x):
        out = self.feature_extractor(x)  # 通过特征提取器
        return self.classifier(out)  # 通过分类器

# 定义一个包含所有类的列表
__all__ = ['MultiHeadReducedResNet18', 'SingleHeadReducedResNet18']