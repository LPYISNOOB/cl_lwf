# 导入必要的库和模块
import torch.nn as nn
from avalanche.models.dynamic_modules import MultiTaskModule, MultiHeadClassifier

# 定义一个针对CIFAR-100的卷积神经网络模型
class ConvCIFAR(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvCIFAR, self).__init__()  # 调用父类的构造函数
        # 定义卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第一个卷积层
            nn.ReLU(inplace=True),  # 第一个卷积层的激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第二个卷积层
            nn.ReLU(inplace=True),  # 第二个卷积层的激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第三个卷积层
            nn.ReLU(inplace=True),  # 第三个卷积层的激活函数
        )
        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 定义全连接层
        self.linear1 = nn.Linear(16*160, 320)  # 第一个全连接层
        self.linear2 = nn.Linear(320, 320)  # 第二个全连接层
        # 定义分类器
        self.classifier = nn.Linear(320, num_classes)  # 分类器

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 2560)  # 重塑x的形状
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x)  # 通过分类器
        return x  # 返回输出


# 定义一个多任务版本的ConvCIFAR模型
class MTConvCIFAR(ConvCIFAR, MultiTaskModule):
    def __init__(self):
        super(MTConvCIFAR, self).__init__()  # 调用父类的构造函数
        # 重新定义分类器为多头分类器
        self.classifier = MultiHeadClassifier(320)

    def forward(self, x, task_labels):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 16*160)  # 重塑x的形状
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x, task_labels)  # 通过多头分类器
        return x  # 返回输出


# 定义一个针对TinyImageNet的卷积神经网络模型
class ConvTinyImageNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvTinyImageNet, self).__init__()  # 调用父类的构造函数
        # 定义卷积层序列
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第一个卷积层
            nn.ReLU(inplace=True),  # 第一个卷积层的激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第二个卷积层
            nn.ReLU(inplace=True),  # 第二个卷积层的激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第三个卷积层
            nn.ReLU(inplace=True),  # 第三个卷积层的激活函数
            nn.Conv2d(160, 160, kernel_size=(3, 3), stride=2, padding=1),  # 第四个卷积层
            nn.ReLU(inplace=True),  # 第四个卷积层的激活函数
        )
        # 定义ReLU激活函数
        self.relu = nn.ReLU(inplace=True)
        # 定义全连接层
        self.linear1 = nn.Linear(16*160, 640)  # 第一个全连接层
        self.linear2 = nn.Linear(640, 640)  # 第二个全连接层
        # 定义分类器
        self.classifier = nn.Linear(640, num_classes)  # 分类器

    def forward(self, x):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 16*160)  # 重塑x的形状
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x)  # 通过分类器
        return x  # 返回输出


# 定义一个多任务版本的ConvTinyImageNet模型
class MTConvTinyImageNet(ConvTinyImageNet, MultiTaskModule):
    def __init__(self):
        super(MTConvTinyImageNet, self).__init__()  # 调用父类的构造函数
        # 重新定义分类器为多头分类器
        self.classifier = MultiHeadClassifier(640)

    def forward(self, x, task_labels):
        x = self.conv_layers(x)  # 通过卷积层
        x = x.view(-1, 16*160)  # 重塑x的形状
        x = self.relu(self.linear1(x))  # 通过第一个全连接层和激活函数
        x = self.relu(self.linear2(x))  # 通过第二个全连接层和激活函数
        x = self.classifier(x, task_labels)  # 通过多头分类器
        return x  # 返回输出