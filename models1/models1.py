# 导入必要的库和模块
import avalanche.models
from avalanche.models import MultiHeadClassifier, MultiTaskModule, BaseModel
from torch import nn

# 定义一个多任务模块，继承自MultiTaskModule
class MultiHeadMLP(MultiTaskModule):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 drop_rate=0, relu_act=True):
        super().__init__()  # 调用父类的构造函数
        self._input_size = input_size  # 输入层的大小

        # 创建隐藏层和激活函数
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        # 添加额外的隐藏层
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)  # 特征提取层
        self.classifier = MultiHeadClassifier(hidden_size)  # 多头部分类器

    def forward(self, x, task_labels):
        x = x.contiguous()  # 确保x是连续的
        x = x.view(x.size(0), self._input_size)  # 重塑x的形状
        x = self.features(x)  # 通过特征提取层
        x = self.classifier(x, task_labels)  # 通过分类器
        return x  # 返回输出


# 定义一个普通的MLP模型，继承自nn.Module和BaseModel
class MLP(nn.Module, BaseModel):
    def __init__(self, input_size=28 * 28, hidden_size=256, hidden_layers=2,
                 output_size=10, drop_rate=0, relu_act=True, initial_out_features=0):
        super().__init__()  # 调用父类的构造函数
        self._input_size = input_size  # 输入层的大小

        # 创建隐藏层和激活函数
        layers = nn.Sequential(*(nn.Linear(input_size, hidden_size),
                                 nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                                 nn.Dropout(p=drop_rate)))
        # 添加额外的隐藏层
        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}", nn.Sequential(
                    *(nn.Linear(hidden_size, hidden_size),
                      nn.ReLU(inplace=True) if relu_act else nn.Tanh(),
                      nn.Dropout(p=drop_rate))))

        self.features = nn.Sequential(*layers)  # 特征提取层

        # 根据initial_out_features的值创建分类器
        if initial_out_features > 0:
            self.classifier = avalanche.models.IncrementalClassifier(in_features=hidden_size,
                                                                     initial_out_features=initial_out_features)
        else:
            self.classifier = nn.Linear(hidden_size, output_size)  # 线性分类器

    def forward(self, x):
        x = x.contiguous()  # 确保x是连续的
        x = x.view(x.size(0), self._input_size)  # 重塑x的形状
        x = self.features(x)  # 通过特征提取层
        x = self.classifier(x)  # 通过分类器
        return x  # 返回输出

    def get_features(self, x):
        x = x.contiguous()  # 确保x是连续的
        x = x.view(x.size(0), self._input_size)  # 重塑x的形状
        return self.features(x)  # 返回特征提取层的输出


# 定义一个CNN模型，继承自MultiTaskModule
class SI_CNN(MultiTaskModule):
    def __init__(self, hidden_size=512):
        super().__init__()  # 调用父类的构造函数
        layers = nn.Sequential(*(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=32, out_channels=32, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Conv2d(in_channels=32, out_channels=64, kernel_size=(3, 3), padding=(1, 1)),
                                 nn.ReLU(inplace=True),
                                 nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3)),
                                 nn.ReLU(inplace=True),
                                 nn.MaxPool2d((2, 2)),
                                 nn.Dropout(p=0.25),
                                 nn.Flatten(),
                                 nn.Linear(2304, hidden_size),
                                 nn.ReLU(inplace=True),
                                 nn.Dropout(p=0.5)
                                 ))
        self.features = nn.Sequential(*layers)  # 特征提取层
        self.classifier = MultiHeadClassifier(hidden_size, initial_out_features=10)  # 多头部分类器

    def forward(self, x, task_labels):
        x = self.features(x)  # 通过特征提取层
        x = self.classifier(x, task_labels)  # 通过分类器
        return x  # 返回输出


# 定义一个用于展平张量的nn模块
class FlattenP(nn.Module):
    '''一个nn模块，用于将多维张量展平为2维张量。'''

    def forward(self, x):
        batch_size = x.size(0)  # 获取批次大小
        return x.view(batch_size, -1)  # 重塑x的形状

    def __repr__(self):
        tmpstr = self.__class__.__name__ + '()'  # 类的名称
        return tmpstr  # 返回类的名称


# 定义一个使用分组稀疏正则化（Group Sparsity Sparse Subnetwork，GSS）的MLP模型
class MLP_gss(nn.Module):
    def __init__(self, sizes, bias=True):
        super(MLP_gss, self).__init__()  # 调用父类的构造函数
        layers = []  # 初始化层列表

        # 遍历sizes列表，创建线性层和激活函数
        for i in range(0, len(sizes) - 1):
            if i < (len(sizes)-2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1]))
                layers.append(nn.ReLU())
            else:
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=bias))

        self.net = nn.Sequential(FlattenP(), *layers)  # 创建网络

    def forward(self, x):
        return self.net(x)  # 返回网络的输出


# 定义一个列表，包含所有定义的类
__all__ = ['MultiHeadMLP', 'MLP', 'SI_CNN', 'MLP_gss']