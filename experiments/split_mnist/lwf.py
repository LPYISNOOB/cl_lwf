# import avalanche as avl
# import torch
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from avalanche.evaluation import metrics as metrics
# from models import MLP
# from experiments.utils import set_seed, create_default_args
#
#
# class LwFCEPenalty(avl.training.LwF):
#     """This wrapper around LwF computes the total loss
#     by diminishing the cross-entropy contribution over time,
#     as per the paper
#     "Three scenarios for continual learning" by van de Ven et. al. (2018).
#     https://arxiv.org/pdf/1904.07734.pdf
#     The loss is L_tot = (1/n_exp_so_far) * L_cross_entropy +
#                         alpha[current_exp] * L_distillation
#     """
#     def _before_backward(self, **kwargs):
#         self.loss *= float(1/(self.clock.train_exp_counter+1))
#         super()._before_backward(**kwargs)
#
#
# def lwf_smnist(override_args=None):
#     """
#     "Learning without Forgetting" by Li et. al. (2016).
#     http://arxiv.org/abs/1606.09282
#     Since experimental setup of the paper is quite outdated and not
#     easily reproducible, this experiment is based on
#     "Three scenarios for continual learning" by van de Ven et. al. (2018).
#     https://arxiv.org/pdf/1904.07734.pdf
#
#     The hyper-parameter alpha controlling the regularization is increased over time, resulting
#     in a regularization of  (1- 1/n_exp_so_far) * L_distillation
#     """
#     args = create_default_args({'cuda': 0,
#                                 # 'lwf_alpha': [0, 0.5, 1.33333, 2.25, 3.2],
#                                 # 'lwf_alpha': [0.5, 1.0, 1.5, 2.0, 2.5],
#                                 # 'lwf_temperature': 2, 'epochs': 21,
#                                 #
#                                 # 'layers': 1, 'hidden_size': 1000,
#                                 # 'learning_rate': 0.001, 'train_mb_size': 128,
#                                 # 'seed': None
#                                 'cuda': 0,
#                                 'lwf_alpha': [0.5, 1.0, 1.5, 2.0, 2.5],
#                                 'lwf_temperature': 1.0, 'epochs': 101,
#                                 'layers': 2, 'hidden_size': 1000,
#                                 'learning_rate': 0.0001, 'train_mb_size': 256,
#                                 'seed': None
#                                 }, override_args)
#     set_seed(args.seed)
#     device = torch.device(f"cuda:{args.cuda}"
#                           if torch.cuda.is_available() and
#                           args.cuda >= 0 else "cpu")
#
#     benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)
#     model = MLP(hidden_size=args.hidden_size, hidden_layers=args.layers,
#                 initial_out_features=0, relu_act=False)
#     criterion = CrossEntropyLoss()
#
#     interactive_logger = avl.logging.InteractiveLogger()
#
#     evaluation_plugin = avl.training.plugins.EvaluationPlugin(
#         metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
#         loggers=[interactive_logger])
#
#     cl_strategy = LwFCEPenalty(
#         model, SGD(model.parameters(), lr=args.learning_rate), criterion,
#         alpha=args.lwf_alpha, temperature=args.lwf_temperature,
#         train_mb_size=args.train_mb_size, train_epochs=args.epochs,
#         device=device, evaluator=evaluation_plugin)
#
#     res = None
#     for experience in benchmark.train_stream:
#         cl_strategy.train(experience)
#         res = cl_strategy.eval(benchmark.test_stream)
#
#     return res
#
#
# if __name__ == '__main__':
#     res = lwf_smnist()
#     print(res)

import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics#指标”或“度量标准
from models import MLP
from experiments.utils import set_seed, create_default_args

# 定义一个LwFCEPenalty类，继承自avl.training.LwF
# 这个类计算总损失时，会根据论文“Three scenarios for continual learning”
# 中的描述，随着时间减少交叉熵的贡献
class LwFCEPenalty(avl.training.LwF):
    def _before_backward(self, **kwargs):
        # 计算总损失，包括交叉熵损失和蒸馏损失
        # 交叉熵损失随着时间推移而减少，蒸馏损失则增加 蒸馏损失是旧任务知识蒸馏后的损失，交叉熵损失是新任务损失
        self.loss *= float(1/(self.clock.train_exp_counter+1))
        # 调用父类的_backward方法来继续执行反向传播
        super()._before_backward(**kwargs)

# 定义lwf_smnist函数，用于运行LwF实验
def lwf_smnist(override_args=None):
    """
    运行“Learning without Forgetting”实验，基于“Three scenarios for continual learning”
    论文中的设置。实验通过增加时间步的alpha参数，实现正则化。
    """
    # 创建默认参数，并允许覆盖
    args = create_default_args({
        # 'cuda': 0,
        # 'lwf_alpha': [0, 0.5, 1.33333, 2.25, 3.2],
        # 'lwf_temperature': 2, 'epochs': 21,
        # 'layers': 1, 'hidden_size': 200,
        # 'learning_rate': 0.001, 'train_mb_size': 128,
        # 'seed': None
        'cuda': 0,
        'lwf_alpha': [0, 0.5, 1.33333, 2.25, 3.2],
        'lwf_temperature': 1.0, 'epochs': 21,
        'layers': 2, 'hidden_size': 500,
        'learning_rate': 0.0001, 'train_mb_size': 256,
        'seed': None

    }, override_args)#可覆盖
    # 设置随机种子
    set_seed(args.seed)
    # 选择设备
    device = torch.device(f"cuda:{args.cuda}"
                          if torch.cuda.is_available() and
                          args.cuda >= 0 else "cpu")

    # 创建SplitMNIST基准测试
    benchmark = avl.benchmarks.SplitMNIST(5, return_task_id=False)
    # 创建MLP模型
    model = MLP(hidden_size=args.hidden_size, hidden_layers=args.layers,
                initial_out_features=0, relu_act=False)
    # 定义交叉熵损失函数
    criterion = CrossEntropyLoss()

    # 创建交互式日志记录器
    interactive_logger = avl.logging.InteractiveLogger()

    # 创建评估插件
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    # 创建LwFCEPenalty策略
    cl_strategy = LwFCEPenalty(
        model, SGD(model.parameters(), lr=args.learning_rate), criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs,
        device=device, evaluator=evaluation_plugin)

    # 运行策略
    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    return res

# 主程序入口
if __name__ == '__main__':
    # 运行实验并打印结果
    res = lwf_smnist()
    print(res)
