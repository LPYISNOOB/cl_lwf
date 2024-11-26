# import avalanche as avl
# import torch
# from torch.nn import CrossEntropyLoss
# from torch.optim import SGD
# from avalanche.evaluation import metrics as metrics
# from models import MultiHeadVGGSmall
# from experiments.utils import set_seed, create_default_args
#
#
# def lwf_stinyimagenet(override_args=None):
#     """
#     "Learning without Forgetting" by Li et. al. (2016).
#     http://arxiv.org/abs/1606.09282
#     Since experimental setup of the paper is quite outdated and not
#     easily reproducible, this experiment is based on
#     "A continual learning survey: Defying forgetting in classification tasks"
#     De Lange et. al. (2021).
#     https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197
#
#     We use a VGG network, which leads a lower performance than the one from
#     De Lange et. al. (2021).
#     """
#     args = create_default_args({'cuda': 0,
#                                 'lwf_alpha': 1, 'lwf_temperature': 2, 'epochs': 70,
#                                 'learning_rate': 1e-3, 'train_mb_size': 200, 'seed': None,
#                                 'dataset_root': None}, override_args)
#     set_seed(args.seed)
#     device = torch.device(f"cuda:{args.cuda}"
#                           if torch.cuda.is_available() and
#                           args.cuda >= 0 else "cpu")
#
#     benchmark = avl.benchmarks.SplitTinyImageNet(
#         10, return_task_id=True, dataset_root=args.dataset_root)
#     model = MultiHeadVGGSmall(n_classes=200)
#     criterion = CrossEntropyLoss()
#
#     interactive_logger = avl.logging.InteractiveLogger()
#
#     evaluation_plugin = avl.training.plugins.EvaluationPlugin(
#         metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
#         loggers=[interactive_logger])
#
#     cl_strategy = avl.training.LwF(
#         model,
#         SGD(model.parameters(), lr=args.learning_rate, momentum=0.9),
#         criterion,
#         alpha=args.lwf_alpha, temperature=args.lwf_temperature,
#         train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
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
# if __name__ == "__main__":
#     res = lwf_stinyimagenet()
#     print(res)

import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall
from experiments.utils import set_seed, create_default_args

# 定义LwF策略在Tiny ImageNet数据集上的应用函数
def lwf_stinyimagenet(override_args=None):
    """
    运行“Learning without Forgetting”实验，基于De Lange等人2021年的论文。
    由于原始论文的实验设置较为过时，不易复现，因此本实验基于
    "A continual learning survey: Defying forgetting in classification tasks"
    论文中的设置。

    使用VGG网络架构，与De Lange等人2021年相比，可能会获得较低的性能。
    """
    # 创建默认参数，并允许覆盖
    args = create_default_args({
        'cuda': 0,
        'lwf_alpha': 1, 'lwf_temperature': 2, 'epochs': 70,
        'learning_rate': 1e-3, 'train_mb_size': 200, 'seed': None,
        'dataset_root': None
    }, override_args)

    # 设置随机种子，确保实验可复现
    set_seed(args.seed)

    # 选择设备，优先使用CUDA设备
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    # 创建SplitTinyImageNet基准测试，10个任务，每个任务包含200个类别
    benchmark = avl.benchmarks.SplitTinyImageNet(10, return_task_id=True, dataset_root=args.dataset_root)

    # 创建MultiHeadVGGSmall模型，适用于Tiny ImageNet数据集
    model = MultiHeadVGGSmall(n_classes=200)
    #model = model.to(device)
    # 定义交叉熵损失函数
    criterion = CrossEntropyLoss()

    # 创建交互式日志记录器
    interactive_logger = avl.logging.InteractiveLogger()

    # 创建评估插件，记录准确率的指标
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
        loggers=[interactive_logger])

    # 创建LwF策略实例
    cl_strategy = avl.training.LwF(
        model,
        SGD(model.parameters(), lr=args.learning_rate, momentum=0.9),
        criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    # 运行LwF策略，训练并评估模型
    res = None
    for experience in benchmark.train_stream:
        cl_strategy.train(experience)
        res = cl_strategy.eval(benchmark.test_stream)

    # 返回评估结果
    return res

# 主程序入口，运行实验并打印结果
if __name__ == "__main__":
    res = lwf_stinyimagenet()
    print(res)