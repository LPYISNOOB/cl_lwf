import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall
# from models import
from spikingjelly.activation_based import neuron, functional
from experiments.utils import set_seed, create_default_args
import os
from torch import nn
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# 定义LwF策略在Tiny ImageNet数据集上的应用函数
from spikingjelly.activation_based.neuron import IFNode

# 假设您已经有了一个模型和一个名为name的层

def convert_to_snn(model):
    """
    将一个标准的ANN模型转换为SNN，通过替换ReLU激活函数为脉冲神经元，
    并移除BatchNorm层。
    """
    # 遍历模型中的所有子模块
    for name, module in model.named_children():
        # 如果模块是BatchNorm2d类型，用Identity模块替换它
        if isinstance(module, nn.BatchNorm2d):
            # 用Identity替换BatchNorm
            setattr(model, name, nn.Identity())
        # 如果模块是ReLU激活函数，用Sigmoid激活函数替换它
        # 这里注释掉了使用LIFNode（漏电积分-火或重置神经元）的代码
        # 可以根据需要选择是否使用
        elif isinstance(module, nn.ReLU):
            # 替换ReLU为Sigmoid激活函数
            #setattr(model, name, nn.Sigmoid())
            # 下面的代码被注释掉了，它们展示了如何使用LIFNode作为替代
            # setattr(model, name, neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.ATan()))
            # setattr(model, name, neuron.LIFNode(tau=2.0, surrogate_function=neuron.surrogate.Sigmoid()))
            #setattr(model, name, neuron.IFNode(v_threshold=1.0, v_reset=0.0, r=1.0))
            setattr(model, name, IFNode(v_threshold=1.0, v_reset=0.0, surrogate_function=neuron.surrogate.Sigmoid()))
        # 如果模块是Linear层，保持不变（如果需要可以修改）
        elif isinstance(module, nn.Linear):
            # 保持Linear层不变
            setattr(model, name, nn.Linear(module.in_features, module.out_features, bias=False))
        # 如果模块是Conv2d层，确保使用适合SNN的参数
        elif isinstance(module, nn.Conv2d):
            # 确保Conv2d使用适合SNN的参数
            setattr(model, name, nn.Conv2d(module.in_channels, module.out_channels,
                                           kernel_size=module.kernel_size,
                                           stride=module.stride,
                                           padding=module.padding,
                                           bias=False))
        else:
            # 对于其他类型的子模块，递归地调用convert_to_snn函数进行转换
            convert_to_snn(module)
    # 返回转换后的模型
    return model




def lwf_stinyimagenet(override_args=None):
    args = create_default_args({
        'cuda': 0,
        'lwf_alpha': 1, 'lwf_temperature': 2, 'epochs': 20,
        'learning_rate': 0.01, 'train_mb_size': 200, 'seed': None,
        'dataset_root': None, 'time_steps': 10
    }, override_args)

    set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 1 else "cpu")
    benchmark = avl.benchmarks.SplitTinyImageNet(10, return_task_id=True, dataset_root=args.dataset_root)
    model1 = MultiHeadVGGSmall(n_classes=200)
    model = convert_to_snn(model1)
    criterion = CrossEntropyLoss()
    interactive_logger = avl.logging.InteractiveLogger()

    # evaluation_plugin = avl.training.plugins.EvaluationPlugin(
    #     metrics.accuracy_metrics(epoch=True, experience=True, stream=True),
    #     loggers=[interactive_logger])

    # 创建评估插件，记录准确率和损失值
    evaluation_plugin = avl.training.plugins.EvaluationPlugin(
        metrics.accuracy_metrics(epoch=True, experience=True, stream=True),  # 准确率
        metrics.loss_metrics(epoch=True, experience=True, stream=True),  # 损失值
        loggers=[interactive_logger]
    )

    cl_strategy = avl.training.LwF(
        model,
        SGD(model.parameters(), lr=args.learning_rate, momentum=0.9),
        criterion,
        alpha=args.lwf_alpha, temperature=args.lwf_temperature,
        train_mb_size=args.train_mb_size, train_epochs=args.epochs, eval_mb_size=128,
        device=device, evaluator=evaluation_plugin)

    res = None
    # for experience in benchmark.train_stream:
    #     cl_strategy.train(experience, time_steps=args.time_steps)  # 传递时间步
    #     res = cl_strategy.eval(benchmark.test_stream)
    # return res
    for experience in benchmark.train_stream:
        print(f"Training on experience {experience.current_experience}")
        cl_strategy.train(experience)

        # 记录训练阶段损失
        train_loss = cl_strategy.loss.item()
        print(f"Train Loss after experience {experience.current_experience}: {train_loss:.4f}")

        # 测试阶段评估
        res = cl_strategy.eval(benchmark.test_stream)
        print("Evaluation results:", res)
    return res


# 主程序入口，运行实验并打印结果lalalalalal
if __name__ == "__main__":
    res = lwf_stinyimagenet()
    print(res)