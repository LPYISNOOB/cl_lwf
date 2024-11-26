import avalanche as avl
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import SGD
from avalanche.evaluation import metrics as metrics
from models import MultiHeadVGGSmall
from experiments.utils import set_seed, create_default_args
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from spikingjelly.activation_based.ann2snn.converter import Converter
from spikingjelly.activation_based.ann2snn.utils import download_url

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
        'lwf_alpha': 1,
        'lwf_temperature': 2,
        'epochs': 70,
        'learning_rate': 1e-3,
        'train_mb_size': 200,
        'seed': None,
        'dataset_root': 'C:\\Users\\86185\\Desktop\\cl\\continual-learning-baselines-main'
    }, override_args)

    # 设置随机种子，确保实验可复现
    set_seed(args.seed)

    # 选择设备，优先使用CUDA设备
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() and args.cuda >= 0 else "cpu")

    # 创建SplitTinyImageNet基准测试，10个任务，每个任务包含200个类别
    benchmark = avl.benchmarks.SplitTinyImageNet(10, return_task_id=True, dataset_root=args.dataset_root)

    # 创建MultiHeadVGGSmall模型，适用于Tiny ImageNet数据集
    model = MultiHeadVGGSmall(n_classes=200)


    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # 数据集路径
    dataset_root = 'C:\\Users\\86185\\Desktop\\cl\\continual-learning-baselines\\tiny-imagenet-200'

    # 加载数据集
    train_dataset = datasets.ImageFolder(root=f'{dataset_root}/train', transform=transform)
    test_dataset = datasets.ImageFolder(root=f'{dataset_root}/val', transform=transform)

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)

    # 定义模型
    model = MultiHeadVGGSmall(n_classes=200)

    # 定义转换器
    converter = Converter(dataloader=train_loader, device='cuda' if torch.cuda.is_available() else 'cpu', mode='Max')

    # 执行转换
    snn_model = converter(model)

    # 保存或使用转换后的模型
    # torch.save(snn_model.state_dict(), 'snn_model.pth')

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
        snn_model,
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