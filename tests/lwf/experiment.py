# import unittest
# from tests.utils import get_average_metric, get_target_result
# from experiments.split_mnist import lwf_smnist
# from experiments.split_tiny_imagenet import lwf_stinyimagenet
#
#
# class LwF(unittest.TestCase):
#     """
#     Reproducing Learning without Forgetting. Original paper is
#     "Learning without Forgetting" by Li et. al. (2016).
#     http://arxiv.org/abs/1606.09282
#     Since experimental setup of the paper is quite outdated and not
#     easily reproducible, this class reproduces LwF experiments
#     on Split MNIST from
#     "Three scenarios for continual learning" by van de Ven et. al. (2018).
#     https://arxiv.org/pdf/1904.07734.pdf
#     We managed to surpass the performances reported in the paper by slightly
#     changing the model architecture or the training hyperparameters.
#     Experiments on Tiny Image Net are taken from
#     "A continual learning survey: Defying forgetting in classification tasks" De Lange et. al. (2021).
#     https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197
#     """
#
#     def test_smnist(self):
#         """Split MNIST benchmark"""
#         res = lwf_smnist({'seed': 0})
#         avg_stream_acc = get_average_metric(res)
#         print(f"LwF-SMNIST Average Stream Accuracy: {avg_stream_acc:.2f}")
#
#         target_acc = float(get_target_result('lwf', 'smnist'))
#         if target_acc > avg_stream_acc:
#             self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.01)
#
#     def test_stinyimagenet(self):
#         """Split Tiny ImageNet benchmark"""
#         res = lwf_stinyimagenet({'seed': 0})
#         avg_stream_acc = get_average_metric(res)
#         print(f"LwF-SplitTinyImageNet Average Stream Accuracy: {avg_stream_acc:.2f}")
#
#         target_acc = float(get_target_result('lwf', 'stiny-imagenet'))
#         if target_acc > avg_stream_acc:
#             self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
#
#
import unittest
from tests.utils import get_average_metric, get_target_result
from experiments.split_mnist import lwf_smnist
from experiments.split_tiny_imagenet import lwf_stinyimagenet

class LwF(unittest.TestCase):
    """
    重现“Learning without Forgetting”实验。原始论文由Li等人于2016年发表。
    http://arxiv.org/abs/1606.09282
    由于原始论文的实验设置已经过时且不易复现，本类在van de Ven等人2018年的
    “Three scenarios for continual learning”的基础上复现了Split MNIST上的LwF实验。
    https://arxiv.org/pdf/1904.07734.pdf
    通过稍微改变模型架构或训练超参数，我们超越了论文中报告的性能。
    关于Tiny Image Net的实验取自De Lange等人2021年的论文，
    “A continual learning survey: Defying forgetting in classification tasks”。
    https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9349197
    """

    def test_smnist(self):
        """
        Split MNIST基准测试
        这个测试用例检查在Split MNIST基准测试上Learning without Forgetting (LwF)方法的平均流准确率，
        并将其与目标准确率进行比较。
        """
        # 使用固定的种子运行Split MNIST上的LwF实验以确保可复现性
        res = lwf_smnist({'seed': 0})
        # 从实验结果中计算平均流准确率
        avg_stream_acc = get_average_metric(res)
        # 打印平均流准确率
        print(f"Split MNIST基准测试上的LwF平均流准确率: {avg_stream_acc:.2f}")

        # 获取Split MNIST基准测试的目标准确率
        target_acc = float(get_target_result('lwf', 'smnist'))
        # 检查目标准确率是否大于平均流准确率
        if target_acc > avg_stream_acc:
            # 使用assertAlmostEqual来考虑准确率的微小变化
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.01)

    def test_stinyimagenet(self):
        """
        Split Tiny ImageNet基准测试
        这个测试用例检查在Split Tiny ImageNet基准测试上Learning without Forgetting (LwF)方法的平均流准确率，
        并将其与目标准确率进行比较。
        """
        # 使用固定的种子运行Split Tiny ImageNet上的LwF实验以确保可复现性
        res = lwf_stinyimagenet({'seed': 0})
        # 从实验结果中计算平均流准确率
        avg_stream_acc = get_average_metric(res)
        # 打印平均流准确率
        print(f"Split Tiny ImageNet基准测试上的LwF平均流准确率: {avg_stream_acc:.2f}")

        # 获取Split Tiny ImageNet基准测试的目标准确率
        target_acc = float(get_target_result('lwf', 'stiny-imagenet'))
        # 检查目标准确率是否大于平均流准确率
        if target_acc > avg_stream_acc:
            # 使用assertAlmostEqual来考虑准确率的微小变化
            self.assertAlmostEqual(target_acc, avg_stream_acc, delta=0.03)
