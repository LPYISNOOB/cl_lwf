# from pathlib import Path
# import inspect
# from pandas import read_csv
# import os
# import tests
#
#
# def pandas_to_list(input_str):
#     return [float(el) for el in input_str.strip('[] ').split(' ')]
#
#
# def get_target_result(strat_name: str, bench_name: str):
#     """
#     Read the target_results.csv file and retrieve the target performance for
#     the given strategy on the given benchmark.
#     :param strat_name: strategy name as found in the target file
#     :param bench_name: benchmark name as found in the target file
#     :return: target performance (either a float or a list of floats)
#     """
#
#     p = os.path.join(Path(inspect.getabsfile(tests)).parent, 'target_results.csv')
#     data = read_csv(p, sep=',', comment='#')
#     target = data[(data['strategy'] == strat_name) & (data['benchmark'] == bench_name)]['result'].values[0]
#     if isinstance(target, str) and target.startswith('[') and target.endswith(']'):
#         target = pandas_to_list(target)
#     else:
#         target = float(target)
#     return target
#
#
# def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
#     """
#     Compute the average of a metric based on the provided metric name.
#     The average is computed across the instance of the metrics containing the
#     given metric name in the input dictionary.
#     :param metric_dict: dictionary containing metric name as keys and metric value as value.
#         This dictionary is usually returned by the `eval` method of Avalanche strategies.
#     :param metric_name: the metric name (or a part of it), to be used as pattern to filter the dictionary
#     :return: a number representing the average of all the metric containing `metric_name` in their name
#     """
#
#     avg_stream_acc = []
#     for k, v in metric_dict.items():
#         if k.startswith(metric_name):
#             avg_stream_acc.append(v)
#     return sum(avg_stream_acc) / float(len(avg_stream_acc))

from pathlib import Path
import inspect
from pandas import read_csv
import os
import tests

# 将字符串表示的列表转换为浮点数列表的辅助函数
def pandas_to_list(input_str):
    # 移除字符串中的'[] '，按空格分割字符串，并将每个元素转换为浮点数
    return [float(el) for el in input_str.strip('[] ').split(' ')]

# 从CSV文件中获取目标结果的函数
def get_target_result(strategy_name: str, benchmark_name: str):
    """
    从target_results.csv文件中读取给定策略在特定基准测试上的目标性能。
    :param strategy_name: 策略名称，应与CSV文件中的名称匹配
    :param benchmark_name: 基准测试名称，应与CSV文件中的名称匹配
    :return: 目标性能（可能是浮点数或浮点数列表）
    """
    # 获取当前tests模块的绝对路径，并构建target_results.csv文件的路径
    p = os.path.join(Path(inspect.getabsfile(tests)).parent, 'target_results.csv')
    # 读取CSV文件，忽略以#开头的注释行
    data = read_csv(p, sep=',', comment='#')
    # 根据策略名称和基准测试名称筛选目标结果
    target = data[(data['strategy'] == strategy_name) & (data['benchmark'] == benchmark_name)]['result'].values[0]
    # 如果目标结果是字符串形式，且以'['开头和']'结尾，则将其转换为浮点数列表
    if isinstance(target, str) and target.startswith('[') and target.endswith(']'):
        target = pandas_to_list(target)
    # 否则，将目标结果转换为浮点数
    else:
        target = float(target)
    return target

# 计算指标平均值的函数
def get_average_metric(metric_dict: dict, metric_name: str = 'Top1_Acc_Stream'):
    """
    根据提供的指标名称计算指标的平均值。
    平均值是根据输入字典中包含给定指标名称的实例计算得出的。
    :param metric_dict: 包含指标名称作为键和指标值作为值的字典。
        这个字典通常由Avalanche策略的`eval`方法返回。
    :param metric_name: 指标名称（或部分名称），用作过滤字典的模式
    :return: 一个数字，表示包含`metric_name`在其名称中的所有指标的平均值
    """
    # 初始化一个空列表来存储所有匹配的指标值
    avg_stream_acc = []
    # 遍历字典，查找以metric_name开头的键
    for k, v in metric_dict.items():
        if k.startswith(metric_name):
            avg_stream_acc.append(v)
    # 计算并返回所有匹配指标值的平均值
    return sum(avg_stream_acc) / float(len(avg_stream_acc))

