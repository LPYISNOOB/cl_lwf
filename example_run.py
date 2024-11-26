"""
This script shows how to run an experiment on a specific strategy and benchmark.
You can override default parameters by providing a dictionary as input to the method.
You can find all the parameters used by the experiment in the source file of the experiment.
"""
# # select the experiment
# from experiments.split_mnist import synaptic_intelligence_smnist
# # run the experiment with custom parameters (do not provide arguments to use default parameters)
# synaptic_intelligence_smnist({'learning_rate': 1e-3, 'si_lambda': 1})

# from experiments.split_mnist import lwf_smnist
# lwf_smnist({'epochs': 21,'learning_rate': 1e-3,'train_mb_size': 128})
# from experiments.split_cifar10.online_replay import online_replay_scifar10
# online_replay_scifar10()

# from experiments.split_tiny_imagenet import packnet_stinyimagenet
# packnet_stinyimagenet()

from tests.lwf.experiment import LwF
mylwf = LwF()
# 使用实例调用方法
#mylwf.test_smnist()
mylwf.test_stinyimagenet()



