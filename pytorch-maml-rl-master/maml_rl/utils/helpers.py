import gym
import torch

from functools import reduce
from operator import mul

from maml_rl.policies import CategoricalMLPPolicy, NormalMLPPolicy


def get_policy_for_env(env, hidden_sizes=(100, 100), nonlinearity='relu'):
    # 判断是连续控制还是离散控制问题
    continuous_actions = isinstance(env.action_space, gym.spaces.Box)
    input_size = get_input_size(env) # 输入是状态空间的维数乘积
    nonlinearity = getattr(torch, nonlinearity) # 返回对象的属性值，这里是一个torch函数

    if continuous_actions:
        output_size = reduce(mul, env.action_space.shape, 1) # 输出是动作空间维数乘积
        policy = NormalMLPPolicy(input_size,
                                 output_size,
                                 hidden_sizes=tuple(hidden_sizes),
                                 nonlinearity=nonlinearity)
    else:
        output_size = env.action_space.n
        policy = CategoricalMLPPolicy(input_size,
                                      output_size,
                                      hidden_sizes=tuple(hidden_sizes),
                                      nonlinearity=nonlinearity)
    return policy

def get_input_size(env): # 返回状态空间维度的乘积
    return reduce(mul, env.observation_space.shape, 1)
