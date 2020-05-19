import numpy as np
import matplotlib.pyplot as plt


class KB_Game:
    def __init__(self, *args, **kwargs):
        self.q = np.array([0.0, 0.0, 0.0])  # 每个臂的平均回报，初始值设为0
        self.action_counts = np.array([0, 0, 0])  # 每个臂被摇动的次数
        self.current_cumulative_rewards = 0.0  # 当前累积回报总和
        self.actions = [1, 2, 3]  # 动作空间
        self.counts = 0  # 玩家玩游戏的次数
        self.counts_history = []  # 玩家玩游戏的次数记录
        self.cumulative_rewards_history = []  # 累积奖励记录
        self.a = 1  # 玩家的当前动作，随机化初始值
        self.r = 0  # 当前回报

    def step(self, a):  # 执行当前动作，得到的奖励
        r = 0
        if a == 1:
            r = np.random.normal(1, 1)  # 不同臂返回的奖励服从某个分布
        if a == 2:
            r = np.random.normal(2, 1)
        if a == 3:
            r = np.random.normal(1.5, 1)
        return r

    def choose_action(self, policy, **kwargs):  # 按照不同的策略选择动作
        action = 0
        if policy == 'e_greedy':
            if np.random.random() < kwargs['epsilon']:
                action = np.random.randint(1, 4)
            else:
                action = np.argmax(self.q) + 1
        if policy == 'ucb':
            c_ratio = kwargs['c_ratio']
            if 0 in self.action_counts:  # np.where(condition) 返回满足条件的元素的坐标，返回的是一个元组，里面有numpy数组
                action = np.where(self.action_counts == 0)[0][0] + 1
            else:
                value = self.q + c_ratio * np.sqrt(np.log(self.counts) / self.action_counts)
                action = np.argmax(value) + 1
        if policy == 'boltzmann':
            tau = kwargs['temperature']
            p = np.exp(self.q / tau) / (np.sum(np.exp(self.q / tau)))
            action = np.random.choice([1, 2, 3], p=p.ravel())
        return action

    def train(self, play_total, policy, **kwargs):
        reward_1 = []
        reward_2 = []
        reward_3 = []
        for i in range(play_total):
            action = 0
            if policy == 'e_greedy':
                action = self.choose_action(policy, epsilon=kwargs['epsilon'])
            if policy == 'ucb':
                action = self.choose_action(policy, c_ratio=kwargs['c_ratio'])
            if policy == 'boltzmann':
                action = self.choose_action(policy, temperature=kwargs['temperature'])
            self.a = action
            self.r = self.step(self.a)
            self.counts += 1
            self.q[self.a-1] = (self.q[self.a-1] * self.action_counts[self.a-1] + self.r) / (
                        self.action_counts[self.a-1] + 1)
            self.action_counts[self.a-1] += 1
            reward_1.append([self.q[0]])
            reward_2.append([self.q[1]])
            reward_3.append([self.q[2]])
            self.current_cumulative_rewards += self.r
            self.cumulative_rewards_history.append(self.current_cumulative_rewards)
            self.counts_history.append(i)

    def reset(self):
        self.q = np.array([0.0, 0.0, 0.0])
        self.action_counts = np.array([0,0,0])
        self.current_cumulative_rewards = 0.0
        self.counts = 0
        self.counts_history = []
        self.cumulative_rewards_history = []
        self.a = 1
        self.r = 0

    def plot(self,colors,policy,hyper):
        plt.figure(1)
        plt.plot(self.counts_history,self.cumulative_rewards_history,colors,label=policy+' hyperparameter= '+str(hyper))
        plt.legend()
        plt.xlabel('n',fontsize=18)
        plt.ylabel('total rewards',fontsize=18)


if __name__ == '__main__':
    np.random.seed(0)
    k_gamble = KB_Game()
    total = 2000
    k_gamble.train(play_total=total,policy='e_greedy',epsilon=0.02)
    k_gamble.plot(colors='r',policy='e_greedy',hyper=0.02)
    k_gamble.reset()
    k_gamble.train(play_total=total, policy='boltzmann', temperature=1)
    k_gamble.plot(colors='b', policy='boltzmann',hyper=1)
    k_gamble.reset()
    k_gamble.train(play_total=total, policy='ucb', c_ratio=6)
    k_gamble.plot(colors='g', policy='ucb',hyper=6)
    plt.show()
