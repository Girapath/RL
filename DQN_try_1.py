"""
    训练好的模型在E:\q_val和E:\q_target
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
import numpy as np
import copy
from collections import namedtuple
import random
import torch.optim as optim

class Net(nn.Module):
    def __init__(self, input_dim=1, output_dim=1, hidden_dim=8):
        super(Net, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.Linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        self.Linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.Linear3 = nn.Linear(self.hidden_dim, self.output_dim)

    def forward(self, x):
        x = F.relu(self.Linear1(x), inplace=True)
        x = F.relu(self.Linear2(x), inplace=True)
        y_pred = self.Linear3(x)
        return y_pred

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'done', 'next_state'))
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):  # 采样
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self, env, Q_eval, Q_target):
        self.env = env
        self.Q_eval = Q_eval
        self.Q_target = Q_target
        # 可选动作数
        self.action_num = self.env.action_space.n
        # 每个状态的参数个数
        self.obs_num = self.env.observation_space.shape[0]
        # 要存放两个状态和reward、action，因此列数为obs_num*2+2
        self.buffer = ReplayMemory(10000)
        self.optim = optim.Adam(self.Q_eval.parameters(), lr=0.01)
        self.loss_func = nn.MSELoss()
        # self.memorySize = 500
        # self.memoryPool = np.zeros((self.memorySize, self.obs_num*2+2))
        # self.memoryNum = 0
        # self.counter = 0

    def put(self, s0, a0, r, t, s1):
        self.buffer.push(s0, a0, r, t, s1)

    def chooseAction(self, s, israndom=True):
        # s为状态，返回选择的action
        # self.counter += 1
        # a = 0.7+0.001*self.counter
        ran = np.random.random()
        s = torch.unsqueeze(torch.FloatTensor(s), 0)
        # greedy
        if ran > 0.9 and israndom:
            a = np.random.choice(self.action_num, size=1)
            # print("chooseaction1: "+str(type(a)))
        else:
            y_pred = self.Q_eval(s)
            # print("chooseaction2 :" + str(type(y_pred)))
            # 出错，y_pred为tensor，不能直接用np.argmax
            a = np.argmax(y_pred.data.numpy())

        return int(a)

    def act(self, a):
        # given an action, return s_, r, done, info
        return self.env.step(a)

    # def learnMemory(self):
    #     # 学习并存储记忆
    #     for i in range(100):
    #         # total_r = 0
    #         s = self.env.reset()
    #         # 不加这行报错,放到choose action中，否则影响storememory
    #         # s = torch.unsqueeze(torch.FloatTensor(s), 0)
    #         # print(type(s))
    #         while True:
    #             a = self.chooseAction(s)
    #             # print("type a:"+str(type(a)))
    #             s_, r, done, info = self.act(int(a))
    #             # 修改reward
    #             x, x_dot, theta, theta_dot = s_
    #             r1 = (self.env.x_threshold - abs(x)) / self.env.x_threshold - 0.8  # env.x_threshold=2.4,在cartpole.py中有写
    #             r2 = (self.env.theta_threshold_radians - abs(
    #                 theta)) / self.env.theta_threshold_radians - 0.5  # env.theta_threshold_radians=12度，同样在cartpole.py中有写
    #             r = r1 + r2
    #             self.storememory(s, a, r, s_)
    #             # total_r += r
    #
    #             if done:
    #                 # print("total_r: {}".format(total_r))
    #                 break
    #             # s更新为下一个状态
    #             s = s_

    def updateQ_target(self):
        # self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target = copy.deepcopy(self.Q_eval)

    # def storememory(self, s, a, r, s_):
    #     # s, [a], [r], s_维度 不同，出错
    #     # print(s.shape)
    #     transition = np.hstack((s, [a, r], s_))
    #     index = self.memoryNum % self.memorySize
    #     self.memoryPool[index, :] = transition
    #     self.memoryNum += 1

    def update_parameters(self):
        batch_size = 32
        GAMMA = 0.9
        if self.buffer.__len__() < batch_size:
            return
        samples = self.buffer.sample(batch_size)
        batch = Transition(*zip(*samples))
        # 将tuple转化为numpy
        tmp = np.vstack(batch.action)
        # 转化成Tensor
        state_batch = torch.Tensor(batch.state)
        action_batch = torch.LongTensor(tmp.astype(int))
        reward_batch = torch.Tensor(batch.reward)
        done_batch = torch.Tensor(batch.done)
        next_state_batch = torch.Tensor(batch.next_state)
        # print(torch.max(self.Q_target(next_state_batch).detach(), dim=1, keepdim=True)[0])
        q_next = torch.max(self.Q_target(next_state_batch).detach(), dim=1,
                           keepdim=True)[0]
        q_eval = self.Q_eval(state_batch).gather(1, action_batch)
        q_tar = reward_batch.unsqueeze(1) + (1-done_batch) * GAMMA * q_next
        loss = self.loss_func(q_eval, q_tar)
        # print(loss)
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

    # def learn(self):
    #     batchsize = 50
    #     if self.memoryNum < batchsize:
    #         return
    #
    #     gamma = 0.9
    #     learning_rate = 0.1
    #     criterion = nn.MSELoss()
    #     optimizer = torch.optim.Adam(self.Q_eval.parameters(), lr=learning_rate)
    #     # 选batchsize个transition进行训练
    #     index = np.random.choice(self.memorySize, size=batchsize)
    #     train_memory = self.memoryPool[index, :]
    #     # 分开存储s, a, r, s_
    #     train_s = torch.FloatTensor(train_memory[:, :self.obs_num])
    #     train_a = torch.LongTensor(train_memory[:, self.obs_num:self.obs_num+1])
    #     train_r = torch.FloatTensor(train_memory[:, self.obs_num+1:self.obs_num+2])
    #     train_s_ = torch.FloatTensor(train_memory[:, self.obs_num+2:])
    #
    #     q_next = self.Q_target(train_s_).detach()
    #     # print(type(q_next.max(dim=1)))
    #     q_target = train_r + gamma * q_next.max(dim=1)[0]
    #     for i in range(20):
    #         q_eval = self.Q_eval(train_s).gather(1, train_a)
    #
    #         loss = criterion(q_eval, q_target)
    #         # print("第{}轮loss: {}".format(i, loss))
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step()

    def test(self, load=False):
        total_r = 0

        # 100次eposide
        for i in range(20):
            s = self.env.reset()
            eposide_r = 0

            while True:
                self.env.render()
                a = self.chooseAction(s)
                s_, r, done, info = self.act(a)
                eposide_r += r

                if done:
                    print("eposide_r: {}".format(eposide_r))
                    break
                s = s_
            total_r += eposide_r

        avg = total_r/20
        print("Average_r: {}".format(total_r/20))
        self.env.close()
        if avg > 200:
            if load:
                torch.save(self.Q_eval.state_dict(), '\q_eval')
                torch.save(self.Q_target.state_dict(), '\q_target')
            return True
        else:
            return False


def main():
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    action_num = env.action_space.n
    # print(type(action_num))
    obs_num = env.observation_space.shape[0]
    # print(type(obs_num))
    Q_eval = Net(obs_num, action_num, 256)
    Q_target = Net(obs_num, action_num, 256)

    dqn = Agent(env, Q_eval, Q_target)
    update_step = 10
    # print("116 ok")
    # dqn.learnMemory()
    for i_episode in range(1000):

        s = env.reset()  # 得到环境的反馈，现在的状态
        tot_reward = 0  # 每个episode的总reward
        tot_time = 0  # 实际每轮运行的时间 （reward的定义可能不一样）
        while True:
            env.render()
            a = dqn.chooseAction(s)
            s_, r, done, info = dqn.act(a)
            tot_time += r  # 计算当前episode的总时间
            # 修改reward
            x, x_dot, theta, theta_dot = s_
            r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8  # env.x_threshold=2.4,在cartpole.py中有写
            r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5  # env.theta_threshold_radians=12度，同样在cartpole.py中有写
            r = r1 + r2
            tot_reward += r  # 计算当前episode的总reward
            if done:
                t = 1
            else:
                t = 0
            dqn.put(s, a, r, t, s_)
            s = s_
            # learn 100遍
            dqn.update_parameters()
            # flag = dqn.test()
            if done:
                print('Episode ', i_episode, 'tot_time: ', tot_time,
                      ' tot_reward: ', tot_reward)
                break

        if i_episode % update_step == 0:
            dqn.updateQ_target()
    env.close()


def source():
    # 用gym.make('CartPole-v0')导入gym定义好的环境，对于更复杂的问题则需要自定义环境
    env = gym.make('CartPole-v0')

    # 第一步不用agent，采用随机策略进行对比
    env.reset()  # 初始化环境
    random_episodes = 0
    reward_sum = 0
    while random_episodes < 10:
        env.render()
        observation, reward, done, _ = env.step(np.random.randint(0, 2))
        # np.random.randint创建随机action，env.step执行action
        reward_sum += reward
        # 最后一个action也获得奖励
        if done:
            random_episodes += 1
            print("Reward for this episodes was:", reward_sum)
            reward_sum = 0  # 重置reward
            env.reset()
    env.close()


def use_model():
    '''
        使用训练好的模型
    :return:
    '''
    env = gym.make('CartPole-v0')
    env = env.unwrapped
    action_num = env.action_space.n
    # print(type(action_num))
    obs_num = env.observation_space.shape[0]
    # print(type(obs_num))
    Q_eval = Net(obs_num, action_num, 20)
    Q_eval.load_state_dict(torch.load('\q_eval'))
    Q_target = Net(obs_num, action_num, 20)
    Q_target.load_state_dict(torch.load('\q_target'))

    dqn = Agent(env, Q_eval, Q_target)
    learn_step = 1
    # print("116 ok")
    for i_episode in range(40):
        s = env.reset()  # 得到环境的反馈，现在的状态
        # ep_r = 0
        # while True:
        env.render()  # 环境渲染，可以看到屏幕上的环境
        # memory 100个episode
        # dqn.learnMemory()

        # ep_r += r
        # learn 100遍
        # dqn.learn()
        flag = dqn.test(load=False)
        # if flag:
        #     break
        # dqn.updateQ_target()
    env.close()


if __name__ == "__main__":
    # use_model()
    main()