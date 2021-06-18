import torch.nn as nn
import torch.nn.functional as F
import torch
import gym
import numpy as np
"""
    损失函数交叉熵杀我，reduction='none'
"""


class NN(nn.Module):
    def __init__(self, input_dim=4, output_dim=2):
        super(NN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = 64
        self.Linear1 = nn.Linear(self.input_dim, self.hidden_dim)
        # nn.init.normal_(self.Linear1.weight, 0, 0.3)
        # nn.init.constant_(self.Linear1.bias, 0.1)
        self.Linear2 = nn.Linear(self.hidden_dim, self.output_dim)
        # nn.init.normal_(self.Linear2.weight, 0, 0.3)
        # nn.init.constant_(self.Linear2.bias, 0.1)

    def forward(self, x):
        # 将numpy转为tensor
        x = torch.from_numpy(x).float()
        x = self.Linear1(x)
        x = F.tanh(x)
        x = self.Linear2(x)
        # print(x)
        y = F.softmax(x, dim=1)
        return x, y


class Agent(object):
    def __init__(self, env, input_dim=4, output_dim=2):
        super(Agent, self).__init__()
        self.env = env
        self.input_dim = input_dim
        self.output_dim = output_dim
        # self.Net = net

    def choose_action(self, s, net, random=True):
        x, y = net(s)
        # 随机性策略
        if random:
            a = np.random.choice(np.arange(self.output_dim), p=y[0].detach().numpy())
        # 确定性策略
        else:
            a = np.argmax(y.detach().numpy())
        # print(a)
        return int(a)

    def act(self, a):
        return self.env.step(a)

    def sample(self, net):
        obs = []
        action = []
        reward = []
        s = self.env.reset()
        total_r = 0
        while True:
            s = np.expand_dims(s, axis=0)
            obs.append(s)
            a = self.choose_action(s, net)
            action.append(a)
            s_, r, done, info = self.act(a)
            reward.append(r)
            total_r += r
            # 结束
            if done:
                break
            s = s_

        obs = np.concatenate(obs, axis=0)
        print(total_r)
        return obs, action, reward

    def modify(self, reward):
        modify_reward = []
        dicount_factor = 1
        for i in range(len(reward)):
            acc_r = 0
            for j in range(i, len(reward)):
                acc_r += dicount_factor ** (j - i) * reward[j]
            modify_reward.append(acc_r)
        print(modify_reward)
        # 归一化
        modify_reward = torch.tensor(modify_reward)
        # modify_reward -= modify_reward.mean()
        # modify_reward /= modify_reward.std()
        return modify_reward

    def learn(self, net, obs, action, reward):
        y_pred = net(obs)
        # 计算交叉熵
        optim = torch.optim.Adam(net.parameters(), lr=0.02)
        loss_func = nn.CrossEntropyLoss(reduction='none')
        loss = loss_func(y_pred[0], torch.tensor(action))

        modify_reward = self.modify(reward)

        # 加权的loss
        loss = loss * modify_reward
        loss = loss.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()

    def test(self, net):
        total_r = 0
        s = self.env.reset()
        while True:
            self.env.render()
            # 加一维
            s = np.expand_dims(s, axis=0)
            a = self.choose_action(s, net, False)
            s_, r, done, info = self.act(a)
            total_r += r

            if done:
                break
            s = s_

        print("total_reward: {}".format(total_r))


def main():
    env = gym.make("CartPole-v0")
    obs_num = env.observation_space.shape[0]
    # print(obs_num)
    act_num = env.action_space.n
    # print(act_num)
    Net = NN()
    agent = Agent(env, obs_num, act_num)
    epoch = 400

    for i in range(epoch):
        # env.render()
        obs, action, reward = agent.sample(Net)
        agent.learn(Net, obs, action, reward)
        agent.test(Net)


if __name__ == "__main__":
    main()