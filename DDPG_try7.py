import torch
import torch.nn as nn
import numpy as np
import gym
import torch.nn.functional as F
import copy
from collections import namedtuple
import random
import torch.optim as optim

env = gym.make("Pendulum-v0")
obs_num = env.observation_space.shape[0]
learning_rate = 0.001
MemorySize = 1000
Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))


# class QNN(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=256):
#         super(QNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.Linear1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.Linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.Linear3 = nn.Linear(self.hidden_dim, self.output_dim)
#
#     def forward(self, obs, action):
#         # print(obs)
#         # print(action)
#         x = np.concatenate((obs.detach().numpy(), action.detach().numpy()), axis=1)
#         # print(x)
#         if isinstance(x, np.ndarray):
#             x = torch.from_numpy(x)
#         if isinstance(x, list):
#             x = torch.tensor(x)
#         x = x.float()
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         x = F.relu(self.Linear1(x))
#         # x = self.Linear2(x).clamp(max=0)
#         x = F.relu(self.Linear2(x))
#         x = self.Linear3(x)
#         return x
#
#     def copy(self):
#         return copy.deepcopy(self)
#
#
# class PNN(nn.Module):
#     def __init__(self, input_dim, output_dim, hidden_dim=256):
#         super(PNN, self).__init__()
#         self.input_dim = input_dim
#         self.output_dim = output_dim
#         self.hidden_dim = hidden_dim
#         self.Linear1 = nn.Linear(self.input_dim, self.hidden_dim)
#         self.Linear2 = nn.Linear(self.hidden_dim, self.hidden_dim)
#         self.Linear3 = nn.Linear(self.hidden_dim, self.output_dim)
#
#     def forward(self, x):
#         if isinstance(x, np.ndarray):
#             x = torch.from_numpy(x)
#         if isinstance(x, list):
#             x = torch.tensor(x)
#         x = x.float()
#         # print(x.shape)
#         if x.dim() == 1:
#             x = x.unsqueeze(0)
#         # print(x.shape)
#         x = F.relu(self.Linear1(x))
#         x = F.relu(self.Linear2(x))
#         x = self.Linear3(x)
#         x = F.tanh(x)
#         x = x*2
#         # print(x)
#         return x
#
#     def copy(self):
#         return copy.deepcopy(self)

class Actor(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super(Actor, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s):
        x = F.relu(self.linear1(s))
        x = F.relu(self.linear2(x))
        x = torch.tanh(self.linear3(x))

        return x


class Critic(nn.Module):
    def __init__(self, input_size, output_size, hidden_size=256):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, output_size)

    def forward(self, s, a):
        x = torch.cat([s, a], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)

        return x


class ReplayMemory(object):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__()
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent(object):
    def __init__(self, env, critic, critic_target, actor, actor_target):
        self.env = env
        self.critic = critic
        self.critic_target = critic_target
        self.actor = actor
        self.actor_target = actor_target
        self.buffer = ReplayMemory(MemorySize)

        # self.actor_target.load_state_dict(self.actor.state_dict())
        # self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_lr = 0.001
        self.critic_lr = 0.001
        self.tau = 0.02
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

    def choose_action(self, s):
        s = torch.tensor(s, dtype=torch.float).unsqueeze(0)
        a = self.actor(s)
        # print(a)
        return a.detach().numpy()

    def push(self, s, a, r, s_, done):
        self.buffer.push(s, a, r, s_, done)

    def act(self, a):
        return self.env.step(a)

    # def update_Q(self):
    #     self.Q_target = self.Q_eval.copy()
    #
    # def update_P(self):
    #     self.P_target = self.P_eval.copy()

    def soft_update(self, net_target, net, tau):
        for target_param, param in zip(net_target.parameters(), net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def learn(self):
        batch_size = 32
        GAMMA = 0.9
        if self.buffer.__len__() < batch_size:
            return
        samples = self.buffer.sample(batch_size)
        batch = Transition(*zip(*samples))
        # 将tuple转化为numpy
        # tmp = np.vstack(batch.action)
        # 转化成Tensor
        state_batch = torch.Tensor(batch.state)

        action_batch = torch.Tensor(batch.action)

        reward_batch = torch.Tensor(batch.reward)
        done_batch = torch.Tensor(batch.done)
        next_state_batch = torch.Tensor(batch.next_state)

        self.critic_learn(state_batch, action_batch, reward_batch, next_state_batch, done_batch)
        self.actor_learn(state_batch)

    def critic_learn(self, obs, action, reward, next_obs, done):
        GAMMA = 0.99
        # print(next_obs.shape)
        next_action = self.actor_target(next_obs).detach()
        next_Q = self.critic_target(next_obs, next_action).detach()

        target_Q = reward + GAMMA * (1.0 - done) * next_Q
        # target_Q.detach()

        Q = self.critic(obs, action)
        criteria = nn.MSELoss()
        loss = criteria(Q, target_Q)
        # optim = torch.optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.critic_optim.zero_grad()
        loss.backward()
        self.critic_optim.step()

    def actor_learn(self, obs):
        action = self.actor(obs)
        Q = self.critic(obs, action)
        loss = -torch.mean(Q)
        # print("loss: {}".format(loss))
        # optim = torch.optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.actor_optim.zero_grad()
        loss.backward()
        self.actor_optim.step()


def main():
    Q_eval = Critic(obs_num+1, 1)
    Q_target = Critic(obs_num+1, 1)
    P_eval = Actor(obs_num, 1)
    P_target = Actor(obs_num, 1)
    agent = Agent(env, Q_eval, Q_target, P_eval, P_target)

    for i in range(400):
        s = agent.env.reset()
        total_r = 0

        while True:
            agent.env.render()

            a = agent.choose_action(s)
            # print(type(a))
            s_, r, done, info = agent.act(a)
            # print("s: {}  r: {} done: {} s_: {} a: {}".format(s, r, done, s_, a))
            # print(s_)
            total_r += r
            if done:
                done = 1
            else:
                done = 0
            agent.push(s, a.squeeze(1), r, s_.squeeze(1), done)
            s = s_
            # s不是一维的，是一个[3, 1]
            s = s.squeeze(1)
            # print(s)


            if done:
                print("{} episode: total_r: {}".format(i, total_r))
                break

            agent.learn()
            agent.soft_update(agent.critic_target, agent.critic, 0.02)
            agent.soft_update(agent.actor_target, agent.actor, 0.02)

        # if i % 10 == 0:
        #     agent.update_P()
        #     agent.update_Q()
    env.close()


if __name__ == "__main__":
    main()