import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_sizes=np.array([64, 64]), activation=nn.Tanh, output_activation=None):
        super(MLP, self).__init__()
        self.hidden_sizes = hidden_sizes
        self.fcs = []
        self.fcs.append(nn.Sequential(
            nn.Linear(input_size, hidden_sizes[0]),
            activation()
        ))
        for i in range(len(hidden_sizes)-1):
            self.fcs.append(nn.Sequential(
                nn.Linear(hidden_sizes[i], hidden_sizes[i+1]),
                activation()
            ))
        self.fcs.append(nn.Linear(hidden_sizes[-1], output_size))
        if output_activation:
           self.fcs.append(output_activation())

        self.layer_module = nn.ModuleList(self.fcs)

    def forward(self, x):
        for fc in self.fcs:
            x = fc(x)
        return x


class A2C(object):
    def __init__(self, env, actor_kwargs, citic_kwargs, learning_rate=0.1, gamma=0.99):
        self.n_action = env.action_space
        self.n_space = env.observation_space
        self.gamma = gamma
        self.lr = learning_rate
        self.discount = 1.

        self.actor_net = MLP(self.n_space, self.n_action, output_activation=nn.Softmax)
        self.critic_net = MLP(self.n_space, 1)
        self.actor_optimizer = torch.optim.Adam(self.actor_net.parameters(), lr=self.lr)
        self.critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.lr)

    def decide(self, observation):
        self.actor_net.eval()
        p_action = self.actor_net(torch.Tensor(observation.reshape((1, -1))))[0].detach().numpy()
        action = np.random.choice(self.n_action, p=p_action)
        return action

    def learn(self, observation, action, reward, next_observation, done):
        self.actor_net.train()
        self.critic_net.train()

        with torch.no_grad():
            s = torch.Tensor(observation.reshape((1, -1)))
            s_prime = torch.Tensor(next_observation.reshape((1, -1)))
            u = reward + (1. - done)*self.gamma*self.critic_net(s_prime)[0, 0]
            td_error = u - self.critic_net(s)[0, 0]

        # train actor_net
        pi = self.actor_net(torch.Tensor(observation.reshape((1, -1))))[0, action]
        log_pi = torch.log(torch.clamp(pi, 1e-6, 1.))
        loss_actor_net = -self.discount * td_error * log_pi
        self.actor_optimizer.zero_grad()
        loss_actor_net.backward()
        self.actor_optimizer.step()

        # train critic_net
        mse_fn = nn.MSELoss(reduce=True, size_average=True)
        loss_critic_net = mse_fn(self.critic_net(torch.Tensor(observation.reshape((1, -1)))), torch.tensor(u))
        self.critic_optimizer.zero_grad()
        loss_critic_net.backward()
        self.critic_optimizer.step()
