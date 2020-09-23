from collections import deque

import torch
import torch.nn as nn
import numpy as np
from algrithmns.networks import MLP
from environments.TorchCrossMaze import TorchCrossMaze
from algrithmns.distributions import *
from algrithmns.storage import RolloutStorage
from algrithmns.evaluation import evaluate

class Policy(nn.Module):
    def __init__(self, env, hidden_size=64):
        super(Policy, self).__init__()
        self.n_action = env.action_space
        self.n_space = env.observation_space
        self.ac_net = MLP(self.n_space, hidden_size)

        self.dist = Categorical(hidden_size, self.n_action)

    def forward(self):
        raise NotImplementedError

    def step(self, inputs, deterministic=False):
        value, actor_features = self.ac_net(inputs)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)

        return value, action, action_log_probs

    def value(self, inputs):
        value, _ = self.ac_net(inputs)
        return value

    def evaluate_actions(self, inputs, action):
        value, actor_features = self.ac_net(inputs)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy


class Model(object):
    def __init__(self,
                 actor_critic,
                 value_loss_coef=0.5,
                 entropy_coef=0.01,
                 lr=7e-4,
                 eps=1e-5,
                 alpha=0.99,
                 gamma=0.99,
                 max_grad_norm=0.5):

        self.actor_critic = actor_critic

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.max_grad_norm = max_grad_norm

        self.optimizer = torch.optim.RMSprop(
            actor_critic.parameters(), lr, eps=eps, alpha=alpha)

    def train(self, rollouts):
        obs_shape = rollouts.obs.size()[2:]
        action_shape = rollouts.actions.size()[-1]
        num_steps, num_processes, _ = rollouts.rewards.size()

        values, action_log_probs, dist_entropy = self.actor_critic.evaluate_actions(
            rollouts.obs[:-1].view(-1, *obs_shape),
            rollouts.actions.view(-1, action_shape))

        values = values.view(num_steps, num_processes, 1)
        action_log_probs = action_log_probs.view(num_steps, num_processes, 1)

        adv = rollouts.returns[:-1] - values
        value_loss = adv.pow(2).mean()

        action_loss = -(adv.detach() * action_log_probs).mean()

        self.optimizer.zero_grad()
        loss = value_loss * self.value_loss_coef + action_loss - dist_entropy * self.entropy_coef
        loss.backward()

        self.optimizer.step()

        return value_loss.item(), action_loss.item(), dist_entropy.item()


# Parameters
nsteps = 5
num_processes = 1
num_env_steps = 80e6
gamma = 0.99
eval_interval = int(1e4)

def main():
    torch.manual_seed(1)

    env = TorchCrossMaze(reward_shaping=True)

    actor_critic = Policy(env)
    agent = Model(actor_critic)

    rollouts = RolloutStorage(nsteps, num_processes, [env.observation_space], env.action_space)

    obs = env.reset()
    rollouts.obs[0].copy_(obs)

    num_updates = int(num_env_steps // nsteps // num_processes)
    for j in range(num_updates):

        for step in range(nsteps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = actor_critic.step(rollouts.obs[step])

            # Step in the environment
            obs, reward, done, info = env.step(action.detach().numpy()[0, 0])

            # create masks
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])

            # if done, reset
            if done:
                obs = env.reset()

            # add info in rollout storage
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        # get next_value
        with torch.no_grad():
            next_value = actor_critic.value(rollouts.obs[-1])

        rollouts.compute_returns(next_value, gamma=gamma)

        value_loss, action_loss, dist_entropy = agent.train(rollouts)

        rollouts.after_update()

        if (j + 1)%eval_interval==0:
            print("Update for {} epoches, ".format(j+1), end="")
            evaluate(actor_critic)

if __name__=="__main__":
    main()
