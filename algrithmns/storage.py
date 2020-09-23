import torch

class RolloutStorage(object):
    def __init__(self, nsteps, num_processes, obs_shape, action_space):
        self.obs = torch.zeros(nsteps + 1, num_processes, *obs_shape)
        self.rewards = torch.zeros(nsteps, num_processes, 1)
        self.value_preds = torch.zeros(nsteps, num_processes, 1)
        self.returns = torch.zeros(nsteps + 1, num_processes, 1)
        self.action_log_probs = torch.zeros(nsteps, num_processes, 1)
        self.actions = torch.zeros(nsteps, num_processes, 1)

        self.masks = torch.ones(nsteps + 1, num_processes, 1)

        self.nsteps = nsteps
        self.step = 0

    def insert(self, obs, actions, action_log_probs,
               value_preds, rewards, masks):
        self.obs[self.step + 1].copy_(obs)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)

        self.step = (self.step + 1) % self.nsteps

    def after_update(self):
        self.obs[0].copy_(self.obs[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value, gamma):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):
            self.returns[step] = self.returns[step + 1] * gamma * self.masks[step + 1] + self.rewards[step]

    def feed_forward_generator(self, advantages):
        raise NotImplementedError
