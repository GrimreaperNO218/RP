from environments.CrossMaze import CrossMaze
import torch

class TorchCrossMaze(CrossMaze):
    def __init__(self, is_render=False, reward_shaping=None, normlize_obs=True):
        super(TorchCrossMaze, self).__init__(is_render, reward_shaping, normlize_obs)

    def reset(self):
        return torch.Tensor(super(TorchCrossMaze, self).reset()).view(1, -1)

    def step(self, action: int):
        obs, reward, done, info = super(TorchCrossMaze, self).step(action)
        return torch.Tensor(obs).view(1, -1), torch.Tensor([reward]).view(1, -1), torch.Tensor([done]).view(1, -1), info
