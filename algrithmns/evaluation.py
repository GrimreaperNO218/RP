import torch

from environments.TorchCrossMaze import TorchCrossMaze

def evaluate(policy):
    eval_episode_reward = []

    eval_env = TorchCrossMaze(reward_shaping=True)
    obs = eval_env.reset()

    while len(eval_episode_reward) < 500:
        with torch.no_grad():
            _, action, _ = policy.step(obs, deterministic=True)
            obs, reward, done, info = eval_env.step(action.detach().numpy()[0, 0])
            eval_episode_reward.append(reward)
            if done:
                obs = eval_env.reset()

    print("evaluate reward:{}".format(sum(eval_episode_reward).detach().numpy()[0, 0]))
