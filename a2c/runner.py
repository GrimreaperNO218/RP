from Project_for_RP.a2c.a2c import A2C
from Project_for_RP.CrossMaze import CrossMaze

EPOCH = int(1e6)

class Runner(object):
def run():
    env = CrossMaze()
    test_env = CrossMaze(is_render=True)

    agent = A2C(env, None, None)
    for epoch in range(EPOCH):
        obs = env.reset()
        done = False
        total_reward = 0
        while not done:
            action = agent.decide(obs)
            new_obs, reward, done, info = env.step(action)
            agent.learn(obs, action, reward, new_obs, done)
            total_reward += reward
            obs = new_obs

        if total_reward>0:
            print("Get reward, epoch {}.".format(epoch+1))
        if (epoch + 1)%1000==0:
            print("Epoch {}.".format(epoch+1))
        #     print("test")
        #     test_env.reset()
        #     done = False
        #     while not done:
        #         action = agent.decide(obs)
        #         print(action, end=" ")
        #         obs, reward, done, _ = test_env.step(action)
        #         print(obs, end=" ")
        #         if reward>0:
        #             print("Get reward!")


if __name__=="__main__":
    run()
