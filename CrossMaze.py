# -*- coding:utf-8 -*-
"""
@author: Gene Lee (Zhao-Hua Li)
@file:TripleRoom.py
@time: 14:31

"""

import numpy as np
import gym
import random
import time

CELL, AGENT, KEY, WALL, DOOR, TREASURE = range(6)
ROOM = np.array([[CELL, CELL, CELL, WALL, CELL, CELL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, WALL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, WALL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, WALL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, WALL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, WALL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, WALL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, CELL, CELL, WALL, CELL, CELL, CELL, ],
                 [CELL, CELL, CELL, WALL, CELL, CELL, CELL, WALL, CELL, CELL, CELL, ],
                 ])
# [0: Up, 1: Down, 2: Left, 3: Right,]
ACTION_MAP = [np.array([1, 0]), np.array([-1, 0]), np.array([0, -1]), np.array([0, 1])]
ACTION_NAME = ['Up', 'Down', 'Left', 'Right']
MAX_COUNT_STEP = 50
UNIT = 40
X = 9
Y = 11

# Observation: [agent_x, agent_y, door_1_x, door_1_y, door_2_x, door_2_y, treasure_x, treasure_y]

class CrossMaze(gym.Env):
    def __init__(self, is_render=False, reward_shaping=None, normlize_obs=True):

        self.action_space = 4
        self.observation_space = 8

        self.room = ROOM
        self.agent_position = np.array([0, 0])
        self.door_position_1 = np.array([0, 0])
        self.door_position_2 = np.array([0, 0])
        self.treasure_position = np.array([0, 0])

        self.on_door = False
        self.count_steps = 0
        self.viewer = None
        self.is_render = is_render
        self.reward_shaping = reward_shaping
        self.first_on_door = [True, True]

        self.normlize_obs = normlize_obs

    def reset(self):
        self.room = ROOM.copy()
        # row, column
        agent_x, agent_y = random.randint(0, 8), random.randint(0, 2)
        treasure_x, treasure_y = random.randint(0, 8), random.randint(8, 10)
        door_x_1 = random.randint(0, 8)
        door_x_2 = random.randint(0, 8)

        self.agent_position = np.array([agent_x, agent_y])
        self.door_position_1 = np.array([door_x_1, 3])
        self.door_position_2 = np.array([door_x_2, 7])
        self.treasure_position = np.array([treasure_x, treasure_y])

        self.room[agent_x][agent_y] = AGENT
        self.room[treasure_x][treasure_y] = TREASURE
        self.room[door_x_1][3] = DOOR
        self.room[door_x_2][7] = DOOR
        self.get_key = False
        self.has_key = False
        self.on_door = False
        self.count_steps = 0
        self.first_on_door = [True, True]
        if self.is_render:
            self.render_initialize()

        return self.get_obs()

    def step(self, action: int):
        self.count_steps += 1
        reward = 0
        done = False
        info = ''
        if self.count_steps >= MAX_COUNT_STEP:
            done = True
        new_position = self.agent_position + ACTION_MAP[action]
        if new_position[0] < 0 or new_position[0] >= X or new_position[1] < 0 or new_position[1] >= Y:
            return self.get_obs(), reward, done, info
        cell_tmp = self.room[new_position[0]][new_position[1]]
        if cell_tmp == WALL:
            return self.get_obs(), reward, done, info
        elif cell_tmp == DOOR:
            reward = 0
            door_id = 0 if new_position[1] == 3 else 1
            if self.reward_shaping is not None:
                if self.first_on_door[door_id]:
                    reward += self.reward_shaping
                    self.first_on_door[door_id] = False
            self.on_door = True
            info = 'on the door '+str(door_id)
        elif cell_tmp == TREASURE:
            reward = 2
            done = True
            info = 'FINISH'
        if self.on_door and cell_tmp != DOOR:
            self.room[self.agent_position[0]][self.agent_position[1]] = DOOR
            self.on_door = False
        else:
            self.room[self.agent_position[0]][self.agent_position[1]] = CELL
        self.room[new_position[0]][new_position[1]] = AGENT
        self.agent_position = new_position

        return self.get_obs(), reward, done, info

    def get_obs(self):
        position_obs = np.vstack((self.agent_position, self.door_position_1, self.door_position_2,
                                      self.treasure_position))
        if self.normlize_obs:
            position_obs = position_obs / np.array([X, Y])
        position_obs = position_obs.flatten()
        return position_obs.copy()


    def render_initialize(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        from gym.envs.classic_control import rendering
        if self.viewer is None:
            self.viewer = rendering.Viewer(Y * UNIT, X * UNIT)

        # render playground
        m = np.copy(self.room)
        for x in (range(X)):
            for y in (range(Y)):
                # the coordinates of 4 angle of the grid
                v = [(y * UNIT, x * UNIT),
                     ((y + 1) * UNIT, x * UNIT),
                     ((y + 1) * UNIT, (x + 1) * UNIT),
                     (y * UNIT, (x + 1) * UNIT)]
                grid = rendering.FilledPolygon(v)

                if m[x, y] == WALL:  # the block
                    grid.set_color(0, 0, 0)  # black
                else:
                    grid.set_color(0, 0.5, 0)  # green

                # render water and home
                if m[x, y] == DOOR:  # water 1
                    grid.set_color(135/255, 206/255, 235/255)  # sky blue
                # if m[x, y] == KEY:  # water 2
                #     grid.set_color(0, 0, 1)  # blue
                if m[x, y] == TREASURE:
                    grid.set_color(255/255, 215/255, 0)  # gold
                self.viewer.add_geom(grid)
                if m[x, y] == KEY:  # water 2
                    # 画一个直径为 30 的园
                    self.key_render = rendering.make_circle(UNIT / 4, 30, True)
                    self.key_render.set_color(0, 0, 1)
                    circle_transform = rendering.Transform(translation=((y+0.5) * UNIT, (x+0.5) * UNIT))
                    self.key_render.add_attr(circle_transform)
                    self.viewer.add_geom(self.key_render)

                # draw outline
                v_outline = v
                outline = rendering.make_polygon(v_outline, False)
                outline.set_linewidth(1)
                outline.set_color(0, 0, 0)
                self.viewer.add_geom(outline)
        # render monk
        self.agent_render = rendering.make_circle(UNIT / 2, 30, True)
        self.agent_render.set_color(1.0, 1.0, 1.0)  # white
        self.viewer.add_geom(self.agent_render)
        self.agent_trans = rendering.Transform()
        self.agent_render.add_attr(self.agent_trans)
        x, y = self.agent_position[0], self.agent_position[1]
        self.agent_trans.set_translation((y + 0.5) * UNIT, (x + 0.5) * UNIT)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def render(self, mode='human', close=False):
        if not self.is_render:
            return
        if self.get_key:
            from gym.envs.classic_control import rendering
            self.key_render.set_color(0, 0.5, 0)
            self.agent_render.set_color(0, 0, 1)  # sky blue
            self.get_key = False
        x, y = self.agent_position[0], self.agent_position[1]
        self.agent_trans.set_translation((y+0.5) * UNIT, (x+0.5) * UNIT)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')


# In order to compute the accomplishment
class PassCrossMaze(CrossMaze):
    def __init__(self, is_render=False, reward_shaping=None, normlize_obs=True, test_function=None, max_steps=50):
        CrossMaze.__init__(self, is_render, reward_shaping, normlize_obs)
        # super(ModifiedCrossMaze, self).__init__(is_render, reward_shaping, normalize_obs)
        # test function candi: go_room1, go_room2, go_treasure
        self.test_function = test_function
        self.max_steps = max_steps

    def reset(self):
        random_obs = super(PassCrossMaze, self).reset()
        if self.test_function == 'go_room1':
            return random_obs
        agent_x, agent_y = self.agent_position.tolist()
        self.room[agent_x][agent_y] = CELL
        if self.test_function == 'go_room2':
            agent_x, agent_y = self.door_position_1
        else:
            agent_x, agent_y = self.door_position_2
        self.room[agent_x][agent_y] = AGENT
        self.agent_position = np.array([agent_x, agent_y])
        return self.get_obs()

    def step(self, action):
        obs, rwd, done, info = super().step(action)
        if self.test_function == 'go_treasure':
            return obs, rwd, done, info
        agent_position = self.agent_position
        if self.test_function == 'go_room1' and (agent_position == self.door_position_1).all():
            return obs, rwd, True, 'FINISH'
        if self.test_function == 'go_room2' and (agent_position == self.door_position_2).all():
            return obs, rwd, True, 'FINISH'
        if self.count_steps >= self.max_steps:
            done = True
        return obs, rwd, done, info

if __name__ == '__main__':
    env = CrossMaze(is_render=True, reward_shaping=False)
    for i in range(100000):
        s = env.reset()
        done = False
        while not done:
            # env.render()
            a = random.randint(0, 3)
            s_, r, done, info = env.step(a)
            if max(s_) > 1:
                print(s[:2], ACTION_NAME[a], s_[:2], s)
            s = s_
            if r != 0:
                print('get reward ', r)
                import time
                time.sleep(1)

        if i % 100 == 0:
            print(i)
