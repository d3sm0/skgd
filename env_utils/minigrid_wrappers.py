import functools
import operator

import gym
import numpy as np

from misc.utils import one_hot


class OneHotEnv(gym.ObservationWrapper):
    n_directions = 4

    def __init__(self, env):
        super(OneHotEnv, self).__init__(env)
        self.observation_space = gym.spaces.Discrete(n=self.grid.width + self.grid.height + OneHotEnv.n_directions * 1)

    def observation(self, obs):
        # pos, goal = np.split(obs, indices_or_sections=2)
        a = self.pos_to_one_hot(obs)
        # b = self.pos_to_one_hot(goal)
        return a  # np.concatenate([a, b], axis=0)

    def pos_to_one_hot(self, pos):
        x, y, z = pos
        x = self._to_one_hot(x, self.grid.width)
        y = self._to_one_hot(y, self.grid.height)
        z = self._to_one_hot(z, self.n_directions)
        v = np.concatenate([x, y, z])
        return v

    @staticmethod
    def _to_one_hot(x, size_x):
        v = np.zeros(size_x, dtype=np.float32)
        v[x] = 1.
        return v


class FlatObs(gym.ObservationWrapper):
    def __init__(self, env):
        super(FlatObs, self).__init__(env)
        img_size = functools.reduce(operator.mul, self.env.observation_space.shape, 1)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,  # TODO max is wrong
            shape=(img_size,),
            dtype='uint8'
        )

    def observation(self, observation):
        return observation.flatten()


class MiniGridWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MiniGridWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n=3)
        self.observation_space = gym.spaces.Box(
            low=0,
            high=19,
            shape=(3,),  # x,y,d, terminal
            dtype=np.float32
        )

    def step(self, action):
        s, r, d, info = super(MiniGridWrapper, self).step(action)
        s = self.observation(s)
        r = self.reward(r)
        return s, r, d, info

    def reward(self, r):
        r = np.sign(r)
        return r

    def reset(self, **kwargs):
        s = super(MiniGridWrapper, self).reset()
        return self.observation(s)

    def observation(self, observation):
        _obs = np.concatenate([self.env.agent_pos, (self.env.agent_dir,)])
        # (x, y, z) = np.argwhere(self.unwrapped.grid.encode() == 8).flatten()
        # _obs = np.concatenate([self.env.agent_pos, (self.env.agent_dir,), (x, y, z)])
        # _obs= _obs / self.observation_space.high
        return _obs


class GridToIndex(gym.ObservationWrapper):
    def __init__(self, env):
        super(GridToIndex, self).__init__(env)
        self._space, self._idx_to_state, self._state_to_idx = self.enumerate_state_space()
        self.n_states = len(self._space)
        self.n_actions = self.action_space.n

    def observation(self, observation):
        return self._state_to_idx[tuple(observation)]

    def enumerate_state_space(self):
        w, h = self.grid.width, self.grid.height
        space = []
        for _w in range(w):
            for _h in range(h):
                for d in range(4):
                    space.append((_w, _h, d))

        idx_to_state = {idx: state for idx, state in enumerate(space)}
        state_to_idx = {state: idx for idx, state in enumerate(space)}
        return space, idx_to_state, state_to_idx


class DistanceReward(gym.Wrapper):
    def __init__(self, env):
        super(DistanceReward, self).__init__(env)

    def step(self, action):
        s, r, d, info = super(DistanceReward, self).step(action)
        r = self.reward(r)
        return s, r, d, info

    @staticmethod
    def reward(reward):
        return -1. if reward == 0. else 0.


class RewardWrapper(gym.Wrapper):

    def __init__(self, env):
        super(RewardWrapper, self).__init__(env)

    def step(self, action):
        s, r, d, info = super(RewardWrapper, self).step(action)
        info["env/reward"] = r
        r = self.reward(r)
        return s, r, d, info

    def reward(self, r):
        return r


def action_features():
    actions = []
    for a in range(3):
        a = one_hot(a)
        actions.append(a)
    actions = np.stack(actions)
    return actions


_A = action_features()


def make_minigrid(env_id):
    env = gym.make(env_id)
    # env = gym_minigrid.envs.FourRoomsEnv(agent_pos=(1, 1), goal_pos=(15, 3))
    # if 'Empty' in env_id:
    env = MiniGridWrapper(env)
    env = OneHotEnv(env)
    # else:
    # env = gym_minigrid.wrappers.FullyObsWrapper(env)
    # env = gym_minigrid.wrappers.ImgObsWrapper(env)
    # env = FlatObs(env)
    # env = RewardWrapper(env)
    # env = DistanceReward(env)
    # env = GridToIndex(env)
    # env = gym_minigrid.wrappers.StateBonus(env)
    return env
