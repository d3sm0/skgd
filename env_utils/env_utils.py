import time

import gym

from misc.torch_utils import RunningMeanStd


class RecordEpisodeStatistics(gym.Wrapper):

    def __init__(self, env):
        super(RecordEpisodeStatistics, self).__init__(env)
        self.t0 = time.perf_counter()
        self.start_time = self.t0
        self.online_stats = RunningMeanStd()
        self.episode_return = 0.0
        self.episode_length = 0
        self.n_episodes = 0

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.episode_return = 0.0
        self.episode_length = 0
        self.start_time = time.perf_counter()
        return observation

    def step(self, action):
        observation, reward, done, info = super(RecordEpisodeStatistics, self).step(action)
        tmp_reward = reward if "env/reward" not in info.keys() else info["env/reward"]
        self.episode_return += tmp_reward
        self.episode_length += 1
        self.online_stats.update((tmp_reward,))
        info['stats'] = {"env/mu": self.online_stats.mean, "env/var": self.online_stats.var }
        if 'param' in info.keys():
            info['stats']['env/param'] = info['param']
        if done:
            self.n_episodes += 1
            total_time = round(time.perf_counter() - self.t0, 6)
            dt = round(time.perf_counter() - self.start_time, 6)
            info['episode'] = {'env/r': self.episode_return,
                               'env/l': self.episode_length,
                               'env/avg': self.episode_return / self.episode_length,
                               'env/fps': self.episode_length / dt,
                               'env/t': total_time,
                               'env/total_ep': self.n_episodes
                               }
            self.episode_return = 0.0
            self.episode_length = 0
        return observation, reward, done, info
