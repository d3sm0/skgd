# from model_prediction import parse_predictions
import gym
import numpy as np


# import config
# from utils.env_utils import TimeLimit


def standardize(x):
    denom = x.std(axis=0, keepdims=True) + 1e-6
    assert denom.sum() != 0.0
    x = (x - x.mean(axis=0, keepdims=True)) / denom
    return x


def get_feature_slice(t, window=4, period=24):
    assert t >= period * 2 + window // 2
    dm1_slice = slice(t - period - window // 2, t - period + window // 2 + 1)
    dm2_slice = slice(t - period * 2 - window // 2, t - period * 2 + window // 2 + 1)
    recent_past_slice = slice(t - window, t + 1)
    return recent_past_slice, dm1_slice, dm2_slice


def min_max_scale(x):
    denom = x.max(axis=0, keepdims=True) - x.min(axis=0, keepdims=True)
    assert denom.sum() != 0.0
    x = (x - x.min(axis=0, keepdims=True)) / denom
    return x


class SequentialFeatures(gym.Wrapper):
    # period = 24
    # window = 4

    def __init__(self, env):
        super(SequentialFeatures, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=(43,)), high=np.ones(shape=(43,))
        )
        self.epv = standardize(np.load("env_utils/microgrid/epv.npy")[1:])
        self.c1 = standardize(np.load("env_utils/microgrid/c1.npy")[1:])

    def step(self, action):
        s, r, d, info = super(SequentialFeatures, self).step(action)
        s = self.observation(s)
        return s, r, d, info

    def observation(self, s):
        epv = self.epv[self.unwrapped.simulator.env_step]
        c1 = self.c1[self.unwrapped.simulator.env_step]
        scale_soc = self.unwrapped.simulator.grid.storage.initial_capacity
        s = np.concatenate([s[-1:] / scale_soc, epv, c1])
        return s

    def reset(self, **kwargs):
        s = super(SequentialFeatures, self).reset()
        return self.observation(s)


class SequentialFeaturesRaw(gym.Wrapper):
    period = 24
    window = 6

    def __init__(self, env):
        super(SequentialFeaturesRaw, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            low=np.zeros(shape=(43,)), high=np.ones(shape=(43,))
        )
        self.t_0 = self.period * 2 + self.window // 2 + 1

    def step(self, action):
        s, r, d, info = super(SequentialFeaturesRaw, self).step(action)
        s = self.observation(s)
        return s, r, d, info

    def observation(self, s):
        a, b, c = get_feature_slice(
            self.unwrapped.simulator.env_step, window=self.window, period=self.period
        )
        s_t = self.unwrapped.simulator.grid.db.values[a]
        s_tm1 = self.unwrapped.simulator.grid.db.values[b]
        s_tm2 = self.unwrapped.simulator.grid.db.values[c]
        s_tilde = np.concatenate([s_t, s_tm1, s_tm2], axis=0).flatten()

        scale_soc = self.unwrapped.simulator.grid.storage.initial_capacity
        s = np.append(s[-1] / scale_soc, s_tilde)
        return s

    def reset(self, **kwargs):
        s = super(SequentialFeaturesRaw, self).reset()
        self.unwrapped.simulator.env_step = self.t_0
        return self.observation(s)


class MaxDischarge(gym.ObservationWrapper):
    def __init__(self, env):
        super(MaxDischarge, self).__init__(env)

        low = np.append(self.observation_space.low, 0)
        high = np.append(
            self.observation_space.high,
            self.env.simulator.grid.storage.max_discharge_rate,
        )

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, observation):
        soc = self.env.simulator.grid_state.soc
        max_discharge = min(
            soc
            * self.env.simulator.grid.storage.discharge_efficiency
            / self.env.simulator.grid.dt,
            self.simulator.grid.storage.max_discharge_rate,
        )

        obs = np.append(observation, max_discharge)
        return obs


class TimeFeatures(gym.ObservationWrapper):
    def __init__(self, env):
        super(TimeFeatures, self).__init__(env)

        high = np.array([24, 7, 1])
        low = np.zeros_like(high)
        low = np.concatenate([self.observation_space.low, low])
        high = np.concatenate([self.observation_space.high, high])

        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

    def observation(self, observation):
        dt = self.env.simulator.grid.db.time_to_idx[
            self.env.simulator.grid_state.time_step
        ]
        is_week_day = 1 if dt.dayofweek > 5 else 0
        time_features = np.array([dt.hour, dt.dayofweek, is_week_day])
        observation = np.concatenate([observation, time_features])
        return observation


class ScaleObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super(ScaleObservation, self).__init__(env)

    def observation(self, observation):
        return observation / self.observation_space.high


class ScaleAction(gym.ActionWrapper):
    def __init__(self, env):
        super(ScaleAction, self).__init__(env)

    def action(self, raw_action):
        action = np.clip(raw_action, -1.0, 1.0)
        return action


class ScaleReward(gym.Wrapper):
    def __init__(self, env):
        super(ScaleReward, self).__init__(env)

    def step(self, action):
        s, r, d, info = super(ScaleReward, self).step(action)
        info["env/reward"] = r
        r = self.reward(r)
        return s, r, d, info

    def reward(self, reward):
        reward = np.clip(reward, -140, reward) / 140
        return reward


def make_microgrid(env_id):
    from microgridRLsimulator.gym_wrapper.microgrid_env import MicrogridEnv

    start_date = "2016-01-31"
    end_date = "2017-07-01"
    data_file = "elespino"
    env_params = dict(
        action_space="Discrete",
        backcast_steps=0,
        forecast_steps=0,
        forecast_type="exact",
        min_stable_generation=0.0,
        prob_failure=0.0000,
    )

    env = MicrogridEnv(
        start_date=start_date, end_date=end_date, case=data_file, params=env_params
    )

    # env = SequentialFeaturesRaw(env)
    env = SequentialFeatures(env)
    env = ScaleReward(env)

    return env


#
