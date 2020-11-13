import random
from collections import namedtuple, deque

import gym
import numpy as np
import torch
from matplotlib import pyplot as plt

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity, recent_queue):
        self.capacity = capacity
        self.latest = deque(maxlen=recent_queue)
        self.memory = []
        self.position = 0
        self.batch_stats = RunningMeanStd()

    def _n_step_return(self, rewards, not_done, gamma, next_value):
        returns = torch.zeros_like(rewards)
        returns = torch.cat([returns, next_value])
        not_done = torch.cat([torch.ones(size=(1,)), not_done])  # assuming is always not terminal
        for step in reversed(range(rewards.shape[0])):
            returns[step] = returns[step + 1] * gamma * not_done[step + 1] + rewards[step]
        return returns[:-1]

    def flush(self):
        self.memory.clear()
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        t = Transition(*args)
        # self.memory.append(t)
        self.latest.append(t)
        if self.position >= len(self.memory):
            self.memory.append(t)
        else:
            self.memory[self.position] = t
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        if len(self.memory) >= batch_size:
            batch = random.sample(self.memory, batch_size)

            s, a, r, s1, done = Transition(*zip(*batch))
            r = torch.stack(r).squeeze()
            done = torch.stack(done).squeeze()
            s = torch.cat(s)
            a = torch.stack(a)
            s1 = torch.cat(s1)
            return s, a, r, s1, done

    def get(self, v_next=None, batch_size=32):

        s, a, r, s1, done = Transition(*zip(*self.latest))
        r = torch.stack(r).squeeze()
        done = torch.stack(done).squeeze()
        R = torch.zeros_like(r)
        p_cont = .99 * (1 - done)
        if v_next is not None:
            R = self._n_step_return(r, (1 - done), .99, v_next.unsqueeze(dim=0))
            self.batch_stats.update(R.detach().numpy())
            R = self.batch_stats.standardize(R)
            if torch.isnan(R.mean()): raise ValueError(f"Found Nan in Return {R.mean()}. Maybe due to v_tp1 ? {v_next}")
        s = torch.cat(s)
        a = torch.stack(a)
        s1 = torch.cat(s1)

        # dataset = data.dataset.TensorDataset(*(s, a, r, s1, p_cont, R))
        # data_loader = data.dataloader.DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False)
        ##print(f"n_batches: {len(data_loader)} with batch_size {batch_size}")
        return s, a, r, s1, p_cont, R

    def __len__(self):
        return len(self.memory)

    def is_available(self, batch_size):
        return len(self.memory) > batch_size


import collections


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), maxlen=20):
        self.mean = np.zeros(shape, 'float32')
        self.var = np.ones(shape, 'float32')
        self.count = epsilon
        self.latest = collections.deque(maxlen=maxlen)

    def standardize(self, x):
        out = x - self.mean
        out = out / (self.var ** .5 + 1e-6)
        return out

    def get_normalized(self):
        return  self.standardize(np.array(self.latest))

    def update(self, x):
        self.latest.append(x)
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = len(x)
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        self.mean, self.var, self.count = self.update_mean_var_count_from_moments(self.mean, self.var, self.count,
                                                                                  batch_mean, batch_var, batch_count)

    def __repr__(self):
        return f"mu:{self.mean:.4f}\tvar:{self.var:.4f}\tn:{self.count:.1f}"

    @staticmethod
    def update_mean_var_count_from_moments(mean, var, count, batch_mean, batch_var, batch_count):
        delta = batch_mean - mean
        tot_count = count + batch_count

        new_mean = mean + delta * batch_count / tot_count
        m_a = var * count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        return new_mean, new_var, new_count


class TorchEnv(gym.Wrapper):
    def __init__(self, env):
        super(TorchEnv, self).__init__(env)

    def step(self, action):
        action = action.numpy()

        s, r, d, info = self.env.step(action)

        s = self._to_torch(s).reshape((1, -1))
        r = self._to_torch((r,))
        d = self._to_torch((d,))
        return s, r, d, info

    def reset(self):
        s = self.env.reset().reshape((1, -1))
        return self._to_torch(s)

    @staticmethod
    def _to_torch(x):
        x = torch.FloatTensor(x)
        return x


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def plot_confusion_matrix(cm):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    # cm = (cm / cm.sum(dim=1, keepdims=True)).numpy()
    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set(title="cov", xlabel="theta", ylabel="theta")
    fig.colorbar()
    plt.tight_layout()
    return fig


def record_stats(writer, stats, step, prefix=None):
    for k, v in stats.items():
        # if v.ndim != 0: continue
        if prefix is not None: k = f"{prefix}/{k}"
        writer.add_scalar(k, v, global_step=step)


def record_img(writer, img, step):
    writer.add_figure('cov', img, step)
