import copy

import torch.distributions as torch_dist
import torch.optim as optim

from ipo import *
from misc.torch_utils import RunningMeanStd


def standardize(x):
    x = (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)
    return x


class PG:
    def __init__(self, actor, critic, *args, **kwargs):
        self.beta = 1.

        self.actor = actor
        self.actor_old = copy.deepcopy(actor)
        self.critic = critic
        self.opt = optim.Adam(self.actor.parameters(), lr=1e-3)
        self.opt_critic = optim.Adam(self.critic.parameters(), lr=1e-3)
        # self.batch_stats = RunningMeanStd()

        self.name = "vpg"

    def train(self, mode=True):
        self.actor.train(mode)
        self.critic.train(mode)

    def act(self, s):
        with torch.no_grad():
            pi = self.actor(s)
            a = pi.sample().squeeze(dim=0)
            v = self.critic(s).squeeze()

            return a, v, pi.entropy()

    def update(self, batch, opt_steps):
        self.actor_old.load_state_dict(self.actor.state_dict())
        s, a, r, s1, p_cont, R = batch

        adv = (R - self.critic(s)).detach()
        pi_stats = self.pi_update(s, a, adv, opt_steps=opt_steps)
        td_stats = self.td_update(s, r, s1, p_cont, R=R, opt_steps=opt_steps)

        pi_stats.update(td_stats)
        return pi_stats, adv

    def td_update(self, s, r, s1, p_cont, R=None, opt_steps=1, **kwargs):
        avg_loss = 0
        for _ in range(opt_steps):
            v = self.critic(s).squeeze()
            adv = (R - v)
            td_loss = .5 * adv.pow(2).mean()
            assert not torch.isnan(td_loss), td_loss
            self.opt_critic.zero_grad()
            td_loss.backward()
            self.opt_critic.step()
            avg_loss += td_loss.detach()

        return {
            "vf/loss": avg_loss / opt_steps
        }

    def pi_update(self, s, a, adv, opt_steps=1):

        avg_kl = 0
        avg_loss = 0
        avg_entropy = 0

        with torch.no_grad():
            pi_old = self.actor_old(s)

        for _ in range(opt_steps):
            pi = self.actor(s)
            pi_loss = - (pi.log_prob(a) - pi_old.log_prob(a)).exp() * adv.detach()
            kl = 1 / self.beta * torch_dist.kl_divergence(pi, pi_old)
            pi_loss = (pi_loss + kl).mean()
            if torch.isnan(pi_loss): raise ValueError(f"Nan in loss {pi_loss}")
            self.opt.zero_grad()
            pi_loss.backward()
            self.opt.step()

            avg_loss += pi_loss.detach()
            avg_kl += kl.detach().mean()
            avg_entropy += pi.entropy().detach().mean()

        stats = {
            "pi/kl": avg_kl / opt_steps,
            "pi/loss": avg_loss / opt_steps,
            "pi/entropy": avg_entropy / opt_steps,
        }
        return stats


import numpy as np

import matplotlib.pyplot as plt


class PPOKTD(PG):
    def __init__(self, actor, critic, **kwargs):
        super(PPOKTD, self).__init__(actor, critic)

        self.name = f"pktd"
        rank = kwargs.get("rank")
        print(f"rank {rank}")
        if rank == 0:
            self.opt_critic = PKTDOptimizer(self.critic.parameters(), **kwargs)
        else:
            self.opt_critic = APKTDOptimizer(self.critic.parameters(), **kwargs)
        self.batch_stats = RunningMeanStd()
        self.global_step = 0

    @torch.no_grad()
    def get_batch_stats(self, residual, stats):
        # test for magnitude bound
        N = len(residual)
        cov = stats["opt/residual/cov"].numpy()
        out = np.correlate(residual, residual, mode="full")
        assert not np.isinf(out).any()
        out = out[len(out) // 2:]
        out = out / out[0]

        try:
            fig, ax = plt.subplots(3, 1, sharex="all")
        except RuntimeWarning:
            plt.close()
            fig, ax = plt.subplots(3, 1, sharex="all")

        cov_residual = residual * 1 / cov * residual
        ax[0].plot(cov_residual, label="residual_normalized")
        ax[0].set_title("residual_normalized")

        ax[0].grid()

        ax[1].plot(residual, label="residual")
        ax[1].plot(residual - 2 * np.sqrt(cov), linestyle="dotted", color="blue")
        ax[1].plot(residual + 2 * np.sqrt(cov), linestyle="dotted", color="red")
        ax[1].set_title("residual")

        ax[1].grid()

        # for a large batch size this should be bounded by +/- 2/sqrt(batch_size)
        ax[2].plot(out, label="rho")
        ax[2].plot(out - 2 / np.sqrt(N), linestyle="dotted", color="blue")
        ax[2].plot(out + 2 / np.sqrt(N), linestyle="dotted", color="red")
        ax[2].grid()

        ax[2].set_title("autocorr")

        return fig

    def td_update(self, s, r, s1, p_cont, R=None, opt_steps=3, **kwargs):
        v_stats = {}
        v_loss = 0
        self.global_step += 1
        for _ in range(opt_steps):
            v = self.critic(s)
            adv = (R - v)
            self.opt_critic.zero_grad()
            v.mean().backward()
            v_stats = self.opt_critic.step(innovation=adv.detach().sum())
            v_loss = v_loss + .5 * adv.pow(2).mean().detach()

        v_stats["vf/loss"] = v_loss / opt_steps

        assert not torch.isnan(v_stats["vf/loss"]), f"Found Nan in td loss {v_loss}"

        # self.opt_critic.param_groups[0]["p_v"] = self.opt_critic.param_groups[0]["p_v"] * np.exp(-(1e-7) * self.global_step)
        # self.get_batch_stats(adv.detach(), v_stats)

        return v_stats
