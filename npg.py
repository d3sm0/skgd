import random

import torch
import torch.optim as optim

optim
import torch.nn.functional as F

from misc.torch_utils import RunningMeanStd

torch.autograd.set_detect_anomaly(True)
random.seed(0)
torch.random.manual_seed(0)

from ipo import *


def _update_target_soft_(target, src, tau=1.):
    for target_param, param in zip(target, src):
        target_param.data.copy_(target_param.data + param.data * tau)


from models import NPGNetwork
from models import Critic


class NPG:
    def __init__(self, obs_space, action_space, hidden_dim=64):
        self.npg = NPGNetwork(obs_space, action_space, hidden_dim)
        # self.actor = Actor(obs_space, action_space, hidden_size=hidden_dim)
        # self.q = Critic(obs_space, action_space=action_space, hidden_size=hidden_dim)
        self.v = Critic(obs_space, action_space=1, hidden_size=hidden_dim)
        self.adv = Critic(obs_space, action_space=action_space, hidden_size=hidden_dim)
        #self.opt_adv = PKTDOptimizer(
        #    list(self.adv.parameters())
        #)
        self.opt_critic = optim.Adam(self.v.parameters())
        self.opt_adv = optim.Adam(self.adv.parameters())

        self.tau = 1.
        self.beta = 10.
        self.gamma = .99
        self.name = f"npg"
        self.batch_stats = RunningMeanStd()

    def act(self, s):
        with torch.no_grad():
            pi = self.npg(s)
            v = self.v(s).squeeze()
            a = pi.sample().squeeze()
            return a, v, 0

    def update(self, batch):
        s, a, r, s1, done, w, R = batch

        v_next = self.v(s1).squeeze()
        v = self.v(s).squeeze()
        adv = self.adv(s)
        a = F.one_hot(a, adv.shape[1])
        adv = (a * adv).sum(dim=1)

        #(adv + v - self.gamma * (1 - done) * v_next).mean().backward()  # gradient
        #innovation = (r - adv - self.gamma * (1 - done) * v_next).mean().detach()

        self.opt_adv.zero_grad()
        td = r + (1-done) * self.gamma * v_next - v
        adv_loss = (.5 * (td.detach() - adv).pow(2)).mean()
        adv_loss.backward()
        self.opt_adv.step()

        #adv.mean().backward()
        #adv_norm = torch.stack([p.grad.norm() for  p in self.adv.parameters()]).norm()
        #td = r + v_next - v
        #innovation = (td - adv).mean().detach()
        #opt_stats = self.opt_adv.step(innovation=innovation)
        td_loss = .5 * (td.pow(2)).mean()
        #td = r + v_next - v
        #adv_loss = (.5 * (td.detach() - adv).pow(2)).mean()
        #td_loss = .5 * (td.pow(2).mean())

        self.opt_critic.zero_grad()
        td_loss.backward()
        self.opt_critic.step()

        with torch.no_grad():
            pi_old = self.npg(s)
        _update_target_soft_(self.npg.parameters(), src=self.adv.parameters(), tau=1e-4)

        with torch.no_grad():
            pi = self.npg(s)
        kl = torch.distributions.kl_divergence(pi, pi_old)

        stats = {"pi/q_loss": adv_loss,
                 #"pi/innovation":innovation,
                 "pi/v_loss": td_loss,
                 "pi/entropy": pi.entropy().mean(),
                 "pi/kl": kl.mean(),
                 #"pi/adv_grad_norm":adv_norm
                 }
        #stats.update(opt_stats)
        return stats
