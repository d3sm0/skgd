import torch
from torch import nn as nn, distributions as dist

from misc.torch_utils import init


class Bias(nn.Module):
    def __init__(self, bias, requires_grad):
        super(Bias, self).__init__()
        self._bias = nn.Parameter(bias, requires_grad=requires_grad)

    def forward(self, x):
        return self._bias


class DiagGaussian(nn.Module):
    def __init__(self, num_inputs, num_outputs):
        super(DiagGaussian, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.mu = init_(nn.Linear(num_inputs, num_outputs))
        bias = torch.zeros(size=(num_outputs,))
        self.log_std = Bias(bias, requires_grad=True)
        self._id_matrix = torch.eye(num_outputs)

    def forward(self, x):
        mu = self.mu(x)
        cov = self.log_std(mu).exp().pow(2) * self._id_matrix
        try:
            pi = dist.MultivariateNormal(mu, cov)
        except RuntimeError as e:
            print(mu, cov)
            raise e
        return pi


class Categorical(nn.Module):
    def __init__(self, input_size, action_space):
        super(Categorical, self).__init__()

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=0.01)

        self.linear = nn.Linear(input_size, action_space, bias=False)

    def forward(self, x):
        x = self.linear(x)
        return torch.distributions.Categorical(logits=x)


class LinearModel(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LinearModel, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.fc = nn.Sequential(
            init_(nn.Linear(input_size, hidden_size)),
            nn.ELU(),
            init_(nn.Linear(hidden_size, hidden_size)),
            nn.ELU(),

        )

    def forward(self, x):
        h = self.fc(x)
        return h


class NPGNetwork(nn.Module):

    def __init__(self, obs_space, action_space, hidden_size=64):
        super().__init__()

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.phi = LinearModel(obs_space, hidden_size)
        # self.project = nn.Linear(hidden_size, hidden_size * action_space, bias=False)
        # self.theta = nn.Linear(obs_space, action_space, bias=False)
        # self.v = nn.Linear(obs_space, 1, bias=False)
        # self.pi = DiagGaussian(hidden_size, action_space)
        # self.pi = Categorical(input_size=hidden_size, action_space=action_space)

        # self.policy_params = list(self.pi.parameters())
        # self.theta_params = list(self.theta.parameters())
        # self.phi_params = list(self.phi.parameters()) + list(self.project.parameters())
        # self.hidden_size = hidden_size
        # self.action_space = action_space

    def forward(self, x):
        # q = self.theta(x)
        h = self.phi(x)
        pi = self.pi(h)

        return pi


class Actor(nn.Module):

    def __init__(self, obs_space, action_space, hidden_size=32):
        super().__init__()

        self.model = LinearModel(obs_space, hidden_size)
        if len(action_space.shape) == 0:
            pi = Categorical(hidden_size, action_space.n)
        else:
            pi = DiagGaussian(hidden_size, action_space.shape[0])
        self.pi = pi

    def forward(self, x):
        h = self.model(x)
        pi = self.pi(h)
        return pi


class Critic(nn.Module):

    def __init__(self, obs_space, action_space, hidden_size=32):
        super().__init__()
        self.model = LinearModel(obs_space, hidden_size)
        self.q = nn.Linear(hidden_size, action_space, bias=False)

    def forward(self, x):
        h = self.model(x)
        q = self.q(h)
        return q.squeeze(dim=1)


class LinearCritic(nn.Module):

    def __init__(self, hidden_size=64):
        super().__init__()

        # init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        # self.model = LinearModel(obs_space, hidden_size)
        self.v = nn.Linear(hidden_size, 1)
        # self.critic_params = list(self.model.parameters()) + list(self.v.parameters())

    def forward(self, x):
        v = self.v(x)
        return v.squeeze()

    def get_value(self, s):
        with torch.no_grad():
            return self.forward(s)
