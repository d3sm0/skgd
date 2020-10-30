import pprint
import time

import numpy as np
import scipy.sparse.linalg as sla
import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer

EPS = 1e-8


class _RequiredParameter(object):
    """Singleton class representing a required parameter for an Optimizer."""

    def __repr__(self):
        return "<required parameter>"


required = _RequiredParameter()


class Conditioner:
    def __init__(self, input_dim, init_value=1.0, q=None):
        self.input_dim = input_dim
        self.p = torch.eye(input_dim) * init_value
        self.Id = torch.eye(input_dim)
        if q > 0:
            self.random_mask = torch.distributions.Bernoulli(probs=torch.ones((1,)) * q)

    def predict(self, p_v):
        self.p.add_(self.Id * p_v)
        if hasattr(self, "random_mask"):
            mask = self.random_mask.sample((self.input_dim,)).squeeze().bool()
            self.p[mask, :] = 0
        return self.p

    def update(self, h, lr):
        # P h h' P
        p_grad = self.p.matmul(torch.ger(h, h)).matmul(self.p)
        self.p.add_(p_grad, alpha=-lr)
        return p_grad


import pickle
def load_matrix(input_dim, rank):
    fname = f"matrix/m:{input_dim}:{rank}.pkl"
    with open(fname, "rb") as m:
        return pickle.load(m)


def save_matrix(r, u, input_dim, rank):
    fname = f"matrix/m:{input_dim}:{rank}.pkl"
    with open(fname, "wb") as m:
        pickle.dump((r, u, rank), m)

class LowRank(Conditioner):
    def __init__(self, *args, **kwargs):
        self.rank = kwargs.pop("rank")
        super().__init__(*args, **kwargs)
        self.Id = torch.eye(self.rank)
        input_dim = self.p.shape[0]
        try:
            r, u, rank = load_matrix(input_dim, self.rank)
        except FileNotFoundError:
            (r, u), rank = may_get_eigen(self.p.numpy(), rank=self.rank)
            save_matrix(r, u, input_dim, rank)
        self.r = torch.FloatTensor(r) * self.Id
        self.u = torch.FloatTensor(u)

    def predict(self, p_v):
        self.r.add_(self.Id * p_v)
        if hasattr(self, "random_mask"):
            mask = self.random_mask.sample((self.rank,)).squeeze().bool()
            self.r[mask, :] = 0
        p = self.u.matmul(self.r).matmul(self.u.T)
        self.p = p
        return p

    def update(self, h, lr):
        outer = torch.ger(h, h)
        r_grad = self.r.matmul(self.u.T).matmul(outer).matmul(self.u).matmul(self.r)
        self.r.add_(r_grad, alpha=-lr)
        self.r = self.r / (self.r.norm() + EPS)
        self.r.set_(self.r)
        return r_grad


class PKTDOptimizer(Optimizer):
    def __init__(self, params, p_v=1e-2, p_n=1, lr=0.5, q=0.1, verbose=True, **kwargs):

        params = list(params)
        input_dim = sum([p.numel() for p in params])
        self.input_dim = input_dim
        self.p = torch.eye(input_dim) * 10
        #self.Id = torch.eye(input_dim)
        #if q > 0:
        #   self.random_mask = torch.distributions.Bernoulli(probs=torch.ones((1,)) * q)
        self.conditioner = Conditioner(input_dim, init_value=10, q=q)
        self.verbose = verbose

        defaults = dict(lr=lr, p_n=p_n, p_v=p_v, q=q)
        print(f"Initialized Optimzer params {input_dim}")
        pprint.pprint(defaults)

        super(PKTDOptimizer, self).__init__(params, defaults)

    # kalman update is
    def _predict(self, theta):
        #p = self.p + self.Id * self.param_groups[0]["p_v"]
        #if hasattr(self, "random_mask"):
        #   mask = self.random_mask.sample((self.input_dim,)).squeeze().bool()
        #   p[mask, :] = 0
        p = self.conditioner.predict(self.param_groups[0]["p_v"])
        return theta, p

    @torch.no_grad()
    def _update(self, theta, h, innovation):
        theta, p = self._predict(theta)
        k, p_r, theta_grad = self._update_theta(h, innovation, p, theta)

        # P h h' P
        #p_grad = p.matmul(torch.ger(h, h)).matmul(p)
        #p.add_(p_grad, alpha=-self.param_groups[0]["lr"]/p_r)
        #self.p.set_(p)

        p_grad = self.conditioner.update(h, lr=self.param_groups[0]["lr"] / p_r)

        if self.verbose:
            stats = {
                "opt/grad/theta": theta_grad.norm(),
                "opt/grad/cov": p_grad.norm(),
                #            "opt/cov/shift": cov_update,
                #   "opt/grad/p": p_grad.norm(),
                #   "opt/grad/h_norm": h_norm,
                #   "opt/grad/h_max":h_max,
                #    "opt/cov/matrix_rank": torch.matrix_rank(self.p),
                #    "opt/cov/update_rank": torch.matrix_rank(p_grad),
                "opt/k/mu": k.mean(),
                "opt/k/std": k.std(),
                "opt/residual/normalized": innovation.pow(2) / p_r,
                "opt/residual/mu": innovation,
                "opt/residual/cov": p_r.mean(),
            }
        self.p = self.conditioner.p

        return theta, stats

    def _update_theta(self, h, innovation, p, theta):
        p_theta = p.mv(h)
        p_r = h.T.matmul(p).matmul(h) + self.param_groups[0]["p_n"]
        k = p_theta / p_r
        theta_grad = k * innovation
        theta.add_(theta_grad, alpha=self.param_groups[0]["lr"])
        return k, p_r, theta_grad

    @torch.no_grad()
    def step(self, innovation, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        stats = {}
        for group in self.param_groups:
            theta = nn.utils.parameters_to_vector(group["params"]).contiguous()
            h = nn.utils.parameters_to_vector([p.grad for p in group["params"]])
            h = h / (h.norm() + EPS)
            theta, stats = self._update(theta, h, innovation)
            nn.utils.vector_to_parameters(theta, group["params"])
            assert not any(
                [torch.isnan(v) for v in stats.values()]
            ), f"Found nan in stats {stats}"

        return stats


def may_get_eigen(p, rank):
    canditadates_rank = list(range(rank, 200, 1))
    r = None
    for rank in canditadates_rank:
        try:
            r, u = sla.eigsh(p, k=rank)
            break
        except Exception as e:
            print(e)
            pass
    if r is None:
        r, u = np.linalg.eig(p)
        rank = p.shape[1]
    u, _ = np.linalg.qr(u - (u @ u.T @ u))
    return (r, u), rank


class APKTDOptimizer(PKTDOptimizer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import warnings; warnings.warn("Running LRPKTD. No explicit regularizer needed")
        self.conditioner = LowRank(self.input_dim, q=0, rank=kwargs["rank"], init_value=10)



#
#class APKTDOptimizer(Optimizer):
#    def __init__(
#            self,
#            params,
#            p_v=1e-2,
#            p_n=1,
#            lambda_=0.0,
#            lr=0.5,
#            q=0.1,
#            rank=100,
#            momentum=0,
#            dampening=0,
#            weight_decay=0,
#            nesterov=False,
#            verbose=True,
#    ):
#        q = 0
#        params = list(params)
#        input_dim = sum([p.numel() for p in params])
#        assert not (q != 0 and 0 != lambda_), "Either lambda or q."
#        self.p = torch.eye(input_dim)  # * 1e-3
#        self.verbose = verbose
#        (r, u), rank = may_get_eigen(self.p.numpy(), rank=rank)
#        u, _ = np.linalg.qr(u - (u @ u.T @ u))
#
#        print(f"Reducing from {input_dim}-> {rank}")
#        self.rank = rank
#        self.Id = torch.eye(rank)
#        self.lr = lr
#        self.r = torch.diag(torch.FloatTensor(r))
#        self.u = torch.FloatTensor(u)
#        defaults = dict(
#            lr=lr,
#            p_n=p_n,
#            p_v=p_v,
#            q=q,
#        )
#        if q > 0:
#            self.random_mask = torch.distributions.Bernoulli(probs=torch.ones((1,)) * q)
#
#        super(APKTDOptimizer, self).__init__(params, defaults)
#
#    # kalman update is
#    @torch.no_grad()
#    def _predict(self, theta):
#        r = self.r + self.Id * self.param_groups[0]["p_v"]
#        if self.param_groups[0]["q"] > 0:
#            r = self._project(r)
#        u = self.u
#        return theta, r, u
#
#    def _project(self, p):
#        mask = self.random_mask.sample(p.shape[:1]).squeeze().bool()
#        p[mask, :] = 0
#        return p
#
#    @torch.no_grad()
#    def _update(self, theta, h, innovation):
#        stats = None
#        h_norm = h.norm()
#        h = h / (h_norm + EPS)
#        theta, r, u = self._predict(theta)
#
#        p = u.matmul(r).matmul(u.T)
#        p_theta = p.matmul(h)
#        p_r = h.T.matmul(p).matmul(h) + self.param_groups[0]["p_n"]
#        k = p_theta / p_r
#
#        theta_grad = self.lr * k * innovation
#        theta.add_(theta_grad, alpha=self.lr)
#
#        outer = torch.ger(h, h)
#        r_grad = (r.matmul(u.T).matmul(outer).matmul(u).matmul(r)) / p_r
#        r.add_(r_grad, alpha=-1 * self.lr)
#        # r.clamp_min_(0)  # symmetric no but psd si
#        r = r / (r.norm() + EPS)
#        self.r.set_(r)
#
#        if self.verbose:
#            stats = {
#                "opt/grad/theta": theta_grad.norm(),
#                "opt/grad/p": r_grad.norm(),
#                "opt/grad/h_norm": h_norm,
#                "opt/r/mu": r.mean(),
#                "opt/p/min": r.min(),
#                # "opt/p/rank": torch.matrix_rank(p),
#                "opt/k/mu": k.mean(),
#                "opt/k/std": k.std(),
#                "opt/k/ratio": p_theta.norm() / p_r.norm(),
#                "opt/residual/normalized": innovation.pow(2) / p_r,
#                "opt/residual/mu": innovation,
#                "opt/residual/cov": p_r.mean(),
#            }
#
#        return theta, stats
#
#    @torch.no_grad()
#    def step(self, innovation, closure=None):
#        """Performs a single optimization step.
#
#        Arguments:
#            closure (callable, optional): A closure that reevaluates the model
#                and returns the loss.
#        """
#        loss = None
#        if closure is not None:
#            with torch.enable_grad():
#                loss = closure()
#        stats = {}
#        for group in self.param_groups:
#            theta = nn.utils.parameters_to_vector(group["params"]).contiguous()
#            h = nn.utils.parameters_to_vector([p.grad for p in group["params"]])
#            theta, stats = self._update(theta, h, innovation)
#            nn.utils.vector_to_parameters(theta, group["params"])
#            assert not any(
#                [torch.isnan(v) for v in stats.values()]
#            ), f"Found nan in stats {stats}"
#
#        return stats
#

def timeit(fn, iterations=100):
    dts = []
    for _ in range(iterations):
        start = time.time()
        fn()
        end = time.time() - start
        dts.append(end)
    dts = torch.FloatTensor(dts)

    print(f"mu:\t{dts.mean()}\tstd:\t{dts.std()}")


def _test_rank():
    shape = 4000
    p = np.eye(shape)
    rank = 20
    while rank != shape:
        (_, u), rank = may_get_eigen(p, rank=rank)
        u = torch.FloatTensor(u)
        torch.qr(u - (u @ u.T @ u))
        print(rank)
        rank += 1


if __name__ == "__main__":
    _test_rank()
