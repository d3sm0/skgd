import gym
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
np.random.seed(0)

from misc.utils import one_hot


def _random_transition(t, mu, std, n):
    p = stats.norm.cdf(t, mu, std)
    x = stats.binom.rvs(n, p)
    return x


class Inventory(gym.Env):
    def __init__(self):
        super(Inventory, self).__init__()
        self.mu = 0
        self.std = 1
        self.min_quantity = 0
        self.max_stock = 20
        self.max_steps = 48
        self.max_order_size = 10  # max order size
        self.stock = None
        self.max_loss = self.max_stock
        self.infos = {"actions": [], "state": [], "demand": [], "next_state": [], "production": [], "reward": []}
        self.t = 0
        self.path = "../logs/"
        print(self.__repr__())

    def _demand_fn(self, t):
        raise NotImplementedError()

    def _production_fn(self, t):
        raise NotImplementedError()

    def step(self, action):
        demand = self.demand
        production = action + self.production
        next_state = self.stock + production - demand
        reward = self.reward(production, next_state)
        done = self.t > self.max_steps
        next_state = max(0, min(next_state, self.max_stock - 1))

        self.infos["state"].append(self.stock)
        self.infos["actions"].append(action)
        self.infos["next_state"].append(next_state)
        self.infos["demand"].append(self.demand)
        self.infos["production"].append(self.production)
        self.infos["reward"].append(reward)

        self.stock = next_state

        self.demand = self._demand_fn(self.t)
        self.production = self._production_fn(self.t)

        info = {"demand": self.demand, "production": self.production, "reward": reward}
        self.t += 1
        reward = reward / self.max_loss
        assert self.stock < self.max_stock
        return self.stock, reward, done, info

    def reward(self, action, next_state):
        import_price = 10
        export_price = 1.5
        order_price = 1

        order_cost = - order_price * action  # product requested
        export_cost = - export_price * max(0, (next_state - self.max_stock))  # product can not be carried in stock
        import_cost = import_price * min(0, next_state)  # unsatisfied demand
        return order_cost + export_cost + import_cost

    def render(self, mode, **kwargs):
        fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharex=True)
        for ax, (k, v) in zip(axs.flatten(), self.infos.items()):
            ax.plot(v, label=k, linestyle='--', marker='o', markersize=4, alpha=.8)
            ax.set_title(k)
        plt.xlabel("time")
        plt.tight_layout()
        # fig.legend(list(self.infos.keys()))
        plt.savefig(self.path + "/render.png")
        plt.close()

    def reset(self):
        self.demand = self._demand_fn(0)
        self.production = self._production_fn(0)
        self.t = 0
        self.stock = self.min_quantity
        self.infos = {"actions": [], "state": [], "demand": [], "next_state": [], "production": [], "reward": []}
        return self.stock

    def __repr__(self):
        return "InventoryEnv"


class DeterministicDemand(Inventory):
    k = 2

    def __init__(self):
        super(DeterministicDemand, self).__init__()

    def _production_fn(self, t):
        return 0

    def _demand_fn(self, t):
        return DeterministicDemand.k

    def __repr__(self):
        return "DeterministicDemandEnv"


class Stochastic(Inventory):
    lambda_demand = 1.
    lambda_production = 1.

    def _demand_fn(self, t):
        x = int(stats.poisson.rvs(Stochastic.lambda_demand))
        return x

    def _production_fn(self, t):
        _action = stats.poisson.rvs(Stochastic.lambda_production)
        return int(_action)

    def __repr__(self):
        return "StochasticDemand"


class BoundedVariation(Inventory):
    alpha = np.array([5])
    w = np.array([1 / 3])

    def _demand_fn(self, t):
        x = periodic_transition(t, self.w, self.alpha, phi=0, c=0)
        x = max(0, x)
        x = min(x, self.max_stock - 1)
        assert self.max_stock > x >= 0
        return int(x)

    def _production_fn(self, t):
        x = periodic_transition(t, self.w / 2, self.alpha, phi=0, c=0)
        x = max(0, x)
        x = min(x, self.max_order_size - 1)
        assert self.max_order_size > x >= 0
        return int(x)

    def __repr__(self):
        return "BoundedVariation"

    def l_coefficient(self):
        p = []
        d = []
        for t in reversed(range(0, self.max_steps + 2)):
            p.append(self._production_fn(t))
            d.append(self._demand_fn(t))
        p = np.abs(np.diff(np.array(p))).max()
        d = np.abs(np.diff(np.array(d))).max()

        return p + d

    def get_optimal_policy(self):
        stock = 0
        actions = []

        for t in reversed(range(self.max_steps + 2)):
            demand = self._demand_fn(t)
            production = self._production_fn(t)
            net = demand - (stock + production)  # over production or unsatisfied demand
            a = 0
            remaining_stock = 0
            requested_stock = 0
            if net > 0:
                # there is unsastified demand than order stuff
                delta_a = net - self.max_order_size
                if delta_a > 0:
                    requested_stock = delta_a
                a = min(net, self.max_order_size)
            elif net < 0:
                # stock and prod are > 0
                delta_s = (stock + production) - demand
                remaining_stock = min(delta_s, self.max_stock)
                lost_stock = max(0, stock - self.max_stock)

            stock = remaining_stock + requested_stock
            actions.append(a)
        np.save("pi_star.npy", actions)
        return actions


class OneHotWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super(OneHotWrapper, self).__init__(env)
        self.action_space = gym.spaces.Discrete(n=self.env.unwrapped.max_order_size)
        self.observation_space = gym.spaces.Discrete(n=self.env.unwrapped.max_stock)

    def observation(self, observation):
        return one_hot(observation, self.env.unwrapped.max_stock)


class CheatWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super(CheatWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Discrete(n=self.env.unwrapped.max_stock + 3)
        self.action_space = gym.spaces.Discrete(n=self.env.unwrapped.max_order_size)

    def observation(self, observation):
        return np.concatenate([observation, [self.env.unwrapped.production, self.env.unwrapped.demand, self.t]])


def action_encoding(max_order_size):
    actions = []
    for a in list(range(max_order_size)):
        actions.append(one_hot(a, c=max_order_size))
    actions = np.stack(actions)
    return actions


def periodic_transition(t, w, alpha, phi, c=0):
    return c + (np.sin(t * w + phi) * alpha).sum()



action_features = action_encoding(max_order_size=10)


# def _reward(stock, action, next_state, demand):
#    recurrent_cost = 1
#    fixed_cost = 0
#    unit_cost = 1
#    price = 1
#
#    total_recurrent_cost = next_state * recurrent_cost
#    total_revenue = np.maximum(0, stock + action - next_state) * price
#    total_cost = action * unit_cost + fixed_cost
#
#    profit = total_revenue - total_cost - total_recurrent_cost
#    return profit

def cv():
    ws = np.array([1, 1 / 2, 1 / 4])
    alpha = np.linspace(0, 5, num=3, dtype=np.int)

    import itertools
    test_params = list(itertools.product(ws, alpha))
    #ls = []
    #for w, alpha in test_params:
    #    env = BoundedVariation()
    #    l = env.l_coefficient()
    #    ls.append((l, w, alpha))

    return test_params

def make_inventory(env_id):
    # env = DeterministicDemand()
    env = BoundedVariation()

    env.get_optimal_policy()
    # env =Stochastic()
    env = OneHotWrapper(env)
    if "Cheat" in env_id:
        env = CheatWrapper(env)
    return env


def test_env(env):
    env.reset()
    rws = 0
    demand = env.stock
    while True:
        action = env.stock - demand
        s, r, done, info = env.step(action)
        rws += r
        demand = info["demand"]
        print(env.t, s, r, info["demand"])
        if done:
            print("Done", rws)
            break


def test_envs():
    env = DeterministicDemand()
    test_env(env)
    # env = RandomProduction(env)
    # test_env(env)


if __name__ == "__main__":
    test_envs()
