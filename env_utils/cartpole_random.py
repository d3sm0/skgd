import collections

import numpy as np
import pybullet as pb
import scipy.stats as stats
from pybullet_envs.bullet.cartpole_bullet import CartPoleContinuousBulletEnv

# Setting perturbation process parameters.
# add new perturb spec for reacher
# debug /test
# check why ppo is shorter
# break ppo
# try directly
# launch the sweep on both pole,reacher and vdp

perturb_spec = {
    'mass': dict(start=0.1, lower_bound=0.1, upper_bound=10., std=0.5),
    # 'jointDamping': dict(start=1e-6, lower_bound=1e-6, upper_bound=1e-3, std=1e-4),
    'angularDamping': dict(start=1e-4, lower_bound=1e-4, upper_bound=3., std=0.3),
}
force_spec = {
    'x': dict(start=0.0, lower_bound=0.0, upper_bound=5., std=0.1),
    'y': dict(start=0.0, lower_bound=0.0, upper_bound=5., std=0.1),
}

import gym
import pybullet_envs

pybullet_envs.registry


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

    def reset(self):
        self.mean.fill(0)
        self.var.fill(0)
        self.count = 1e-4
        self.latest.clear()

    def get_normalized(self):
        return self.standardize(np.array(self.latest))

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


class DetDist:
    def __init__(self, spec, resolution=250):
        self.__dict__.update(spec)
        self.dx = (self.upper_bound - self.lower_bound) / resolution
        print(f'resolution drift {self.dx}')

    def rvs(self):
        return self.dx

# this is per attribute
class NonStationaryDrift:
    def __init__(self, spec, resolution=10000):
        self.__dict__.update(spec)
        self.state = self.start
        self.dx = (self.upper_bound - self.lower_bound) / resolution
        self.std = 1e-6
        print(self.dx)

    def sample(self):
        self.std += self.dx
        # variance increases as t-> infinity
        delta = stats.halfnorm(scale=self.std).rvs()
        state = self._clip(self.state + delta)
        self.state = state
        return state

    def reset(self):
        self.state = self.start
        self.std = 1e-6
        return self.state

    def _clip(self, delta):
        return np.clip(delta, self.lower_bound, self.upper_bound)


class Drift:
    def __init__(self, spec):
        self.dist = stats.halfnorm(scale=spec["std"])
        self.state = spec["start"]
        self.__dict__.update(spec)

    def sample(self):
        delta = self.dist.rvs()
        state = self._clip(self.state + delta)
        self.state = state
        return state

    def reset(self):
        self.state = self.start
        return self.state

    def _clip(self, delta):
        return np.clip(delta, self.lower_bound, self.upper_bound)


class  DetDrift(Drift):
    def __init__(self, spec):
        super().__init__( spec)
        self.dist = DetDist(spec=spec)

class ManualForce:
    def __init__(self, p):
        self.x = p.addUserDebugParameter("x", 0.0, 1.0, 0)
        self.y = p.addUserDebugParameter("y", 0.0, 1.0, 0)
        self.z = p.addUserDebugParameter("z", 0., 1.0, 0)

    def get(self, p):
        x = p.readUserDebugParameter(self.x)
        y = p.readUserDebugParameter(self.y)
        z = p.readUserDebugParameter(self.z)
        return [x, y, z]


class ManualDrift:
    def __init__(self, p):
        self.mass_slider = p.addUserDebugParameter("mass", 0.1, 10, 0)
        self.joint_slider = p.addUserDebugParameter("jointDamping", 2e-6, 1e-3, 0)
        self.angular_slider = p.addUserDebugParameter("angularDamping", 5e-4, 3, 0)
        # self.linear_slider = p.addUserDebugParameter("linearDamping", 0, 1, 0)
        # self.friction = p.addUserDebugParameter("lateralFriction", 0, 1, 0)

    def get(self, p):
        mass = p.readUserDebugParameter(self.mass_slider)
        angular_damping = p.readUserDebugParameter(self.angular_slider)
        joint_slider = p.readUserDebugParameter(self.joint_slider)
        # linear_slider = p.readUserDebugParameter(self.linear_slider)
        # friction = p.readUserDebugParameter(self.friction)
        return {
            "mass": mass,
            "angularDamping": angular_damping,
            "jointDamping": joint_slider,
            # "linearDamping": linear_slider,
            # "lateralFriction": friction,
        }


class Entity:
    def __init__(self, body_idx, joint_idx, use_mass=False, entity_name=None):
        self.objects = {}
        self.entity_name = entity_name
        for k, v in perturb_spec.items():
            if "mass" in k and use_mass is False:
                continue
            self.objects[k] = Drift(v)  # Drift(v)
        self.body_idx = body_idx
        self.joint_idx = joint_idx
        # print(f"Init entity {entity_name}")

    def randomize(self, reset_state=False):
        change = {}
        for k, v in self.objects.items():
            if reset_state:
                new_state = v.reset()
            else:
                new_state = v.sample()
            change[k] = new_state
        return change

class ForceEntity:
    def __init__(self, body_idx, joint_idx, force_spec, entity_name, randomize):
        self.objects = {}
        self.entity_name = entity_name
        if randomize == "random":
            drift_class = Drift
        elif randomize == "sequential":
            drift_class = NonStationaryDrift
        elif randomize == "deterministic":
            drift_class = DetDrift
        else:
            raise NameError("wrong drift")
        for k, v in force_spec.items():
            self.objects[k] = drift_class(v)
        self.body_idx = body_idx
        self.joint_idx = joint_idx
        # print(f"Init entity {entity_name}")

    # TODO duplicate code remove me
    def randomize(self, reset_state=False):
        change = {}
        for k, v in self.objects.items():
            if reset_state:
                new_state = v.reset()
            else:
                new_state = v.sample()
            change[k] = new_state
        return list(change.values()) + [0]


class NormalizeObservation(gym.ObservationWrapper):
    def __init__(self, env, clip=(-10, 10)):
        super(NormalizeObservation, self).__init__(env)
        shape = self.observation_space.shape
        self._stats = RunningMeanStd(shape=shape)
        self._clip = clip

    def observation(self, observation):
        o = np.clip(observation, *self._clip)
        self._stats.update(o)
        o = self._stats.standardize(o)
        return o

    def reset(self):
        self._stats.reset()
        return super(NormalizeObservation, self).reset()


class RandomizeEnv(gym.Wrapper):
    def __init__(self, env):
        super(RandomizeEnv, self).__init__(env)
        self.mutable_entities = {"terrain": Entity(body_idx=0, joint_idx=-1, use_mass=False, entity_name="terrain")}
        self.forces = {}
        self.reset()

    def step(self, action):
        s, r, d, info = super(RandomizeEnv, self).step(action)
        specs = self.randomize_env()
        info["param"] = specs
        return s, r, d, info

    def reset(self, **kwargs):
        s = super(RandomizeEnv, self).reset()
        self.randomize_env(reset_state=True)
        return s

    def randomize_env(self, reset_state=False):
        _specs = {}
        #for entity_name, entity in self.mutable_entities.items():
        #    new_spec = entity.randomize(reset_state=reset_state)
        #    self.unwrapped._p.changeDynamics(entity.body_idx, entity.joint_idx, **new_spec)
        #    _specs.update(new_spec)

        for entity_name, entity in self.forces.items():
            new_spec = entity.randomize(reset_state=reset_state)
            self.unwrapped._p.applyExternalForce(entity.body_idx, entity.joint_idx, new_spec, [0, 0, 0],
                                                 flags=pb.WORLD_FRAME)
            _specs.update(dict(force=new_spec))

        return _specs

class RandomizePole(RandomizeEnv):
    def __init__(self, env, randomize="random"):
        super(RandomizePole, self).__init__(env)
        # self.mutable_entities["cart"] = Entity(body_idx=0, joint_idx=0, use_mass=True, entity_name="cart")
        # self.mutable_entities["pole"] = Entity(body_idx=0, joint_idx=1, use_mass=True, entity_name="pole")
        self.forces["cart"] = ForceEntity(body_idx=0, joint_idx=0, force_spec=force_spec, entity_name="cart", randomize=randomize)
        self.forces["pole"] = ForceEntity(body_idx=0, joint_idx=1, force_spec=force_spec, entity_name="pole", randomize=randomize)


class RandomizeReacher(RandomizeEnv):
    def __init__(self, env):
        super(RandomizeReacher, self).__init__(env)

        joint_name = self.robot.central_joint.joint_name
        self.mutable_entities[joint_name] = Entity(
            body_idx=self.robot.central_joint.bodyIndex,
            joint_idx=self.robot.central_joint.jointIndex, use_mass=False
        )
        joint_name = self.robot.elbow_joint.joint_name
        self.mutable_entities[joint_name] = Entity(
            body_idx=self.robot.elbow_joint.bodyIndex,
            joint_idx=self.robot.elbow_joint.jointIndex, use_mass=False
        )


def make_bullet(env_id, randomize=False):
    # Continuous
    # env_id = "Pendulum-v0"
    # import ipdb; ipdb.set_trace()
    env = gym.make(env_id)
    # env = NormalizeObservation(env)
    np.random.seed(0)
    env.seed(0)
    print(randomize)
    if randomize != 0:
        print(f"Init:\t{env_id}\trandomize:\t{randomize}")
        if "CartPole" in env_id:
            env = RandomizePole(env, randomize)
        elif "Reacher" in env_id:
            env = RandomizeReacher(env, randomize)
    # assert type(env) is RandomizeEnv
    return env


def _test_randomize():
    env = CartPoleContinuousBulletEnv(renders=True)
    # env = AntBulletEnv(render=True)
    # env = HopperBulletEnv(render=True)
    # env = ReacherBulletEnv(render=True)
    # _run_env_manual(env)
    env = RandomizePole(env,randomize="deterministic")
    # env = NormalizeObservation(env)
    _run_env(env)


def _run_env_manual(env):
    env.reset()
    p = env.unwrapped._p
    drifter = ManualDrift(p)
    # force = ManualForce(p)
    p.configureDebugVisualizer(pb.COV_ENABLE_GUI, 1)
    body_idx = 0
    link_idx = 14
    # action_shape = 2
    while p.isConnected():
        new_dynamics = drifter.get(p)
        p.changeDynamics(body_idx, link_idx, **new_dynamics)
        ##p.applyExternalTorque(body_idx, link_idx, force.get(p), p.WORLD_FRAME)
        ##p.applyExternalTorque(body_idx, 2, [-x for x in force.get(p)], p.WORLD_FRAME)
        # state = p.getJointState(body_idx, link_idx)
        # print(f"p:{state[0]}\tv:{state[1]}")
        # s, r, done, _ = env.step(np.zeros(shape=(action_shape,)))
        # print(np.round(s, 2))
        # time.sleep(0.01)
        p.stepSimulation()
        # if done:
        #    break


def _run_env(env):
    done = False
    env.reset()
    a = np.zeros_like(env.action_space.shape)
    forces = []
    import time
    import matplotlib.pyplot as plt
    t= 0
    while t < 200:
        s, r, done, info = env.step(a)
        forces.append(info["param"]["force"])
        print(info)
        if done:
            env.unwrapped.reset()
        t+=1
        #time.sleep(0.1)
        
    forces = np.stack(forces)[:, :-1]
    plt.scatter(*forces.T)
    plt.show(block=False)
    plt.pause(5)
    plt.close()


if __name__ == '__main__':
    _test_randomize()
