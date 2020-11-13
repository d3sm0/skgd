#!/usr/bin/env python
import datetime
import functools
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch

from env_utils.env_utils import RecordEpisodeStatistics
from misc.metric_logger import MetricLogger
from misc.torch_utils import ReplayMemory, TorchEnv, record_stats
from models import Critic, Actor
from pg import PPOKTD, PG

from env_utils.cartpole_random import NormalizeObservation


class SummaryWriter:
    def __init__(self, path, *args):
        self.log_dir = path
        os.makedirs(self.log_dir, exist_ok=True)

    def add_hparams(self, config, *args):
        connected = False
        trial = 0
        total_trials = 10
        while not connected:
            if connected:
                break
            elif trial == total_trials:
                from wandb.sdk.wandb_settings import Settings

                Settings.disabled = True
                connected = True
            try:
                _name = f"{config['env']['env_id']}:{config['agent']['agent_id']}"
                wandb.init(project=config["project_id"], name=_name, config=config)
                connected = True
            except wandb.errors.error.UsageError:
                print(f"Failed to connect {trial}/{total_trials}")
            trial += 1

    def add_scalar(self, k, v, global_step):
        wandb.log({k: v, "global_step": global_step})

    def add_figure(self, k, v, global_step):
        wandb.log({k: v, "global_step": global_step})

    def close(self):
        return


def eval_agent(agent, env):
    agent.train(mode=False)
    s = env.reset()
    while True:
        a, v, _ = agent.act(s)
        s1, r, done, info = env.step(a)
        s = s1
        if done:
            break
    agent.train(mode=True)
    return info


def run_once(agent, env, memory, n_steps, s):
    done = False

    last_info = {}
    for step in range(n_steps):
        a, v, log_pi = agent.act(s)
        s1, r, done, info = env.step(a)
        memory.push(s, a, r, s1, done)
        s = s1
        if done:
            s = env.reset()
            last_info.update(info)
            # break
    if len(last_info.keys()) == 0:
        last_info.update(info)
    return s, done, last_info


def stop_by_samples(step, max_samples=1):
    if step >= max_samples:
        return True


def make_env(env_id, randomize=False, seed=0, monitor_path=None):
    if "bullet" in env_id.lower():
        from env_utils.cartpole_random import make_bullet

        env = make_bullet(env_id, randomize=randomize)
    elif env_id == "vdp":
        # interestingly an std of .04 makes the system much harder to control
        import gym_vdp

        env = gym_vdp.make_vdp(env_id, randomize=randomize, constrained=False, std=0.04)
    elif env_id == "microgrid":
        from env_utils import make_microgrid

        env = make_microgrid(env_id)
    else:
        raise Exception(env_id)

    # env = NormalizeObservation(env)
    env = RecordEpisodeStatistics(env)
    env = TorchEnv(env)

    env.seed(seed)
    env.env_id = env_id
    return env


WANDB = True if os.uname().nodename != "roach" else False
DEBUG = False if os.uname().nodename != "roach" else True
# import ipdb

if not WANDB:

    from torch.utils.tensorboard import SummaryWriter

    class wandb:
        def log(self, *args, **kwargs):
            return

        def watch(self, *args, **kwargs):
            return


else:
    import wandb


def main(config):
    seed = config.env.seed
    env_id = config.env.env_id
    agent_id = config.agent.agent_id
    batch_size = config.train.batch_size

    torch.random.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    config.project_id = (
        "ktd-single-env-reboot" if WANDB is True and not DEBUG else "test"
    )
    dtm = datetime.datetime.now().strftime("%d-%H-%M-%S-%f")
    path = f"{config.project_id}/{config.env.env_id}/{config.agent.agent_id}/{dtm}"
    print(f"\nResults saved at :{path}\n")
    writer = SummaryWriter(os.path.join(path, "tb"))
    try:
        writer.add_hparams(config.get_dict(), {})  # just fake
    except ValueError:
        # this condition rises if using tensorboard
        pass

    if not WANDB:
        os.makedirs(f"{writer.log_dir}/plot/", exist_ok=True)
    logger = MetricLogger(output_dir=os.path.join(path, "metric"))

    env = make_env(env_id, randomize=config.env.randomize, seed=seed)
    eval_env = make_env(
        env_id,
        randomize=config.env.randomize,
        seed=seed,
        monitor_path=os.path.join(path, "monitor"),
    )

    obs_space = env.observation_space.shape[0]

    critic = Critic(obs_space, action_space=1, hidden_size=config.agent.params)

    actor = Actor(obs_space, action_space=env.action_space)

    if agent_id == "ktd":
        agent = PPOKTD(
            actor,
            critic,
            p_n=config.agent.p_n,
            p_v=config.agent.p_v,
            lr=config.agent.lr,
            q=config.agent.q,
            rank=config.agent.rank,
        )

    else:
        agent = PG(actor, critic)

    memory = ReplayMemory(capacity=int(1e4), recent_queue=config.train.n_steps)
    # if config.env.env_id == "microgrid":
    #    stop_fn = functools.partial(stop_by_episode, max_episodes=config.train.max_episodes)
    # else:
    stop_fn = functools.partial(stop_by_samples, max_samples=config.train.max_samples)
    train(env, agent, memory, writer, logger, eval_env, config.train, stop_fn)


def train(env, agent, memory, writer, logger, eval_env, train_config, stop_fn):
    global_step = 0
    s = env.reset()
    # for _ in logger.log_every(itertools.count(), print_freq=train_config.print_every, max_steps=train_config.max_steps):
    while True:
        s, done, info = run_once(agent, env, memory, n_steps=train_config.n_steps, s=s)
        _, v_next, _ = agent.act(s)
        assert not torch.isnan(v_next), f"Found nan in v_next {v_next}, at state {s}"
        assert v_next < 1e4, f"v_next to big {v_next}"
        batch = memory.get(v_next, batch_size=train_config.n_steps)
        train_stats, adv = agent.update(batch, opt_steps=train_config.opt_steps)

        if global_step % train_config.print_every == 0:
            if "opt/k/mu" in train_stats.keys() and not WANDB:
                train_stats["opt/lambda/max"] = torch.svd(agent.opt_critic.p)[1].max()
                fig = agent.get_batch_stats(residual=adv, stats=train_stats)
                fig.savefig(f"{writer.log_dir}/plot/stats_{global_step}.png")
                plt.close()

        record_stats(writer, train_stats, step=global_step * train_config.n_steps)

        if env.env_id == "microgrid":
            logger.update(**info["stats"], write=True)
            record_stats(writer, info["stats"], step=global_step * train_config.n_steps)
        if "episode" in info.keys():
            record_stats(
                writer, info["episode"], step=global_step * train_config.n_steps
            )
            logger.update(**info["episode"], write=True)
        if stop_fn(global_step * train_config.n_steps):
            break

        if global_step % train_config.print_every == 0:
            print(
                f"n_steps:{global_step}\tprogress:{global_step * train_config.n_steps / train_config.max_samples:.2}\t{logger}"
            )

        global_step += 1

        if global_step % train_config.eval_every == 0 and env.env_id != "microgrid":
            info = eval_agent(agent, eval_env)
            record_stats(
                writer,
                info["episode"],
                step=global_step * train_config.n_steps,
                prefix="test",
            )
            torch.save(
                (agent.actor, agent.critic), os.path.join(writer.log_dir, "latest.pt")
            )

        # import pickle
        # with open("cov.pkl", "wb")  as f:
        #    pickle.dump((covariances, grads), f)
    writer.close()
    env.close()
    eval_env.close()


if __name__ == "__main__":
    from slurm_tools.parse_args import parse_args

    config = parse_args()

    main(config)
