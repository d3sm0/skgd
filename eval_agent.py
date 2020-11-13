import gym_vdp
import torch
from gym.wrappers import Monitor

from env_utils.env_utils import RecordEpisodeStatistics
from misc.torch_utils import RunningMeanStd
from misc.torch_utils import TorchEnv
from pg import (PG)


def _eval_agent(env, path):
    actor, critic = torch.load(path)
    agent = PG(actor, critic)
    agent.train(mode=False)
    avg = RunningMeanStd()
    avg_v = RunningMeanStd()
    for _ in range(1):
        env.seed(0)
        done = False
        s = env.reset()
        v = agent.critic(s)
        avg_v.update(v.detach().numpy())
        while not done:
            pi = agent.actor(s)
            a = pi.sample()
            s, r, done, info = env.step(a)
            if done:
                avg.update((info['episode']['env/r'],))
                break
    print(path, avg, avg_v)
    fig = env.render(mode="static")
    fig.savefig(path.split('/')[3] + '_ns.png')
    env.close()


def main():
    paths = [
        "vdp_logs/logs/vdp/ppo/07-23-45-51-499009/tb/latest.pt",
        "vdp_logs/logs/vdp/ktd/07-23-46-06-088769/tb/latest.pt",
        "vdp_logs_new/logs/vdp/ktd_low_rank/07-23-44-56-865730/tb/latest.pt"
             ]

    for path in paths:
        env = gym_vdp.make_vdp("vdp", randomize=True, constrained=False, std=0.05)
        #env = Monitor(env, force=True, directory=f"videos/{path.split('/')[3]}")
        env = RecordEpisodeStatistics(env)
        env = TorchEnv(env)
        _eval_agent(env, path)


if __name__ == '__main__':
    main()
