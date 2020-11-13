#!/bin/python3.7

import itertools
import multiprocessing as mp
import subprocess as sp
import time


def do_work(args):
    env_id, seed, agent = args
    print(f"starting processs seed:{seed}, env_id:{env_id}")
    time.sleep(5)
    # session = sp.Popen(['python','main.py'], stdout=sp.PIPE, stderr=sp.PIPE)
    session = sp.Popen(
        ["sbatch", "./cmd/train.sh", env_id, agent, str(seed)],
        stdout=sp.PIPE,
        stderr=sp.PIPE,
    )
    stdout, stderr = session.communicate()
    print(stdout.decode())
    if stderr:
        raise Exception("Error " + str(stderr))


if __name__ == "__main__":
    seeds = list(range(10))
    n_processes = 1
    agents = ["ktd", "ppo"]
    env_ids = ["bullet:cartpole"]  # ["vdp:vdp",
    args = list(itertools.product(env_ids, seeds, agents))
    with mp.get_context("spawn").Pool(n_processes) as pool:
        pool.map(do_work, args)
