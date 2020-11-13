import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from misc.tb_to_plot import smooth


def get_rw(files):
    rw = []
    max_shape = 0
    for f in files:
        r = pd.read_csv(f).loc[:, ["env/r"]].values.flatten()
        max_shape = max(max_shape, len(r))
        #if len(r) < 3906: continue
        print(len(r))
        rw.append(r)
    rw = np.stack([np.pad(r, (0, max_shape - len(r)), constant_values=max(r)) for r in rw])
    #rw = np.stack(rw)
    return rw

env_id = "bullet"

files = glob.glob(f"../logs/{env_id}/ktd/*/metric/metric.csv")
ktd = get_rw(files)
files = glob.glob(f"../logs/{env_id}/ppo/*/metric/metric.csv")
ppo = get_rw(files)
fig, ax = plt.subplots(1, 1)

ax.plot(smooth(ktd.mean(axis=0), weight=.8), label="ktd")
ax.plot(smooth(ppo.mean(axis=0), weight=.8), label="ppo")
plt.legend()
plt.savefig(env_id)
# rw.std()
