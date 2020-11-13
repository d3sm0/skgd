from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import multiprocessing as mp

import matplotlib.pyplot as plt
from pathlib import Path
from misc.utils import smooth

_KEY = "env/r"

# Loading too much data is slow...
_TF_CONFIG = {
    'compressedHistograms': 0,
    'images': 0,
    'scalars': 1000,
    'histograms': 0
}


def get_scalar(path):
    assert "tfevents" in path.name, "check path"
    event_acc = EventAccumulator(path.as_posix(), _TF_CONFIG)
    event_acc.Reload()
    _, x, y = list(zip(*event_acc.Scalars(_KEY)))
    return y

def plot_tensorflow_log(path):
    paths = list(Path(path).rglob("*.0"))

    with mp.Pool(5) as p:
        results = p.map(get_scalar, paths)

    fig, ax = plt.subplots(1,1,figsize=(12,6))
    ax.set(xlabel="steps (k)", ylabel=_KEY, title="training")
    for y, path  in zip(results,paths):
        y = smooth(y, weight=.9)
        agent = path.parts[3]
        env_id = path.parts[4]
        label = f"{agent}/{env_id}"
        ax.plot(y, label=label)

    ax.legend(loc='upper left', frameon=True)
    plt.tight_layout()
    plt.savefig(path.parts[2])


if __name__ == '__main__':
    import sys
    plot_tensorflow_log(sys.argv[1])
