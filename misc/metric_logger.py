import csv
import datetime
import time
from collections import defaultdict, deque

import torch
import tqdm
import os



class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.time = time.time()
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n
        self.time = time.time()

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)



class MetricLogger(object):
    def __init__(self, delimiter="\t", output_dir=None):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "metric.csv")
        self.output_file = open(output_file, 'w') if output_file is not None else None

    def update(self, write=False, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)
        if write:
            self._write()

    def _write(self):
        if self.output_file is None:
            return
        writer = csv.writer(self.output_file)
        if self.output_file.tell() == 0:
            writer.writerow(self.meters.keys())
        writer.writerow((v.value for v in self.meters.values()))
        self.output_file.flush()

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, max_steps, header='\t'):
        i = 0
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        log_msg = self.delimiter.join([header,
                                       'eta:{eta:.2f}',
                                       '{meters}',
                                       # 'time: {time}',
                                       # 'data: {data}'
                                       ])
        MB = 1024.0 * 1024.0
        for obj in tqdm.tqdm(iterable):
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0:
                tqdm.tqdm.write(log_msg.format(eta=i / max_steps, meters=str(self)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print(f'{header} Total time: {total_time_str}')
