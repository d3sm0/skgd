import glob
import os

import yaml
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(logging.StreamHandler())


class Config:
    def __init__(self, data):
        for k, v in data.items():
            setattr(self, k, self._wrap(v))

    @staticmethod
    def _wrap(v):
        if isinstance(v, dict):
            return Config(v)
        return v

    def get_dict(self):
        out = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Config):
                v = v.get_dict()
            out[k] = v
        return out

    def __repr__(self):
        return '{%s}' % (", ".join(f"{k}: {v}" for k, v in self.__dict__.items()))


def _to_number(x):
    try:
        x = int(x)
    except ValueError:
        x = float(x)
    return x


def args_from_list(args):
    # following sweep args --key=value
    _args = {}

    for arg in args:
        assert "=" in arg, "Following convention --key=value for unspecified args."
        k, v = arg.split('=')
        k = k.split('--')[1]
        assert "-" not in k, f"Bad format k {k}"
        try:
            v = _to_number(v)
        except ValueError as e:
            logger.error(f"Check argument type {k}:{v}. Can not cast in float.")
        _args[k] = v
    return _args


def parse(parser, verbose=True):
    _config = {}
    args, uknown_args = parser.parse_known_args()
    args = vars(args)
    uknown_args = args_from_list(uknown_args)
    args.update(uknown_args)
    # TODO this path should be absolute
    config_fname = os.path.join(args['config_dir'], f"{args['bench_id']}:{args['agent_id']}.yaml")
    try:
        config_file = glob.glob(config_fname)[0]
        default = _load_yaml(config_file)
        _config.update(default)
    except IndexError:
        logger.error(f"Config file not found for {config_fname}")
    update_config(_config, args)
    #if verbose:
    #    logger.info("Config summary: \n")
    #    logger.info("\n".join(
    #        [f"{k}:\t{v}" for k, v in _config.items()]
    #    ))
    #    logger.info("\n" + "=" * 10 + "\n")
    return Config(_config)


def update_config(config, data):
    if len(config.keys()) == 0:
        config.update(data)
    else:
        data_keys = set(data.keys())
        _update(config, data, data_keys)
        if len(data_keys):
            logger.info(f"Found {data_keys} not in config file.")
            config.update({k: data[k] for k in data_keys})
    return config


def _update(config, data, data_keys):
    for k, v in config.items():
        if isinstance(v, dict):
            _update(v, data, data_keys)
        else:
            try:
                if data[k] is not None:  # use None default in argparse
                    config[k] = data[k]
                data_keys.remove(k)
            except KeyError:
                pass


def _load_yaml(fname):
    with open(fname, "r") as f:
        item = yaml.safe_load(f)
    return item


def _test_handler():
    from slurm_tools.parse_args import parse_args
    config = parse_args()

    with open("test.yaml", "w") as f:
        yaml.safe_dump(config.get_dict(), f)

    import wandb
    wandb.init(project="debug", config=config.get_dict())

    print("done")


