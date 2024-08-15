
import numpy as np
import misc


def _hparams(algorithm, dataset, random_seed):
    """
    Global registry of hyperparams. Each entry is a (default, random) tuple.
    New algorithms / networks / etc. should add entries here.
    """
    hparams = {}

    def _hparam(name, default_val, random_val_fn):
        """Define a hyperparameter. random_val_fn takes a RandomState and
        returns a random hyperparameter value."""
        assert(name not in hparams)
        random_state = np.random.RandomState(
            misc.seed_hash(random_seed, name)
        )
        hparams[name] = (default_val, random_val_fn(random_state))


    _hparam('dm_idx', True, lambda r: True)
    _hparam("tau_temp", 4, lambda r: r.choice([0.1, 0.01, 0.001]))
    _hparam("clip", 0.05, lambda r: r.choice([1, 0.5, 0.1, 0.01]))
    _hparam('train_step', 5, lambda r: r.choice([1, 3, 5]))
    _hparam('test_step', 5, lambda r: r.choice([1, 3, 5]))
    _hparam("alpha", 0.1, lambda r: r.choice([0.1, 0.01, 0.001]))
    _hparam("lambda", 0.5, lambda r: r.choice([0.1, 0.01, 0.001]))
    _hparam('beta', 0.001, lambda r: r.choice([0.01, 0.05, 0.001]))
    _hparam('attn_width', 128, lambda r: 32)
    _hparam('attn_depth', 3, lambda r: 3)
    _hparam('mlp_width', 128, lambda r: int(2 ** r.uniform(6, 10)))
    _hparam('mlp_depth', 3, lambda r: int(r.choice([3, 4, 5])))  # 5
    _hparam('mlp_dropout', 0.2, lambda r: 0)
    _hparam("env_sample_number", 800, lambda r: int(r.choice([300, 400, 500, 800])))
    _hparam("env_distance", 15, lambda r: int(r.choice([10, 15, 30])))
    _hparam('total_sample_number', 0, lambda r: 0)

    return hparams

def default_hparams(algorithm, dataset):  # this function to choose the default one
    return {a: b for a, (b, c) in _hparams(algorithm, dataset, 0).items()}
