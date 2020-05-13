import numpy
import random
import torch
import os


def environment_setup(seed=0, max_threads=4, deterministic_cuda=True, benchmark_cuda=True):
    """
    Fixes seeds for all possible random value generators and limits the amount of OMP threads. You may also select
    to use deterministic cuda (significantly slower) and turn off benchmark cuda.

    Arguments:
        seed (int): Seed to use in ```numpy.random```, ```random``` and ```torch.manual_seed```.
        max_threads (int): maximum amount of OMP threads.
        deterministic_cuda (bool): use deterministic CUDA backend if true
        benchmark_cuda (bool): use benchmark CUDA backend if true
    """

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

    if torch.cuda.is_available():
        # torch.cuda.seed(seed)
        # torch.cuda.seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = deterministic_cuda
        torch.backends.cudnn.benchmark = benchmark_cuda


def collect_random_states():
    return {
        'python': random.getstate(),
        'numpy': numpy.random.get_state(),
        'torch': torch.random.get_rng_state(),
        'torch.cuda': torch.cuda.random.get_rng_state_all()
    }


def set_random_states(states):
    if 'python' in states:
        random.setstate(states['python'])
    if 'numpy' in states:
        numpy.random.set_state(states['numpy'])
    if 'torch' in states:
        torch.random.set_rng_state(states['torch'])
    if 'torch.cuda' in states:
        torch.cuda.random.set_rng_state_all(states['torch.cuda'])
