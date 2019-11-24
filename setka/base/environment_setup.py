import numpy
import random
import torch
import os

def environment_setup(seed=0,
                      max_threads=4,
                      deterministic_cuda=False,
                      benchmark_cuda=True):

    '''
    Fixes seeds for all possible random value generators and limits the amount of OMP threads. You may also select
    to use deterministic cuda (significantly slower) and turn off benchmark cuda.

    :param seed: Seed to use in ```numpy.random```, ```random``` and ```torch.manual_seed```.
    :param max_threads: maximum amount of OMP threads.
    :param deterministic_cuda: use deterministic CUDA backend if true
    :param benchmark_cuda: use benchmark CUDA backend if true
    '''

    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    # torch.seed()

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