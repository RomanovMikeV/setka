import numpy
import random
import torch
import os

def environment_setup(seed=0,
                      max_threads=4,
                      deterministic_cuda=False,
                      benchmark_cuda=True):
    numpy.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    os.environ["OMP_NUM_THREADS"] = str(max_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(max_threads)
    os.environ["MKL_NUM_THREADS"] = str(max_threads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(max_threads)
    os.environ["NUMEXPR_NUM_THREADS"] = str(max_threads)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = deterministic_cuda
        torch.backends.cudnn.benchmark = benchmark_cuda