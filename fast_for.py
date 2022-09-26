from multiprocessing import Pool, cpu_count
from typing import Callable


def fast_for(func: Callable, list: list):
    # unfortunately is still slower than expected
    with Pool(cpu_count()) as p:
        new_list = p.map(func, list)
    return new_list
