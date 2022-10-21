from functools import wraps
import time
import pickle
import os
from typing import Callable


def time_func(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end-start:.3f} seconds")
        return result

    return wrapped


cache_count = 0


def cache_result(func: Callable) -> Callable:
    @wraps(func)
    def wrapped(*args, **kwargs):
        if hasattr(wrapped, "has_run"):
            wrapped.has_run += 1
            # raise Exception(
            #    f"{func.__name__} cache function was called twice, this should not be possible")
        else:
            wrapped.has_run = 1
        global cache_count
        pickle_file = f"./pickle_cache/{cache_count}_{func.__name__}_{wrapped.has_run}.pickle"
        cache_count += 1

        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        if os.path.isfile(pickle_file):
            # check if cached
            old_state = pickle.load(open(pickle_file, "rb"))
            return old_state
        # recalculate and cache
        result = func(*args, **kwargs)
        file = open(pickle_file, "wb")
        pickle.dump(result, file)
        print(f"{func.__name__} cached as {pickle_file}")
        return result

    return wrapped
