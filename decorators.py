from functools import wraps
import time
import pickle
import os


def time_func(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__}: {end-start:.3f} seconds")
        return result
    return wrapped


def cache_result(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        if hasattr(wrapped, "has_run"):
            raise Exception(
                f"{func.__name__} cache function was called twice, this should not be possible")
        else:
            wrapped.has_run = True

        pickle_file = f'./pickle_cache/{func.__name__}.pickle'
        os.makedirs(os.path.dirname(pickle_file), exist_ok=True)
        if os.path.isfile(pickle_file):
            # check if cached
            old_state = pickle.load(open(pickle_file, 'rb'))
            return old_state
        # recalculate and cache
        result = func(*args, **kwargs)
        file = open(pickle_file, 'wb')
        pickle.dump(result, file)
        print(f"{func.__name__} cached as {pickle_file}")
        return result
    return wrapped
