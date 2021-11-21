#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep 22 22:00:54 2021

@author: demust
"""

import functools
import time

time_its = []


def time_it(func):
    """Print the runtime of the decorated function"""

    @functools.wraps(func)
    def wrapper_time_it(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        time_its.append(str(f"Finished {func.__name__!r} in {run_time:.4f} secs"))
        return value

    return wrapper_time_it
