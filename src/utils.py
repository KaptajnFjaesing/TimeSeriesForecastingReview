"""
Created on Wed Sep 11 13:25:03 2024

@author: Jonas Petersen
"""

import numpy as np
import time
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

class CustomBackTransformation:
    def __init__(self, constants0,consants1):
        self.constants0 = constants0
        self.constants1 = consants1

    def __call__(self, forecast, index):
        # Apply the transformation with the constant for the specific forecast index
        forecast.samples = np.exp(np.cumsum(forecast.samples, axis=1) + self.constants0[index] + self.constants1[index])
        return forecast

def suppress_output(func, *args, **kwargs):
    f = StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        result = func(*args, **kwargs)
    return result

def log_execution_time(func, log_file, log_key, *args, **kwargs):
    """
    Logs the execution time of a function to a JSON file.

    Parameters:
        func (callable): The function to execute and time.
        log_file (str): Path to the JSON log file.
        log_key (str): Key under which to store the computation time.
        *args: Positional arguments to pass to the function.
        **kwargs: Keyword arguments to pass to the function.
    """
    tic = time.time()  # Start timing
    result = func(*args, **kwargs)  # Execute the function
    toc = time.time()  # End timing

    # Load existing log data or initialize an empty dictionary
    log_data = {}
    try:
        with open(log_file, "r") as log:
            log_data = json.load(log)
    except FileNotFoundError:
        pass

    # Add the new computation time
    log_data[log_key] = toc - tic

    # Save the updated log data
    with open(log_file, "w") as log:
        json.dump(log_data, log, indent=4)

    return result