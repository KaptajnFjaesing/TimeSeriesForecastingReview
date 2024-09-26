"""
Created on Wed Sep 11 13:25:03 2024

@author: Jonas Petersen
"""
import pandas as pd
import datetime
import numpy as np


class CustomBackTransformation:
    def __init__(self, constants0,consants1):
        self.constants0 = constants0
        self.constants1 = consants1

    def __call__(self, forecast, index):
        # Apply the transformation with the constant for the specific forecast index
        forecast.samples = np.exp(np.cumsum(forecast.samples, axis=1) + self.constants0[index] + self.constants1[index])
        return forecast
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

def suppress_output(func, *args, **kwargs):
    f = StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        result = func(*args, **kwargs)
    return result

