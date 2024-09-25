"""
Created on Wed Sep 11 13:25:03 2024

@author: Jonas Petersen
"""


from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

def suppress_output(func, *args, **kwargs):
    f = StringIO()
    with redirect_stdout(f), redirect_stderr(f):
        result = func(*args, **kwargs)
    return result

