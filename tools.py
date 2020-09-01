"""
@ tools.py: Basic tools
@ Thesis: Geographical Data and Predictions of Windmill Energy Production
@ Weisi Li
@ liwb@itu.dk, liweisi8121@hotmail.com
"""

import pandas as pd
import numpy as np


def normalize_zcenter(df):
    """
    Normalize value to zero centered.
    """
    return (df - df.mean()) / df.std()

def normalize_maxmin(df, max=1, min=0):
    """
    Normalize the data to real values(from new max 0 to new min 1).
    """
    return (df - df.min()) / (df.max() - df.min()) * (max - min) + min

def save_list(lst, path):
    with open(path, 'w+') as f:
        for i in lst:
            f.write("%s\n" % i) 
