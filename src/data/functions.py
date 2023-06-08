import math
import datetime
import xarray as xr
import numpy as np
import pandas as pd


def safe_log(x):
    if x > 0:
        return math.log(x)
    else:
        return float('nan')
    
    
def convert_datetime(dataset):
    if type(dataset.time.values[0]) is not np.datetime64:
        dataset["time"] = xr.apply_ufunc(pd.to_datetime, dataset["time"], vectorize=True)
    return dataset
