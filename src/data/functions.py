import math
import datetime
import xarray as xr
import numpy as np
import pandas as pd


def safe_log(x):
    """
    Calculates the logarithm of a positive number.

    Args:
        x (float): The input value.

    Returns:
        float: The logarithm of the input value, or NaN if the input value is
        not positive.
    """
    # Only take logarithms of positive numbers
    if x > 0:
        return math.log(x)
    # Return NaN if it cannot take the logarithm
    else:
        return float('nan')


def convert_datetime(dataset):
    """
    Converts the 'time' variable in a dataset to datetime format.

    Args:
        dataset (xarray.Dataset): The input dataset containing a variable
        named 'time'.

    Returns:
        xarray.Dataset: The modified dataset with the 'time' variable
        converted to datetime format.
    """
    if type(dataset.time.values[0]) is not np.datetime64:
        # Use pandas to_datetime function if time is not in the correct format
        # Vectorise data series to perform operation to entire column
        dataset["time"] = xr.apply_ufunc(pd.to_datetime, dataset["time"],
                                         vectorize=True)
    return dataset
