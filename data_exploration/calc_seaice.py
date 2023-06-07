import netCDF4 as nc
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# gsw oceanic toolbox: http://www.teos-10.org/pubs/Getting_Started.pdf
import gsw
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import warnings
import datetime as datetime
warnings.filterwarnings("ignore", category=RuntimeWarning)


def interpolate_sic_values(Hadi_SI, time_values, latitude_values,
                           longitude_values):
    """
    Interpolate sea ice concentration values,
    considering land values do not exist.

    Parameters:
    - Hadi_SI (xarray.Dataset): Dataset containing sea ice concentration
    values.
    - time_values (numpy.ndarray): Array of time values.
    - latitude_values (numpy.ndarray): Array of latitude values.
    - longitude_values (numpy.ndarray): Array of longitude values.

    Returns:
    - numpy.ndarray: Interpolated sea ice concentration values.
    """

    # Retrieve sea ice concentration values from Hadi_SI dataset
    sic_values = Hadi_SI.sel(time=time_values, latitude=latitude_values,
                             longitude=longitude_values,
                             method="nearest")["sic"]
    sic_values = sic_values.values.reshape((-1,) + sic_values.shape[1:])

    # Create a mask of NaN values
    nan_mask = np.isnan(sic_values)

    # Compute the indices of the nearest non-NaN values in the nearest
    # longitude
    left_indices = np.maximum(np.arange(sic_values.shape[2]) - 1, 0)
    right_indices = np.minimum(np.arange(sic_values.shape[2]) + 1,
                               sic_values.shape[2] - 1)

    # Find the nearest non-NaN values by indexing with the mask and indices
    left_values = sic_values[:, :, left_indices]
    right_values = sic_values[:, :, right_indices]
    valid_values = np.where(nan_mask, np.where(np.isnan(left_values),
                                               right_values, left_values),
                            sic_values)

    # Update the sic_values with the interpolated values
    sic_values[nan_mask] = valid_values[nan_mask]

    return sic_values


def calc_SIC(dataset, Hadi_SI):
    """
    Calculate sea ice concentration values.

    Parameters:
    - dataset (xarray.Dataset): Dataset containing the profile data.
    - Hadi_SI (xarray.Dataset): Dataset containing the sea ice data.

    Returns:
    - xarray.Dataset: Dataset with sea ice concentration values.

    """
    # Define the base date
    # base_date = datetime(1870, 1, 1)

    # Round the longitude and latitude values to the nearest half degree
    rounded_longitude = np.ceil(dataset['longitude'] * 2) / 2
    rounded_latitude = np.ceil(dataset['latitude'] * 2) / 2

    # Calculate the number of months between the base date and all target dates
    # months_after_base = (dataset["time"].dt.year - base_date.year) * 12 +
    # (dataset["time"].dt.month - base_date.month)

    dataset["nearest_lon"] = rounded_longitude
    dataset["nearest_lat"] = rounded_latitude

    # Convert the time, latitude, and longitude variables to arrays
    time_values = dataset["time"].values
    latitude_values = dataset["nearest_lat"].values
    longitude_values = dataset["nearest_lon"].values

    # Check the shape of latitude_values and longitude_values
    if latitude_values.ndim == 2 and longitude_values.ndim == 2:
        latitude_values = latitude_values[0, :]
        longitude_values = longitude_values[0, :]
    elif latitude_values.ndim == 1 and longitude_values.ndim == 1:
        pass  # No need to modify the arrays
    else:
        raise ValueError("Invalid shape for latitude_values and",
                         "longitude_values. "
                         "Expected 1D or 2D arrays.")

    # Use the arrays of indices to retrieve the sic values from the dataset
    sic_values = Hadi_SI.sel(time=time_values, latitude=latitude_values,
                             longitude=longitude_values,
                             method="nearest")["sic"]

    # Check if sic_values contains any NaN values
    nan_percentage = np.isnan(sic_values).mean() * 100

    if nan_percentage > 10:
        # Call the function for interpolated sic values
        dataset = calc_seaice(dataset, Hadi_SI, time_values, latitude_values,
                              longitude_values)
        return dataset

    else:
        # Create an empty array to store the sic scalar values
        sic_scalar = np.empty(dataset["profile"].shape)

        # Loop over each profile in the dataset
        for i, profile in enumerate(dataset["profile"]):
            # Get the corresponding indices of time, latitude, and longitude
            # print(dataset["time"].values.shape)

            try:
                time_index = np.where(dataset["time"].values ==
                                      time_values[i])[0][0]
                lat_index = np.where(dataset["nearest_lat"].values ==
                                     latitude_values[i])[0][0]
                lon_index = np.where(dataset["nearest_lon"].values ==
                                     longitude_values[i])[0][0]

                # Extract the sic scalar value for the profile
                sic_scalar[i] = sic_values[time_index, lat_index, lon_index]

            except IndexError:
                # Assign NaN to sic_scalar if any index results in an error
                sic_scalar[i] = np.nan

        # Assign the sic scalar values to a new variable in the dataset
        dataset["sea_ice_concentration"] = (("profile",), sic_scalar)
    return dataset


def calc_seaice(dataset, Hadi_SI, time_values, latitude_values,
                longitude_values):
    """
    Calculate sea ice concentration values with interpolated sic values.

    Parameters:
    - dataset (xarray.Dataset): Dataset containing the profile data.
    - Hadi_SI (xarray.Dataset): Dataset containing the sea ice
    concentration data.

    Returns:
    - xarray.Dataset: Dataset with sea ice concentration values.

    """

    # Interpolate sic_values using longitude values
    sic_values_interpolated = interpolate_sic_values(Hadi_SI, time_values,
                                                     latitude_values,
                                                     longitude_values)

    sic_values_interpolated = xr.DataArray(sic_values_interpolated,
                                           coords=[time_values,
                                                   latitude_values,
                                                   longitude_values],
                                           dims=["time", "latitude",
                                                 "longitude"])

    sic_scalar = np.empty(dataset["profile"].shape)

    # Loop over each profile in dataset
    for i, profile in enumerate(dataset["profile"]):
        # Get the corresponding indices of time, latitude, and longitude

        try:
            time_index = np.where(dataset["time"].values ==
                                  time_values[i])[0][0]
            lat_index = np.where(dataset["nearest_lat"].values ==
                                 latitude_values[i])[0][0]
            lon_index = np.where(dataset["nearest_lon"].values ==
                                 longitude_values[i])[0][0]

            # Extract the sic scalar value for the profile
            sic_scalar[i] = sic_values_interpolated[time_index,
                                                    lat_index,
                                                    lon_index]

        except IndexError:
            # Assign NaN to sic_scalar if any index results in an error
            sic_scalar[i] = np.nan

    # Assign the sic scalar values to a new variable in the dataset
    dataset["sea_ice_concentration"] = (("profile",), sic_scalar)
    return dataset
