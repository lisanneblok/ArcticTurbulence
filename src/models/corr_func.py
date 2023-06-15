# Load up packages
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import xarray as xr
import pandas as pd

import sys
import pickle
import warnings
from tqdm import tqdm
import cartopy.crs as ccrs
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})


def calc_correlation(merged_df):
    merged_df["absolute_residuals"] = np.abs(merged_df["log_eps"] - merged_df["eps_pred"])
    absolute_residuals = merged_df["absolute_residuals"]
    # Extract the relevant variables from merged_df
    depth = merged_df['depth']
    latitude = merged_df['latitude']
    longitude = merged_df['longitude']
    tu_label = merged_df['Tu_label']

    # Convert the other variables to numpy arrays
    depth = np.array(depth)
    latitude = np.array(latitude)
    longitude = np.array(longitude)
    tu_label = np.array(tu_label)

    # Reshape the variables to have compatible dimensions
    depth = depth.reshape(-1, 1)  # Reshape to have shape (n_samples, 1)
    latitude = latitude.reshape(-1, 1)
    longitude = longitude.reshape(-1, 1)
    tu_label = tu_label.reshape(-1, 1)

    # Concatenate the variables and absolute residuals
    data = np.concatenate((depth, latitude, longitude, tu_label, absolute_residuals), axis=1)

    # Calculate correlation coefficients
    correlation_matrix = np.corrcoef(data, rowvar=False)

    # Extract the correlations with absolute residuals
    correlation_depth = correlation_matrix[-1, 0]
    correlation_latitude = correlation_matrix[-1, 1]
    correlation_longitude = correlation_matrix[-1, 2]
    correlation_tu_label = correlation_matrix[-1, 3]

    print("Correlation with Absolute Residuals:")
    print("Depth:", correlation_depth)
    print("Latitude:", correlation_latitude)
    print("Longitude:", correlation_longitude)
    print("Tu Label:", correlation_tu_label)

    return correlation_matrix


def plot_correlations(merged_df):
    merged_df["absolute_residuals"] = np.abs(
        merged_df["log_eps"] - merged_df["eps_pred"])

    # Scatter plot of Absolute Residuals vs. Depth
    plt.scatter(merged_df['depth'], merged_df['absolute_residuals'])
    plt.xlabel('Depth')
    plt.ylabel('Absolute Residuals')
    plt.title('Absolute Residuals vs. Depth')
    plt.show()

    # Scatter plot of Absolute Residuals vs. Latitude
    plt.scatter(merged_df['latitude'], merged_df['absolute_residuals'])
    plt.xlabel('Latitude')
    plt.ylabel('Absolute Residuals')
    plt.title('Absolute Residuals vs. Latitude')
    plt.show()

    # Scatter plot of Absolute Residuals vs. Longitude
    plt.scatter(merged_df['longitude'], merged_df['absolute_residuals'])
    plt.xlabel('Longitude')
    plt.ylabel('Absolute Residuals')
    plt.title('Absolute Residuals vs. Longitude')
    plt.show()

    # Scatter plot of Absolute Residuals vs. Tu Label
    plt.scatter(merged_df['Tu_label'], merged_df['absolute_residuals'])
    plt.xlabel('Tu Label')
    plt.ylabel('Absolute Residuals')
    plt.title('Absolute Residuals vs. Tu Label')
    plt.show()
