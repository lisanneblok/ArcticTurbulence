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
from scipy import stats
import seaborn as sns
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})


def create_testdf(X_test, y_test, y_pred):
    X_test['index'] = X_test.index
    y_test['index'] = y_test.index
    # Convert y_pred2 to a dataframe
    y_pred_df = pd.DataFrame(y_pred, columns=['eps_pred'])
    y_pred_df["index"] = y_pred_df.index
    # Merge X_test2_reset with arctic_df based on the index column
    test_df = X_test.merge(y_test, on='index')
    test_df = test_df.merge(y_pred_df, on="index")
    return test_df


def plot_residuals(test_df):
    abs_residual = np.abs(test_df['log_eps'] - test_df['eps_pred'])

    # Define the x and y coordinates for the scatter plot
    latitude = test_df['latitude'].values
    depth = test_df['depth'].values

    # Create a scatter plot of the absolute residuals
    plt.figure(figsize=(10, 6))
    plt.scatter(latitude, depth, c=abs_residual, cmap='plasma', s=50)
    plt.colorbar(label='Absolute Residual')
    plt.xlabel('Latitude')
    plt.ylabel('Depth')
    plt.title('Absolute Residual between Epsilon and Epsilon_pred')

    # Invert the y-axis
    plt.gca().invert_yaxis()

    # Show the plot
    plt.show()


def plot_std_lat(test_df):
    # Calculate the absolute residuals between 'kappa' and 'kappa_pred'
    abs_residual = np.abs(test_df['log_eps'] - test_df['eps_pred'])

    # Define the latitude and absolute residual values
    latitude = test_df['latitude'].values
    abs_residual_values = abs_residual.values

    # Define the binning for latitude
    bin_width = 1.0  # Adjust the bin width as desired
    lat_bins = np.arange(latitude.min(), latitude.max() + bin_width, bin_width)

    # Compute the mean and standard deviation of the absolute residuals for each latitude bin
    mean_resid, bin_edges, _ = stats.binned_statistic(latitude, abs_residual_values, statistic='mean', bins=lat_bins)
    std_resid, _, _ = stats.binned_statistic(latitude, abs_residual_values, statistic='std', bins=lat_bins)

    # Compute the bin centers for plotting
    lat_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Create a line plot of the mean and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(lat_centers, mean_resid, marker='o', label='Mean', color='blue')
    plt.plot(lat_centers, std_resid, marker='o', label='Standard Deviation', color='orange')

    # Add labels and title
    plt.xlabel('Latitude')
    plt.ylabel('Absolute Residual')
    plt.title('Mean and Standard Deviation of Absolute Residual by Latitude')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_historgram(dataframe):
    # Plot histograms of latitude per "cruise" variable
    sns.histplot(data=dataframe, x="latitude", hue="cruise", bins=30)
    plt.xlabel("Latitude")
    plt.ylabel("Count")
    plt.title("Histograms of Latitude per Cruise")
    plt.legend(title="Cruise", labels=dataframe["cruise"].unique(), loc="upper left")
    plt.show()
    
