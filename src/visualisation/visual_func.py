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
    """
    Creates a dataframe for evaluating the performance of a regression model
    on test data.

    Parameters:
        X_test (pd.DataFrame): The feature matrix of the test data.
        y_test (pd.Series or np.ndarray): The true target values of the
            test data.
        y_pred (np.ndarray): The predicted target values of the test data.

    Returns:
        pd.DataFrame: A dataframe containing the test data features,
            true target values, and predicted target values.
    """
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
    """
    Plots the absolute residuals between the true and predicted epsilon value
    as a scatter plot.

    Parameters:
        test_df (pd.DataFrame): A dataframe containing the test data features,
        true epsilon values, and predicted epsilon values.

    Returns:
        None

    Example:
        # Assuming test_df is a dataframe with columns 'latitude', 'depth',
        # 'log_eps' (true epsilon values), and 'eps_pred'
        # (predicted epsilon values)
        plot_residuals(test_df)
    """
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
    """
    Plots the mean and standard deviation of absolute residuals by latitude.

    Parameters:
        test_df (pd.DataFrame): A dataframe containing the test data features,
        true epsilon values, and predicted epsilon values.

    Returns:
        None

    Example:
        # Assuming test_df is a dataframe with columns 'latitude', 'depth',
        # 'log_eps' (true epsilon values), and 'eps_pred' (predicted epsilon
        # values)
        plot_std_lat(test_df)
    """
    # Calculate the absolute residuals between 'kappa' and 'kappa_pred'
    abs_residual = np.abs(test_df['log_eps'] - test_df['eps_pred'])

    # Define the latitude and absolute residual values
    latitude = test_df['latitude'].values
    abs_residual_values = abs_residual.values

    # Define the binning for latitude
    bin_width = 1.0  # Adjust the bin width as desired
    lat_bins = np.arange(latitude.min(), latitude.max() + bin_width, bin_width)

    # Compute the mean and standard deviation of the absolute residuals
    # for each latitude bin
    mean_resid, bin_edges, _ = stats.binned_statistic(latitude,
                                                      abs_residual_values,
                                                      statistic='mean',
                                                      bins=lat_bins)
    std_resid, _, _ = stats.binned_statistic(latitude, abs_residual_values,
                                             statistic='std', bins=lat_bins)

    # Compute the bin centers for plotting
    lat_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # Create a line plot of the mean and standard deviation
    plt.figure(figsize=(10, 6))
    plt.plot(lat_centers, mean_resid, marker='o', label='Mean', color='blue')
    plt.plot(lat_centers, std_resid, marker='o', label='Standard Deviation',
             color='orange')

    # Add labels and title
    plt.xlabel('Latitude')
    plt.ylabel('Absolute Residual')
    plt.title('Mean and Standard Deviation of Absolute Residual by Latitude')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def plot_historgram(dataframe):
    """
    Plots histograms of latitude per "cruise" variable.

    Parameters:
        dataframe (pd.DataFrame): A dataframe containing latitude and cruise
        information.

    Returns:
        None

    Example:
        # Assuming dataframe is dataframe with columns 'latitude' and 'cruise'
        plot_histogram(dataframe)
    """
    # Plot histograms of latitude per "cruise" variable
    sns.histplot(data=dataframe, x="latitude", hue="cruise", bins=30)
    plt.xlabel("Latitude")
    plt.ylabel("Count")
    plt.title("Histograms of Latitude per Cruise")
    plt.legend(title="Cruise", labels=dataframe["cruise"].unique(),
               loc="upper left")
    plt.show()


def stereo_plot(merged_df, variable, name_var, vmin=False, vmax=False):
    """
    Creates a polar stereographic plot of longitude and latitude with colored
    residuals.

    Parameters:
        merged_df (pd.DataFrame): A dataframe containing longitude and
            latitude information.
        variable (np.ndarray): An array of values representing the variable
            of interest.
        name_var (str): The name of the variable for the colorbar label.
        vmin (int, optional): The minimum value for the colorbar.
            Default: False.
        vmax (int, optional): The maximum value for the colorbar.
            Default: False.

    Returns:
        None

    Example:
        # Assuming merged_df is a dataframe with columns 'longitude' and
        # 'latitude' variable is an array of values, and name_var is a string
        stereo_plot(merged_df, variable, name_var)
    """

    # Create a polar stereographic projection centered on the Arctic pole
    projection = ccrs.NorthPolarStereo(central_longitude=0)

    # Create a figure and axes using the polar stereographic projection
    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw={'projection': projection})

    # Plot longitude and latitude with colored residuals
    if isinstance(vmin, int):
        sc = ax.scatter(merged_df['longitude'], merged_df['latitude'],
                        c=np.abs(variable), vmin=vmin, vmax=vmax,
                        cmap='viridis', transform=ccrs.PlateCarree())
    else:
        sc = ax.scatter(merged_df['longitude'], merged_df['latitude'],
                        c=np.abs(variable), cmap='viridis',
                        transform=ccrs.PlateCarree())
    # Add colorbar
    cbar = plt.colorbar(sc, ax=ax, label=name_var)

    # Set map extent to focus on the Arctic region
    ax.set_extent([-180, 180, 60, 90], ccrs.PlateCarree())

    # Add map features
    ax.coastlines()
    ax.gridlines()

    # Set plot title and labels
    plt.title(f"{name_var} based on Longitude and Latitude")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')

    # Show the plot
    plt.show()
