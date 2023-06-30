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
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error
import seaborn as sns
import numpy as np
from sklearn.utils import resample
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})


def calc_correlation(merged_df):
    """
    Calculate correlation coefficients between the variables and
    absolute residuals.

    Args:
        merged_df (DataFrame): Merged dataframe containing the relevant
        variables and absolute residuals.

    Returns:
        correlation_matrix (DataFrame): Correlation matrix showing the
        correlation coefficients.

    """
    merged_df["absolute_residuals"] = np.abs(
        merged_df["log_eps"] - merged_df["eps_pred"])

    # Select the relevant columns from merged_df
    variables = ['depth', 'latitude', 'longitude',
                 'Tu_x', 'absolute_residuals']
    data = merged_df[variables]

    # Calculate correlation coefficients
    correlation_matrix = data.corr()

    # Extract the correlations with absolute residuals
    correlation_with_residuals = correlation_matrix[
        'absolute_residuals'][:-1]

    print("Correlation with Absolute Residuals:")
    print(correlation_with_residuals)

    return correlation_matrix


def plot_correlations(merged_df):
    """
    Plot scatter plots of Absolute Residuals against different variables.

    Args:
        merged_df (DataFrame): Merged dataframe containing the Absolute
        Residuals and other variables.

    Returns:
        None

    """
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


def score_metrics(y_test, y_pred):
    """
    Calculate and print various evaluation metrics for regression model
    predictions.

    Args:
        y_test (array-like): True values of the target variable.
        y_pred (array-like): Predicted values of the target variable.

    Returns:
        None

    """
    # Calculate the R2 score
    r2 = r2_score(y_test.values.flatten(), y_pred)

    # Calculate the mean error (ME)
    me = np.mean(y_pred - y_test.values.flatten())

    # Calculate the mean absolute error (MAE)
    mae = mean_absolute_error(y_test.values.flatten(), y_pred)

    # Calculate the residuals
    residuals = y_test.values.flatten() - y_pred

    # Calculate the standard deviation of the residuals
    residual_std = np.std(residuals)

    # Print the metrics
    print("R2 Score: {:.4f}".format(r2))
    print("Mean Error (ME): {:.4f}".format(me))
    print("Mean Absolute Error (MAE): {:.4f}".format(mae))
    print("Residual Standard Deviation: {:.4f}".format(residual_std))


def confidence_metrics(y_test, y_pred, num_bootstraps=1000, alpha=0.05):
    """
    Calculate and print various evaluation metrics for regression model
    predictions, along with their bootstrap confidence intervals, and plot
    the bootstrap distributions.

    Args:
        y_test (array-like): True values of the target variable.
        y_pred (array-like): Predicted values of the target variable.
        num_bootstraps (int, optional): Number of bootstrap resamples.
            Default is 1000.
        alpha (float, optional): Significance level for the confidence
            intervals. Default is 0.05.

    Returns:
        None

    """

    # Calculate the original metrics
    r2 = r2_score(y_test.values.flatten(), y_pred)
    me = np.mean(y_pred - y_test.values.flatten())
    mae = mean_absolute_error(y_test.values.flatten(), y_pred)
    residuals = y_test.values.flatten() - y_pred
    residual_std = np.std(residuals)

    # Bootstrap resampling to calculate confidence intervals
    r2_bootstraps = []
    me_bootstraps = []
    mae_bootstraps = []
    residual_std_bootstraps = []

    for _ in range(num_bootstraps):
        # Resample the data
        indices = np.random.choice(len(y_test), size=len(y_test), replace=True)
        y_test_resampled = y_test.values[indices].flatten()
        y_pred_resampled = y_pred[indices]

        # Calculate the metrics on the resampled data
        r2_bootstraps.append(r2_score(y_test_resampled, y_pred_resampled))
        me_bootstraps.append(np.mean(y_pred_resampled - y_test_resampled))
        mae_bootstraps.append(mean_absolute_error(y_test_resampled,
                                                  y_pred_resampled))
        residuals_resampled = y_test_resampled - y_pred_resampled
        residual_std_bootstraps.append(np.std(residuals_resampled))

    # Calculate the confidence intervals
    r2_ci = np.percentile(r2_bootstraps,
                          [100 * alpha / 2, 100 * (1 - alpha / 2)])
    me_ci = np.percentile(me_bootstraps,
                          [100 * alpha / 2, 100 * (1 - alpha / 2)])
    mae_ci = np.percentile(mae_bootstraps,
                           [100 * alpha / 2, 100 * (1 - alpha / 2)])
    residual_std_ci = np.percentile(residual_std_bootstraps,
                                    [100 * alpha / 2, 100 * (1 - alpha / 2)])

    # Print the metrics and confidence intervals
    print("R2 Score: {:.4f}, 95% CI: [{:.4f}, {:.4f}]".format(r2, r2_ci[0],
                                                              r2_ci[1]))
    print("Mean Error (ME): {:.4f}, 95% CI: [{:.4f}, {:.4f}]".format(
        me, me_ci[0], me_ci[1]))
    print("Mean Absolute Error (MAE): {:.4f}, 95% CI: [{:.4f}, {:.4f}]".format(
        mae, mae_ci[0], mae_ci[1]))
    print(
        "Residual Standard Deviation: {:.4f}, 95% CI: [{:.4f}, {:.4f}]".format(
            residual_std, residual_std_ci[0], residual_std_ci[1]))

    # Plot the bootstrap distributions
    plt.figure(figsize=(12, 6))
    plt.subplot(2, 2, 1)
    sns.histplot(r2_bootstraps, kde=True)
    plt.axvline(x=r2, color='r', linestyle='--')
    plt.title("R2 Score")
    plt.subplot(2, 2, 2)
    sns.histplot(me_bootstraps, kde=True)
    plt.axvline(x=me, color='r', linestyle='--')
    plt.title("Mean Error (ME)")
    plt.subplot(2, 2, 3)
    sns.histplot(mae_bootstraps, kde=True)
    plt.axvline(x=mae, color='r', linestyle='--')
    plt.title("Mean Absolute Error (MAE)")
    plt.subplot(2, 2, 4)
    sns.histplot(residual_std_bootstraps, kde=True)
    plt.axvline(x=residual_std, color='r', linestyle='--')
    plt.title("Residual Standard Deviation")
    plt.tight_layout()
    plt.show()


def correlation_matrix(arctic_df, xstringlist):
    """
    Generates a correlation matrix heatmap for the specified columns
    in the Arctic DataFrame.

    Args:
        arctic_df (pandas.DataFrame): The input DataFrame containing
            the data.
        xstringlist (list): A list of column names to include in the
            correlation matrix.
    """
    # Calculate the correlation matrix
    corr = arctic_df[xstringlist].corr()

    # Set the color palette
    cmap = sns.diverging_palette(220, 10, as_cmap=True)

    # Create a figure and axes
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create the heatmap with customized styling
    sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool),
                cmap=cmap, square=True, ax=ax,
                annot=True, annot_kws={"fontsize": 12},
                fmt=".2f", cbar=True, cbar_kws={"shrink": 0.8})

    # Set the title
    plt.title('Correlation Matrix', fontsize=16)

    # Rotate x-axis labels if needed
    plt.xticks(rotation=45, ha='right')

    # Adjust the layout for better readability
    plt.tight_layout()

    # Show the plot
    plt.show()
