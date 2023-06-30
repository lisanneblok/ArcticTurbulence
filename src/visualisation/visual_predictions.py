# Load up packages
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import xarray as xr
import pandas as pd
import warnings
from tqdm import tqdm
import os
import sys
warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})

sys.path.append('../..')


def plot_predictions(cruise, parent_dir):
    """
    Plots the mean predicted epsilon values along with the ground truth
    epsilon values for different models (Random Forest, XGBoost, and XGBoost
    7 features) as a function of depth.

    Parameters:
        cruise (str): The name of the cruise.
        parent_dir (str): The parent directory where the pickle files are
            located.

    Returns:
        None

    Example:
        plot_predictions("cruise1", "/path/to/parent_dir")
    """
    # Specify the path and filename for the pickle file
    cruise_file = cruise + "_testdf.pkl"
    pickle_path_RF = os.path.join(parent_dir, "models/RandomForest",
                                  cruise_file)
    RF_testdf = pd.read_pickle(pickle_path_RF)

    # Specify the path and filename for the pickle file
    pickle_path_XGB = os.path.join(parent_dir, "models/XGBoost", cruise_file)
    XGB_testdf = pd.read_pickle(pickle_path_XGB)

    # Specify the path and filename for the pickle file
    pickle_path_XGB_7 = os.path.join(parent_dir, "models/XGBoost_7",
                                     cruise_file)
    XGB_7_testdf = pd.read_pickle(pickle_path_XGB_7)

    sorted_df_XGB = XGB_testdf.sort_values('depth')
    # Sample every 10 meters
    sampled_df_XGB = sorted_df_XGB[sorted_df_XGB['depth'] % 8 == 0]
    # Group the data by depth and calculate the mean of XGBoost
    # epsilon predictions
    mean_df_XGB = sampled_df_XGB.groupby(
        'depth')['eps_pred'].mean().reset_index()
    # Calculate the standard deviation for XGBoost
    std_dev_XGB = mean_df_XGB['eps_pred'].std()
    # Lower bound of the confidence interval
    ci_lower_XGB = mean_df_XGB['eps_pred'] - std_dev_XGB
    # Upper bound of the confidence interval
    ci_upper_XGB = mean_df_XGB['eps_pred'] + std_dev_XGB

    sorted_df_RF = RF_testdf.sort_values('depth')
    # Sample every 10 meters
    sampled_df_RF = sorted_df_RF[sorted_df_RF['depth'] % 8 == 0]
    # Group the data by depth and calculate the mean of epsilon predictions
    mean_df_RF = sampled_df_RF.groupby(
        'depth')['eps_pred'].mean().reset_index()
    # Calculate the standard deviation
    std_dev_RF = mean_df_RF['eps_pred'].std()
    # Lower bound of the confidence interval
    ci_lower_RF = mean_df_RF['eps_pred'] - std_dev_RF
    # Upper bound of the confidence interval
    ci_upper_RF = mean_df_RF['eps_pred'] + std_dev_RF

    sorted_df_XGB_7 = XGB_7_testdf.sort_values('depth')
    # Sample every 10 meters
    sampled_df_XGB_7 = sorted_df_XGB_7[sorted_df_XGB_7['depth'] % 8 == 0]
    # Group the data by depth and calculate the mean of XGBoost 7 features
    # epsilon predictions
    mean_df_XGB_7 = sampled_df_XGB_7.groupby(
        'depth')['eps_pred'].mean().reset_index()
    # Calculate the standard deviation for XGBoost 7 features
    std_dev_XGB_7 = mean_df_XGB_7['eps_pred'].std()
    # Lower bound of the confidence interval
    ci_lower_XGB_7 = mean_df_XGB_7['eps_pred'] - std_dev_XGB_7
    # Upper bound of the confidence interval
    ci_upper_XGB_7 = mean_df_XGB_7['eps_pred'] + std_dev_XGB_7

    sorted_df_GT = XGB_testdf.sort_values('depth')
    # Sample every 10 meters
    sampled_df_GT = sorted_df_GT[sorted_df_GT['depth'] % 8 == 0]
    # Group the data by depth and calculate the mean of XGBoost epsilon
    # predictions
    mean_df_GT = sampled_df_GT.groupby('depth')['log_eps'].mean().reset_index()

    plt.plot(mean_df_RF['eps_pred'],
             mean_df_RF['depth'], label='RandomForest Mean Epsilon')
    plt.fill_betweenx(mean_df_RF['depth'], ci_lower_RF, ci_upper_RF,
                      alpha=0.1, label='Random Forest Std')

    plt.plot(mean_df_XGB['eps_pred'], mean_df_XGB['depth'],
             label='XGBoost Mean Epsilon')
    plt.fill_betweenx(mean_df_XGB['depth'], ci_lower_XGB,
                      ci_upper_XGB, alpha=0.1, label='XGBoost Std')

    plt.plot(mean_df_XGB_7['eps_pred'], mean_df_XGB_7['depth'],
             label='XGBoost 7 features Mean Epsilon')
    plt.fill_betweenx(mean_df_XGB_7['depth'], ci_lower_XGB_7, ci_upper_XGB_7,
                      alpha=0.1, label='XGBoost 7 Std')

    plt.plot(mean_df_GT['log_eps'], mean_df_GT['depth'],
             label='Ground Truth Epsilon')

    plt.xlabel('$log_{10}(\epsilon)$')
    plt.ylabel('Depth (m)')
    plt.title(cruise + ' mean Predicted $log_{10} \epsilon$')
    plt.grid(True)
    plt.legend()

    plt.gca().invert_yaxis()  # Invert the y-axis

    plt.show()
