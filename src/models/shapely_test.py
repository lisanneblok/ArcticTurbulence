import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import xarray as xr
import pandas as pd
import shap


def shap_plot(pipeline, X_test, xfeatures):
    """
    Generates a Shapley summary plot based on the provided pipeline and test
        data.

    Args:
        pipeline: The trained pipeline model.
        X_test (pandas.DataFrame): The test data containing the input features
        xfeatures (list): A list of column names representing the input
            features.
    """
    # Calculate Shapley values
    explainer = shap.Explainer(pipeline)
    shap_values = explainer(X_test)

    # Plot Shapley summary plot
    shap.summary_plot(shap_values, X_test, feature_names=xfeatures)


def plot_shap_waterfall(X_test, shap_values, instance_idx):
    """
    Generate a waterfall plot to visualize SHAP values for a specific instance.

    Parameters:
        X_test (pd.DataFrame): Input features.
        shap_values (np.ndarray): SHAP values for the instance.
        instance_idx (int): Index of the instance in the test/validation set.

    Returns:
        None

    """
    # Select the instance for the waterfall plot
    instance = X_test.iloc[instance_idx]
    print(pd.DataFrame(instance))

    # Sort features based on absolute SHAP values for the selected instance
    sorted_features = np.argsort(np.abs(shap_values[instance_idx]))

    # Calculate cumulative sum of sorted SHAP values
    cumulative_shap_values = np.cumsum(
        shap_values[instance_idx][sorted_features])

    # Define colors for positive and negative SHAP values
    positive_color = 'salmon'
    negative_color = 'steelblue'

    # Plot the waterfall plot with custom styling
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(
        X_test.columns[sorted_features], cumulative_shap_values,
        color=[positive_color if value >= 0 else negative_color
               for value in shap_values[instance_idx][sorted_features]])

    ax.set_xlabel("Cumulative SHAP Value", fontsize=12)
    ax.set_ylabel("Features", fontsize=12)
    ax.set_title("Shapley Waterfall Plot", fontsize=14)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.tick_params(axis='both', which='both', length=0)

    # Add the actual Shapley values as text next to each bar
    for i, bar in enumerate(bars):
        value = shap_values[instance_idx][sorted_features][i]
        value_str = "{:.4f}".format(value)
        ax.text(
            bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
            value_str, ha='left' if value >= 0 else 'right', va='center',
            fontsize=10, color='white' if np.abs(value) > 0.5 else 'black')

    plt.tight_layout()
    plt.show()
