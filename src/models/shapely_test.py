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
