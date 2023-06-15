import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import xarray as xr
import pandas as pd
import shap


def shap_plot(pipeline, X_test, xfeatures):
    # Calculate Shapley values
    explainer = shap.Explainer(pipeline)
    shap_values = explainer(X_test)

    # Plot Shapley summary plot
    shap.summary_plot(shap_values, X_test, feature_names=xfeatures)
