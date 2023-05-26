import matplotlib.pyplot as plt
import numpy as np
import matplotlib
import xarray as xr
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import warnings
from tqdm import tqdm
import cartopy.crs as ccrs
import math
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})


def Tu_label(data_series):
    """
    Apply labels to Turner angle values based on
    https://www.teos-10.org/pubs/gsw/pdf/Turner_Rsubrho.pdf

    Parameters:
        data_series (pd.Series): Series of Turner angle values.

    Returns:
        pd.Series: Series with labels assigned to Turner angle values.
    """
    # Define the conditions and labels
    conditions = [
        data_series.isnull(),
        (data_series >= -90) & (data_series < -45),
        (data_series >= -45) & (data_series < 45),
        (data_series >= 45) & (data_series < 90),
        (data_series >= 90) & (data_series < -90)
    ]
    labels = ['NaN', 'Diffusive Convection', 'Doubly stable',
              'Salt fingering', 'Statically unstable']

    # Apply the conditions and labels to create a new series with the labels
    result = np.select(conditions, labels, default=0)

    # Create a new series with the labels
    labeled_series = pd.Series(result, index=data_series.index)

    return labeled_series


def encode_tulabel(data):
    # Create an instance of the LabelEncoder
    label_encoder = LabelEncoder()
    # Fit the encoder on the Tu_label column
    label_encoder.fit(data['Tu_label'])
    # Transform the Tu_label column into numeric representation
    numeric_labels = label_encoder.transform(data['Tu_label'])
    # Replace the Tu_label column with the numeric labels
    data['Tu_label'] = numeric_labels
    return data


def plot_importances(
    var_col_names,
    importances,
    num_params_to_show: int = None,
    ax=None
):
    """Visualise feature importance"""
    # initialise axes if necessary
    ax = ax or plt.gca()

    data = dict(zip(var_col_names, importances))
    sorted_data = dict(sorted(data.items(), key=lambda x: x[1], reverse=False))
    
    # if specified to show fewer, remove all but greatest n values
    if type(num_params_to_show) == int:
        nth_val = sorted(sorted_data.values(), reverse=True)[num_params_to_show-1]
        sorted_data = {k: v for k, v in sorted_data.items() if v >= nth_val}
        
    ax.barh(list(sorted_data.keys()), list(sorted_data.values()))

    # formatting
    ax.set_ylabel('Input variable')
    ax.set_xlabel('Feature importance')

    if type(num_params_to_show) == int:
        ax.set_title(f'Feature importance for model\nTop {num_params_to_show} most significant features', fontsize=18)
    else:
        ax.set_title('Feature importance for model\nAll features', fontsize=18)

    ax.grid(which='both', linewidth=0.3)
    ax.set_xlim(right=1.15*max(importances))

    for i, v in enumerate(sorted_data.values()):
        ax.text(v+.02*max(importances), i, f'{v:.3f}', ha='left', va='center_baseline')
    return ax


def RF_regressor(dataframe, xfeatures, yfeatures):
    if "Tu_label" in xfeatures:
        hallo = Tu_label(dataframe.Tu)
        dataframe["Tu_label"] = hallo

        dataframe = encode_tulabel(dataframe)

    dataframe['log_eps'] = dataframe['eps'].apply(lambda x: math.log(x))

    # stop depth at 300m
    dataframe = dataframe[dataframe["depth"] <= 300]

    x = dataframe[xfeatures]
    y = dataframe[yfeatures]

    # Split into train and test sets
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=SEED)

    # Define the MinMaxScaler
    scaler = MinMaxScaler()

    # Define the RandomForestRegressor
    rfr = RandomForestRegressor(random_state=SEED)

    # Create a pipeline
    pipeline = Pipeline([('scaler', scaler), ('rfr', rfr)])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    return pipeline, y_test, X_test
    #return r2_score, rfr, y_pred, y_test, X_test, importances, pipeline

def later(X_test):
    # Predict the test set labels
    y_pred = pipeline.predict(X_test)
    r2_score = r2_score(y_test, y_pred)

    # investigate importance of features
    importances = rfr.feature_importances_
    feature_names = xfeatures
    # Sort feature importances in descending order
    sorted_indices = np.argsort(importances)[::-1]

    # Print feature importance rankings with feature names
    for i, index in enumerate(sorted_indices):
        feature_name = feature_names[index]
        print(f"Feature #{i+1}: {feature_name} ({importances[index]})")


def XGBoost_regressor(dataframe, xfeatures, yfeatures):
    if "Tu_label" in xfeatures:
        hallo = Tu_label(dataframe.Tu)
        dataframe["Tu_label"] = hallo

        dataframe = encode_tulabel(dataframe)

    if 'log_eps' not in dataframe.columns:
        dataframe['log_eps'] = dataframe['eps'].apply(lambda x: math.log(x))

    # Stop depth at 300m
    dataframe = dataframe[dataframe["depth"] <= 300]

    x = dataframe[xfeatures].values
    y = dataframe[yfeatures].values

    # Split into train and test sets
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    # Define the XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(random_state=SEED)

    # Fit the regressor on the training data
    xgb_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_regressor.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Plot feature importances
    feature_importances = xgb_regressor.feature_importances_
    sorted_indices = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
    plt.yticks(range(len(feature_importances)), [xfeatures[i] for i in sorted_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importances')
    plt.show()
    return xgb_regressor, r2, y_test, y_pred, X_test, feature_importances


def XGBoost_regressor1m(dataframe, xfeatures, yfeatures):
    if "Tu_label" in xfeatures:
        hallo = Tu_label(dataframe.Tu)
        dataframe["Tu_label"] = hallo

        dataframe = encode_tulabel(dataframe)

    if 'log_eps' not in dataframe.columns:
        dataframe['log_eps'] = dataframe['eps'].apply(lambda x: math.log(x))

    # Stop depth at 300m
    dataframe = dataframe[dataframe["depth"] <= 30]

    x = dataframe[xfeatures].values
    y = dataframe[yfeatures].values

    # Split into train and test sets
    SEED = 42
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=SEED)

    # Define the XGBoost regressor
    xgb_regressor = xgb.XGBRegressor(random_state=SEED)

    # Fit the regressor on the training data
    xgb_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_regressor.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Plot feature importances
    feature_importances = xgb_regressor.feature_importances_
    sorted_indices = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)), feature_importances[sorted_indices], align='center')
    plt.yticks(range(len(feature_importances)), [xfeatures[i] for i in sorted_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importances')
    plt.show()
    return xgb_regressor, r2, y_test, y_pred, X_test, feature_importances
