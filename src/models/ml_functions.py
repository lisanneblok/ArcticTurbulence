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
import sys

sys.path.append('../..')
from src.models.encode_features import Tu_label, encode_tulabel

warnings.filterwarnings('ignore')
plt.rcParams['figure.figsize'] = (10, 10)
plt.rcParams.update({'font.size': 14})
sys.path.append('../..')


def plot_importances(var_col_names, importances, num_params_to_show=None,
                     ax=None):
    """
    Visualize feature importances.

    Parameters:
    - var_col_names (list): List of input variable column names.
    - importances (list): List of feature importances corresponding to
        var_col_names.
    - num_params_to_show (int, optional): Number of top most significant
        features to show. Default is None.
    - ax (matplotlib.axes.Axes, optional): Axes object to plot on. If not
        provided, a new one will be created.

    Returns:
    - ax (matplotlib.axes.Axes): Axes object containing the feature
        importance plot.
    """

    if ax is None:
        fig, ax = plt.subplots()

    # Sort the importances and variable names in descending order
    sorted_indices = sorted(range(len(importances)),
                            key=lambda i: importances[i], reverse=True)
    sorted_var_names = [var_col_names[i]
                        for i in sorted_indices]
    sorted_importances = [importances[i] for i in sorted_indices]

    # Select the top most significant features if specified
    if isinstance(num_params_to_show, int):
        sorted_var_names = sorted_var_names[:num_params_to_show]
        sorted_importances = sorted_importances[:num_params_to_show]

    # Plot the feature importances as horizontal bars
    ax.barh(sorted_var_names, sorted_importances)

    # Set labels and title
    ax.set_xlabel('Feature Importance')
    ax.set_ylabel('Input Variable')
    ax.set_title('Feature Importance')

    # Add value annotations next to each bar
    for i, imp in enumerate(sorted_importances):
        ax.text(imp + 0.01, i, f'{imp:.3f}', ha='left', va='center_baseline')

    return ax


def RF_regressor(dataframe, xfeatures, yfeatures, cruise_test=False):
    """
    Trains a Random Forest Regressor model on the provided dataset and
    predicts the target variable.

    Args:
        dataframe (pandas.DataFrame): The input dataset containing the features
            and target variable.
        xfeatures (list): A list of column names representing the input
            features.
        yfeatures (list): A list of column names representing the target
            variable(s).
        cruise_test (bool or str, optional): If a string is provided, the
            function splits the dataset into training and testing sets based
            on the specified cruise name. If False (default), the dataset is
            randomly split into train and test sets using a 80:20 ratio.
            Defaults to False.

    Returns:
        tuple: A tuple containing the trained pipeline, actual target values
            for the test set, input features of the test set, feature
            importances, input features of the training set, and target values
            of the training set.

    """
    if "Tu_label" not in dataframe.columns:
        hallo = Tu_label(dataframe.Tu)
        dataframe["Tu_label"] = hallo

        dataframe = encode_tulabel(dataframe)

    # dataframe['log_eps'] = dataframe['eps'].apply(lambda x: math.log(x))

    # stop depth at 300m
    # dataframe = dataframe[dataframe["depth"] <= 300]

    SEED = 42

    if isinstance(cruise_test, str):
        # split test and train dataframe accordingly to test
        train_df = dataframe[dataframe["cruise"] != cruise_test]
        test_df = dataframe[dataframe["cruise"] == cruise_test]

        X_train = train_df[xfeatures]
        X_test = test_df[xfeatures]
        y_train = train_df[yfeatures]
        y_test = test_df[yfeatures]

    else:
        x = dataframe[xfeatures]
        y = dataframe[yfeatures]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=SEED)

    # Split into train and test sets
    SEED = 42

    # Define the MinMaxScaler
    scaler = MinMaxScaler()

    # Define the RandomForestRegressor
    rfr = RandomForestRegressor(random_state=SEED)

    # Create a pipeline
    pipeline = Pipeline([('scaler', scaler), ('rfr', rfr)])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Obtain feature importances
    importances = rfr.feature_importances_
    return pipeline, y_test, X_test, importances, X_train, y_train


def XGBoost_regressor(dataframe, xfeatures, yfeatures, cv=False,
                      cruise_test=False, manual_regressor=None):
    """
    Perform XGBoost Regression on the given dataset.

    The function applies XGBoost regression to predict the target variable
    specified by yfeatures using the features specified by xfeatures.

    Parameters:
    - dataframe (pandas.DataFrame): The input dataset.
    - xfeatures (list): List of column names representing the input features.
    - yfeatures (list): List of column names representing the target variable.
    - cv (int or None): Number of cross-validation folds. If None,
        no cross-validation is performed.
    - cruise_test (bool or str): If bool, indicates whether to split the
        data into train and test sets randomly. If str, indicates the cruise
        name to use as the test set.
    - manual_regressor (xgboost.XGBRegressor): Optional manual XGBoost
        regressor object. If provided, this regressor will be used instead of
        creating a new one.

    Returns:
    - xgb_regressor (xgboost.XGBRegressor): The trained XGBoost regressor.
    - r2 (float): R2 score on the test set.
    - y_test (pandas.Series): The true values of the target variable for the
        test dataset.
    - y_pred (numpy.ndarray): The predicted values of the target variable for
        the test dataset.
    - X_test (pandas.DataFrame): The input features of the test dataset.
    - feature_importances (numpy.ndarray): The feature importances calculated
        by the XGBoost regressor.
    - X_train (pandas.DataFrame): The input features of the train dataset.
    - y_train (pandas.Series): The true values of the target variable for the
        train dataset.
    """

    if "Tu_label" not in dataframe.columns:
        hallo = Tu_label(dataframe.Tu)
        dataframe["Tu_label"] = hallo

        dataframe = encode_tulabel(dataframe)

        dataframe, label = encode_tulabel(dataframe)

    if 'log_eps' not in dataframe.columns:
        dataframe['log_eps'] = dataframe['eps'].apply(lambda x: math.log(x))

    # Stop depth at 300m
    # dataframe = dataframe[dataframe["depth"] <= 300]

    SEED = 42

    if isinstance(cruise_test, str):
        # split test and train dataframe accordingly to test
        train_df = dataframe[dataframe["cruise"] != cruise_test]
        test_df = dataframe[dataframe["cruise"] == cruise_test]

        X_train = train_df[xfeatures]
        X_test = test_df[xfeatures]
        y_train = train_df[yfeatures]
        y_test = test_df[yfeatures]

    else:
        x = dataframe[xfeatures]
        y = dataframe[yfeatures]

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(x,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=SEED)

    if manual_regressor is None:
        # Create the XGBoost regressor
        xgb_regressor = xgb.XGBRegressor(
            max_depth=7,
            n_estimators=300,
            learning_rate=0.1,
            random_state=SEED
        )
    else:
        xgb_regressor = manual_regressor

    # Fit the regressor on the training data
    xgb_regressor.fit(X_train, y_train)

    # Predict on the test set
    y_pred = xgb_regressor.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Perform cross-validation if cv is not None
    # cv = None
    if cv is not None:
        r2 = cross_val_score(xgb_regressor, X_train, y_train, cv=cv,
                             scoring='r2')

    # Plot feature importances
    feature_importances = xgb_regressor.feature_importances_
    sorted_indices = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)),
             feature_importances[sorted_indices], align='center')
    plt.yticks(range(len(feature_importances)),
               [xfeatures[i] for i in sorted_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importances')
    plt.show()

    return (xgb_regressor, r2, y_test, y_pred, X_test,
            feature_importances, X_train, y_train)


def XGBoost_regressor_tuned(dataframe, xfeatures, yfeatures):
    """Perform fine-tuned XGBoost regression on the given dataset.

    The function applies XGBoost regression with tuned parameters to predict
    the target variable specified by yfeatures
    using the features specified by xfeatures.

    Parameters:
    - dataframe (pandas.DataFrame): The input dataset.
    - xfeatures (list): List of column names representing the input features.
    - yfeatures (list): List of column names representing the target variable.

    Returns:
    - pipeline (sklearn.pipeline.Pipeline): The trained pipeline containing
        preprocessing steps and the XGBoost regressor.
    - r2 (float): The R-squared score for the model's performance on the
        test set.
    - y_test (numpy.ndarray): The true values of the target variable for the
        test dataset.
    - y_pred (numpy.ndarray): The predicted values of the target variable for
        the test dataset.
    - X_test (numpy.ndarray): The input features of the test dataset.
    - feature_importances (numpy.ndarray): The feature importances calculated
        by the XGBoost regressor.
    - X_train (numpy.ndarray): The input features of the training dataset.
    - y_train (numpy.ndarray): The true values of the target variable for the
        training dataset.
    """
    if "Tu_label" not in dataframe.columns:
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
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2,
                                                        random_state=SEED)

    # Define the XGBoost regressor with best parameters
    xgb_regressor = xgb.XGBRegressor(learning_rate=0.1, max_depth=7,
                                     n_estimators=300, random_state=SEED)

    # Perform k-fold cross-validation
    cv = KFold(n_splits=5, shuffle=True, random_state=SEED)

    # Create a pipeline with preprocessing steps and XGBoost regressor
    pipeline = Pipeline([
        ('scaler', MinMaxScaler()),
        ('regressor', xgb_regressor)
    ])

    # Fit the pipeline on the training data using cross-validation
    pipeline.fit(X_train, y_train)

    # Predict on the test set
    y_pred = pipeline.predict(X_test)

    # Calculate R2 score
    r2 = r2_score(y_test, y_pred)

    # Plot learning curves
    train_scores = []
    val_scores = []

    for train_index, val_index in cv.split(X_train):
        X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Fit the pipeline on the training fold
        pipeline.fit(X_train_fold, y_train_fold)

        # Calculate R2 scores for training and validation folds
        train_score = r2_score(y_train_fold, pipeline.predict(X_train_fold))
        val_score = r2_score(y_val_fold, pipeline.predict(X_val_fold))

        train_scores.append(train_score)
        val_scores.append(val_score)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(train_scores) + 1), train_scores, label='Training')
    plt.plot(range(1, len(val_scores) + 1), val_scores, label='Validation')
    plt.xlabel('Number of Folds')
    plt.ylabel('R2 Score')
    plt.title('Learning Curves')
    plt.legend()
    plt.show()

    # Plot feature importances
    feature_importances = xgb_regressor.feature_importances_
    sorted_indices = feature_importances.argsort()

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_importances)),
             feature_importances[sorted_indices], align='center')
    plt.yticks(range(len(feature_importances)),
               [xfeatures[i] for i in sorted_indices])
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('XGBoost Feature Importances')
    plt.show()
    return (pipeline, r2, y_test, y_pred, X_test,
            feature_importances, X_train, y_train)
