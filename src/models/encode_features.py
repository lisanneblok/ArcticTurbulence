import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


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
    """
    Encodes the values in the 'Tu_label' column of a pandas DataFrame
    into numeric representations.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a column
        named 'Tu_label'.

    Returns:
        pandas.DataFrame: The modified DataFrame with the 'Tu_label' column
        replaced by numeric labels.
    """
    # Create an instance of the LabelEncoder
    label_encoder = LabelEncoder()
    # Fit the encoder on the Tu_label column
    label_encoder.fit(data['Tu_label'])
    # Transform the Tu_label column into numeric representation
    numeric_labels = label_encoder.transform(data['Tu_label'])
    # Replace the Tu_label column with the numeric labels
    data['Tu_label'] = numeric_labels
    return data
