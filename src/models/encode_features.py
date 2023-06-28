import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb


def Tu_label(data_series):
    """
    Apply labels to Turner angle values based on
    https://www.teos-10.org/pubs/gsw/pdf/Turner_Rsubrho.pdf
    (McDougall et al, 1998)

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
        LabelEncoder
    """
    # Create an instance of the LabelEncoder
    label_encoder = LabelEncoder()
    # Fit the encoder on the Tu_label column
    label_encoder.fit(data['Tu_label'])
    # Transform the Tu_label column into numeric representation
    numeric_labels = label_encoder.transform(data['Tu_label'])
    # Replace the Tu_label column with the numeric labels
    data['Tu_label'] = numeric_labels
    return data, label_encoder


def decode_tulabel(data, label_encoder):
    """
    Decodes the numeric labels in the 'Tu_label' column of a pandas DataFrame
    back into their original categorical values.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a column
        named 'Tu_label'.
        label_encoder (sklearn.preprocessing.LabelEncoder): The fitted
        LabelEncoder used for encoding the labels.

    Returns:
        pandas.DataFrame: The modified DataFrame with the 'Tu_label' column
        replaced by the original categorical values.
    """
    # Transform the numeric labels back into the original categorical values
    decoded_labels = label_encoder.inverse_transform(data['Tu_label'])
    # Replace the Tu_label column with the decoded labels
    data['Tu_label'] = decoded_labels
    return data


def encode_tulabeldict(data):
    """
    Encodes the values in the 'Tu_label' column of a pandas DataFrame
    into numeric representations using a dictionary.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a column
        named 'Tu_label'.

    Returns:
        pandas.DataFrame: The modified DataFrame with the 'Tu_label' column
        replaced by numeric labels.
        dict: Dictionary containing the label mappings.
    """
    # Create a dictionary to map labels to numeric representations
    label_map = {'NaN': 0,
                 'Diffusive Convection': 1,
                 'Doubly stable': 2,
                 'Salt fingering': 3,
                 'Statically unstable': 4}

    # Map the labels to their numeric representations
    numeric_labels = data['Tu_label'].map(label_map)

    # Replace the Tu_label column with the numeric labels
    data['Tu_label'] = numeric_labels

    return data, label_map


def decode_tulabeldict(data, label_map):
    """
    Decodes the numeric labels in the 'Tu_label' column of a pandas DataFrame
    back into their original categorical values using a dictionary.

    Args:
        data (pandas.DataFrame): The input DataFrame containing a column
        named 'Tu_label'.
        label_map (dict): Dictionary containing the label mappings.

    Returns:
        pandas.DataFrame: The modified DataFrame with the 'Tu_label' column
        replaced by the original categorical values.
    """
    # Reverse the label_map dictionary to map numeric representations back
    # to labels
    reverse_label_map = {v: k for k, v in label_map.items()}

    # Map the numeric labels back to their original categorical values
    decoded_labels = data['Tu_label'].map(reverse_label_map)

    # Replace the Tu_label column with the decoded labels
    data['Tu_label'] = decoded_labels

    return data
