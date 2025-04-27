import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import QuantileTransformer


# Function to subsample every 10 meters
def subsample(group):
    # Round depth to nearest 10 meters
    group = group.copy()
    group['depth_rounded'] = (group['depth'] / 10).round() * 10
    # Drop duplicates on the rounded depth
    return group.drop_duplicates(subset='depth_rounded')


def process_dataframe():
    SEED = 42

    df = pd.read_pickle('/Users/lb962/Documents/GitHub/ArcticTurbulence/data/ml_ready/merged_arctic.pkl')
    df.reset_index(drop=True, inplace=True)

    # group by 'profile' and 'cruise'
    grouped = df.groupby(['profile', 'cruise'])

    # deal with class imbalance by subsampling
    df = grouped.apply(subsample).reset_index(drop=True)

    # Define the list of features and target variable
    xstringlist = ['S', 'T', 'latitude', 'dSdz', 'dTdz', 'log_N2']
    ystringlist = ['log_eps']

    # Get the unique profiles
    profiles = df['profile'].unique()

    # Split by profiles
    profiles_train, profiles_temp = train_test_split(
        profiles, test_size=0.4, random_state=SEED)

    profiles_val, profiles_test = train_test_split(
        profiles_temp, test_size=0.5, random_state=SEED)

    # Select rows where profile is in the corresponding set
    train_df = df[df['profile'].isin(profiles_train)]
    val_df = df[df['profile'].isin(profiles_val)]
    test_df = df[df['profile'].isin(profiles_test)]

    # select target and predictor features
    X_train = train_df[xstringlist]
    y_train = train_df[ystringlist]

    X_val = val_df[xstringlist]
    y_val = val_df[ystringlist]

    X_test = test_df[xstringlist]
    y_test = test_df[ystringlist]

    scaler_range = MinMaxScaler()
    scaler_range.fit(X_train)
    df_train_scaled_range = pd.DataFrame(scaler_range.transform(X_train))
    df_val_scaled_range = pd.DataFrame(scaler_range.transform(X_val))
    df_test_scaled_range = pd.DataFrame(scaler_range.transform(X_test))

    # Target transformation to deal with class imbalance
    qt = QuantileTransformer(output_distribution='normal', random_state=SEED)
    y_train_transformed = qt.fit_transform(y_train)
    y_val_transformed = qt.transform(y_val)
    y_test_transformed = qt.transform(y_test)

    return df_train_scaled_range, df_val_scaled_range, df_test_scaled_range, y_train_transformed, y_val_transformed, y_test_transformed


if __name__ == "__main__":
    process_dataframe()
