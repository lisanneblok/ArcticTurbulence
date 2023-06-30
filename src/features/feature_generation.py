import netCDF4 as nc
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import numpy as np
import sys
import os
import tqdm

from src.features.processing_func import mld
from src.features.processing_func import calc_hab, arctic_calchab
from src.features.processing_func import mld, calc_N2_kappa_sorted
from src.features.calc_seaice import calc_SIC
from src.utils.directories import get_parent_directory

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

sys.path.append('../..')


def select_variables(data, variables):
    """
    Select specific variables from the given dataset and reduce dimensions and
        coordinates to depth and profile.
    Args:
        data (xarray.Dataset): Input dataset containing variables.
        variables (list): List of variable names to select.
    Returns:
        xarray.Dataset: Dataset with selected variables and reduced
            dimensions/coordinates.
    """
    # Select the desired variables from the dataset
    selected_data = data[variables]
    # Reduce dimensions and coordinates to depth and profile
    selected_data = selected_data.squeeze().reset_coords(drop=True)
    return selected_data


def processing_functions(dataset, selected_columns, Hadi_SI, bathy_ds,
                         arctic=False, ASBO=False):
    """
    Perform processing on the input dataset.

    Parameters:
    - dataset (xarray.Dataset): Input dataset to be processed.
    - selected_columns (list): List of columns/variables to be selected
        from the dataset.
    - Hadi_SI (float): Value for Hadi_SI calculation.
    - bathy_ds (xarray.Dataset): Bathymetry dataset for
        calculations.
    - arctic (bool, optional): Flag indicating whether to perform
        Arctic-specific calculations.
                                Default is False.

    Returns:
    - dataset: Processed dataset.

    Processing Steps:
    1. Calculate N2 and kappa sorted values in the dataset.
    2. Calculate SIC (Sea Ice Concentration) in the dataset using Hadi_SI.
    3. Calculate MLD (Mixed Layer Depth) in the dataset.
    4. Perform Arctic-specific calculations if 'arctic' flag is True.
    5. Calculate habitat index (Hab) in the dataset.
    6. Select the desired variables/columns from the dataset.
    7. Convert the processed dataset to a pandas DataSet.

    Notes:
    - The 'arctic' flag determines whether the Arctic-specific calculations
        are performed. If set to True, 'arctic_calchab' is called; otherwise,
        'calc_hab' is used.

    Example Usage:
    processed_data = processing_func(dataset, selected_columns, Hadi_SI,
        bathy_ds, arctic=True)
    """
    dataset = calc_N2_kappa_sorted(dataset)
    # ABSO SIC values assigned manually
    #  if ASBO is not False:
    dataset = calc_SIC(dataset, Hadi_SI)
    dataset = mld(dataset)
    if arctic is True:
        dataset = arctic_calchab(dataset, bathy_ds)
    else:
        dataset = calc_hab(dataset, bathy_ds)

    dataset = select_variables(dataset, selected_columns)
    # dataframe = dataset.to_dataframe().reset_index()
    return dataset


def main():
    """
    Function to process datasets and generate a combined dataframe.

    This function performs the following steps:
    1. Retrieves the parent directory using the `get_parent_directory()`
        function.
    2. Constructs file paths for various NetCDF files using the parent
        directory.
    3. Opens the NetCDF files and assigns the resulting `xr.Dataset` objects
        to corresponding variables.
    4. Opens the bathymetry dataset and sea ice fraction data.
    5. Adds a "cruise" variable to each dataset to specify the cruise name.
    6. Defines a list of selected columns for processing.
    7. Modifies the variable name in the mosaic dataset to match the expected
        name.
    8. Creates an empty list to store the processed dataframes.
    9. Iterates over the datasets and calls the `processing_functions()`
        function on each dataset.
    10. Combines all the resulting dataframes into a single dataframe.
    11. Drops rows with missing values from the combined dataframe.
    12. Saves the processed dataframe as a pickle file.

    Returns:
    combined_nona (pandas.DataFrame): The combined and processed dataframe
        without missing values. This dataframe can then be used as input to
        a different model.
    """
    parent_dir = get_parent_directory()

    arctic_mix = os.path.join(parent_dir, "data/interim/arctic_mix.nc")
    asbo_nc = os.path.join(parent_dir, "data/interim/ASBO-TEACOSI_ds.nc")
    mosaic_nc = os.path.join(parent_dir, "data/interim/mosaic_ds.nc")
    nice_nc = os.path.join(parent_dir, "data/interim/nice_ds.nc")
    HM_nc = os.path.join(parent_dir, "data/interim/HM_ds.nc")
    barneo2007_nc = os.path.join(parent_dir, "data/interim/barneo2007_ds.nc")
    barneo2008_nc = os.path.join(parent_dir, "data/interim/barneo2008_ds.nc")
    KB2018616_nc = os.path.join(parent_dir, "data/interim/KB2018616.nc")
    KH2018709_nc = os.path.join(parent_dir, "data/interim/KH2018709.nc")
    ascos_nc = os.path.join(parent_dir, "data/interim/ascos_ds.nc")

    arctic_ds = xr.open_dataset(arctic_mix)
    asbo_ds = xr.open_dataset(asbo_nc)
    mosaic_ds = xr.open_dataset(mosaic_nc)
    nice_ds = xr.open_dataset(nice_nc)
    HM_ds = xr.open_dataset(HM_nc)
    barneo2007_ds = xr.open_dataset(barneo2007_nc)
    barneo2008_ds = xr.open_dataset(barneo2008_nc)
    KB2018616_ds = xr.open_dataset(KB2018616_nc)
    KH2018709_ds = xr.open_dataset(KH2018709_nc)
    ascos_ds = xr.open_dataset(ascos_nc)

    # Bathymetry dataset
    GEBCO_ds = os.path.join(
        parent_dir,
        "data/external/GEBCO/gebco_2022_n80.0_s63.0_w-170.0_e-130.0.nc")
    bathy_ds = xr.open_dataset(GEBCO_ds)

    # Sea ice fraction data
    SI_HadISST = os.path.join(
        parent_dir, "data/external/SI-area/HadISST_ice.nc")
    Hadi_SI = xr.open_dataset(SI_HadISST)

    # Add cruise name as a variable name
    arctic_ds["cruise"] = "ArcticMix"
    nice_ds["cruise"] = "NICE-2015"
    mosaic_ds["cruise"] = "Mosaic"
    asbo_ds["cruise"] = "ASBO"
    HM_ds["cruise"] = "Haakon Mosby"
    barneo2007_ds["cruise"] = "IPY Barneo 2007"
    barneo2008_ds["cruise"] = "IPY Barneo 2008"
    KB2018616_ds["cruise"] = "Nansen Legacy 2018"
    KH2018709_ds["cruise"] = "Nansen Legacy 2019"
    ascos_ds["cruise"] = "ASCOS"

    selected_columns = ["depth", "profile", "cruise", "latitude",
                        "longitude", "S", "T", "log_eps", "log_N2", "dTdz",
                        "dSdz", "hab", "Tu", "Tu_label", "time", "Rsubrho",
                        "sea_ice_concentration", "MLDJ", "MLDI"]

    # Mosaic dataset only includes the log epsilon
    mosaic_ds["log_eps"] = mosaic_ds["eps"]

    # Create a list of datasets
    datasets = [nice_ds, mosaic_ds, HM_ds, arctic_ds, asbo_ds, barneo2007_ds,
                barneo2008_ds, KB2018616_ds, KH2018709_ds, ascos_ds]
    # datasets = [nice_ds, ascos_ds]
    # Call the function to process the datasets

    dataframes = []
    for dataset in tqdm(datasets, desc='Processing Datasets', unit='dataset'):
        dataframe = processing_functions(dataset, selected_columns, Hadi_SI,
                                         bathy_ds)
        dataframes.append(dataframe)

    combined_df = pd.concat(dataframes)
    combined_nona = combined_df.dropna()
    combined_nona.to_pickle(
        parent_dir, "/data/processed_data/ml_ready/processed_df.pkl")


if __name__ == '__main__':
    main()
