import netCDF4 as nc
from netCDF4 import Dataset
import xarray as xr
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# gsw oceanic toolbox: http://www.teos-10.org/pubs/Getting_Started.pdf
import gsw
from scipy.io import loadmat
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from tqdm import tqdm
from datetime import datetime


def check_coords(data):
    """
    Check the coordinate variables in the given data and add missing
    coordinates if necessary.
    Args:
        data (xarray.DataArray or xarray.Dataset): Input data containing
        variables.
    Returns:
        xarray.DataArray or xarray.Dataset: Data with updated coordinates.
    """

    # Check if latitude coordinate is present, add if missing
    if "latitude" not in data.coords:
        data = data.set_coords("latitude")
    # Check if longitude coordinate is present, add if missing
    if "longitude" not in data.coords:
        data = data.set_coords("longitude")
    # Check if time coordinate is present, add if missing
    if "time" not in data.coords:
        data = data.set_coords("time")
    # Check if depth coordinate is present, add if missing
    if "depth" not in data.coords:
        data["depth"] = data.depth
    # Return data with updated coordinates
    return data


def TS_derivative(dataset):
    """
    Calculate the derivatives of temperature (dT/dz) and salinity (dS/dz) with
    respect to depth in the given dataset.
    Parameters:
        dataset (xarray.Dataset): Input dataset containing temperature (T) and
        salinity (S) variables.
    Returns:
        xarray.Dataset: Dataset with additional variables 'dTdz' and 'dSdz'
        representing the derivatives.
    Raises:
        ValueError: If the required variables 'T' or 'S' are not present in the
        dataset.
    """
    dataset["dTdz"] = dataset.T.differentiate('depth')
    dataset['dSdz'] = dataset.S.differentiate('depth')
    return dataset


def interpolate_pmid(dataset, variable):
    """
    Interpolate a variable along the 'depth' dimension to match the original
    depth values in the given dataset.

    Parameters:
        dataset (xarray.Dataset): Input dataset containing the original
            'depth' dimension.
        variable (numpy.ndarray): Variable array with dimensions
            (depth, profile).

    Returns:
        xarray.DataArray: Interpolated variable array with dimensions
            ('depth', 'profile').
    """
    original_depth = dataset.depth
    depth_old = np.arange(variable.shape[0])  # Depth values of the variable

    # Extend the depth array to include the boundary value
    depth_new = np.append(depth_old, original_depth[-1])

    # Extend the variable array to include the boundary value
    new_array = np.vstack((variable, variable[-1, :]))

    # Create a 1D interpolation function along the depth dimension
    interp_func = interp1d(depth_new, new_array, axis=0, kind='linear')
    # Interpolate the variable to match the original depth dimension
    var_interp = interp_func(original_depth)

    # Create a DataArray with explicit dimension names
    var_dataarray = xr.DataArray(var_interp, dims=('depth', 'profile'))
    return var_dataarray


def Tu_label1(data_arr):
    """
    Apply labels to Turner angle values based on
    https://www.teos-10.org/pubs/gsw/pdf/Turner_Rsubrho.pdf

    Parameters:
        data_arr (numpy.ndarray): Array of Turner angle values.

    Returns:
        numpy.ndarray: Array with labels assigned to Turner angle values.
    """
    # Define the conditions and labels
    conditions = [
        (data_arr >= -90) & (data_arr < -45),
        (data_arr >= -45) & (data_arr < 45),
        (data_arr >= 45) & (data_arr < 90),
        (data_arr >= 90) & (data_arr < -90)
    ]
    labels = ['Diffusive Convection', 'Doubly stable', 'Salt fingering',
              'Statically unstable']

    # Apply the conditions and labels to create a new array with the labels
    result = np.select(conditions, labels, default='NaN')

    # Create a new DataArray with the label
    labeled_arr = data_arr.copy()
    labeled_arr.values = result

    return labeled_arr


def Tu_label(data_arr):
    """
    Apply labels to Turner angle values based on
    https://www.teos-10.org/pubs/gsw/pdf/Turner_Rsubrho.pdf

    Parameters:
        data_arr (numpy.ndarray): Array of Turner angle values.

    Returns:
        numpy.ndarray: Array with labels assigned to Turner angle values.
    """
    # Define the conditions and labels
    conditions = [
        np.isnan(data_arr),
        (data_arr >= -90) & (data_arr < -45),
        (data_arr >= -45) & (data_arr < 45),
        (data_arr >= 45) & (data_arr < 90),
        (data_arr >= 90) & (data_arr < -90)
    ]
    labels = ['NaN', 'Diffusive Convection', 'Doubly stable',
              'Salt fingering', 'Statically unstable']

    # Apply the conditions and labels to create a new array with the labels
    result = np.select(conditions, labels, default=0)

    # Create a new DataArray with the label
    labeled_arr = data_arr.copy()
    labeled_arr.values = result

    return labeled_arr


def calc_N2_kappa(dataset):
    """N2 and kappa both independent of epsilon

    Parameters
    ----------
    dataset : dataset
        Microstructure dataset, where insitu temperature is named as "T" in
        degrees Celcius.
        Salinity is named as "S" in .., and depth is called "depth" in meters.
    """
    S = dataset.S
    P = dataset.P
    lon = dataset.longitude.squeeze()
    lat = dataset.latitude.squeeze()
    T = dataset.T
    eps = dataset.eps
    z = dataset.depth

    # Add a dummy axis to pressure (P) to match the shape of other variables
    if T.shape != P.shape:
        # Project the lower-dimensional variable onto the target dimension
        P = np.expand_dims(P, axis=1)

    # convert measured T to potential T, which is seen as temperature now
    # potential temperature is the temperature a water parcel would have
    # if it were brought to the surface adiabatically (no pressure effects)
    dataset = dataset.rename({"T": "insituT"})
    dataset["T"] = gsw.conversions.pt_from_t(S, T, P, p_ref=0)
    dataset['rho'] = gsw.rho(S, T, P)
    dataset["SA"] = gsw.SA_from_SP(S, P, lon, lat)
    dataset["CT"] = gsw.CT_from_t(dataset.SA, T, P)

    # calculate the turner angle
    # The values of Turner Angle Tu and density ratio Rrho are calculated
    # at mid-point pressures, p_mid.
    # https://teos-10.org/pubs/gsw/html/gsw_Turner_Rsubrho.html
    [Tu, Rsubrho, p_mid] = gsw.Turner_Rsubrho(dataset.SA, dataset.CT, P)

    dataset["Tu"] = interpolate_pmid(dataset, Tu)
    dataset["Rsubrho"] = interpolate_pmid(dataset, Rsubrho)

    dataset["Tu_label"] = Tu_label(dataset.Tu)

    # Calculate N^2 using gsw_Nsquared
    # https://teos-10.org/pubs/gsw/html/gsw_Nsquared.html
    [N2, p_mid] = gsw.Nsquared(SA=dataset["SA"], CT=dataset["CT"], p=P,
                               lat=dataset["latitude"])

    dataset["N2"] = interpolate_pmid(dataset, N2)
    # calculate kappa like in Mashayek et al, 2022
    # assume chi is 0.2 in standard turbulence regime
    dataset['kappa'] = 0.2*dataset.eps/dataset.N2
    # assume mixing efficiency of 1 in double diffusion regime
    dataset["kappa_AT"] = dataset.eps/dataset.N2

    dataset["log_N2"] = np.log10(dataset.N2)
    dataset["log_kappa"] = np.log10(dataset.kappa)
    dataset["log_eps"] = np.log10(dataset.eps)

    dataset = TS_derivative(dataset)
    return dataset


def calc_N2_kappa_sorted(dataset):
    """N2 and kappa both independent of epsilon.
    Assume dataset is in profile, depth shape

    Parameters
    ----------
    dataset : dataset
        Microstructure dataset, where insitu temperature is named as "T" in
        degrees Celcius.
        Salinity is named as "S" in .., and depth is called "depth" in meters.
    """
    S = dataset.S
    P = dataset.P
    lon = dataset.longitude.squeeze()
    lat = dataset.latitude.squeeze()
    T = dataset.T
    eps = dataset.eps
    z = dataset.depth

    # Add a dummy axis to pressure (P) to match the shape of other variables
    if T.shape != P.shape:
        # Project the lower-dimensional variable onto the target dimension
        P = np.expand_dims(P, axis=-1)

    # convert measured T to potential T, which is seen as temperature now
    # potential temperature is the temperature a water parcel would have
    # if it were brought to the surface adiabatically (no pressure effects)

    # dataset = dataset.rename({"T": "insituT"})
    # dataset["T"] = gsw.conversions.pt_from_t(S, T, P, p_ref=0)
    # dataset['rho'] = gsw.rho(S, T, P)

    dataset["SA"] = gsw.SA_from_SP(S, P, lon, lat)
    # calculate conservative temeprature from absolute salinity and insitu-T
    dataset["CT"] = gsw.CT_from_t(dataset.SA, T, P)

    dataset["rho"] = gsw.rho(dataset["S"], dataset["T"], 0)
    dataset['drhodz'] = dataset.rho.differentiate('depth')
    dataset['d2rhodz2'] = dataset.drhodz.differentiate('depth')

    # The values of Turner Angle Tu and density ratio Rrho are calculated
    # at mid-point pressures, p_mid.
    # https://teos-10.org/pubs/gsw/html/gsw_Turner_Rsubrho.html
    [Tu, Rsubrho, p_mid] = gsw.Turner_Rsubrho(dataset.SA, dataset.CT, P)
    dataset["Tu"] = interpolate_pmid(dataset, Tu)
    dataset["Rsubrho"] = interpolate_pmid(dataset, Rsubrho)
    dataset["Tu_label"] = Tu_label(dataset.Tu)

    CT_values = dataset["CT"].values
    CT_sort = np.empty_like(CT_values) * np.nan

    # Iterate over each profile
    for i in range(dataset.profile.size):
        # Get the CT values for the current profile
        temp_CT = dataset.CT[:, i].values
        # Sort the CT values in ascending order
        sorted_CT = np.sort(temp_CT)
        # Store the sorted CT values in the CT_sort array
        CT_sort[:, i] = sorted_CT

    # Use the sorted CT values in the gws.Nsquared function
    N2, p_mid = gsw.Nsquared(SA=dataset.SA, CT=CT_sort, p=P,
                             lat=dataset["latitude"])
    dataset["N2_sort"] = interpolate_pmid(dataset, N2)

    # calculate kappa like in Mashayek et al, 2022
    # assume chi is 0.2 in standard turbulence regime
    dataset['kappa'] = 0.2*dataset.eps/dataset.N2_sort

    dataset["log_N2"] = np.log10(dataset.N2_sort)
    dataset["log_kappa"] = np.log10(dataset.kappa)
    
    # Perform column check and calculation
    if "log_eps" not in dataset:
        dataset["log_eps"] = np.log10(dataset["eps"])
    dataset = TS_derivative(dataset)
    return dataset


def calc_hab(data, bathy_ds):
    """
    Calculate the height above bottom (hab) based on the bathymetry dataset
    and the depth value in the dataset.

    Parameters:
        data (xarray.Dataset): Input dataset containing variables 'longitude',
            'latitude', 'profile', 'depth'.
        bathy_ds (xarray.Dataset): Bathymetry dataset with variables
            'elevation', 'lon', 'lat'.

    Returns:
        xarray.Dataset: Updated dataset with added variables 'bathymetry' and
        'hab'.

    """
    bathy_interp = bathy_ds.interp_like(data, method='nearest')
    n_depths = data.profile.shape[0]
    depth = np.zeros(n_depths)

    for i in (range(n_depths)):
        microlon = data.longitude[i].values.flatten()
        microlat = data.latitude[i].values.flatten()
        depth[i] = bathy_interp.elevation.sel(lon=microlon, lat=microlat,
                                              method='nearest')
    data['bathymetry'] = data.profile.copy(data=depth)

    data["hab"] = data.bathymetry + abs(data.depth)
    # Set positive depth values to zero
    data["hab"] = data["hab"].where(data["hab"] <= 0, 0)
    return data


def arctic_calchab(data, bathy_ds):
    """
    Calculate the height above bottom (hab) based on the bathymetry dataset
    and the depth value in the dataset. Specifically for the arctic mix, which
    has a different dimension shapes.

    Parameters:
        data (xarray.Dataset): Input dataset containing variables 'longitude',
            'latitude', 'profile', 'depth'.
        bathy_ds (xarray.Dataset): Bathymetry dataset with variables
            'elevation', 'lon', 'lat'.

    Returns:
        xarray.Dataset: Updated dataset with added variables 'bathymetry' and
        'hab'.

    """
    # group data by the 'profile' dimension
    profile_groups = data.groupby('profile')

    bathy_interp = bathy_ds.interp_like(data, method='nearest')
    n_profiles = len(data.depth)

    profile = np.zeros(len(profile_groups))

    # loop over each group
    for i, (_, profile_data) in (enumerate(profile_groups)):
        microlat = profile_data.latitude.values.flatten()[0]
        microlon = profile_data.longitude.values.flatten()[0]
        profile[i] = bathy_interp.elevation.sel(lon=microlon, lat=microlat,
                                                method='nearest').values

    # Create a DataArray for bathymetry with the 'profile' dimension
    profile_arr = xr.DataArray(profile, coords=[range(len(profile_groups))],
                               dims=['profile'])
    data['bathymetry'] = profile_arr
    data["hab"] = data.bathymetry + abs(data.depth)
    # Set positive depth values to zero
    data["hab"] = data["hab"].where(data["hab"] <= 0, 0)
    return data


def mld(dataset, outfile=False, save_mld=False, threshold=0.01):
    """
    out = mld(dataset, outfile, save_mld, threshold)

    Reads:
        dataset    :: xarray dataset containing rho, drhodz, and d2rhodz2
            variables
        outfile    :: output file to save the snapshot of the computed MLD
        save_mld   :: if True, the computed MLD will be saved in the output
            file
        threshold  :: threshold value for the potential density criterion
            (default: 0.01)

    Returns:
        MLDI, MLDJ  :: xarray DataArrays representing the MLD indices and
            values
    """
    # Extract variables from the dataset
    rho = dataset["rho"]
    drhodz = dataset["drhodz"]
    d2rhodz2 = dataset["d2rhodz2"]

    # Extract dimensions from the dataset
    depth_size, profile_size = rho.shape

    MLDI = np.ones((profile_size,)) * np.nan
    MLDJ = np.ones((profile_size,)) * np.nan

    for profile_num in range(profile_size):
        tmp = drhodz[:, profile_num]
        tmp2 = d2rhodz2[:, profile_num]

        if np.isnan(tmp).all():
            continue

        # Find the index of minimum stratification
        min_strat = np.nanargmin(tmp)
        MLDI[profile_num] = min_strat

        if min_strat > 2:
            # Check for non-NaN values in the slice
            valid_slice = tmp2[:min_strat][~np.isnan(tmp2[:min_strat])]
            if len(valid_slice) > 0:
                J = np.nanargmax(valid_slice)
                min_strat = min(min_strat, J)
        if tmp2[min_strat] < 0. and (min_strat > 4 and min_strat == J):
            MLDJ[profile_num] = 0
        else:
            # Use threshold-based approach if no clear minimum stratification
            if np.nanmin(tmp) > threshold:
                # Find the index where drhodz first exceeds the threshold
                indices_above_threshold = np.where(tmp > threshold)[0]
                if len(indices_above_threshold) > 0:
                    min_strat = indices_above_threshold[0]

            MLDJ[profile_num] = min_strat

    if save_mld:
        # Save the MLD
        MLD_dataset = xr.Dataset({"MLDI": (("profile",), MLDI),
                                  "MLDJ": (("profile",), MLDJ)})
        MLD_dataset.to_netcdf(outfile)

    # MLDI = xr.DataArray(MLDI, dims=("profile",),
    # coords={"profile": range(profile_size)})
    # MLDJ = xr.DataArray(MLDJ, dims=("profile",),
    # coords={"profile": range(profile_size)})

    # dataset["MLDI"] = MLDI
    # dataset["MLDJ"] = MLDJ

    MLD_dataset = xr.Dataset({"MLDI": (("profile",), MLDI),
                              "MLDJ": (("profile",), MLDJ)})

    # Combine MLD_dataset with the original dataset
    dataset.update(MLD_dataset)
    return dataset


def seasonal_sin(dataset, time_var='time', profile_var='profile'):
    """
    Add a seasonal sine feature to a NetCDF dataset.

    Parameters:
        dataset (xarray.Dataset): Input NetCDF dataset.
        time_var (str): Name of the datetime variable in the dataset.
        profile_var (str): Name of the profile variable in the dataset.

    Returns:
        xarray.Dataset: Updated dataset with the seasonal sine feature added.
    """

    # Extract the datetime array
    datetime_array = pd.to_datetime(dataset[time_var].values)

    # Extract the month and day from the datetime array
    months = datetime_array.month
    days = datetime_array.day

    # Convert month and day to radians
    month_radians = 2 * np.pi * (months - 1) / 12
    day_radians = 2 * np.pi * (days - 1) / 31

    # Create seasonal sine
    seasonal_feature_sin = np.sin(month_radians + day_radians)

    # Add the seasonal feature to the dataset
    dataset['seasonal_sin'] = (profile_var, seasonal_feature_sin)

    return dataset
