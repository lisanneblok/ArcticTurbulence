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


def check_coords(data):
    if "latitude" not in data.coords:
        data = data.set_coords("latitude")
    if "longitude" not in data.coords:
        data = data.set_coords("longitude")
    if "time" not in data.coords:
        data = data.set_coords("time")
    if "depth" not in data.coords:
        data["depth"] = data.depth
    return data


def TS_derivative(dataset):
    dataset["dTdz"] = dataset.T.differentiate('depth')
    dataset['dSdz'] = dataset.S.differentiate('depth')
    return dataset


def interpolate_pmid(dataset, variable):
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

    dataset = TS_derivative(dataset)
    return dataset


def calc_hab(data, bathy_ds):
    bathy_interp = bathy_ds.interp_like(data, method='nearest')
    n_depths = data.profile.shape[0]
    depth = np.zeros(n_depths)

    for i in tqdm(range(n_depths)):
        microlon = data.longitude[i].values.flatten()
        microlat = data.latitude[i].values.flatten()
        depth[i] = bathy_interp.elevation.sel(lon=microlon, lat=microlat,
                                              method ='nearest')
    data['bathymetry'] = data.profile.copy(data=depth)
    
    data["hab"] = data.bathymetry + abs(data.depth)
    # Set positive depth values to zero
    data["hab"] = data["hab"].where(data["hab"] <= 0, 0)
    return data


def arctic_calchab(data, bathy_ds):
    # group data by the 'profile' dimension
    profile_groups = data.groupby('profile')

    bathy_interp = bathy_ds.interp_like(data, method='nearest')
    n_profiles = len(data.depth)

    profile = np.zeros(len(profile_groups))

    # loop over each group
    for i, (_, profile_data) in tqdm(enumerate(profile_groups)):
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
