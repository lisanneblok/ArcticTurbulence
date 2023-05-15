function [gamma_n] = get_gamma_n(nc_file)

%'get_neut_dens' gets neutral density in kg m^-^3 from microstructure
% profiles using the routines of Jackett & McDougall (1997). 

%INPUT = .nc file
%OUTPUT = .nc file of neutral density variable

%first we get data from .nc file
file = nc_file
S = ncread(file,'S');
T = ncread(file,'T');
p = ncread(file, 'PRESSURE');
lon = ncread(file, 'longitude');
lat = ncread(file, 'latitude');

%then we make all dimensions consistent (no. profiles x depth in 1m)
lon = squeeze(lon);
lat = squeeze (lat);
p = ones(size(lon,1),size(lon,2)).*p;

%next we calculated neutral density 
gamma_n = eos80_legacy_gamma_n(S,T,p,lon,lat);

%finally we save gamma_n as a .nc file
ncfilename = 'gamma_n.nc'
nccreate(ncfilename, 'data', 'Dimensions', ...      %create netCDF file
    {'profile', size(lon,1), 'depth', size(lon,2)});
ncwrite(ncfilename, 'data', gamma_n);               %write matrix to netCDF file
gamma_n = ncfilename                                %return netcdf filename

end
