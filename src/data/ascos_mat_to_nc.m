ASCOS = load('/Users/Lisanne/Documents/AI4ER/Mres/ArcticTurbulence/data/external/Microstructure/MSS_ASCOS.mat')

% Access the MSS data within the ASCOS structure
mssData = ASCOS.MSS;

% Define the dimensions of the data
nTime = size(mssData.P, 2);
nDepth = size(mssData.P, 1);
nLatLon = size(mssData.latlon, 1);

% Create a NetCDF file
ncFile = 'data/interim/ASCOS_data.nc';
ncid = netcdf.create(ncFile, 'NC_WRITE');

% Define the dimensions in the NetCDF file
timeDimID = netcdf.defDim(ncid, 'time', nTime);
depthDimID = netcdf.defDim(ncid, 'depth', nDepth);
latlonDimID = netcdf.defDim(ncid, 'latlon', nLatLon);

% Define the variables in the NetCDF file
timeVarID = netcdf.defVar(ncid, 'time', 'double', timeDimID);
depthVarID = netcdf.defVar(ncid, 'depth', 'double', depthDimID);
latVarID = netcdf.defVar(ncid, 'latitude', 'double', latlonDimID);
lonVarID = netcdf.defVar(ncid, 'longitude', 'double', latlonDimID);
pVarID = netcdf.defVar(ncid, 'P', 'double', [depthDimID, timeDimID]);
tVarID = netcdf.defVar(ncid, 'T', 'double', [depthDimID, timeDimID]);
sVarID = netcdf.defVar(ncid, 'S', 'double', [depthDimID, timeDimID]);
sigthVarID = netcdf.defVar(ncid, 'SIGTH', 'double', [depthDimID, timeDimID]);
epsilonVarID = netcdf.defVar(ncid, 'epsilon', 'double', [depthDimID, timeDimID]);

% End the definition phase
netcdf.endDef(ncid);

% Write the data to the variables in the NetCDF file
netcdf.putVar(ncid, timeVarID, mssData.decday);
netcdf.putVar(ncid, depthVarID, 1:nDepth);
netcdf.putVar(ncid, latVarID, mssData.latlon(:, 1));
netcdf.putVar(ncid, lonVarID, mssData.latlon(:, 2));
netcdf.putVar(ncid, pVarID, mssData.P);
netcdf.putVar(ncid, tVarID, mssData.T);
netcdf.putVar(ncid, sVarID, mssData.S);
netcdf.putVar(ncid, sigthVarID, mssData.SIGTH);
netcdf.putVar(ncid, epsilonVarID, mssData.epsilon);

% Close the NetCDF file
netcdf.close(ncid);
