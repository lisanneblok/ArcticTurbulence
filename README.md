# ArcticTurbulence MRes Report

_Lisanne Blok_, AI4ER student at AI4ER, University of Cambridge.

[![License: MIT](https://img.shields.io/badge/License-MIT-orange.svg)](https://opensource.org/licenses/MIT)

## Running the model:
- Find model under models/XGBoost or RandomForest
- Run notebooks/models/mlp_model.py or notebooks/models/MLP_model.ipynb

### Abstract
This project explores whether Arctic ocean turbulence, specifically ε, the dissipation rate of
energy, can be predicted using machine learning in the Arctic.
In previous machine learning turbulence models, Arctic data has been excluded, due
to its distinct mixing and low magnitude of turbulence. Double diffusion, (seasonal) sea
ice cover, internal wave breakage and tides make the mixing regimes unique in the Arctic.
Understanding Arctic turbulence is critical for understanding Arctic sea ice melting in a changing
climate.
However, measuring ε in the Arctic poses challenges, due to measurements being close to
noise levels and the presence of sea ice.
In this report is shown that machine learning models can accurately predict ε from CTD
microstructure measurements.
An XGBoost model and Random Forest had R2 scores of 0.885 and 0.726 respectively, in
a random train-test split. The XGBoost demonstrated a better fit to the data, however, this
model did not generalise well when tested on specific cruises. Specifically the Arctic shelf
regions showed absolute residuals. The accuracy of the model also increased with increasing
depth, demonstrating the upper ocean mixing regime is complex and not well captured. Double
diffusion turned out to be an important predictor of ε, specifically an encoded turner angle.
Moreover, a ResMLP was trained on both Arctic data and data used in Mashayek et al., however,
this model overfitted the data severly. This shows that turbulence regimes all over the world
are quite distinct and hard to generalise.
Overall, this research demonstrates the potential of machine learning in predicting Arctic
turbulence and provides valuable insights into the unique characteristics and challenges associated
with turbulence in this region. More microstructure data from the shelf regions could allow for
better characterisation of depth.

### Licence
This project falls under the MIT licence


### Environment
Install packages using `pip install -r requirements.txt` in a new environment, created by `conda create -n $ENV_NAME`.

### Loading datasets
Three types of datasets where used in this research. 

`/data/external/`

```
 ├── Microstructure  
 │      ├──   <arctic mictrostructure profile>.nc  
 │      │      
 ├── SI-area
 │      ├──   HadISST_ice.nc
 │         
 ├── GEBCO
 │      ├──   <gebco_2022>.nc
 │         
```
The data can be downloaded as follows:

#### Microstructure
To process the microstructure data, NetCDF files can be downloaded and processed in notebooks/preprocessing/ notebooks, since the format of each file is different.

- ArcticMix can be downloaded here as NetCDF: https://microstructure.ucsd.edu/#/cruise/33BI20150823
- ABSO-TEACOSI needs to be requested at the National Oceanography Centre, https://doi.org/10.1038/ngeo2350 
- ASCOS can be found here https://doi.org/10.1007/s00382-010-0937-5
- Haakon Mosby can be found here at the Norwegian Data Centre: http://metadata.nmdc.no/metadata-api/landingpage/dc9ce71f02d06191a5b1387b160a60f1
- Barneo Cruises can be found at the Norwegian Data Centre: https://doi.org/10.21335/NMDC-927288030 and https://doi.org/10.21335/NMDC-1809685365 
- Mosaic Cruise can be downloaded from PANGEA, https://doi.pangaea.de/10.1594/PANGAEA.939819
- N-ICE2015 can be found at the Norwegian Polar Institute: https://data.npolar.no/dataset/774bf6ab-b27e-51ab-bf8c-eb866cf61be2
- Nansen Legacy cruises can be found at the Norwegian Data Centre http://metadata.nmdc.no/metadata-api/landingpage/efb5d2e2d1f5b147ad828aa48b337205, http://metadata.nmdc.no/metadata-api/landingpage/1a5407acf9bacf7edea36b344c03b631, http://metadata.nmdc.no/metadata-api/landingpage/17dc16e9dbe5412c31b1ba86c996ccee

#### Sea Ice cover dataset
The full dataset covering 1870 to 2023 with monthly resolution can be downloaded as a ZIP file at https://doi.org/10.5065/r33v-sv91 for version 6. This dataset is under the Creative Commons Attribution 4.0 International License. The format is in NetCDF.

Citation: Shea, Dennis, Hurrell, Jim, Phillips, Adam. (2022). Merged Hadley-OI sea surface temperature and sea ice concentration data set. Version 6.0. UCAR/NCAR - GDEX. https://doi.org/10.5065/r33v-sv91. Accessed 29 Jun 2023.

### GEBCO dataset
Bathymetric data from The General bathymetric Chart of the Oceans (GEBCO) can be downloaded here https://download.gebco.net/. Select GEBCO 2023 and enter the boundaries, -180 degrees to 180 degrees in longitude and 60\degree N to 90 \degree N for the Arctic. The format of the dataset will be in NetCDF.

Citation: GEBCO Compilation Group (2023) GEBCO 2023 Grid (doi:10.5285/f98b053b-0cbc-6c23-e053-6c86abc0af7b)
