# constants.py

import numpy as np

# Gas constants
R_UNIVERSAL = 8.31446261815324 # Universal gas constant [J/(mol·K)]

Rd = 287.05  # specific gas constant for dry air [J/(kg.K)]

# Molar masses [kg/mol]
GAS_DATA = {    
    "helium":    { "molar_mass": 4.002602e-3 },
    "hydrogen":  { "molar_mass": 2.01588e-3 },
    "nitrogen":  { "molar_mass": 28.0134e-3 },
    "oxygen":    { "molar_mass": 31.9988e-3 },
}


# Ellipsoid constants (WGS-84) 

g = 9.81                         # Gravity
RE = 6_378_137.0                 # Equatorial radius [m]  
f = 1 / 298.257223563            # Flattening                 
eE   = np.sqrt(f * (2 - f))      # First eccentricity ≈ 0.08181919             
eE2  = eE**2                     # Eccentricity²                           
RP   = RE * np.sqrt(1 - eE2)     # Polar radius [m]     
ep2  = (RE**2 - RP**2) / RP**2   # Second eccentricity²


OPENDAP_URL = (
    "https://thredds.ucar.edu/thredds/dodsC/grib/NCEP/GFS/Global_0p25deg/Best"
)

ISOBARIC_VARS = [
    "u-component_of_wind_isobaric",
    "v-component_of_wind_isobaric",
    "Vertical_velocity_geometric_isobaric",   # w  (m s‑1)
    "Temperature_isobaric",
    "Specific_humidity_isobaric",             # q  (kg kg‑1)
    "Geopotential_height_isobaric",
]

SURFACE_VARS = [
    "u-component_of_wind_height_above_ground",  # 10 m AGL
    "v-component_of_wind_height_above_ground",
]      