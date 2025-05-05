from datetime import datetime, timezone, timedelta
from weather import get_forecast
from geo_utils import ecef_to_geodetic_newton
import numpy as np
from typing import Dict, List, Any
from constants import R_UNIVERSAL, GAS_DATA
from dataclasses import dataclass


@dataclass
class Gas:
    name: str
    molar_mass: float
    R_specific: float
    
    @classmethod
    def from_name(cls, name: str):
        """
        Factory: look up molar_mass by name, compute specific R,
        and return a new Gas instance.
        """
        key = name.lower()
        if key not in GAS_DATA:
            raise ValueError(f"Unknown gas '{name}'. "
                             f"Available: {list(GAS_DATA)}")
        M = GAS_DATA[key]["molar_mass"]
        R_spec = R_UNIVERSAL / M
        return cls(name=key, molar_mass=M, R_specific=R_spec)




class Balloon:
    def __init__(self, 
                 radius: float, 
                 envelope_mass: float, 
                 gas: str):
        pass
    

class Payload:
    pass

class Tether:
    pass




def compute_drag(C_d: float,
                 rho: float,
                 A: float,
                 v_rel: np.ndarray) -> np.ndarray:
    """
    Compute the aerodynamic drag force vector.

    Parameters
    ----------
    C_d : float
        Drag coefficient (dimensionless).
    rho : float
        Ambient fluid density (kg/m³).
    A : float
        Reference area (m²).
    v_rel : np.ndarray
        Relative velocity vector of the object w.r.t. the fluid (m/s).

    Returns
    -------
    np.ndarray
        Drag force vector (N), pointing opposite to v_rel.
    """
    # Magnitude of relative velocity
    v_mag = np.linalg.norm(v_rel)
    
    # Dynamic pressure q = 0.5 * rho * v^2
    q = 0.5 * rho * v_mag**2
    
    # Drag force magnitude = q * C_d * A
    drag_mag = q * C_d * A
    
    # Return vector opposite to motion 
    if v_mag > 0:
        return -drag_mag * (v_rel / v_mag)
    else:
        return np.zeros_like(v_rel)




def get_statedot(state: np.ndarray, t: datetime, t_0: datetime) -> np.ndarray:
    current_time = t_0 + timedelta(seconds=t)
    x_b, y_b, z_b = state[0:3]
    x_p, y_p, z_p = state[6:9]
    lat_b, lon_b, h_b =  ecef_to_geodetic_newton(x_b,y_b,z_b)
    lat_p, lon_p, h_p = ecef_to_geodetic_newton(x_p,y_p,z_p)
    forecast_dict_b = get_forecast(lat_b,lon_b, h_b, current_time)
    forecast_dict_p = get_forecast(lat_p,lon_p, h_p, current_time)
    
    
    statedot: np.ndarray = [] 
    statedot[0:3] = state[3:6]
    statedot[6:9] = state[9:12]    
    pass 



if __name__ == '__main__':
    print('Start of Simulation')