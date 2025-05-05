from datetime import datetime, timezone, timedelta
from weather import get_forecast
from geo_utils import ecef_to_geodetic_newton
import numpy as np
from typing import Dict, List, Any, Union
from constants import R_UNIVERSAL, GAS_DATA, g
from dataclasses import dataclass


@dataclass
class Gas:
    name: str
    molar_mass: float
    R_specific: float
    mass: float
    
    @classmethod
    def from_name(cls, name: str, mass: float):
        """
        Factory: look up molar_mass by name, compute specific R,
        and return a new Gas instance.
        """
        key = name.lower()
        if key not in GAS_DATA:
            raise ValueError(f"Unknown gas '{name}'. Available: {list(GAS_DATA)}")
        M = GAS_DATA[key]["molar_mass"]
        R_spec = R_UNIVERSAL / M
        return cls(name=key, molar_mass=M, R_specific=R_spec, mass=mass)




class Balloon:
    def __init__(self,
                 radius: float,
                 envelope_mass: float,
                 gas: Union[str, Gas],
                 gas_mass: float = None):
        self.radius        = radius
        self.envelope_mass = envelope_mass

        if isinstance(gas, Gas):
            # user passed in a fully‐formed Gas (with its .mass)
            self.gas = gas
        elif isinstance(gas, str):
            # they gave us a name; they *must* supply gas_mass
            if gas_mass is None:
                raise ValueError("When `gas` is a string you must also pass `gas_mass`")
            self.gas = Gas.from_name(gas, gas_mass)
        else:
            raise ValueError("`gas` must be either a Gas instance or a string")

    def __repr__(self):
        return (f"<Balloon r={self.radius} m, "
                f"env_mass={self.envelope_mass} kg, "
                f"gas={self.gas.name}, "
                f"gas_mass={self.gas.mass} kg, "
                f"R_spec={self.gas.R_specific:.4f} J/(kg·K)>")
    
    def get_reynolds(self, 
                     rho: float, 
                     T: float, 
                     v_rel_norm: float) -> float:
        mu = sutherland_viscosity(T)
        return rho*v_rel_norm*(2*self.radius)/mu
    
    def get_drag_coeff(self,
                       rho: float,
                       T: float,
                       v_rel_norm: float) -> float:
        #  Reynolds number
        Re = self.get_reynolds(rho, T, v_rel_norm)

        # Piecewise Morsi–Alexander fits
        if Re < 0.1:
            return 24.0 / Re
        elif Re < 1.0:
            return 24.0 / Re * (1.0 + 0.1315 * Re**0.82)
        elif Re < 10.0:
            return 24.0 / Re * (1.0 + 0.1935 * Re**0.6305)
        elif Re < 1000.0:
            return 24.0 / Re * (1.0 + 0.3315 * Re**0.5)
        else:
            return 0.44
        
    def get_volume(self, T: float, P: float) -> float:
        return self.gas.mass * self.gas.R_specific * T / P
    
    def get_added_mass(self, rho: float, T: float, P: float) -> float:
        V = self.get_volume(T, P)
        Cm = 0.5
        return Cm * rho * V
    
    def drag_force(self,
                   rho: float,
                   T: float,
                   v_rel: np.ndarray[float]) -> np.ndarray[float]:
        v_rel_norm = np.linalg.norm(v_rel)
        if v_rel_norm < 1e-12:
            return np.zeros_like(v_rel)
        C_d = self.get_drag_coeff(rho, T, v_rel_norm)
        A = np.pi*self.radius**2
        return -0.5*C_d*rho*A*v_rel_norm*v_rel
    
    def buoyant_force(self, 
                      rho: float, 
                      T: float, 
                      P: float) -> np.ndarray[float]:
        V = self.get_volume(self, T, P)
        return np.array([0.0, 0.0, 1.0]) * (rho * V * g)
    
    def gravity_force(self, 
                      rho: float, 
                      T: float, 
                      P: float) -> np.ndarray[float]:
        total_mass = self.envelope_mass+self.gas.mass+self.get_added_mass(rho, T, P)
        return - total_mass * g * np.array([0.0, 0.0, 1.0])

          

class Payload:
    def __init__(self, 
                 radius: float, 
                 length: float, 
                 mass: float):
        self.radius = radius
        self.length = length
        self.mass = mass
        
    def get_reynolds(self, 
                     rho: float, 
                     T: float, 
                     v_rel_norm: float) -> float:
        mu = sutherland_viscosity(T)
        return rho * v_rel_norm * (2*self.radius) / mu
    
    def get_drag_coeff(self,
                       rho: float,
                       T: float,
                       v_rel_norm: float) -> float:
        """
        Piecewise drag‐coefficient for a smooth circular
        cylinder in cross‐flow (axis ⟂ flow).

        Re < 1e3 : transitional fit, Cd ~ 1.2 + 10 / sqrt(Re)
        1e3 ≤ Re < 4e5 : fully turbulent wake, Cd ≈ 1.0
        Re ≥ 4e5       : post‐critical drag crisis, Cd ≈ 0.5
        """
        Re = self.get_reynolds(rho, T, v_rel_norm)

        if Re < 1.0:
            # creeping‐flow regime
            return 4.0 * np.pi / Re
        elif Re < 1e3:
            # empirical transitional fit 
            return 1.2 + 10.0 / np.sqrt(Re)
        elif Re < 4e5:
            # subcritical, nearly constant form drag
            return 1.0
        else:
            # after critical‐Re drag crisis
            return 0.5

    def drag_force(self,
                   rho: float,
                   T: float,
                   v_rel: np.ndarray[float]) -> np.ndarray[float]:
        v_rel_norm = np.linalg.norm(v_rel)
        if v_rel_norm < 1e-12:
            return np.zeros_like(v_rel)
        C_d = self.get_drag_coeff(rho, T, v_rel_norm)
        A = 2 * self.radius * self.length 
        return -0.5*C_d*rho*A*v_rel_norm*v_rel
    
    def gravity_force(self) -> np.ndarray[float]:
        total_mass = self.mass
        return - total_mass * g * np.array([0.0, 0.0, 1.0])


class Tether:
    def __init__(self, 
                 lenght: float):
        self.lenght = lenght



def sutherland_viscosity(T: float) -> float:
    """
    Compute dynamic viscosity mu [Pa·s] of air at temperature T [K]
    using Sutherland's law.
    """
    # Reference conditions
    mu0 = 1.716e-5   # Pa·s at T0 = 273.15 K
    T0  = 273.15     # reference temperature [K]
    S   = 110.4      # Sutherland's constant [K]

    # Sutherland's law
    return mu0 * (T / T0)**1.5 * (T0 + S) / (T + S)




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