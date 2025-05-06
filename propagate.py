from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from weather import get_forecast
from geo_utils import ecef_to_geodetic_newton, geodetic_to_ecef, enu_vector_to_ecef, ecef_to_enu
from gas_utils import sutherland_viscosity
from scipy.integrate import solve_ivp
import numpy as np
from typing import Dict, Any, Union
from constants import R_UNIVERSAL, GAS_DATA, g, RE
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import sys
import json
import os


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


class ForecastCache:
    def __init__(self, max_size=10000):
        self.cache = {}
        self.max_size = max_size
        self.hits = 0
        self.misses = 0
    
    def get(self, lat: float, lon: float, alt: float, time: datetime):
        key = (round(lat, 3), round(lon, 3), round(alt, 1), time.replace(microsecond=0, second=0))
        if key in self.cache:
            self.hits += 1
        else:
            self.misses += 1
        return self.cache.get(key)
    
    def set(self, lat: float, lon: float, alt: float, time: datetime, forecast: Dict[str, Any]):
        if len(self.cache) >= self.max_size:
            self.cache.pop(next(iter(self.cache)))
        key = (round(lat, 3), round(lon, 3), round(alt, 1), time.replace(microsecond=0, second=0))
        self.cache[key] = forecast
    
    def stats(self):
        total = self.hits + self.misses
        return f"Cache hits: {self.hits}, misses: {self.misses}, hit rate: {self.hits / total:.2%}" if total > 0 else "Cache not used"


class Balloon:
    def __init__(self,
                 radius: float,
                 envelope_mass: float,
                 gas: Union[str, Gas],
                 gas_mass: float = None):
        self.radius        = radius
        self.envelope_mass = envelope_mass

        if isinstance(gas, Gas):
            self.gas = gas
        elif isinstance(gas, str):
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
        V_ideal = self.gas.mass * self.gas.R_specific * T / P
        V_max = (4/3) * np.pi * self.radius**3  
        # print(f"V_ideal={V_ideal:.4f} m³, V_max={V_max:.4f} m³")
        return min(V_ideal, V_max)
    
    def get_added_mass(self, rho: float, T: float, P: float) -> float:
        V = self.get_volume(T, P)
        Cm = 0.5
        return Cm * rho * V
    
    def get_total_mass(self, rho: float, T: float, P: float) -> float:
        return self.envelope_mass+self.gas.mass+self.get_added_mass(rho, T, P)
    
    def drag_force(self,
                   rho: float,
                   T: float,
                   v_rel: np.ndarray[float]) -> np.ndarray[float]:
        v_rel_norm = np.linalg.norm(v_rel)
        if v_rel_norm < 1e-12:
            return np.zeros_like(v_rel)
        C_d = self.get_drag_coeff(rho, T, v_rel_norm)
        A = np.pi*self.radius**2
        # print(f"Drag force={-0.5*C_d*rho*A*v_rel_norm*v_rel}")
        return -0.5*C_d*rho*A*v_rel_norm*v_rel
    
    def buoyant_force(self, 
                      rho: float, 
                      T: float, 
                      P: float,
                      lat: float,
                      lon: float) -> np.ndarray[float]:
        V = self.get_volume(T, P)
        up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        # print(f"Buoyant force={up * (rho * V * g)}")
        return up * (rho * V * g)
    
    def gravity_force(self, 
                      rho: float, 
                      T: float, 
                      P: float,
                      lat: float,
                      lon: float) -> np.ndarray[float]:
        total_mass = self.get_total_mass(rho, T, P)
        up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        # print(f"Gravity force={- total_mass * g * up}")
        return - total_mass * g * up

          

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
    
    def get_total_mass(self) -> float:
        return self.mass

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
    
    def gravity_force(self, 
                      lat: float, 
                      lon:float) -> np.ndarray[float]:
        total_mass = self.get_total_mass()
        up = np.array([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
        return - total_mass * g * up


class Tether:
    def __init__(self, 
                 length: float):
        self.length = length
        


class System:
    def __init__(self, balloon: Balloon, payload: Payload, tether: Tether, t0: datetime):
        self.balloon = balloon
        self.payload = payload
        self.tether = tether
        self.t0 = t0
        self.forecast_cache = ForecastCache()
    
    def __call__(self, t, state):
        return get_statedot(state, t, self.t0, balloon=self.balloon, payload=self.payload,
                            tether=self.tether, forecast_cache=self.forecast_cache)


def get_statedot(state: np.ndarray, t: float, t0: datetime, balloon: Balloon, payload: Payload, tether: Tether, forecast_cache: ForecastCache) -> np.ndarray:
    current_time = t0 + timedelta(seconds=t)
    x_b, v_b = state[0:3], state[3:6]
    x_p, v_p = state[6:9], state[9:12]
    lat_b, lon_b, h_b = ecef_to_geodetic_newton(*x_b)
    lat_p, lon_p, h_p = ecef_to_geodetic_newton(*x_p)
    
    # print(f"Balloon height={h_b}, Payload height={h_p}")
    
    
    # print total simulation time evry 10 minutes
    # if t % 600 < 1e-6:
    #     print(f"Balloon height={h_b:.2f} m, Payload height={h_p:.2f} m, Time={t:.2f} s")
    
    print(f"Simulation time={t:.2f} s, Balloon height={h_b:.2f} m")
    
    
    # if t % 600 < 1e-6:
    #     print(f"Balloon height={h_b:.2f} m, Payload height={h_p:.2f} m, Time={t:.2f} s")    

    f_b = forecast_cache.get(lat_b, lon_b, h_b, current_time)
    if f_b is None:
        f_b = get_forecast(lat_b, lon_b, h_b, current_time)
        forecast_cache.set(lat_b, lon_b, h_b, current_time, f_b)

    f_p = forecast_cache.get(lat_p, lon_p, h_p, current_time)
    if f_p is None:
        f_p = get_forecast(lat_p, lon_p, h_p, current_time)
        forecast_cache.set(lat_p, lon_p, h_p, current_time, f_p)
    
    
    wind_b = enu_vector_to_ecef(np.array([f_b["u"], f_b["v"], f_b["w"]]), lat_b, lon_b)
    wind_p = enu_vector_to_ecef(np.array([f_p["u"], f_p["v"], f_p["w"]]), lat_p, lon_p)
    # print(f"wind_b={wind_b}, wind_p={wind_p}")
    
    vrel_b = v_b - wind_b
    vrel_p = v_p - wind_p
    # print(f"vrel_b={vrel_b}, vrel_p={vrel_p}")
    
    Fb = ( balloon.buoyant_force(f_b["rho"], f_b["T"], f_b["p"]*100, lat_b, lon_b)
         + balloon.gravity_force(f_b["rho"], f_b["T"], f_b["p"]*100, lat_b, lon_b)
         + balloon.drag_force(f_b["rho"], f_b["T"], vrel_b))
    Fp = ( payload.drag_force(f_p["rho"], f_p["T"], vrel_p)
         + payload.gravity_force(lat_p, lon_p) )
    
    mb = balloon.get_total_mass(f_b["rho"], f_b["T"], f_b["p"]*100)
    mp = payload.get_total_mass()
    
    # print(f"mb={mb}, mp={mp}")
    d_vec = x_p - x_b
    L = tether.length
    t_hat = d_vec / L
    dv2 = np.dot(v_p - v_b, v_p - v_b)
    
    num = dv2 + np.dot(d_vec, (Fp/mp - Fb/mb))
    den = L * (1.0/mp + 1.0/mb)
    Tmag = num / den
    
    # test
    # Spring-damper tether model
    # d_norm = np.linalg.norm(d_vec)
    # k = 1e3  # Stiffness [N/m]
    # c = 10  # Damping [N·s/m]
    # d_dot = np.dot(v_p - v_b, d_vec) / d_norm if d_norm > 1e-6 else 0.0
    # Tmag = k * (d_norm - L) + c * d_dot
    # Tmag = max(Tmag, 0.0)  # Tether cannot push
    # t_hat = d_vec / d_norm if d_norm > 1e-6 else np.zeros(3)
    
    # print(f"Tether length={d_norm}, Tmag={Tmag}, t_hat={t_hat}")
    
    
    a_b = (Fb +  Tmag * t_hat) / mb
    a_p = (Fp -  Tmag * t_hat) / mp
    
    # print(f"a_b={a_b}, a_p:{a_p}")
    
    statedot = np.zeros_like(state)
    statedot[0:3] = v_b
    statedot[3:6] = a_b
    statedot[6:9] = v_p
    statedot[9:12] = a_p
     
    return statedot 


if __name__ == '__main__':
    print('Start of Simulation')
    lat0, lon0, h0 = np.deg2rad(40.0), np.deg2rad(-75.0), 100.0
    
    balloon = Balloon(radius=5, envelope_mass=1.5, gas="helium", gas_mass=4.0) # 1.745
    payload = Payload(radius=0.2, length=0.5, mass=3)
    tether = Tether(length=20.0)
    t0 = datetime.now(ZoneInfo("UTC"))
    system = System(balloon, payload, tether, t0)

    x_b0 = geodetic_to_ecef(lat0, lon0, h0)
    up = np.array([np.cos(lat0) * np.cos(lon0), np.cos(lat0) * np.sin(lon0), np.sin(lat0)])
    x_p0 = x_b0 - tether.length * up
    y0 = np.hstack([x_b0, np.zeros(3), x_p0, np.zeros(3)])

    t_span = (0.0, 5*3600.0)
    t_eval = np.linspace(0.0, t_span[1], 360)
    
    start_time = time.time()
    sol = solve_ivp(system, t_span, y0, method="Radau", t_eval=t_eval, rtol=1e-4, atol=1e-5, max_step=10.0)
    print(f"Simulation took {time.time() - start_time:.2f} seconds")

    Xb, Yb, Zb = sol.y[0], sol.y[1], sol.y[2]
    Vbx, Vby, Vbz = sol.y[3], sol.y[4], sol.y[5]
    Xp, Yp, Zp = sol.y[6], sol.y[7], sol.y[8]
    t = sol.t
    

    traj_log = {
        "t": sol.t.tolist(),
        "balloon_ecef":  {
            "X": Xb.tolist(),
            "Y": Yb.tolist(),
            "Z": Zb.tolist()
        },
        "payload_ecef":  {
            "X": Xp.tolist(),
            "Y": Yp.tolist(),
            "Z": Zp.tolist()
        },
        "origin": {
            "lat0": float(lat0),
            "lon0": float(lon0),
            "h0":   float(h0)
        },
        "tether_length": tether.length
    }

    with open("trajectory.json", "w") as f:
        json.dump(traj_log, f)
    print("Trajectory written to trajectory.json")



    # Compute altitudes and ENU velocities
    heights = np.array([ecef_to_geodetic_newton(x, y, z)[2] for x, y, z in zip(Xb, Yb, Zb)])
    heights_p = np.array([ecef_to_geodetic_newton(x, y, z)[2] for x, y, z in zip(Xp, Yp, Zp)])
    v_enu = np.array([ecef_to_enu(np.array([vx, vy, vz]).reshape(1, 3), lat0, lon0, h0)[0]
                      for vx, vy, vz in zip(Vbx, Vby, Vbz)])
    speed = np.sqrt(v_enu[:, 0]**2 + v_enu[:, 1]**2 + v_enu[:, 2]**2)

    # Compute tether length in ENU coordinates
    enu_b = np.array([ecef_to_enu(np.array([x, y, z]).reshape(1, 3), lat0, lon0, h0)[0]
                      for x, y, z in zip(Xb, Yb, Zb)])
    enu_p = np.array([ecef_to_enu(np.array([x, y, z]).reshape(1, 3), lat0, lon0, h0)[0]
                      for x, y, z in zip(Xp, Yp, Zp)])
    tether_lengths = np.sqrt(np.sum((enu_b - enu_p)**2, axis=1))
    print("Tether lengths (first 10):", tether_lengths[:10])
    print(f"Tether lengths stats: min={tether_lengths.min():.4f}, max={tether_lengths.max():.4f}, mean={tether_lengths.mean():.4f}")

    # Plot 1: Altitude vs Time
    plt.figure()
    plt.plot(t/60.0, heights, lw=1.5, label='Balloon')
    plt.plot(t/60.0, heights_p, lw=1.5, linestyle='--', label='Payload')
    plt.xlabel('Time [min]')
    plt.ylabel('Altitude [m]')
    plt.title('Balloon and Payload Altitude vs. Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('altitude_vs_time.png')

    # Plot 2: ENU Velocity Components
    fig, axs = plt.subplots(4, 1, figsize=(8, 12), sharex=True)

    # Debug: Print sample v_enu values to verify
    print("Sample v_enu (first 5 points, [East, North, Up]):", v_enu[:5])

    # Explicitly plot East, North, Up components
    axs[0].plot(t/60.0, v_enu[:, 0], color='C0', lw=1.5, label='East')
    axs[0].set_ylabel('East [m/s]')
    axs[0].grid(True)
    axs[0].legend()

    axs[1].plot(t/60.0, v_enu[:, 1], color='C1', lw=1.5, label='North')
    axs[1].set_ylabel('North [m/s]')
    axs[1].grid(True)
    axs[1].legend()

    axs[2].plot(t/60.0, v_enu[:, 2], color='C2', lw=1.5, label='Up')
    axs[2].set_ylabel('Up [m/s]')
    axs[2].grid(True)
    axs[2].legend()

    axs[3].plot(t/60.0, speed, color='k', lw=1.5, label='Speed')
    axs[3].set_ylabel('Speed [m/s]')
    axs[3].set_xlabel('Time [min]')
    axs[3].grid(True)
    axs[3].legend()

    # Set y-axis limits to improve readability (adjust based on data)
    for ax in axs[:3]:
        ax.set_ylim(np.min(v_enu) - 0.5, np.max(v_enu) + 0.5)  # Adjust based on data range
    axs[3].set_ylim(0, np.max(speed) + 0.5)

    fig.suptitle('Balloon Velocity in ENU Coordinates')
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig('velocity_enu.png')

    # Plot 3: Tether Length
    plt.figure()
    plt.plot(t/60.0, tether_lengths, lw=1.5, label='Tether Length')
    plt.axhline(tether.length, color='r', linestyle='--', label='Nominal Length (20 m)')
    plt.ylim(19.9, 20.1)  # Zoom in around 20 m
    plt.xlabel('Time [min]')
    plt.ylabel('Tether Length [m]')
    plt.title('Tether Length vs. Time')
    plt.grid(True)
    plt.legend()
    plt.savefig('tether_length.png')

    # Plot 4: 3D Trajectory in ENU
    enu_traj = np.array([ecef_to_enu(np.array([x, y, z]).reshape(1, 3), lat0, lon0, h0)[0]
                         for x, y, z in zip(Xb, Yb, Zb)])
    enu_traj_p = np.array([ecef_to_enu(np.array([x, y, z]).reshape(1, 3), lat0, lon0, h0)[0]
                           for x, y, z in zip(Xp, Yp, Zp)])
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(enu_traj[:, 0], enu_traj[:, 1], enu_traj[:, 2], lw=2, label='Balloon')
    ax.plot(enu_traj_p[:, 0], enu_traj_p[:, 1], enu_traj_p[:, 2], lw=2, linestyle='--', label='Payload')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('Balloon and Payload Trajectory in ENU')
    ax.legend()
    plt.tight_layout()
    plt.savefig('trajectory_enu.png')

    plt.show()
    print('End of Simulation')
    