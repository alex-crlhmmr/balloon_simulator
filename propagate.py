from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from weather import get_forecast
from geo_utils import ecef_to_geodetic_newton, geodetic_to_ecef, enu_vector_to_ecef
from gas_utils import sutherland_viscosity
from scipy.integrate import solve_ivp
from scipy.integrate import RK23
import numpy as np
from typing import Dict, List, Any, Union
from constants import R_UNIVERSAL, GAS_DATA, g, RE
from dataclasses import dataclass
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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
        return self.gas.mass * self.gas.R_specific * T / P
    
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
        return -0.5*C_d*rho*A*v_rel_norm*v_rel
    
    def buoyant_force(self, 
                      rho: float, 
                      T: float, 
                      P: float) -> np.ndarray[float]:
        V = self.get_volume(T, P)
        return np.array([0.0, 0.0, 1.0]) * (rho * V * g)
    
    def gravity_force(self, 
                      rho: float, 
                      T: float, 
                      P: float) -> np.ndarray[float]:
        total_mass = self.get_total_mass(rho, T, P)
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
    
    def gravity_force(self) -> np.ndarray[float]:
        total_mass = self.get_total_mass()
        return - total_mass * g * np.array([0.0, 0.0, 1.0])


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
    
    def __call__(self, t, state):
        return get_statedot(state, t, self.t0,
                            balloon=self.balloon,
                            payload=self.payload,
                            tether=self.tether)



def get_statedot(state: np.ndarray, t: datetime, t0: datetime, balloon: Balloon, payload: Payload, tether: Tether) -> np.ndarray:
    
    current_time = t0 + timedelta(seconds=t)
    
    x_b, v_b = state[0:3], state[3:6]
    x_p, v_p = state[6:9], state[9:12]
    
    lat_b, lon_b, h_b = ecef_to_geodetic_newton(*x_b)
    lat_p, lon_p, h_p = ecef_to_geodetic_newton(*x_p)
    
    
    # f_b = get_forecast(lat_b, lon_b, h_b, current_time)
    # f_p = get_forecast(lat_p, lon_p, h_p, current_time)
    
    print(f"h_b={h_b}, h_p:{h_p}")
    fake_forecast = {
        "u": -0.26,
        "v": -3.82,
        "w": 0.01,
        "T": 279.52,
        "p": 993.088,
        "rho": 1.2343,
    }
    f_b = fake_forecast
    f_p = fake_forecast
    
    wind_b = enu_vector_to_ecef(np.array([f_b["u"], f_b["v"], f_b["w"]]), lat_b, lon_b)
    wind_p = enu_vector_to_ecef(np.array([f_p["u"], f_p["v"], f_p["w"]]), lat_p, lon_p)
    
    vrel_b = v_b - wind_b
    vrel_p = v_p - wind_p
    
    Fb = ( balloon.buoyant_force(f_b["rho"], f_b["T"], f_b["p"]*100)
         + balloon.gravity_force(f_b["rho"], f_b["T"], f_b["p"]*100)
         + balloon.drag_force(f_b["rho"], f_b["T"], vrel_b))
    Fp = ( payload.drag_force(f_p["rho"], f_p["T"], vrel_p)
         + payload.gravity_force() )
    
    mb = balloon.get_total_mass(f_b["rho"], f_b["T"], f_b["p"]*100)
    mp = payload.get_total_mass()
    
    d_vec = x_p - x_b
    L = tether.length
    t_hat = d_vec / L
    dv2 = np.dot(v_p - v_b, v_p - v_b)
    
    num = dv2 + np.dot(d_vec, (Fp/mp - Fb/mb))
    den = L * (1.0/mp + 1.0/mb)
    Tmag = num / den
    
    a_b = (Fb +  Tmag * t_hat) / mb
    a_p = (Fp -  Tmag * t_hat) / mp
    
    print(f"a_b={a_b}, a_p:{a_p}")
    
    statedot = np.zeros_like(state)
    statedot[0:3] = v_b
    statedot[3:6] = a_b
    statedot[6:9] = v_p
    statedot[9:12] = a_p
     
    return statedot 



# if __name__ == '__main__':
#     print('Start of Simulation')
    
#     balloon = Balloon(
#         radius=1.5,            
#         envelope_mass=0.5,     
#         gas="helium",
#         gas_mass=0.67            
#     )
    
#     payload = Payload(
#         radius=0.2,             
#         length=0.5,             
#         mass=1.2               
#     )
    
    
#     tether = Tether(length=10.0)
    
#     t0 = datetime.now(ZoneInfo("UTC"))
#     system = System(balloon, payload, tether, t0)
    
#     # lat0, lon0, h0 = np.deg2rad(40.0), np.deg2rad(-75.0), tether.length
#     lat0, lon0, h0 = np.deg2rad(40.0), np.deg2rad(-75.0), 30
#     x0_ecef = geodetic_to_ecef(lat0, lon0, h0)
    
#     x_b0 = x0_ecef
#     # x_p0 = geodetic_to_ecef(lat0, lon0, 0)
#     x_p0 = geodetic_to_ecef(lat0, lon0, 30-tether.length)
    
#     v_b0 = np.zeros(3)
#     v_p0 = np.zeros(3)
    
#     y0 = np.hstack([x_b0, v_b0, x_p0, v_p0])
    
#     t_span = (0.0, 10*3600.0)
#     t_eval = np.linspace(0.0, 10*3600.0, 360)
    
#     sol = solve_ivp(
#         fun=system, 
#         t_span=t_span, 
#         y0=y0, 
#         method="RK23",      #"RK45" 
#         t_eval=t_eval, 
#         rtol=1e-6, 
#         atol=1e-9
#     )
    
#    # unpack balloon positions in ECEF
#     Xb = sol.y[0]
#     Yb = sol.y[1]
#     Zb = sol.y[2]

#     # convert final point back to geodetic just for printing
#     x_b_final = sol.y[0:3, -1]
#     lat_b_f, lon_b_f, h_b_f = ecef_to_geodetic_newton(*x_b_final)
#     print(f"Final balloon height: {h_b_f:.1f} m")

#     # --- 1) plot height vs time ----------------------------------------------
#     # first convert every (X,Y,Z) back to height
#     heights = []
#     for x, y, z in zip(Xb, Yb, Zb):
#         _, _, h = ecef_to_geodetic_newton(x, y, z)
#         heights.append(h)
#     heights = np.array(heights)

#     plt.figure()
#     plt.plot(sol.t / 60.0, heights)
#     plt.xlabel('Time [min]')
#     plt.ylabel('Altitude [m]')
#     plt.title('Balloon Altitude vs. Time')
#     plt.grid(True)

#     # --- 2) 3D trajectory recentered on initial position ---------------------
#     # subtract the very first ECEF point so it sits at the origin
#     X0, Y0, Z0 = Xb[0], Yb[0], Zb[0]
#     Xr, Yr, Zr = Xb - X0, Yb - Y0, Zb - Z0

#     fig = plt.figure(figsize=(8,6))
#     ax = fig.add_subplot(111, projection='3d')
#     ax.plot(Xr, Yr, Zr, lw=2, label='Trajectory')

#     # draw a small reference sphere at the origin (the launch point)
#     # here we draw a tiny sphere of radius equal to your tether length
#     u = np.linspace(0, 2*np.pi, 30)
#     v = np.linspace(0, np.pi, 15)
#     U, V = np.meshgrid(u, v)
#     r_ref = tether.length * 1.1  # just for scale
#     Xs = r_ref * np.cos(U)*np.sin(V)
#     Ys = r_ref * np.sin(U)*np.sin(V)
#     Zs = r_ref * np.cos(V)
#     ax.plot_surface(Xs, Ys, Zs, color='lightgray', alpha=0.3, linewidth=0)

#     ax.set_xlabel('ΔX (m)')
#     ax.set_ylabel('ΔY (m)')
#     ax.set_zlabel('ΔZ (m)')
#     ax.set_title('Recentered Balloon Trajectory')
#     ax.legend()
#     plt.tight_layout()
#     plt.show()



if __name__ == '__main__':
    print('Start of Simulation')
    
    # — set up your system as before —
    balloon = Balloon(radius=1.5, envelope_mass=0.5, gas="helium", gas_mass=0.67)
    payload = Payload(radius=0.2, length=0.5, mass=1.2)
    tether  = Tether(length=10.0)
    t0 = datetime.now(ZoneInfo("UTC"))
    system = System(balloon, payload, tether, t0)

    # initial positions/velocities
    lat0, lon0, h0 = np.deg2rad(40.0), np.deg2rad(-75.0), 30
    x0 = geodetic_to_ecef(lat0, lon0, h0)
    y0 = np.hstack([x0, np.zeros(3),
                    geodetic_to_ecef(lat0, lon0, h0 - tether.length),
                    np.zeros(3)])

    # integrate
    t_span = (0.0, 24*3600.0)
    t_eval = np.linspace(0.0, 24*3600.0, 360)
    sol = solve_ivp(system, t_span, y0,
                    method="RK23", t_eval=t_eval,
                    rtol=1e-6, atol=1e-9)

    # unpack balloon trajectory and compute speed + accel
    Xb, Yb, Zb = sol.y[0], sol.y[1], sol.y[2]
    Vbx, Vby, Vbz = sol.y[3], sol.y[4], sol.y[5]
    t = sol.t
    speed = np.sqrt(Vbx**2 + Vby**2 + Vbz**2)

    # approximate accel by finite diff
    abx = np.gradient(Vbx, t)
    aby = np.gradient(Vby, t)
    abz = np.gradient(Vbz, t)

    # — 1) Altitude vs time —
    heights = np.array([ecef_to_geodetic_newton(x,y,z)[2]
                        for x,y,z in zip(Xb, Yb, Zb)])
    plt.figure()
    plt.plot(t/60.0, heights, lw=1.5)
    plt.xlabel('Time [min]')
    plt.ylabel('Altitude [m]')
    plt.title('Balloon Altitude vs. Time')
    plt.grid(True)

    # — 2) Acceleration components —
    fig, axs = plt.subplots(3, 1, figsize=(6, 8), sharex=True)
    comp = ['aₓ', 'aᵧ', 'a_z']
    for i, data in enumerate((abx, aby, abz)):
        axs[i].plot(t/60.0, data, color=f'C{i}', lw=1.5)
        axs[i].set_ylabel(f'{comp[i]} [m/s²]')
        axs[i].grid(True)
    axs[-1].set_xlabel('Time [min]')
    fig.suptitle('Balloon Acceleration Components')
    plt.tight_layout(rect=[0,0,1,0.96])

    # — 3) Velocity components + speed —
    fig, axs = plt.subplots(4, 1, figsize=(6, 10), sharex=True)
    comp_v = ['vₓ', 'vᵧ', 'v_z']
    for i, data in enumerate((Vbx, Vby, Vbz)):
        axs[i].plot(t/60.0, data, color=f'C{i}', lw=1.5)
        axs[i].set_ylabel(f'{comp_v[i]} [m/s]')
        axs[i].grid(True)
    axs[3].plot(t/60.0, speed, color='k', lw=1.5)
    axs[3].set_ylabel('|v| [m/s]')
    axs[3].set_xlabel('Time [min]')
    axs[3].grid(True)
    fig.suptitle('Balloon Velocity Components & Speed')
    plt.tight_layout(rect=[0,0,1,0.96])

    # — 4) Recentered 3D trajectory —
    X0, Y0, Z0 = Xb[0], Yb[0], Zb[0]
    Xr, Yr, Zr = Xb - X0, Yb - Y0, Zb - Z0
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(Xr, Yr, Zr, lw=2, label='Trajectory')
    # reference sphere at launch
    u = np.linspace(0,2*np.pi,30)
    v = np.linspace(0,np.pi,15)
    U, V = np.meshgrid(u,v)
    r_ref = tether.length * 1.1
    Xs = r_ref*np.cos(U)*np.sin(V)
    Ys = r_ref*np.sin(U)*np.sin(V)
    Zs = r_ref*np.cos(V)
    ax.plot_surface(Xs, Ys, Zs, color='lightgray', alpha=0.3, linewidth=0)
    ax.set_xlabel('ΔX (m)'); ax.set_ylabel('ΔY (m)'); ax.set_zlabel('ΔZ (m)')
    ax.set_title('Recentered Balloon Trajectory')
    ax.legend()
    plt.tight_layout()
    plt.show()

