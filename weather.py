from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any 
import sys
import numpy as np
import xarray as xr

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

Rd = 287.05  # J kg‑1 K‑1


def rho_from_ptq(p_hPa: float | np.ndarray,
                 T_K: float | np.ndarray,
                 q: float | np.ndarray) -> np.ndarray:
    """
    Moist-air density rho [kg m-3] from pressure p [hPa], temperature T [K],
    specific humidity q [kg kg-1].
    """
    Tv = T_K * (1.0 + 0.61 * q)
    return (p_hPa * 100.0) / (Rd * Tv)


def gfs025_tile(lat: float, lon: float):
    """Return the bounding 0.25° tile around (lat,lon) and its four corners."""
    lat_step = lon_step = 0.25
    lon = lon % 360.0
    lat_lo = np.floor(lat / lat_step) * lat_step
    lon_lo = np.floor(lon / lon_step) * lon_step
    lat_lo = max(-90.0, lat_lo)
    lat_hi = min(90.0, lat_lo + lat_step)
    return {
        "lat_lo": lat_lo,
        "lat_hi": lat_hi,
        "lon_lo": lon_lo,
        "lon_hi": lon_lo + lon_step,
    }


def get_latest_cycle(dt: datetime) -> datetime:
    """Round UTC time down to the latest 00/06/12/18 Z synoptic cycle."""
    return dt.replace(hour=(dt.hour // 6) * 6,
                      minute=0, second=0, microsecond=0)


def get_column(lat: float, lon: float, when: datetime):
    """
    Bilinearly-interpolated GFS column at (lat,lon).
    Returns z, p, u, v, w, T, q profiles, 10 m winds (u10,v10) and valid time.
    """
    ds = xr.open_dataset(OPENDAP_URL,
                         decode_times=True)[ISOBARIC_VARS + SURFACE_VARS]

    time64 = np.datetime64(when.replace(tzinfo=None), "ns")
    ds_t = ds.sel(time=time64, method="nearest",
                  tolerance=np.timedelta64(3, "h"))

    tile = gfs025_tile(lat, lon)
    lat_lo, lat_hi = tile["lat_lo"], tile["lat_hi"]
    lon_lo, lon_hi = tile["lon_lo"], tile["lon_hi"]

    ds_4 = ds_t.sel(lat=[lat_lo, lat_hi], lon=[lon_lo, lon_hi]).sortby(["lat", "lon"])

    dy = (lat - lat_lo) / (lat_hi - lat_lo)
    dx = ((lon % 360.0) - lon_lo) / (lon_hi - lon_lo)
    w00 = (1 - dy) * (1 - dx)
    w01 = (1 - dy) * dx
    w10 = dy * (1 - dx)
    w11 = dy * dx

    print(f"Bilinear box  : [{lat_lo:.2f}, {lat_hi:.2f}]° x "
          f"[{lon_lo:.2f}, {lon_hi:.2f}]°  "
          f"weights (w00…w11)={w00:.2f},{w01:.2f},{w10:.2f},{w11:.2f}")

    def collapse(var: str):
        v = ds_4[var].values
        return (
            w00 * v[:, 0, 0] +
            w01 * v[:, 0, 1] +
            w10 * v[:, 1, 0] +
            w11 * v[:, 1, 1]
        )

    z = collapse("Geopotential_height_isobaric")          # m
    u = collapse("u-component_of_wind_isobaric")          # m s‑1
    v = collapse("v-component_of_wind_isobaric")         # m s‑1
    w = collapse("Vertical_velocity_geometric_isobaric")  # m s‑1
    T = collapse("Temperature_isobaric")                  # K
    q = collapse("Specific_humidity_isobaric")            # kg/kg

    def collapse_surface(var):
        da = ds_4[var]
        # pick the only non‑lat/lon dimension (length 1)
        for d in da.dims:
            if d not in ("lat", "lon"):
                da = da.isel({d: 0})
                break
        v = da.values            # shape (2, 2)
        return (w00 * v[0, 0] + w01 * v[0, 1] +
                w10 * v[1, 0] + w11 * v[1, 1])

    u10 = float(collapse_surface("u-component_of_wind_height_above_ground"))
    v10 = float(collapse_surface("v-component_of_wind_height_above_ground"))
       


    vert_dim = ds_4["Geopotential_height_isobaric"].dims[0]
    p = ds_4.coords[vert_dim].values.astype(float)        # hPa by attr check
    if "pa" in ds_4.coords[vert_dim].attrs.get("units", "").lower():
        p /= 100.0

    if z[0] > z[-1]:
        z, u, v, w, T, q, p = (arr[::-1] for arr in (z, u, v, w, T, q, p))

    valid_dt = np.datetime64(ds_4.time.values).astype("datetime64[ms]").astype(datetime)
    return z, p, u, v, w, T, q, u10, v10, valid_dt


def interp_at(z: np.ndarray, prof: np.ndarray, alt_m: float) -> float:
    """1-D linear interpolation helper."""
    return np.interp(alt_m, z, prof)



def get_forecast(lat: float, lon: float, alt: float, time: datetime, local_tz: ZoneInfo = None) -> Dict[str, Any]:
    
    # get latest forcast time available in UTC
    if time.tzinfo is None:
        if local_tz is None:
            raise ValueError('datetime passed without `local_tz`')
        else:
            time = time.replace(tzinfo=local_tz)
    time_utc = time.astimezone(timezone.utc)
    latest_cycle = get_latest_cycle(time_utc)
    
    z, p, u, v, w, T, q, u10, v10, valid_dt = get_column(lat, lon, time_utc)
           
    # surface-layer parameters
    z0 = 0.03
    z_low = 100.0
    z_high = float(z[0])
    
    # compute log-profile wind (normalized at 10 m)
    def log_wind(u_ref, z_ref, z):
        return u_ref * np.log(z / z0) / np.log(z_ref / z0)
    u_log = log_wind(u10, 10.0, alt) if alt > z0 else 0.0
    v_log = log_wind(v10, 10.0, alt) if alt > z0 else 0.0
    
    # model-interpolated wind
    u_int = interp_at(z, u, alt)
    v_int = interp_at(z, v, alt)
    
    # blend over [z_low, z_high] to avoid discontinuity
    if alt <= z_low:
        u_at, v_at = u_log, v_log
    elif alt >= z_high:
        u_at, v_at = u_int, v_int
    else:
        frac = (alt - z_low) / (z_high - z_low)
        u_at = (1 - frac) * u_log + frac * u_int
        v_at = (1 - frac) * v_log + frac * v_int
    
    # interpolate profiles to requested geometric altitude    
    w_at = interp_at(z, w, alt)
    T_at = interp_at(z, T, alt)
    p_at = interp_at(z, p, alt)
    q_at = interp_at(z, q, alt)
    rho_at = rho_from_ptq(p_at, T_at, q_at)
    
    print(f"GFS cycle      : {latest_cycle:%Y-%m-%d %H:%MZ}")
    print(f"Forecast valid : {valid_dt:%Y-%m-%d %H:%MZ}")
    print(f"Wind 10 m AGL  : u10={u10:.2f} m/s  v10={v10:.2f} m/s")
    print(f"Wind @ {alt/1000:.3f} km : "
          f"u={u_at:.2f} m/s  v={v_at:.2f} m/s  w={w_at:.2f} m/s")
    print(f"Temperature    : {T_at:.2f} K ({T_at-273.15:.2f} °C)")
    print(f"Pressure       : {p_at:.3f} hPa")
    print(f"Density        : {rho_at:.4f} kg m⁻³")
    
    return {
        "u": u_at,
        "v": v_at,
        "w": w_at,
        "T": T_at,
        "p": p_at,
        "q": q_at,
        "rho": rho_at,
        "u10": u10,
        "v10": v10,
        "valid_dt": valid_dt,
        "cycle_dt": latest_cycle,
    }
    


if __name__ == "__main__":
    now_pdt = datetime.now()
    print("Current PDT time:", now_pdt)
    #get_forecast(45.49,-122.7,2500,datetime.now(timezone.utc))
    get_forecast(45.49,-122.7,250,now_pdt,ZoneInfo("America/Los_Angeles"))
