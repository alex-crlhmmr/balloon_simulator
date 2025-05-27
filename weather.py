from __future__ import annotations
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any
import numpy as np
import xarray as xr
from multiprocessing import Lock
from constants import OPENDAP_URL, ISOBARIC_VARS, SURFACE_VARS, Rd
import time
import os

class GFSDataCache:
    def __init__(self):
        self.full_ds = None
        self.last_fetch_time = None
        self.time_dim = None
        self.ds_t = None
        self.cached_time = None
        self.cached_time_window = None
        self.tile_bounds = None
        self.ds_4 = None
        self.lock = Lock()
        self.was_tile_reused = False
        self.bilinear_cache = {}  # Cache for bilinear-interpolated columns
        self.bilinear_cache_max_size = 1000  # Limit cache size

    def _update_cycle(self, when: datetime):
        with self.lock:
            start = time.time()
            current_real_time = datetime.now(timezone.utc)
            if (self.full_ds is None or 
                self.last_fetch_time is None or 
                (current_real_time - self.last_fetch_time) >= timedelta(hours=6)):
                print("Fetching new GFS dataset")
                try:
                    self.full_ds = xr.open_dataset(OPENDAP_URL, decode_times=True)[ISOBARIC_VARS + SURFACE_VARS]
                    self.last_fetch_time = current_real_time
                    self.time_dim = None
                    for dim in self.full_ds.dims:
                        if dim.startswith('time'):
                            self.time_dim = dim
                            break
                    if self.time_dim is None:
                        raise ValueError(f"No time dimension found in dataset. Available dimensions: {self.full_ds.dims}")
                    self.ds_t = None
                    self.cached_time = None
                    self.cached_time_window = None
                    self.tile_bounds = None
                    self.ds_4 = None
                    self.bilinear_cache = {}
                except Exception as e:
                    print(f"Error fetching dataset: {e}")
                    raise
            # print(f"_update_cycle took {time.time() - start:.4f} s")

            start = time.time()
            time64 = np.datetime64(when.replace(tzinfo=None), "ns")
            when_utc = when.astimezone(timezone.utc)
            time_window = np.datetime64(when_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=(when_utc.hour % 3)))
            if self.ds_t is None or self.cached_time_window != time_window:
                print(f"Selecting forecast time for {when}")
                time_values = self.full_ds[self.time_dim].values
                if not any(np.abs(time_values - time64) <= np.timedelta64(3, "h")):
                    raise ValueError(f"No forecast available for {when}. Latest forecast: {time_values[-1]}")
                self.ds_t = self.full_ds.sel(
                    {self.time_dim: time64},
                    method="nearest",
                    tolerance=np.timedelta64(3, "h")
                )
                self.cached_time = time64
                self.cached_time_window = time_window
                print(f"Selected forecast time: {self.ds_t[self.time_dim].values}")
                self.tile_bounds = None
                self.ds_4 = None
                self.bilinear_cache = {}
            # print(f"Time slice selection took {time.time() - start:.4f} s")

    def _update_tile(self, lat: float, lon: float):
        with self.lock:
            start = time.time()
            tile = gfs025_tile(lat, lon)
            if tile != self.tile_bounds or self.ds_4 is None:
                print(f"Updating tile for lat={lat}, lon={lon}")
                self.was_tile_reused = False
                if 'lat' not in self.ds_t.dims or 'lon' not in self.ds_t.dims:
                    raise ValueError(f"Missing 'lat' or 'lon' dimensions in dataset. Available dimensions: {self.ds_t.dims}")
                lat_lo, lat_hi = tile["lat_lo"], tile["lat_hi"]
                lon_lo, lon_hi = tile["lon_lo"], tile["lon_hi"]
                available_lats = self.ds_t['lat'].values
                available_lons = self.ds_t['lon'].values
                if not (np.any(np.isclose(available_lats, lat_lo)) and np.any(np.isclose(available_lats, lat_hi)) and
                        np.any(np.isclose(available_lons, lon_lo)) and np.any(np.isclose(available_lons, lon_hi))):
                    raise ValueError(f"Tile coordinates {tile} not found in dataset coordinates: lat={available_lats[:10]}..., lon={available_lons[:10]}...")
                ds_4 = self.ds_t.sel(
                    lat=[lat_lo, lat_hi],
                    lon=[lon_lo, lon_hi]
                ).sortby(["lat", "lon"])
                self.tile_bounds = tile
                self.ds_4 = ds_4
                self.bilinear_cache = {}
            else:
                # print(f"Reusing cached tile for lat={lat}, lon={lon}")
                self.was_tile_reused = True
            # print(f"_update_tile took {time.time() - start:.4f} s")

    def get_column(self, lat: float, lon: float, when: datetime):
        start = time.time()
        self._update_cycle(when)
        self._update_tile(lat, lon)
        ds_4 = self.ds_4
        tile = gfs025_tile(lat, lon)
        lat_lo, lat_hi = tile["lat_lo"], tile["lat_hi"]
        lon_lo, lon_hi = tile["lon_lo"], tile["lon_hi"]
        
        # Compute weights for this lat/lon
        dy = (lat - lat_lo) / (lat_hi - lat_lo)
        dx = ((lon % 360.0) - lon_lo) / (lon_hi - lon_lo)
        w00 = (1 - dy) * (1 - dx)
        w01 = (1 - dy) * dx
        w10 = dy * (1 - dx)
        w11 = dy * dx
        # print(f"Weights for lat={lat:.6f}, lon={lon:.6f}: w00={w00:.4f}, w01={w01:.4f}, w10={w10:.4f}, w11={w11:.4f}")
        was_tile_reused = self.was_tile_reused

        # Check bilinear cache for nearby lat/lon (within 0.001°)
        cache_key = (round(lat, 3), round(lon, 3), self.cached_time_window)
        if cache_key in self.bilinear_cache:
            # print(f"Using bilinear cache for lat={lat:.6f}, lon={lon:.6f} (cached lat={cache_key[0]:.6f}, lon={cache_key[1]:.6f})")
            z, p, u, v, w, T, q, u_surface, v_surface = self.bilinear_cache[cache_key]
            # print(f"Bilinear cache retrieval took {time.time() - start:.4f} s")
            return z, p, u, v, w, T, q, u_surface, v_surface, ds_4[self.time_dim].values, self.last_fetch_time, was_tile_reused

        def collapse(var: str):
            # start_collapse = time.time()
            v = ds_4[var].values
            if v.ndim == 3:
                result = w00 * v[:, 0, 0] + w01 * v[:, 0, 1] + w10 * v[:, 1, 0] + w11 * v[:, 1, 1]
                # print(f"Interpolation for {var} took {time.time() - start_collapse:.4f} s")
                return result
            else:
                raise ValueError(f"Unexpected shape for {var}: {v.shape}")

        z = collapse("Geopotential_height_isobaric")
        u = collapse("u-component_of_wind_isobaric")
        v = collapse("v-component_of_wind_isobaric")
        w = collapse("Vertical_velocity_geometric_isobaric")
        T = collapse("Temperature_isobaric")
        q = collapse("Specific_humidity_isobaric")

        def collapse_surf(var: str, height=None):
            # start_collapse = time.time()
            da = ds_4[var]
            if height is not None:
                try:
                    idx = np.where(ds_4['height_above_ground2'].values == height)[0][0]
                    da = da.isel({ds_4['height_above_ground2'].dims[0]: idx})
                except IndexError:
                    raise ValueError(f"Height {height} not found in height_above_ground2")
            else:
                da = da.isel({ds_4['height_above_ground2'].dims[0]: 0})
            m = da.values
            if m.shape == (2, 2):
                result = w00 * m[0, 0] + w01 * m[0, 1] + w10 * m[1, 0] + w11 * m[1, 1]
                # print(f"Interpolation for {var} at height {height} took {time.time() - start_collapse:.4f} s")
                return result
            else:
                raise ValueError(f"Unexpected shape for {var}: {m.shape}")

        heights = [10, 20, 30, 40, 50, 80, 100]
        u_surface = [float(collapse_surf("u-component_of_wind_height_above_ground", height=h)) for h in heights]
        v_surface = [float(collapse_surf("v-component_of_wind_height_above_ground", height=h)) for h in heights]

        vert_dim = ds_4["Geopotential_height_isobaric"].dims[0]
        p = ds_4.coords[vert_dim].values.astype(float)
        if "pa" in ds_4.coords[vert_dim].attrs.get("units", "").lower():
            p /= 100.0

        if z[0] > z[-1]:
            z, u, v, w, T, q, p = [arr[::-1] for arr in (z, u, v, w, T, q, p)]

        valid_dt = np.datetime64(ds_4[self.time_dim].values).astype("datetime64[ms]").astype(datetime)
        cycle_dt = self.last_fetch_time

        # Store in bilinear cache
        if len(self.bilinear_cache) >= self.bilinear_cache_max_size:
            self.bilinear_cache.pop(next(iter(self.bilinear_cache)))
        self.bilinear_cache[cache_key] = (z, p, u, v, w, T, q, u_surface, v_surface)
        # print(f"get_column took {time.time() - start:.4f} s")
        # print(f"Interpolated u[0]={u[0]:.4f}, v[0]={v[0]:.4f} at lat={lat:.6f}, lon={lon:.6f}")
        return z, p, u, v, w, T, q, u_surface, v_surface, valid_dt, cycle_dt, was_tile_reused

# singleton instance
gfs_cache = GFSDataCache()

def rho_from_ptq(p_hPa: float | np.ndarray, T_K: float | np.ndarray, q: float | np.ndarray) -> np.ndarray:
    Tv = T_K * (1 + 0.61 * q)
    return (p_hPa * 100.0) / (Rd * Tv)

def gfs025_tile(lat: float, lon: float):
    lat_step = lon_step = 0.25
    lon = lon % 360.0
    lat_lo = np.floor(lat / lat_step) * lat_step
    lon_lo = np.floor(lon / lon_step) * lon_step
    lat_lo = max(-90.0, lat_lo)
    lat_hi = min(90.0, lat_lo + lat_step)
    return {"lat_lo": lat_lo, "lat_hi": lat_hi, "lon_lo": lon_lo, "lon_hi": lon_lo + lon_step}

def get_latest_cycle(dt: datetime) -> datetime:
    return dt.replace(hour=(dt.hour // 6) * 6, minute=0, second=0, microsecond=0)

def interp_at(z: np.ndarray, prof: np.ndarray, alt_m: float) -> float:
    return np.interp(alt_m, z, prof)

def get_forecast(lat: float, lon: float, alt: float, when: datetime, local_tz: ZoneInfo = None) -> Dict[str, Any]:
    if when.tzinfo is None:
        if local_tz is None:
            raise ValueError("datetime passed without `local_tz`")
        when = when.replace(tzinfo=local_tz)
    time_utc = when.astimezone(timezone.utc)

    # start = time.time()
    z, p, u_prof, v_prof, w_prof, T_prof, q_prof, u_surface, v_surface, valid_dt, cycle_dt, was_tile_reused = gfs_cache.get_column(lat, lon, time_utc)
    
    # save in a cvs file z, p, u_prof, v_prof, w_prof, T_prof, q_prof
    # with open("gfs_data.csv", "w") as f:
    #     f.write("z,p,u_prof,v_prof,w_prof,T_prof,q_prof\n")
    #     for i in range(len(z)):
    #         f.write(f"{z[i]},{p[i]},{u_prof[i]},{v_prof[i]},{w_prof[i]},{T_prof[i]},{q_prof[i]}\n")
    
    # column_time = time.time() - start
    # print(f"GFSDataCache.get_column took {column_time:.4f} s")

    z_low = 100.0
    z_high = float(z[0])
    surface_heights = [10, 20, 30, 40, 50, 80, 100]

    # start_interp = time.time()
    def interp_surface(alt, heights, values):
        if alt <= heights[0]:
            return values[0]
        return np.interp(alt, heights, values)

    if alt <= z_low:
        u_at = interp_surface(alt, surface_heights, u_surface)
        v_at = interp_surface(alt, surface_heights, v_surface)
    elif alt < z_high:
        frac = (alt - z_low) / (z_high - z_low)
        u_at = (1 - frac) * u_surface[-1] + frac * interp_at(z, u_prof, alt)
        v_at = (1 - frac) * v_surface[-1] + frac * interp_at(z, v_prof, alt)
    else:
        u_at = interp_at(z, u_prof, alt)
        v_at = interp_at(z, v_prof, alt)

    w_at = interp_at(z, w_prof, alt)
    T_at = interp_at(z, T_prof, alt)
    p_at = interp_at(z, p, alt)
    q_at = interp_at(z, q_prof, alt)
    rho_at = rho_from_ptq(p_at, T_at, q_at)
    # interp_time = time.time() - start_interp
    # print(f"Interpolation in get_forecast took {interp_time:.4f} s")
    # print(f"Final u={u_at:.4f}, v={v_at:.4f} at lat={lat:.6f}, lon={lon:.6f}, alt={alt:.1f}")

    # total_time = time.time() - start
    # print(f"get_forecast took {total_time:.4f} s")
    return {
        "u": u_at, "v": v_at, "w": w_at,
        "T": T_at, "p": p_at, "q": q_at, "rho": rho_at,
        "u10": u_surface[0], "v10": v_surface[0],
        "valid_dt": valid_dt, "cycle_dt": cycle_dt,
        "was_tile_reused": was_tile_reused
    }
