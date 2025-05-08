from __future__ import annotations
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, Any
import numpy as np
import xarray as xr
from multiprocessing import Lock
from constants import OPENDAP_URL, ISOBARIC_VARS, SURFACE_VARS, Rd


class GFSDataCache:
    def __init__(self):
        self.full_ds = None  # Full dataset with all forecast times
        self.last_fetch_time = None  # Real-time when dataset was fetched
        self.time_dim = None  # Time dimension name (e.g., 'time')
        self.ds_t = None  # Current time slice
        self.cached_time = None  # Selected forecast time
        self.cached_time_window = None  # 3-hour window for forecast time
        self.tile_bounds = None
        self.ds_4 = None
        self.weights = None
        self.lock = Lock()

    def _update_cycle(self, when: datetime):
        with self.lock:
            print(f"Checking cycle for {when}")
            # Check if we need a new dataset (real-time based)
            current_real_time = datetime.now(timezone.utc)
            if (self.full_ds is None or 
                self.last_fetch_time is None or 
                (current_real_time - self.last_fetch_time) >= timedelta(hours=6)):
                print("Fetching new GFS dataset")
                try:
                    self.full_ds = xr.open_dataset(OPENDAP_URL, decode_times=True)[ISOBARIC_VARS + SURFACE_VARS]
                    self.last_fetch_time = current_real_time
                    print(f"Dataset dimensions: {self.full_ds.dims}")
                    print(f"Dataset variables: {list(self.full_ds.variables)}")
                    if 'height_above_ground2' in self.full_ds.coords:
                        print(f"Height above ground levels: {self.full_ds['height_above_ground2'].values}")
                    self.time_dim = None
                    for dim in self.full_ds.dims:
                        if dim.startswith('time'):
                            self.time_dim = dim
                            break
                    if self.time_dim is None:
                        raise ValueError(f"No time dimension found in dataset. Available dimensions: {self.full_ds.dims}")
                    print(f"Selected time dimension: {self.time_dim}")
                    # Invalidate time slice and tile
                    self.ds_t = None
                    self.cached_time = None
                    self.cached_time_window = None
                    self.tile_bounds = None
                    self.ds_4 = None
                    self.weights = None
                except Exception as e:
                    print(f"Error fetching dataset: {e}")
                    raise

            # Select forecast time closest to 'when'
            time64 = np.datetime64(when.replace(tzinfo=None), "ns")
            # Round to nearest 3-hour window for caching
            when_utc = when.astimezone(timezone.utc)
            time_window = np.datetime64(when_utc.replace(minute=0, second=0, microsecond=0) - timedelta(hours=(when_utc.hour % 3)))
            if self.ds_t is None or self.cached_time_window != time_window:
                print(f"Selecting forecast time for {when}")
                # Validate forecast range
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
                # Invalidate tile cache
                self.tile_bounds = None
                self.ds_4 = None
                self.weights = None

    def _update_tile(self, lat: float, lon: float):
        with self.lock:
            tile = gfs025_tile(lat, lon)
            if tile != self.tile_bounds:
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
                print(f"Tile dataset shape: {ds_4.dims}")
                dy = (lat - lat_lo) / (lat_hi - lat_lo)
                dx = ((lon % 360.0) - lon_lo) / (lon_hi - lon_lo)
                w00 = (1 - dy) * (1 - dx)
                w01 = (1 - dy) * dx
                w10 = dy * (1 - dx)
                w11 = dy * dx
                self.tile_bounds = tile
                self.ds_4 = ds_4
                self.weights = (lat_lo, lat_hi, lon_lo, lon_hi, dy, dx, w00, w01, w10, w11)

    def get_column(self, lat: float, lon: float, when: datetime):
        self._update_cycle(when)
        self._update_tile(lat, lon)
        ds_4 = self.ds_4
        lat_lo, lat_hi, lon_lo, lon_hi, dy, dx, w00, w01, w10, w11 = self.weights

        def collapse(var: str):
            v = ds_4[var].values
            if v.ndim == 3:
                return w00 * v[:, 0, 0] + w01 * v[:, 0, 1] + w10 * v[:, 1, 0] + w11 * v[:, 1, 1]
            else:
                raise ValueError(f"Unexpected shape for {var}: {v.shape}")

        z = collapse("Geopotential_height_isobaric")
        u = collapse("u-component_of_wind_isobaric")
        v = collapse("v-component_of_wind_isobaric")
        w = collapse("Vertical_velocity_geometric_isobaric")
        T = collapse("Temperature_isobaric")
        q = collapse("Specific_humidity_isobaric")

        def collapse_surf(var: str):
            da = ds_4[var]
            for d in da.dims:
                if d not in ("lat", "lon"):
                    da = da.isel({d: 0})
                    break
            m = da.values
            if m.shape == (2, 2):
                return w00 * m[0, 0] + w01 * m[0, 1] + w10 * m[1, 0] + w11 * m[1, 1]
            else:
                raise ValueError(f"Unexpected shape for {var}: {m.shape}")

        u10 = float(collapse_surf("u-component_of_wind_height_above_ground"))
        v10 = float(collapse_surf("v-component_of_wind_height_above_ground"))

        vert_dim = ds_4["Geopotential_height_isobaric"].dims[0]
        p = ds_4.coords[vert_dim].values.astype(float)
        if "pa" in ds_4.coords[vert_dim].attrs.get("units", "").lower():
            p /= 100.0

        if z[0] > z[-1]:
            z, u, v, w, T, q, p = [arr[::-1] for arr in (z, u, v, w, T, q, p)]

        valid_dt = np.datetime64(ds_4[self.time_dim].values).astype("datetime64[ms]").astype(datetime)
        cycle_dt = self.last_fetch_time  # Dataset fetch time

        return z, p, u, v, w, T, q, u10, v10, valid_dt, cycle_dt

# singleton instance
gfs_cache = GFSDataCache()


def rho_from_ptq(p_hPa: float | np.ndarray,
                 T_K: float | np.ndarray,
                 q: float | np.ndarray) -> np.ndarray:
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


def get_forecast(lat: float, lon: float, alt: float, time: datetime, local_tz: ZoneInfo = None) -> Dict[str, Any]:
    if time.tzinfo is None:
        if local_tz is None:
            raise ValueError("datetime passed without `local_tz`")
        time = time.replace(tzinfo=local_tz)
    time_utc = time.astimezone(timezone.utc)

    z, p, u_prof, v_prof, w_prof, T_prof, q_prof, u10, v10, valid_dt, cycle_dt = gfs_cache.get_column(lat, lon, time_utc)

    # Debug shapes to ensure 1D profiles
    # print(f"u_prof shape: {u_prof.shape}, v_prof shape: {v_prof.shape}, w_prof shape: {w_prof.shape}")

    z0 = 0.03
    z_low = 100.0
    z_high = float(z[0])

    def log_wind(u_ref, z_ref, z):
        return u_ref * np.log(z / z0) / np.log(z_ref / z0)

    u_log = log_wind(u10, 10.0, alt) if alt > z0 else 0.0
    v_log = log_wind(v10, 10.0, alt) if alt > z0 else 0.0

    u_int = interp_at(z, u_prof, alt)
    v_int = interp_at(z, v_prof, alt)

    if alt <= z_low:
        u_at, v_at = u_log, v_log
    elif alt >= z_high:
        u_at, v_at = u_int, v_int
    else:
        frac = (alt - z_low) / (z_high - z_low)
        u_at = (1 - frac) * u_log + frac * u_int
        v_at = (1 - frac) * v_log + frac * v_int

    w_at = interp_at(z, w_prof, alt)
    T_at = interp_at(z, T_prof, alt)
    p_at = interp_at(z, p, alt)
    q_at = interp_at(z, q_prof, alt)
    rho_at = rho_from_ptq(p_at, T_at, q_at)

    # Debug scalar values
    # print(f"u_at: {u_at}, v_at: {v_at}, w_at: {w_at}")
    # print(f"GFS cycle      : {cycle_dt:%Y-%m-%d %H:%MZ}")
    # print(f"Forecast valid : {valid_dt:%Y-%m-%d %H:%MZ}")
    # print(f"Wind 10 m AGL  : u10={u10:.2f} m/s  v10={v10:.2f} m/s")
    # print(f"Wind @ {alt/1000:.3f} km : "
    #       f"u={u_at:.2f} m/s  v={v_at:.2f} m/s  w={w_at:.2f} m/s")
    # print(f"Temperature    : {T_at:.2f} K ({T_at-273.15:.2f} °C)")
    # print(f"Pressure       : {p_at:.3f} hPa")
    # print(f"Density        : {rho_at:.4f} kg m⁻³")
    
    return {
        "u": u_at, "v": v_at, "w": w_at,
        "T": T_at, "p": p_at, "q": q_at, "rho": rho_at,
        "u10": u10, "v10": v10,
        "valid_dt": valid_dt, "cycle_dt": cycle_dt,
    }


if __name__ == "__main__":
    now_pdt = datetime.now(ZoneInfo("America/Los_Angeles"))
    print("Current PDT time:", now_pdt)
    fc = get_forecast(45.49, -122.7, 250, now_pdt)
    print(fc)
