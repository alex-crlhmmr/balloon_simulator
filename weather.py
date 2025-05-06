from __future__ import annotations
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
from typing import Dict, Any
import numpy as np
import xarray as xr

from constants import OPENDAP_URL, ISOBARIC_VARS, SURFACE_VARS, Rd


class GFSDataCache:
    """Two-level cache: synoptic cycle + 0.25° tile."""

    def __init__(self):
        self.cached_cycle = None      # datetime of last loaded cycle
        self.ds_t         = None      # sliced dataset at that cycle
        self.tile_bounds  = None      # dict of lat/lon corners
        self.ds_4         = None      # 4-cell subset
        self.weights      = None      # bilinear weights

    def _update_cycle(self, when: datetime):
        cycle = get_latest_cycle(when)
        if cycle != self.cached_cycle:
            # load once per new cycle
            full_ds = xr.open_dataset(OPENDAP_URL, decode_times=True)[ISOBARIC_VARS + SURFACE_VARS]
            time64  = np.datetime64(cycle.replace(tzinfo=None), "ns")
            self.ds_t = full_ds.sel(
                time=time64,
                method="nearest",
                tolerance=np.timedelta64(3, "h")
            )
            self.cached_cycle = cycle
            # invalidate tile cache
            self.tile_bounds = None
            self.ds_4        = None
            self.weights     = None

    def _update_tile(self, lat: float, lon: float):
        tile = gfs025_tile(lat, lon)
        if tile != self.tile_bounds:
            lat_lo, lat_hi = tile["lat_lo"], tile["lat_hi"]
            lon_lo, lon_hi = tile["lon_lo"], tile["lon_hi"]
            ds_4 = self.ds_t.sel(
                lat=[lat_lo, lat_hi],
                lon=[lon_lo, lon_hi]
            ).sortby(["lat", "lon"])

            dy = (lat - lat_lo) / (lat_hi - lat_lo)
            dx = ((lon % 360.0) - lon_lo) / (lon_hi - lon_lo)
            w00 = (1 - dy) * (1 - dx)
            w01 = (1 - dy) * dx
            w10 = dy * (1 - dx)
            w11 = dy * dx

            self.tile_bounds = tile
            self.ds_4        = ds_4
            self.weights     = (lat_lo, lat_hi, lon_lo, lon_hi, dy, dx, w00, w01, w10, w11)

    def get_column(self, lat: float, lon: float, when: datetime):
        self._update_cycle(when)
        self._update_tile(lat, lon)

        ds_4 = self.ds_4
        lat_lo, lat_hi, lon_lo, lon_hi, dy, dx, w00, w01, w10, w11 = self.weights

        def collapse(var: str):
            v = ds_4[var].values  # Expected shape: (lev, 2, 2)
            if v.ndim == 3:  # Ensure 3D array (lev, lat, lon)
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
            m = da.values  # Expected shape: (2, 2)
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

        valid_dt = np.datetime64(ds_4.time.values).astype("datetime64[ms]").astype(datetime)
        cycle_dt = self.cached_cycle

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
