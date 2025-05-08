# Balloon Trajectory Simulation

## Overview
This project simulates the trajectories of tethered balloons using weather data from the Global Forecast System (GFS). It models balloon and payload dynamics under wind, buoyancy, drag, and tether forces, producing trajectories in ECEF coordinates saved as JSON files. The simulation leverages GFS data for accurate wind, temperature, pressure, and humidity, with precise interpolation and automated forecast fetching.

### Key Features
- **Weather Data**: Fetches GFS data (0.25° resolution, 3-hour cycles) via OPENDAP.
- **Wind Interpolation**: Combines bilinear (horizontal) and linear (vertical) interpolation across three altitude regimes for precise wind profiles.
- **Forecast Fetching**: Automatically retrieves the nearest forecast and checks for updated datasets.
- **Configurable Parameters**: Initial latitude, longitude, height, number of simulations, duration, CPU usage, and perturbation range.
- **Parallel Processing**: Runs multiple simulations concurrently using Python’s `multiprocessing`.
- **Output**: Saves trajectories as `trajectory_{sim_id}.json`.


## Usage
1. **Run Simulations**:
   - Execute `propagate.py` with default parameters:
     ```bash
     python propagate.py
     ```
   - Default settings:
     - Initial position: `lat=37.428230`, `lon=-122.168861`, `height=50.0 m`
     - Simulations: 1
     - Duration: 5 hours
     - CPUs: 8
     - Perturbation: ±0.1°
   - Outputs JSON files (`trajectory_{sim_id}.json`) with balloon and payload trajectories.

2. **Customize Parameters**:
   - Modify `propagate.py`’s `if __name__ == "__main__":` block:
     ```python
     main(initial_lat=40.7128, initial_lon=-74.0060, initial_height=150.0, num_simulations=6, duration_hours=2.5, num_cpus='all')
     ```

3. **Test Weather Data**:
   - Run `weather.py` to test wind components:
     ```bash
     python weather.py
     ```
   - Example output:
     ```
     Wind at lat=59.21, lon=-42.24, alt=30.0 m:
     u=0.90 m/s, v=0.59 m/s, w=-0.01 m/s
     ```

## Wind Interpolation Regimes
The simulation uses precise wind interpolation in `weather.py` to compute wind components (`u`, `v`, `w`) at any `lat`, `lon`, and `alt`:

1. **Horizontal Bilinear Interpolation**:
   - GFS data is on a 0.25° grid. For a given `lat`, `lon`, a 2x2 tile is selected (e.g., for `lat=37.428230`, `lon=-122.168861`: `[37.25°, 37.50°] × [-122.25°, -122.00°]`).
   - Weights (`w00`, `w01`, `w10`, `w11`) are computed based on the fractional position within the tile:
     ```
     dy = (lat - lat_lo) / (lat_hi - lat_lo)
     dx = (lon - lon_lo) / (lon_hi - lon_lo)
     w00 = (1 - dy) * (1 - dx), ...
     ```
   - Winds are interpolated for each `height_above_ground2` level (10 m, 20 m, 30 m, 40 m, 50 m, 80 m, 100 m) and isobaric level (41 levels, e.g., 1000 hPa to 100 hPa) using:
     ```
     value = w00 * value[lat_lo, lon_lo] + w01 * value[lat_lo, lon_hi] + w10 * value[lat_hi, lon_lo] + w11 * value[lat_hi, lon_hi]
     ```

2. **Vertical Linear Interpolation (Three Regimes)**:
   - **0–100 m**:
     - Uses surface winds from `height_above_ground2` levels (10 m, 20 m, 30 m, 40 m, 50 m, 80 m, 100 m).
     - Linearly interpolates `u`, `v` across these levels (e.g., for `alt=50.0`, uses exact 50 m wind; for `alt=45.0`, interpolates between 40 m and 50 m).
     - Below 10 m, uses 10 m wind.
     - Vertical wind (`w`) is interpolated from isobaric levels.
   - **100 m to First Isobaric Level**:
     - Blends the 100 m surface wind with the first isobaric wind level (e.g., ~100–200 m for 1000 hPa) using:
       ```
       frac = (alt - 100) / (z_high - 100)
       u_at = (1 - frac) * u_surface[100 m] + frac * u_isobaric[alt]
       ```
     - Ensures a smooth transition from surface to isobaric winds.
   - **Above First Isobaric Level**:
     - Interpolates `u`, `v`, `w` across 41 isobaric levels using geopotential heights (`z`).
     - Example: For `alt=1000.0 m`, interpolates between isobaric levels (e.g., 925 hPa, 850 hPa).

## Forecast Fetching Regimes
The simulation automatically fetches GFS forecasts in `weather.py` using `GFSDataCache` with two regimes:

1. **Nearest Forecast Fetch**:
   - For a given simulation time (`t0 + t`), the code selects the nearest 3-hour GFS forecast cycle (e.g., for 2025-05-08 09:22:38, selects 09:00:00).
   - Uses `get_column` to fetch data for the requested `lat`, `lon`, and time, caching the forecast for efficiency.
   - Example: If simulation time is 09:22:38, it fetches the 09:00:00 dataset, as seen in:
     ```
     Selecting forecast time for 2025-05-08 09:22:38.408607+00:00
     Selected forecast time: 2025-05-08T09:00:00.000000000
     ```

2. **New Dataset Check**:
   - Every 6 hours, the code checks if a new, more precise GFS dataset is available on the server:
     ```python
     if (self.full_ds is None or 
         self.last_fetch_time is None or 
         (current_real_time - self.last_fetch_time) >= timedelta(hours=6)):
         print("Fetching new GFS dataset")
         self.full_ds = xr.open_dataset(OPENDAP_URL, decode_times=True)[ISOBARIC_VARS + SURFACE_VARS]
     ```
   - If the dataset is outdated, it fetches a new one, ensuring the simulation uses the latest available forecast.

## Code Structure
- **propagate.py**: Simulates balloon dynamics, runs parallel simulations, and saves trajectories.
- **weather.py**: Fetches and interpolates GFS data for wind, temperature, pressure, and density.
- **constants.py**: Defines GFS URL and variables.
- **geo_utils.py**: Handles coordinate conversions (ECEF, geodetic).
- **gas_utils.py**: Computes gas properties (e.g., viscosity).


## Future Improvements
- Improve ballon dynamics modeling (balloon structural analysis, internal temperature, pressure model, material)
- Initialize balloon position and atitude controller + simulate sensor readings to compare with real data
- Add pipeline to vizualize live/post-flight data
- Ability to ingest complex inertial and drag profiles
- Improve simulation speed
- Investigate Deep RL framework for policy training
- Improve tether model (mass, drag, material, stiffness/damp)
- Optimize dataset fetching
- Add command-line argument support for dynamic parameters.
- Incorporate additional GFS variables (e.g., 2 m temperature).
- Use spline interpolation for smoother wind profiles.


## Contact
For issues or contributions, contact acarlham@stanford.edu