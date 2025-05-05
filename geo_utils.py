import numpy as np
from typing import Tuple, Union
from constants import RE, f, eE, eE2, RP, ep2

def geodetic_to_ecef(lat: float,
                     lon: float,
                     h: float
                    ) -> np.ndarray:
    """
    (lat, lon, h) → (X, Y, Z) in ECEF
    """
    N = RE / np.sqrt(1 - eE2 * np.sin(lat)**2)
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = ((1 - eE2) * N + h) * np.sin(lat)
    return np.array([X, Y, Z])


def ecef_to_geodetic_newton(X: float,
                            Y: float,
                            Z: float,
                            tol: float = 1e-12,
                            maxiter: int = 50
                           ) -> Tuple[float, float, float]:
    """
    Convert ECEF (X,Y,Z) → geodetic (lat, lon, h) using
    Newton-Raphson on Bowring's κ-equation.
    """
    
    # Constanst
    p = np.hypot(X, Y)          # distance in equatorial plane
    A = p*p
    B = (1 - eE2) * Z*Z
    c = eE2 * RE

    # Initial guess κ0 = 1/(1 - eE2)
    k = 1.0 / (1 - eE2)

    # Newton–Raphson loop
    for _ in range(maxiter):
        D  = np.sqrt(A + B * k*k)
        f  = k - 1 - c * k / D
        fp = 1 - (c * A) / (A + B * k*k)**1.5
        k_new = k - f/fp
        if abs(k_new - k) < tol * k:
            k = k_new
            break
        k = k_new

    # Recover lat, lon, h
    lat = np.arctan2(k * Z, p)
    lon = np.arctan2(Y, X)
    N   = RE / np.sqrt(1 - eE2 * np.sin(lat)**2)
    h   = p / np.cos(lat) - N

    return lat, lon, h



def enu_to_ecef(enu: np.ndarray,
                lat0: float,
                lon0: float,
                h0: float) -> np.ndarray:
    """
    Transform local ENU vector(s) to ECEF point(s).

    Uses:
        [Xp, Yp, Zp]^T
        = R_ENU2ECEF @ [e, n, u]^T + [Xr, Yr, Zr]^T

    where R_ENU2ECEF is
        [[-sin(lon0),         -sin(lat0)*cos(lon0),   cos(lat0)*cos(lon0)],
         [ cos(lon0),         -sin(lat0)*sin(lon0),   cos(lat0)*sin(lon0)],
         [      0   ,                   cos(lat0),           sin(lat0)     ]]

    and [Xr, Yr, Zr] = geodetic_to_ecef(lat0, lon0, h0).
    """
    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    R_ENU2ECEF = np.array([
        [-sin_lon,         -sin_lat * cos_lon,    cos_lat * cos_lon],
        [ cos_lon,         -sin_lat * sin_lon,    cos_lat * sin_lon],
        [      0.0,                cos_lat,             sin_lat   ]
    ])

    # origin in ECEF
    Xr, Yr, Zr = geodetic_to_ecef(lat0, lon0, h0)

    # apply rotation (supports both (3,) and (...,3) enu)
    ecef_vec = (R_ENU2ECEF @ enu.T).T

    # translate to absolute ECEF position
    return ecef_vec + np.array([Xr, Yr, Zr])


def ecef_to_enu(xyz: np.ndarray,
                lat0: float,
                lon0: float,
                h0: float) -> np.ndarray:
    """
    Transform ECEF point(s) to local ENU vector(s).

    Uses:
        [e, n, u]^T
        = R_ECEF2ENU @ ([Xp, Yp, Zp]^T - [Xr, Yr, Zr]^T)

    where R_ECEF2ENU is
        [[-sin(lon0),            cos(lon0),             0       ],
         [-sin(lat0)*cos(lon0), -sin(lat0)*sin(lon0),  cos(lat0)],
         [ cos(lat0)*cos(lon0),  cos(lat0)*sin(lon0),  sin(lat0)]]
    """
    sin_lat = np.sin(lat0)
    cos_lat = np.cos(lat0)
    sin_lon = np.sin(lon0)
    cos_lon = np.cos(lon0)

    R_ECEF2ENU = np.array([
        [-sin_lon,             cos_lon,           0.0      ],
        [-sin_lat * cos_lon,  -sin_lat * sin_lon, cos_lat  ],
        [ cos_lat * cos_lon,   cos_lat * sin_lon, sin_lat  ]
    ])

    # origin in ECEF
    Xr, Yr, Zr = geodetic_to_ecef(lat0, lon0, h0)

    # offset from origin
    delta = xyz - np.array([Xr, Yr, Zr])

    # apply rotation
    return (R_ECEF2ENU @ delta.T).T



def enu_vector_to_ecef(enu: np.ndarray,
                       lat: float,
                       lon: float) -> np.ndarray:
    """
    Rotate a vector from local ENU -> global ECEF axes.

    enu : [east, north, up]
    lat, lon : geodetic latitude & longitude in radians
    returns [vx, vy, vz] in ECEF axes
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # rotation matrix ENU->ECEF
    R = np.array([
        [-sin_lon,          -sin_lat*cos_lon,   cos_lat*cos_lon],
        [ cos_lon,          -sin_lat*sin_lon,   cos_lat*sin_lon],
        [      0.0,                 cos_lat,          sin_lat   ]
    ])

    return R @ enu


def ecef_vector_to_enu(ecef: np.ndarray,
                       lat: float,
                       lon: float) -> np.ndarray:
    """
    Rotate a vector from global ECEF -> local ENU axes.

    ecef : [vx, vy, vz] in ECEF axes
    lat, lon : geodetic latitude & longitude in radians
    returns [east, north, up]
    """
    sin_lat = np.sin(lat)
    cos_lat = np.cos(lat)
    sin_lon = np.sin(lon)
    cos_lon = np.cos(lon)

    # rotation matrix ECEF->ENU
    R = np.array([
        [-sin_lon,           cos_lon,          0.0    ],
        [-sin_lat*cos_lon,  -sin_lat*sin_lon,  cos_lat],
        [ cos_lat*cos_lon,   cos_lat*sin_lon,  sin_lat]
    ])

    return R @ ecef




def geocentric_to_geodetic(r: float,
                           lat_gc: float,
                           lon_gc: float,
                           tol: float = 1e-12
                          ) -> Tuple[float, float, float]:
    """
    Spherical geocentric (r, lat_gc, lon) → geodetic (lat, lon, h)

    1) Back to ECEF:
         X     = r * cos(lat_gc) * cos(lon)
         Y     = r * cos(lat_gc) * sin(lon)
         Z     = r * sin(lat_gc)
         r_xy  = sqrt(X² + Y²)

    2) Initial guess for lat:
         lat = arcsin(Z / r)

    3) Iterate until convergence:
         N       = RE / sqrt(1 - eE2 * sin(lat)**2)
         lat_new = atan2(Z + eE2 * N * sin(lat), r_xy)

    4) Compute height:
         h = r_xy / cos(lat) - N
    """
    # back to ECEF
    X    = r * np.cos(lat_gc) * np.cos(lon_gc)
    Y    = r * np.cos(lat_gc) * np.sin(lon_gc)
    Z    = r * np.sin(lat_gc)
    r_xy = np.hypot(X, Y)

    # lon unchanged
    lon = lon_gc 
    
    # initial guess
    lat = np.arcsin(Z / r)

    # iterate on lat
    while True:
        N = RE / np.sqrt(1 - eE2 * np.sin(lat)**2)
        lat_new = np.arctan2(Z + eE2 * N * np.sin(lat), r_xy)
        if abs(lat_new - lat) < tol:
            lat = lat_new
            break
        lat = lat_new

    # final height
    N_final = RE / np.sqrt(1 - eE2 * np.sin(lat)**2)
    h = r_xy / np.cos(lat) - N_final

    return lat, lon, h


def geodetic_to_geocentric(lat: float,
                           lon: float,
                           h: float
                          ) -> Tuple[float, float, float]:
    """
    (lat, lon, h) → spherical geocentric (r, lat_gc, lon)
    
    1) Convert geodetic → ECEF:
         N    = RE / sqrt(1 - eE2 * sin(lat)**2)
         X    = (N + h) * cos(lat) * cos(lon)
         Y    = (N + h) * cos(lat) * sin(lon)
         Z    = ((1 - eE2)*N + h) * sin(lat)

    2) Convert ECEF → spherical:
         r     = sqrt(X² + Y² + Z²)
         lat_gc = arcsin(Z / r)      # geocentric latitude
         lon    = lon                # same longitude
    """
    # lon unchanged
    lon_gc = lon 
    
    # geodetic → ECEF
    N = RE / np.sqrt(1 - eE2 * np.sin(lat)**2)
    X = (N + h) * np.cos(lat) * np.cos(lon)
    Y = (N + h) * np.cos(lat) * np.sin(lon)
    Z = ((1 - eE2) * N + h) * np.sin(lat)

    # ECEF → spherical geocentric
    r      = np.sqrt(X*X + Y*Y + Z*Z)
    lat_gc = np.arcsin(Z / r)
    return r, lat_gc, lon_gc



if __name__ == "__main__":
    import numpy as np

    # Sample geodetic point (40°N, 75°W, 100 m)
    lat_deg, lon_deg, h = 40.0, -75.0, 100.0
    lat = np.deg2rad(lat_deg)
    lon = np.deg2rad(lon_deg)

    print("=== Geodetic <-> ECEF ===")
    X, Y, Z = geodetic_to_ecef(lat, lon, h)
    print(f"geodetic_to_ecef({lat_deg}°, {lon_deg}°, {h}m) ->", X, Y, Z)
    lat2, lon2, h2 = ecef_to_geodetic_newton(X, Y, Z)
    print("ecef_to_geodetic_newton ->",
          f"{np.rad2deg(lat2):.6f}°", f"{np.rad2deg(lon2):.6f}°", f"{h2:.6f} m")

    print("\n=== Geodetic <-> Geocentric ===")
    r, lat_gc, lon_gc = geodetic_to_geocentric(lat, lon, h)
    print("geodetic_to_geocentric ->",
          f"r={r:.3f}m", f"lat_gc={np.rad2deg(lat_gc):.6f}°", f"lon_gc={np.rad2deg(lon_gc):.6f}°")
    lat3, lon3, h3 = geocentric_to_geodetic(r, lat_gc, lon_gc)
    print("geocentric_to_geodetic ->",
          f"{np.rad2deg(lat3):.6f}°", f"{np.rad2deg(lon3):.6f}°", f"{h3:.6f} m")

    print("\n=== ENU Vector Rotation ===")
    v_enu = np.array([1.0, 2.0, 3.0])
    v_ecef = enu_vector_to_ecef(v_enu, lat, lon)
    v_enu_back = ecef_vector_to_enu(v_ecef, lat, lon)
    print("enu_vector_to_ecef([1,2,3]) ->", v_ecef)
    print("ecef_vector_to_enu ->", v_enu_back)

    print("\n=== ENU <-> ECEF Point ===")
    p_enu = np.array([100.0, 200.0, 50.0])
    p_ecef = enu_to_ecef(p_enu, lat, lon, h)
    p_enu_back = ecef_to_enu(p_ecef, lat, lon, h)
    print("enu_to_ecef_point ->", p_ecef)
    print("ecef_to_enu_point ->", p_enu_back)
