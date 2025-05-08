from geo_utils import ecef_to_geodetic_newton, ecef_to_enu
import matplotlib.pyplot as plt
import numpy as np
import folium
import json


def plot_with_folium(json_file='trajectory_0.json',
                     out_html='trajectory_map.html'):
    # load ECEF trajectory
    traj = json.load(open(json_file))
    Xb = np.array(traj['balloon_ecef']['X'])
    Yb = np.array(traj['balloon_ecef']['Y'])
    Zb = np.array(traj['balloon_ecef']['Z'])
    ecef = np.column_stack((Xb, Yb, Zb))

    # convert each point to geodetic (lat, lon) in DEGREES
    lats, lons = [], []
    for X, Y, Z in ecef:
        lat_rad, lon_rad, _ = ecef_to_geodetic_newton(X, Y, Z)
        lats.append(np.rad2deg(lat_rad))
        lons.append(np.rad2deg(lon_rad))

    # center map on start point
    m = folium.Map(location=[lats[0], lons[0]], zoom_start=12)

    # draw the trajectory line
    folium.PolyLine(
        list(zip(lats, lons)),
        color='red', weight=3, opacity=0.8
    ).add_to(m)

    # start/end markers
    folium.Marker([lats[0], lons[0]], popup='Start').add_to(m)
    folium.Marker([lats[-1], lons[-1]], popup='End').add_to(m)

    # write out the HTML file
    m.save(out_html)
    print(f"Map saved to {out_html}")
    
    

def plot_enu_trajectory(json_file='trajectory_0.json'):
    # Load trajectory data
    with open(json_file, 'r') as f:
        traj = json.load(f)

    # Extract origin geodetic coordinates
    origin = traj['origin']
    lat0 = np.deg2rad(origin['lat0'])
    lon0 = np.deg2rad(origin['lon0'])
    h0 = origin['h0']

    # Extract time and balloon ECEF
    t = np.array(traj['t'])
    Xb = np.array(traj['balloon_ecef']['X'])
    Yb = np.array(traj['balloon_ecef']['Y'])
    Zb = np.array(traj['balloon_ecef']['Z'])
    ecef = np.column_stack((Xb, Yb, Zb))

    # Convert to local ENU
    enu = ecef_to_enu(ecef, lat0, lon0, h0)
    east, north, up = enu[:, 0], enu[:, 1], enu[:, 2]

    # === 3D Trajectory ===
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(east, north, up, label='Balloon ENU Trajectory')
    ax.set_xlabel('East [m]')
    ax.set_ylabel('North [m]')
    ax.set_zlabel('Up [m]')
    ax.set_title('3D Balloon Trajectory (ENU)')
    ax.legend()
    plt.tight_layout()
    plt.show()

    # === Evolution of E/N/U over time ===
    plt.figure()
    plt.plot(t, east, label='East [m]')
    plt.plot(t, north, label='North [m]')
    plt.plot(t, up, label='Up [m]')
    plt.xlabel('Time [s]')
    plt.ylabel('Displacement [m]')
    plt.title('Balloon Displacement in ENU Frame')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # === Compute and plot ENU Speed ===
    dt = np.diff(t)                       # time steps
    denu = np.diff(enu, axis=0)          # ENU displacement per step
    speed = np.linalg.norm(denu, axis=1) / dt  # m/s

    t_mid = 0.5 * (t[1:] + t[:-1])  # time at midpoints
   
    print(f"Max ENU speed: {np.max(speed):.2f} m/s")

    
    plt.figure()
    plt.plot(t_mid, speed)
    plt.xlabel('Time [s]')
    plt.ylabel('Speed [m/s]')
    plt.title('Estimated Balloon Speed in ENU Frame')
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    plot_enu_trajectory("trajectory_0.json")
    plot_with_folium("trajectory_0.json")
