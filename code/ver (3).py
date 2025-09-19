import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# === Constants ===
g0 = 9.80665        # gravitational acceleration (m/s^2)
Rp = 6371000.0      # planet radius (m)
Isp = 278           # specific impulse (s)

# === Nominal Initial Conditions ===
state_nominal = {
    'h': 0.0,       # altitude (m)
    'x': 0.0,       # downrange distance (m)
    'v': 100.0,     # velocity (m/s)
    'gamma': np.deg2rad(85.0),  # flight path angle (radians)
    'm': 50000.0    # mass (kg)
}

# === Monte Carlo Settings ===
num_samples = 5000
simulation_time = (0, 500)  # seconds
time_points = np.linspace(simulation_time[0], simulation_time[1], 50)

# === Uncertainty Parameters ===
sigma_atm_density = 0.05      # 5% std dev
sigma_thrust = 0.03            # 3% std dev
sigma_wind = 20.0              # m/s
sigma_ic = np.array([100, 0, 10, np.deg2rad(1), 0])  # [h, x, v, gamma, m]

# === Functions ===
def atmospheric_density(h):
    rho0 = 1.225  # kg/m³ at sea level
    H = 8500.0    # scale height (m)
    return rho0 * np.exp(-h / H)

def thrust_force(mass, u, eps_T):
    return u * mass * g0 * (1 + eps_T)

def equations_of_motion(t, y, u, eps_rho, eps_T, wind_x, wind_y):
    h, x, v, gamma, m = y
    g = g0 * (Rp / (Rp + h))**2
    rho = atmospheric_density(h) * (1 + eps_rho)

    Cd = 0.5
    A = 1.0  # reference area (m^2)
    D = 0.5 * rho * v**2 * Cd * A
    L = 0.0  # assuming no lift

    T = thrust_force(m, u, eps_T)

    v_rel_x = v * np.cos(gamma) - wind_x
    v_rel_y = v * np.sin(gamma) - wind_y
    v_rel = np.hypot(v_rel_x, v_rel_y)
    gamma_rel = np.arctan2(v_rel_y, v_rel_x)

    dh_dt = v_rel * np.sin(gamma_rel)
    dx_dt = v_rel * np.cos(gamma_rel) / (1 + h / Rp)
    dv_dt = (T - D) / m - g * np.sin(gamma)
    dgamma_dt = (L / (m * v)) + (v / (Rp + h) - g / v) * np.cos(gamma)
    dm_dt = -T / (Isp * g0)

    return [dh_dt, dx_dt, dv_dt, dgamma_dt, dm_dt]

# === Monte Carlo Storage ===
Q_max_list = []
acc_max_list = []
h_dot_max_list = []
gamma_max_deg_list = []

# === Monte Carlo Simulation ===
for trial in range(num_samples):
    eps_rho = np.random.normal(0, sigma_atm_density)
    eps_T = np.random.normal(0, sigma_thrust)
    wind_x = np.random.normal(0, sigma_wind)
    wind_y = np.random.normal(0, sigma_wind)
    delta_ic = np.random.normal(0, sigma_ic)

    # Initial conditions
    y0 = np.array([
        state_nominal['h'],
        state_nominal['x'],
        state_nominal['v'],
        state_nominal['gamma'],
        state_nominal['m']
    ]) + delta_ic

    u_nominal = 2.0  # nominal thrust-to-weight ratio

    sol = solve_ivp(
        fun=lambda t, y: equations_of_motion(t, y, u_nominal, eps_rho, eps_T, wind_x, wind_y),
        t_span=simulation_time,
        y0=y0,
        t_eval=time_points,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )

    h_traj = sol.y[0]
    v_traj = sol.y[2]
    gamma_traj = sol.y[3]
    m_traj = sol.y[4]

    rho_traj = atmospheric_density(h_traj) * (1 + eps_rho)
    q_dyn = 0.5 * rho_traj * v_traj**2
    thrust_traj = thrust_force(m_traj, u_nominal, eps_T)
    drag_traj = 0.5 * rho_traj * v_traj**2 * 0.5 * 1.0
    acc_traj = (thrust_traj - drag_traj) / m_traj

    h_dot = v_traj * np.sin(gamma_traj)
    gamma_deg = np.rad2deg(gamma_traj)

    # Record max values
    Q_max_list.append(np.max(q_dyn))
    acc_max_list.append(np.max(acc_traj))
    h_dot_max_list.append(np.max(h_dot))
    gamma_max_deg_list.append(np.max(gamma_deg))

# === Plots ===
fig, axs = plt.subplots(2, 2, figsize=(12,10))

axs[0, 0].hist(Q_max_list, bins=30, color='deepskyblue', edgecolor='black')
axs[0, 0].set_title('Peak Dynamic Pressure (Qmax)')
axs[0, 0].set_xlabel('Dynamic Pressure (Pa)')
axs[0, 0].set_ylabel('Frequency')

axs[0, 1].hist(acc_max_list, bins=30, color='coral', edgecolor='black')
axs[0, 1].set_title('Maximum Acceleration (amax)')
axs[0, 1].set_xlabel('Acceleration (m/s²)')
axs[0, 1].set_ylabel('Frequency')

axs[1, 0].hist(h_dot_max_list, bins=30, color='lightgreen', edgecolor='black')
axs[1, 0].set_title('Maximum Climb Rate ($\dot{h}_{max}$)')
axs[1, 0].set_xlabel('Vertical Speed (m/s)')
axs[1, 0].set_ylabel('Frequency')

axs[1, 1].hist(gamma_max_deg_list, bins=30, color='orchid', edgecolor='black')
axs[1, 1].set_title('Maximum Flight Path Angle ($\\gamma_{max}$)')
axs[1, 1].set_xlabel('Angle (deg)')
axs[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
