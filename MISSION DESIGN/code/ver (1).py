import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ==== PARAMETERS ====
g0 = 9.80665         # m/s^2, standard gravity
Rp = 6371000         # m, planetary radius
Isp = 278            # s, specific impulse

# Nominal initial conditions
h0 = 0.0             # m, altitude
x0 = 0.0             # m, downrange distance
v0 = 100.0           # m/s, initial speed
gamma0 = np.deg2rad(85.0)  # radians, flight path angle
m0 = 50000           # kg, mass

# Simulation settings
N_mc = 1000           # Number of Monte Carlo runs
t_span = (0, 500)    # Simulation time window (seconds)
t_eval = np.linspace(t_span[0], t_span[1], 1000)

# Uncertainty settings
sigma_rho = 0.05     # 5% atmospheric density error
sigma_T = 0.03       # 3% thrust perturbation
sigma_wind = 20.0    # m/s, random wind gusts
sigma_x0 = np.array([100, 0, 10, np.deg2rad(1), 0])  # [alt, x, vel, gamma, mass]

# ==== ATMOSPHERE MODEL ====
def atmospheric_density(h):
    rho0 = 1.225  # kg/m^3
    H = 8500     # m
    return rho0 * np.exp(-h / H)

# ==== THRUST MODEL ====
def thrust_profile(t, m, T_nominal, epsilon_T):
    return T_nominal * (1 + epsilon_T)

# ==== EOM DEFINITION ====
def ascent_eom(t, y, u, epsilon_rho, epsilon_T, wind_long, wind_lat):
    h, x, v, gamma, m = y
    g = g0 * (Rp / (Rp + h))**2
    rho = atmospheric_density(h) * (1 + epsilon_rho)

    Cd = 0.5
    Cl = 0.0
    A_ref = 1.0
    D = 0.5 * rho * v**2 * Cd * A_ref
    L = 0.5 * rho * v**2 * Cl * A_ref

    T = u * m * g0 * (1 + epsilon_T)

    v_wind_x = wind_long
    v_wind_y = wind_lat
    v_rel_x = v * np.cos(gamma) - v_wind_x
    v_rel_y = v * np.sin(gamma) - v_wind_y
    v_rel = np.hypot(v_rel_x, v_rel_y)
    gamma_rel = np.arctan2(v_rel_y, v_rel_x)

    dhdt = v_rel * np.sin(gamma_rel)
    dxdt = v_rel * np.cos(gamma_rel) / (1 + h / Rp)
    dvdt = (T - D) / m - g * np.sin(gamma)
    dgammadt = (L / (m * v)) + (v / (Rp + h) - g / v) * np.cos(gamma)
    dmdt = -T / (Isp * g0)

    return [dhdt, dxdt, dvdt, dgammadt, dmdt]

# ==== MONTE CARLO SIMULATION ====
max_q_list = []
max_acc_list = []
max_hdot_list = []
max_gamma_deg_list = []

for i in range(N_mc):
    epsilon_rho = np.random.normal(0, sigma_rho)
    epsilon_T = np.random.normal(0, sigma_T)
    wind_long = np.random.normal(0, sigma_wind)
    wind_lat = np.random.normal(0, sigma_wind)
    delta_x0 = np.random.normal(0, sigma_x0)

    y0 = np.array([h0, x0, v0, gamma0, m0]) + delta_x0

    u_nominal = 2.0

    sol = solve_ivp(ascent_eom, t_span, y0, args=(u_nominal, epsilon_rho, epsilon_T, wind_long, wind_lat), t_eval=t_eval, rtol=1e-6, atol=1e-9)

    h = sol.y[0, :]
    v = sol.y[2, :]
    gamma = sol.y[3, :]
    m = sol.y[4, :]

    rho = atmospheric_density(h) * (1 + epsilon_rho)
    q = 0.5 * rho * v**2
    T = u_nominal * m * g0 * (1 + epsilon_T)
    D = 0.5 * rho * v**2 * 0.5 * 1.0
    acc = (T - D) / m

    hdot = v * np.sin(gamma)
    gamma_deg = np.rad2deg(gamma)

    max_q_list.append(np.max(q))
    max_acc_list.append(np.max(acc))
    max_hdot_list.append(np.max(hdot))
    max_gamma_deg_list.append(np.max(gamma_deg))

# ==== PLOTS ====
plt.figure(figsize=(12,10))

plt.subplot(2,2,1)
plt.hist(max_q_list, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Max Dynamic Pressure (Q)')
plt.xlabel('Max Q (Pa)')
plt.ylabel('Frequency')

plt.subplot(2,2,2)
plt.hist(max_acc_list, bins=30, color='salmon', edgecolor='black')
plt.title('Histogram of Max Acceleration')
plt.xlabel('Max Acceleration (m/s²)')
plt.ylabel('Frequency')

plt.subplot(2,2,3)
plt.hist(max_hdot_list, bins=30, color='lightgreen', edgecolor='black')
plt.title('Histogram of Max Rate of Climb (ḣ)')
plt.xlabel('Max ḣ (m/s)')
plt.ylabel('Frequency')

plt.subplot(2,2,4)
plt.hist(max_gamma_deg_list, bins=30, color='violet', edgecolor='black')
plt.title('Histogram of Max Degree of Climb (γ)')
plt.xlabel('Max γ (deg)')
plt.ylabel('Frequency')

plt.tight_layout()
plt.show()
