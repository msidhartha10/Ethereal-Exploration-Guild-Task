import numpy as np

def direct_staging_solver(M0_target, Vmission, C, epsilon):
    etta = lagrange_NR(Vmission, C, epsilon)
    
    if np.isnan(etta):
        raise ValueError("Lagrange multiplier solver failed to converge.")
    
    # Mass ratios and inverse mass fractions
    n = (1 + etta * C) / (etta * C * epsilon)
    p = (n * epsilon - 1) / (1 - n)

    # Optimal payload mass
    A = ((1 + 1/p[1]) / p[0]) + (1 / p[1]) + 1
    m_pl = M0_target / A

    # Stage masses
    m2 = m_pl / p[1]
    m1 = (m2 + m_pl) / p[0]

    # Structural and propellant masses
    ms1 = epsilon[0] * m1
    mp1 = m1 - ms1

    ms2 = epsilon[1] * m2
    mp2 = m2 - ms2

    # Initial mass at each stage
    mi1 = m1 + m2 + m_pl
    mi2 = m2 + m_pl
    mi = [mi1, mi2]

    # ΔV for each stage
    dV1 = C[0] * np.log(mi1 / mi2)
    dV2 = C[1] * np.log(mi2 / m_pl)

    return m_pl, m1, m2, ms1, ms2, mp1, mp2, [m1, m2], mi, dV1, dV2

def lagrange_NR(Vmission, C, epsilon):
    etta_0 = -1 / min(C * (1 - epsilon))
    etta = etta_0
    tol = 1e-4
    max_iter = 100
    iter_count = 0

    while iter_count < max_iter:
        V = Vmission - sum(C * np.log((1 + etta * C) / (etta * C * epsilon)))
        if abs(V) < tol:
            break
        dV = sum(C / etta / (1 + etta * C))
        etta += -V / dV
        iter_count += 1

    if iter_count == max_iter:
        return np.nan
    return etta

# Example usage
M0_target = 672470
Vmission = 10464
Isp = np.array([278, 302])
g0 = 9.8065
C = Isp * g0
epsilon = np.array([0.0513, 0.0404])

m_pl, m1, m2, ms1, ms2, mp1, mp2, mk, mi, dV1, dV2 = direct_staging_solver(M0_target, Vmission, C, epsilon)

print(f"Optimal Payload Mass: {m_pl:.2f} kg\n")

print("--- Stage Mass Breakdown ---")
print(f"Stage 1 Total Mass: {m1:.2f} kg")
print(f"  → Structural Mass (ms1): {ms1:.2f} kg")
print(f"  → Propellant Mass (mp1): {mp1:.2f} kg\n")

print(f"Stage 2 Total Mass: {m2:.2f} kg")
print(f"  → Structural Mass (ms2): {ms2:.2f} kg")
print(f"  → Propellant Mass (mp2): {mp2:.2f} kg\n")

print("--- Total Liftoff Mass Check ---")
print(f"Total Liftoff Mass (mi1): {mi[0]:.2f} kg")
print(f"Initial Mass at Stage 2 (mi2): {mi[1]:.2f} kg\n")

print("--- ΔV Distribution ---")
print(f"Stage 1 ΔV: {dV1:.2f} m/s")
print(f"Stage 2 ΔV: {dV2:.2f} m/s")

