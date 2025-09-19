import numpy as np

# Constants
g0 = 9.80665  # gravity, m/s^2
delta_v_total = 10464  # total delta V in m/s
Isp1 = 278  # Stage 1 specific impulse (s)
Isp2 = 302  # Stage 2 specific impulse (s)
epsilon1 = 0.0513  # Stage 1 structural coefficient
epsilon2 = 0.0404  # Stage 2 structural coefficient
m0 = 672470  # Liftoff mass (kg)

# Function to calculate payload mass for a given ΔV split ratio
def compute_payload(delta_v1_ratio):
    delta_v1 = delta_v1_ratio * delta_v_total
    delta_v2 = delta_v_total - delta_v1

    MR1 = np.exp(delta_v1 / (g0 * Isp1))
    MR2 = np.exp(delta_v2 / (g0 * Isp2))

    # Assume unit payload and scale back to match total mass later
    m_payload = 1.0
    m2_initial = MR2 * m_payload
    m2_mass = m2_initial - m_payload
    prop2 = (1 - epsilon2) * m2_mass
    struct2 = epsilon2 * m2_mass

    m1_burnout = m2_initial
    m1_initial = MR1 * m1_burnout
    m1_mass = m1_initial - m1_burnout
    prop1 = (1 - epsilon1) * m1_mass
    struct1 = epsilon1 * m1_mass

    total_mass_unit_payload = m1_initial
    scale = m0 / total_mass_unit_payload

    return {
        "payload": m_payload * scale,
        "stage1_prop": prop1 * scale,
        "stage1_struct": struct1 * scale,
        "stage2_prop": prop2 * scale,
        "stage2_struct": struct2 * scale,
        "delta_v1_ratio": delta_v1_ratio
    }

# Sweep ΔV split ratios and find optimal payload
ratios = np.linspace(0.3, 0.8, 100)
results = [compute_payload(r) for r in ratios]
best_result = max(results, key=lambda x: x["payload"])

# Output
print("Optimal ΔV₁ / ΔV_total ratio:", round(best_result["delta_v1_ratio"], 3))
print("Payload Mass:", round(best_result["payload"], 2), "kg")
print("Stage 1 Propellant:", round(best_result["stage1_prop"], 2), "kg")
print("Stage 1 Structure:", round(best_result["stage1_struct"], 2), "kg")
print("Stage 2 Propellant:", round(best_result["stage2_prop"], 2), "kg")
print("Stage 2 Structure:", round(best_result["stage2_struct"], 2), "kg")
