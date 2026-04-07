"""
PUBLISH AND PERISH — Core Model
================================
Minimal 2-variable ODE model: Queue pressure → reviewer AI adoption → quality erosion

Causal chain (one-directional):
  φ_w(t) [external] → S↑ → Q↑ → φ_r(Q)↑ → q↓ → K↓

State variables: Q(t) — review queue, q(t) — verification quality
External input: φ_w(t) — writing AI adoption (logistic, fitted to data)
Endogenous coupling: φ_r(Q) = Q/(Q + Q_c) — review AI adoption driven by queue

Author: S. Joon Kwon, SKKU
"""

import numpy as np
from scipy.integrate import solve_ivp
import json

# ═══════════════════════════════════════════════
# EXTERNAL INPUT
# ═══════════════════════════════════════════════
def logistic(t, k, t_mid):
    """Logistic function for AI adoption dynamics."""
    return 1.0 / (1.0 + np.exp(-k * (t - t_mid)))

# Writing AI adoption: >50% by 2025 (t=3)
K_W, T_W = 1.0, 3.0
def phi_w(t):
    """Writing AI penetration (prescribed, external)."""
    return logistic(t, K_W, T_W)

# ═══════════════════════════════════════════════
# ENDOGENOUS COUPLING
# ═══════════════════════════════════════════════
def phi_r(Q, Q_c):
    """Review AI adoption driven by queue pressure (Michaelis-Menten type)."""
    return Q / (Q + Q_c)

# ═══════════════════════════════════════════════
# DEFAULT PARAMETERS
# ═══════════════════════════════════════════════
DEFAULT_PARAMS = dict(
    S0=1.0,       # baseline submission rate (normalized)
    R_max=1.0,    # max reviewer capacity (normalized: pre-AI S₀ = R_max)
    gamma=2.0,    # writing acceleration factor
    delta=0.5,    # review acceleration from AI
    Q_c=2.0,      # half-saturation: at Q=Q_c, 50% of reviewers adopt AI
    lam=0.3,      # quality degradation rate
    mu=0.5,       # quality restoration rate
    eta=0.8,      # fraction of human review displaced by AI
    q_min=0.2,    # institutional quality floor
)

# ═══════════════════════════════════════════════
# ODE SYSTEM
# ═══════════════════════════════════════════════
def model_rhs(t, y, p):
    """Right-hand side of the ODE system.
    
    Args:
        t: time (years since Nov 2022)
        y: [Q, q] state vector
        p: parameter dictionary
    
    Returns:
        [dQ/dt, dq/dt]
    """
    Q, q = y
    Q = max(Q, 0.0)
    q = np.clip(q, p['q_min'], 1.0)
    
    pw = phi_w(t)
    pr = phi_r(Q, p['Q_c'])
    
    S = p['S0'] * (1 + p['gamma'] * pw)
    R = p['R_max'] * (1 + p['delta'] * pr)
    
    dQ = max(S - R, -Q / 0.01)  # prevent Q < 0
    dq = -p['lam'] * pr * (q - p['q_min']) + p['mu'] * (1 - p['eta'] * pr) * (1 - q)
    
    return [dQ, dq]

# ═══════════════════════════════════════════════
# SIMULATION
# ═══════════════════════════════════════════════
def simulate(params=None, t_end=20, n_points=2000, y0=None):
    """Run the model simulation.
    
    Args:
        params: parameter dictionary (defaults to DEFAULT_PARAMS)
        t_end: simulation end time in years
        n_points: number of evaluation points
        y0: initial conditions [Q0, q0] (defaults to [0.0, 1.0])
    
    Returns:
        dict with t, Q, q, K, R, S, phi_w, phi_r arrays and params
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    if y0 is None:
        y0 = [0.0, 1.0]
    
    t_eval = np.linspace(0, t_end, n_points)
    sol = solve_ivp(model_rhs, (0, t_end), y0, args=(params,),
                    t_eval=t_eval, method='RK45', max_step=0.02,
                    rtol=1e-10, atol=1e-12)
    
    t = sol.t
    Q_t, q_t = sol.y
    
    pw_t = np.array([phi_w(ti) for ti in t])
    pr_t = np.array([phi_r(Qi, params['Q_c']) for Qi in Q_t])
    S_t = params['S0'] * (1 + params['gamma'] * pw_t)
    R_t = params['R_max'] * (1 + params['delta'] * pr_t)
    K_t = R_t * q_t
    K0 = params['R_max'] * 1.0
    
    return dict(
        t=t, Q=Q_t, q=q_t, K=K_t / K0, R=R_t, S=S_t,
        phi_w=pw_t, phi_r=pr_t, params=params, K0=K0
    )

# ═══════════════════════════════════════════════
# ANALYTICAL STEADY STATE
# ═══════════════════════════════════════════════
def analytical_steady_state(params=None):
    """Compute analytical steady-state values.
    
    Returns:
        dict with S_ss, R_max_eff, q_ss, K_ss, K_ss_over_K0, paradox (bool),
        delta_critical
    """
    if params is None:
        params = DEFAULT_PARAMS.copy()
    
    S_ss = params['S0'] * (1 + params['gamma'])
    R_max_eff = params['R_max'] * (1 + params['delta'])
    K0 = params['R_max'] * 1.0
    
    paradox = S_ss > R_max_eff
    
    if paradox:
        # Queue diverges → φ_r → 1
        lam, mu, eta, q_min = params['lam'], params['mu'], params['eta'], params['q_min']
        q_ss = (lam * q_min + mu * (1 - eta)) / (lam + mu * (1 - eta))
        K_ss = R_max_eff * q_ss
    else:
        q_ss = 1.0  # simplified; exact requires numerical solution
        K_ss = R_max_eff * q_ss
    
    delta_critical = params['gamma'] * params['S0'] / params['R_max'] + params['S0'] / params['R_max'] - 1
    
    return dict(
        S_ss=S_ss, R_max_eff=R_max_eff, q_ss=q_ss,
        K_ss=K_ss, K_ss_over_K0=K_ss / K0,
        paradox=paradox, delta_critical=delta_critical
    )

# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════
if __name__ == '__main__':
    # Run baseline simulation
    results = simulate()
    ss = analytical_steady_state()
    
    t, K, Q, q = results['t'], results['K'], results['Q'], results['q']
    R, S = results['R'], results['S']
    pw, pr = results['phi_w'], results['phi_r']
    
    print("═══ ANALYTICAL STEADY STATE ═══")
    print(f"  S_ss = {ss['S_ss']:.2f}, R_max_eff = {ss['R_max_eff']:.2f}")
    print(f"  Paradox: {ss['paradox']}")
    print(f"  q_ss = {ss['q_ss']:.3f}, K_ss/K₀ = {ss['K_ss_over_K0']:.3f}")
    print(f"  δ_critical = {ss['delta_critical']:.2f}")
    
    print(f"\n═══ SIMULATION RESULTS ═══")
    print(f"{'t':>4} {'Year':>5} {'K/K₀':>6} {'R':>6} {'q':>6} {'Q':>6} {'φ_w':>5} {'φ_r':>5}")
    for yr in [0, 1, 2, 3, 3.5, 5, 7, 10, 15, 20]:
        i = np.argmin(np.abs(t - yr))
        print(f"{t[i]:4.1f} {t[i]+2022:5.0f} {K[i]:6.3f} {R[i]:6.3f} {q[i]:6.3f} {Q[i]:6.2f} {pw[i]:5.3f} {pr[i]:5.3f}")
    
    print(f"\n═══ MILESTONES ═══")
    print(f"  K_peak = {np.max(K):.3f} at t = {t[np.argmax(K)]:.1f} yr ({t[np.argmax(K)]+2022:.0f})")
    for thresh in [1.0, 0.95, 0.90, 0.80, 0.70]:
        idx = np.argmax(K < thresh)
        if K[idx] < thresh:
            print(f"  K < {thresh:.2f} at t = {t[idx]:.1f} yr ({t[idx]+2022:.0f})")
    
    # Save results
    output = dict(
        t=t.tolist(), Q=Q.tolist(), q=q.tolist(), K=K.tolist(),
        R=R.tolist(), S=S.tolist(), phi_w=pw.tolist(), phi_r=pr.tolist(),
        params=results['params'], K0=results['K0'],
        analytical=ss
    )
    with open('data/model_results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print("\n✓ Results saved to data/model_results.json")
