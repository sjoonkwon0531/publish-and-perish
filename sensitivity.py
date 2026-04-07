"""
Sensitivity analysis and Monte Carlo simulation for Publish and Perish model.
Covers: one-at-a-time (OAT) sensitivity, Latin Hypercube Sampling (LHS),
and stochastic Monte Carlo with parameter noise.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import json

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9, 'axes.labelsize': 11, 'figure.dpi': 300,
    'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'legend.fontsize': 8, 'legend.frameon': False,
})

# ═══ Model functions ═══
def logistic(t, k, t_mid):
    return 1.0 / (1.0 + np.exp(-k * (t - t_mid)))
k_w, t_w = 1.0, 3.0
phi_w = lambda t: logistic(t, k_w, t_w)
def phi_r(Q, Q_c):
    return Q / (Q + Q_c)

def model(t, y, p):
    Q, q = y
    Q = max(Q, 0.0)
    q = np.clip(q, p['q_min'], 1.0)
    pw = phi_w(t)
    pr = phi_r(Q, p['Q_c'])
    S = p['S0'] * (1 + p['gamma'] * pw)
    R = p['R_max'] * (1 + p['delta'] * pr)
    dQ = max(S - R, -Q/0.01)
    dq = -p['lam'] * pr * (q - p['q_min']) + p['mu'] * (1 - p['eta']*pr) * (1 - q)
    return [dQ, dq]

# Stochastic version with Wiener process noise
def model_stochastic(t, y, p, dt, rng):
    Q, q = y
    Q = max(Q, 0.0)
    q = np.clip(q, p['q_min'], 1.0)
    pw = phi_w(t)
    pr = phi_r(Q, p['Q_c'])
    S = p['S0'] * (1 + p['gamma'] * pw)
    R = p['R_max'] * (1 + p['delta'] * pr)
    dQ_det = S - R
    dq_det = -p['lam'] * pr * (q - p['q_min']) + p['mu'] * (1 - p['eta']*pr) * (1 - q)
    # Add noise proportional to signal
    sigma_Q = p.get('sigma_Q', 0.1) * S  # submission noise
    sigma_q = p.get('sigma_q', 0.02)     # quality noise
    dW_Q = rng.normal(0, np.sqrt(dt))
    dW_q = rng.normal(0, np.sqrt(dt))
    dQ = dQ_det * dt + sigma_Q * dW_Q
    dq = dq_det * dt + sigma_q * dW_q
    Q_new = max(Q + dQ, 0.0)
    q_new = np.clip(q + dq, p['q_min'], 1.0)
    return Q_new, q_new

DEFAULT_PARAMS = dict(S0=1.0, R_max=1.0, gamma=2.0, delta=0.5,
                      Q_c=2.0, lam=0.3, mu=0.5, eta=0.8, q_min=0.2)

def run_deterministic(params, t_end=20):
    sol = solve_ivp(model, (0, t_end), [0.0, 1.0], args=(params,),
                    t_eval=np.linspace(0, t_end, 500), method='RK45',
                    max_step=0.05, rtol=1e-9, atol=1e-11)
    Q, q = sol.y
    pr_arr = np.array([phi_r(Qi, params['Q_c']) for Qi in Q])
    R = params['R_max'] * (1 + params['delta'] * pr_arr)
    K = R * q
    return sol.t, K, Q, q

# ═══ 1. OAT Sensitivity Analysis ═══
print("═══ ONE-AT-A-TIME SENSITIVITY ═══\n")
param_ranges = {
    'gamma': np.linspace(0.5, 4.0, 30),
    'delta': np.linspace(0.0, 3.0, 30),
    'Q_c':   np.linspace(0.5, 5.0, 20),
    'lam':   np.linspace(0.05, 1.0, 20),
    'mu':    np.linspace(0.1, 1.5, 20),
    'eta':   np.linspace(0.2, 1.0, 20),
    'q_min': np.linspace(0.05, 0.6, 20),
}

fig_oat, axes_oat = plt.subplots(2, 4, figsize=(14, 6))
axes_flat = axes_oat.flatten()

sensitivity_results = {}
for idx, (pname, pvals) in enumerate(param_ranges.items()):
    K20_vals = []
    Kpeak_vals = []
    for pv in pvals:
        p2 = dict(DEFAULT_PARAMS)
        p2[pname] = pv
        t2, K2, _, _ = run_deterministic(p2)
        K20_vals.append(K2[-1])
        Kpeak_vals.append(np.max(K2))
    
    K20_vals = np.array(K20_vals)
    Kpeak_vals = np.array(Kpeak_vals)
    
    ax = axes_flat[idx]
    ax.plot(pvals, K20_vals, 'b-', lw=2, label='K(20yr)')
    ax.plot(pvals, Kpeak_vals, 'g--', lw=1.5, label='K peak')
    ax.axhline(1.0, color='gray', ls=':', lw=0.7)
    ax.axvline(DEFAULT_PARAMS[pname], color='red', ls='--', lw=1, alpha=0.5)
    ax.set_xlabel(pname)
    ax.set_ylabel('K/K₀')
    ax.set_title(pname, fontweight='bold')
    if idx == 0:
        ax.legend(fontsize=7)
    
    # Elasticity at baseline
    base_idx = np.argmin(np.abs(pvals - DEFAULT_PARAMS[pname]))
    if base_idx > 0 and base_idx < len(pvals)-1:
        dK = K20_vals[base_idx+1] - K20_vals[base_idx-1]
        dp = pvals[base_idx+1] - pvals[base_idx-1]
        elasticity = (dK / K20_vals[base_idx]) / (dp / pvals[base_idx])
        sensitivity_results[pname] = elasticity
        print(f"  {pname:6s}: elasticity = {elasticity:+.3f} (at baseline {DEFAULT_PARAMS[pname]})")

axes_flat[-1].axis('off')
fig_oat.suptitle('One-at-a-Time Sensitivity Analysis', fontsize=13, fontweight='bold')
fig_oat.tight_layout()
fig_oat.savefig('/home/claude/sensitivity_oat.png', dpi=300)
print("\nOAT figure saved.")

# ═══ 2. Monte Carlo with stochastic model ═══
print("\n═══ MONTE CARLO SIMULATION (Stochastic) ═══\n")

N_MC = 500
t_end = 20
dt = 0.01
n_steps = int(t_end / dt)
t_mc = np.linspace(0, t_end, n_steps + 1)

K_ensemble = np.zeros((N_MC, n_steps + 1))
Q_ensemble = np.zeros((N_MC, n_steps + 1))
q_ensemble = np.zeros((N_MC, n_steps + 1))

rng = np.random.default_rng(42)
params_mc = dict(DEFAULT_PARAMS)
params_mc['sigma_Q'] = 0.1   # 10% noise on submissions
params_mc['sigma_q'] = 0.02  # quality noise

for mc in range(N_MC):
    Q_curr, q_curr = 0.0, 1.0
    Q_ensemble[mc, 0] = Q_curr
    q_ensemble[mc, 0] = q_curr
    
    for step in range(n_steps):
        t_curr = step * dt
        Q_curr, q_curr = model_stochastic(t_curr, [Q_curr, q_curr], params_mc, dt, rng)
        Q_ensemble[mc, step+1] = Q_curr
        q_ensemble[mc, step+1] = q_curr
    
    # Compute K for this trajectory
    for step in range(n_steps + 1):
        pr_val = phi_r(Q_ensemble[mc, step], params_mc['Q_c'])
        R_val = params_mc['R_max'] * (1 + params_mc['delta'] * pr_val)
        K_ensemble[mc, step] = R_val * q_ensemble[mc, step]

# Deterministic baseline
t_det, K_det, Q_det, q_det = run_deterministic(DEFAULT_PARAMS, t_end)

# Statistics
K_mean = np.mean(K_ensemble, axis=0)
K_median = np.median(K_ensemble, axis=0)
K_p5 = np.percentile(K_ensemble, 5, axis=0)
K_p25 = np.percentile(K_ensemble, 25, axis=0)
K_p75 = np.percentile(K_ensemble, 75, axis=0)
K_p95 = np.percentile(K_ensemble, 95, axis=0)

years_mc = t_mc + 2022

fig_mc, axes_mc = plt.subplots(1, 3, figsize=(12, 4))

# K trajectories
ax = axes_mc[0]
for mc in range(min(50, N_MC)):
    ax.plot(years_mc, K_ensemble[mc], color='blue', alpha=0.05, lw=0.5)
ax.fill_between(years_mc, K_p5, K_p95, alpha=0.15, color='blue', label='5-95%')
ax.fill_between(years_mc, K_p25, K_p75, alpha=0.3, color='blue', label='25-75%')
ax.plot(years_mc, K_mean, 'b-', lw=2, label='Mean')
ax.plot(t_det + 2022, K_det, 'r--', lw=2, label='Deterministic')
ax.axhline(1.0, color='gray', ls=':', lw=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('K(t)/K₀')
ax.set_title('(a) K(t) ensemble', fontweight='bold', loc='left')
ax.legend(fontsize=7)
ax.set_ylim(0.4, 1.3)

# K(20) distribution
ax = axes_mc[1]
K20_dist = K_ensemble[:, -1]
ax.hist(K20_dist, bins=40, density=True, color='steelblue', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(K20_dist), color='red', ls='-', lw=2, label=f'Mean={np.mean(K20_dist):.3f}')
ax.axvline(np.median(K20_dist), color='orange', ls='--', lw=2, label=f'Median={np.median(K20_dist):.3f}')
ax.axvline(K_det[-1], color='green', ls=':', lw=2, label=f'Determ.={K_det[-1]:.3f}')
ax.set_xlabel('K(t=20)/K₀')
ax.set_ylabel('Density')
ax.set_title('(b) K(20yr) distribution', fontweight='bold', loc='left')
ax.legend(fontsize=7)

# Paradox onset distribution
ax = axes_mc[2]
paradox_times = []
for mc in range(N_MC):
    K_traj = K_ensemble[mc]
    # Find first time K < 1 after initial rise
    peak_idx = np.argmax(K_traj)
    cross_idx = None
    for i in range(peak_idx, len(K_traj)):
        if K_traj[i] < 1.0:
            cross_idx = i
            break
    if cross_idx is not None:
        paradox_times.append(t_mc[cross_idx])

paradox_times = np.array(paradox_times)
ax.hist(paradox_times + 2022, bins=40, density=True, color='indianred', alpha=0.7, edgecolor='white')
ax.axvline(np.mean(paradox_times) + 2022, color='red', ls='-', lw=2, label=f'Mean={np.mean(paradox_times)+2022:.1f}')
ax.axvline(np.median(paradox_times) + 2022, color='orange', ls='--', lw=2, label=f'Median={np.median(paradox_times)+2022:.1f}')
ax.set_xlabel('Year of paradox onset')
ax.set_ylabel('Density')
ax.set_title('(c) Paradox onset timing', fontweight='bold', loc='left')
ax.legend(fontsize=7)

fig_mc.tight_layout()
fig_mc.savefig('/home/claude/monte_carlo.png', dpi=300)

print(f"  N simulations: {N_MC}")
print(f"  σ_Q = {params_mc['sigma_Q']}, σ_q = {params_mc['sigma_q']}")
print(f"  K(20yr): mean={np.mean(K20_dist):.3f}, median={np.median(K20_dist):.3f}, std={np.std(K20_dist):.3f}")
print(f"  K(20yr): 5th={np.percentile(K20_dist,5):.3f}, 95th={np.percentile(K20_dist,95):.3f}")
print(f"  Deterministic K(20yr) = {K_det[-1]:.3f}")
print(f"  Paradox onset: mean={np.mean(paradox_times)+2022:.1f}, std={np.std(paradox_times):.1f} yr")
print(f"  Paradox occurs in {len(paradox_times)}/{N_MC} ({100*len(paradox_times)/N_MC:.0f}%) of trajectories")

# ═══ 3. LHS parameter uncertainty ═══
print("\n═══ LATIN HYPERCUBE SAMPLING ═══\n")

N_LHS = 500
param_bounds = {
    'gamma': (1.0, 3.5),
    'delta': (0.1, 1.5),
    'Q_c':   (0.5, 4.0),
    'lam':   (0.1, 0.8),
    'mu':    (0.2, 1.0),
    'eta':   (0.4, 1.0),
    'q_min': (0.1, 0.4),
}

rng_lhs = np.random.default_rng(123)
lhs_samples = {}
for pname, (lo, hi) in param_bounds.items():
    # Stratified sampling
    cuts = np.linspace(0, 1, N_LHS + 1)
    u = rng_lhs.uniform(cuts[:-1], cuts[1:])
    rng_lhs.shuffle(u)
    lhs_samples[pname] = lo + (hi - lo) * u

K20_lhs = np.zeros(N_LHS)
Kpeak_lhs = np.zeros(N_LHS)
paradox_lhs = np.zeros(N_LHS, dtype=bool)

for i in range(N_LHS):
    p2 = dict(DEFAULT_PARAMS)
    for pname in param_bounds:
        p2[pname] = lhs_samples[pname][i]
    t2, K2, _, _ = run_deterministic(p2)
    K20_lhs[i] = K2[-1]
    Kpeak_lhs[i] = np.max(K2)
    paradox_lhs[i] = K2[-1] < 1.0

print(f"  N samples: {N_LHS}")
print(f"  K(20yr): mean={np.mean(K20_lhs):.3f}, std={np.std(K20_lhs):.3f}")
print(f"  K(20yr): min={np.min(K20_lhs):.3f}, max={np.max(K20_lhs):.3f}")
print(f"  Paradox occurs: {np.sum(paradox_lhs)}/{N_LHS} ({100*np.mean(paradox_lhs):.0f}%)")
print(f"  K_peak: mean={np.mean(Kpeak_lhs):.3f}, range=[{np.min(Kpeak_lhs):.3f}, {np.max(Kpeak_lhs):.3f}]")

# Correlation with K(20)
print("\n  Pearson correlations with K(20yr):")
for pname in param_bounds:
    corr = np.corrcoef(lhs_samples[pname], K20_lhs)[0, 1]
    print(f"    {pname:6s}: r = {corr:+.3f}")

fig_lhs, ax_lhs = plt.subplots(1, 1, figsize=(5, 4))
ax_lhs.hist(K20_lhs, bins=40, density=True, color='teal', alpha=0.7, edgecolor='white')
ax_lhs.axvline(1.0, color='red', ls='--', lw=2, label='K₀')
ax_lhs.axvline(np.mean(K20_lhs), color='black', ls='-', lw=2, label=f'Mean={np.mean(K20_lhs):.3f}')
ax_lhs.set_xlabel('K(t=20)/K₀')
ax_lhs.set_ylabel('Density')
ax_lhs.set_title('LHS Parameter Uncertainty: K(20yr) Distribution', fontweight='bold')
ax_lhs.legend()
fig_lhs.tight_layout()
fig_lhs.savefig('/home/claude/lhs_distribution.png', dpi=300)

print("\nAll analysis complete. Figures saved.")
