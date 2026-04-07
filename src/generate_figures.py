"""
Generate all 4 figures for Publish and Perish paper.
Uses final model results from JSON + runs sensitivity/policy sweeps.
"""
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import json

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans', 'Arial'],
    'font.size': 9, 'axes.labelsize': 11, 'axes.titlesize': 11,
    'figure.dpi': 300, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'axes.spines.top': False, 'axes.spines.right': False,
    'mathtext.default': 'regular',
    'legend.fontsize': 8, 'legend.frameon': False,
})

# Load results
with open('/mnt/user-data/uploads/final_model_results.json') as f:
    data = json.load(f)

t = np.array(data['t'])
Q_t = np.array(data['Q'])
q_t = np.array(data['q'])
K_t = np.array(data['K'])
R_t = np.array(data['R'])
S_t = np.array(data['S'])
pw_t = np.array(data['phi_w'])
pr_t = np.array(data['phi_r'])
params = data['params']

# Model functions (for sweeps)
def logistic(t, k, t_mid):
    return 1.0 / (1.0 + np.exp(-k * (t - t_mid)))

k_w, t_w = 1.0, 3.0
phi_w = lambda t: logistic(t, k_w, t_w)

def phi_r_func(Q, Q_c):
    return Q / (Q + Q_c)

def model(t, y, p):
    Q, q = y
    Q = max(Q, 0.0)
    q = np.clip(q, p['q_min'], 1.0)
    pw = phi_w(t)
    pr = phi_r_func(Q, p['Q_c'])
    S = p['S0'] * (1 + p['gamma'] * pw)
    R = p['R_max'] * (1 + p['delta'] * pr)
    dQ = max(S - R, -Q/0.01)
    dq = -p['lam'] * pr * (q - p['q_min']) + p['mu'] * (1 - p['eta']*pr) * (1 - q)
    return [dQ, dq]

y0 = [0.0, 1.0]

# Colors
C_BLUE = '#2166AC'
C_RED = '#B2182B'
C_ORANGE = '#E08214'
C_GREEN = '#1B7837'
C_PURPLE = '#762A83'
C_GRAY = '#666666'
C_LIGHTBLUE = '#4393C3'
C_LIGHTRED = '#D6604D'

years = t + 2022

# ===== FIGURE 1: Model dynamics (4 panels) =====
fig1, axes1 = plt.subplots(2, 2, figsize=(7.0, 5.5))

# (a) phi_w(t) and phi_r(t)
ax = axes1[0, 0]
ax.plot(years, pw_t, color=C_BLUE, lw=2, label=r'$\phi_w(t)$ (writing)')
ax.plot(years, pr_t, color=C_RED, lw=2, label=r'$\phi_r(t)$ (review)')
ax.set_xlabel('Year')
ax.set_ylabel('AI adoption fraction')
ax.set_ylim(-0.05, 1.05)
ax.legend(loc='center right')
ax.set_title('(a) AI adoption dynamics', fontsize=10, fontweight='bold', loc='left')
ax.axhline(0.5, color='gray', ls=':', lw=0.7, alpha=0.5)

# (b) S(t), R(t), Q(t)
ax = axes1[0, 1]
ax.plot(years, S_t, color=C_GREEN, lw=2, label='S(t)')
ax.plot(years, R_t, color=C_ORANGE, lw=2, label='R(t)')
ax2 = ax.twinx()
ax2.plot(years, Q_t, color=C_GRAY, lw=1.5, ls='--', label='Q(t)')
ax2.set_ylabel('Queue Q', color=C_GRAY)
ax2.tick_params(axis='y', labelcolor=C_GRAY)
ax2.spines['right'].set_visible(True)
ax2.spines['right'].set_color(C_GRAY)
ax.set_xlabel('Year')
ax.set_ylabel('Rate (normalized)')
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='center left')
ax.set_title('(b) Submissions, reviews, and queue', fontsize=10, fontweight='bold', loc='left')

# (c) q(t)
ax = axes1[1, 0]
ax.plot(years, q_t, color=C_PURPLE, lw=2)
ax.axhline(params['q_min'], color=C_RED, ls='--', lw=1, alpha=0.7)
ax.text(2040, params['q_min'] + 0.02, r'$q_{min}$', color=C_RED, fontsize=8)
ax.axhline(data['analytical']['q_ss'], color=C_GRAY, ls=':', lw=1, alpha=0.7)
ax.text(2040, data['analytical']['q_ss'] + 0.02, r'$q_{ss}$', color=C_GRAY, fontsize=8)
ax.set_xlabel('Year')
ax.set_ylabel('Verification quality q')
ax.set_ylim(0, 1.05)
ax.set_title('(c) Verification quality', fontsize=10, fontweight='bold', loc='left')

# (d) K(t)/K0
ax = axes1[1, 1]
ax.plot(years, K_t, color=C_BLUE, lw=2.5)
ax.axhline(1.0, color='gray', ls='--', lw=1, alpha=0.7)
ax.text(2040, 1.015, r'$K_0$', color='gray', fontsize=8)
i_peak = np.argmax(K_t)
ax.plot(years[i_peak], K_t[i_peak], 'o', color=C_GREEN, ms=6, zorder=5)
ax.annotate(f'Peak: {K_t[i_peak]:.2f}$K_0$\n({years[i_peak]:.0f})',
            xy=(years[i_peak], K_t[i_peak]), xytext=(years[i_peak]+3, K_t[i_peak]+0.02),
            fontsize=7, color=C_GREEN, arrowprops=dict(arrowstyle='->', color=C_GREEN, lw=0.8))
i_cross = np.argmax(K_t < 1.0)
ax.plot(years[i_cross], K_t[i_cross], 's', color=C_RED, ms=5, zorder=5)
ax.annotate(f'Paradox onset\n({years[i_cross]:.0f})',
            xy=(years[i_cross], K_t[i_cross]), xytext=(years[i_cross]+2, 0.92),
            fontsize=7, color=C_RED, arrowprops=dict(arrowstyle='->', color=C_RED, lw=0.8))
ax.axvspan(2022, years[i_cross], alpha=0.06, color='green')
ax.axvspan(years[i_cross], 2042, alpha=0.06, color='red')
ax.set_xlabel('Year')
ax.set_ylabel(r'$K(t)/K_0$')
ax.set_ylim(0.5, 1.20)
ax.set_title(r'(d) Knowledge output $K/K_0$', fontsize=10, fontweight='bold', loc='left')
ax.axhline(data['analytical']['K_ss_over_K0'], color=C_RED, ls=':', lw=1, alpha=0.5)
ax.text(2040, data['analytical']['K_ss_over_K0'] - 0.03, r'$K_{ss}$', color=C_RED, fontsize=8)

fig1.tight_layout()
fig1.savefig('/home/claude/figure1.png', dpi=300)
print("Figure 1 saved.")

# ===== FIGURE 2: Empirical validation (4 panels) =====
fig2, axes2 = plt.subplots(2, 2, figsize=(7.0, 5.5))

neurips = {2008:1235, 2009:1105, 2010:1678, 2011:1400, 2012:1467, 2013:1420,
           2014:1678, 2015:1838, 2016:2403, 2017:3240, 2018:4856, 2019:6743,
           2020:9467, 2021:9122, 2022:10411, 2023:13330, 2024:15671, 2025:21575}

iclr = {2013:30, 2014:79, 2015:170, 2016:507, 2017:490, 2018:935,
        2019:1580, 2020:2594, 2021:2997, 2022:3391, 2023:4966,
        2024:7304, 2025:11565, 2026:19631}

arxiv = {2008:5200, 2009:5500, 2010:5900, 2011:6200, 2012:6700,
         2013:7200, 2014:7800, 2015:8500, 2016:9400, 2017:10500,
         2018:11800, 2019:13000, 2020:15000, 2021:16500,
         2022:17500, 2023:19500, 2024:21500, 2025:24000}

biorxiv = {2014:824, 2015:1823, 2016:4113, 2017:9871, 2018:17874,
           2019:27773, 2020:38100, 2021:42653, 2022:44382,
           2023:45500, 2024:47000, 2025:48000}

chatgpt_year = 2022

def plot_venue(ax, data_dict, title, label, chatgpt_yr=2022):
    yrs = sorted(data_dict.keys())
    vals = [data_dict[y] for y in yrs]
    pre = [(y, v) for y, v in zip(yrs, vals) if y <= chatgpt_yr]
    post = [(y, v) for y, v in zip(yrs, vals) if y >= chatgpt_yr]
    ax.plot([p[0] for p in pre], [p[1] for p in pre], 'o-', color=C_BLUE, ms=4, lw=1.5, label='Pre-ChatGPT')
    ax.plot([p[0] for p in post], [p[1] for p in post], 's-', color=C_RED, ms=4, lw=1.5, label='Post-ChatGPT')
    ax.axvline(chatgpt_yr, color='gray', ls=':', lw=0.8, alpha=0.5)
    if 2020 in data_dict and chatgpt_yr in data_dict:
        n_pre = chatgpt_yr - 2020
        cagr_pre = (data_dict[chatgpt_yr] / data_dict[2020]) ** (1/n_pre) - 1
        ax.text(0.05, 0.92, f'Pre CAGR: {cagr_pre*100:.0f}%', transform=ax.transAxes,
                fontsize=7, color=C_BLUE, va='top')
    latest_yr = max(yrs)
    if chatgpt_yr in data_dict:
        n_post = latest_yr - chatgpt_yr
        if n_post > 0:
            cagr_post = (data_dict[latest_yr] / data_dict[chatgpt_yr]) ** (1/n_post) - 1
            ax.text(0.05, 0.82, f'Post CAGR: {cagr_post*100:.0f}%', transform=ax.transAxes,
                    fontsize=7, color=C_RED, va='top')
    ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
    ax.set_ylabel(label)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x/1000:.0f}k' if x >= 1000 else f'{x:.0f}'))

plot_venue(axes2[0,0], neurips, '(a) NeurIPS', 'Submissions')
plot_venue(axes2[0,1], iclr, '(b) ICLR', 'Submissions')
plot_venue(axes2[1,0], arxiv, '(c) arXiv (monthly)', 'Monthly submissions')
plot_venue(axes2[1,1], biorxiv, '(d) bioRxiv (annual)', 'Annual preprints')
axes2[0,0].legend(loc='upper left', fontsize=7)

fig2.tight_layout()
fig2.savefig('/home/claude/figure2.png', dpi=300)
print("Figure 2 saved.")

# ===== FIGURE 3: gamma-delta parameter space (heatmap) =====
fig3, ax3 = plt.subplots(1, 1, figsize=(4.5, 4.0))

gamma_range = np.linspace(0.2, 4.0, 60)
delta_range = np.linspace(0.0, 3.0, 60)
K_grid = np.zeros((len(delta_range), len(gamma_range)))

for i, d in enumerate(delta_range):
    for j, g in enumerate(gamma_range):
        p2 = dict(params)
        p2['gamma'] = g
        p2['delta'] = d
        sol2 = solve_ivp(model, (0, 20), y0, args=(p2,),
                         t_eval=[20], method='RK45', max_step=0.1,
                         rtol=1e-8, atol=1e-10)
        Q20, q20 = sol2.y[:, -1]
        pr20 = phi_r_func(Q20, params['Q_c'])
        R20 = params['R_max'] * (1 + d * pr20)
        K_grid[i, j] = R20 * q20

im = ax3.pcolormesh(gamma_range, delta_range, K_grid,
                     cmap='RdBu', vmin=0.4, vmax=1.6, shading='auto')
cb = fig3.colorbar(im, ax=ax3, label=r'$K(t=20)/K_0$', shrink=0.9)

cs = ax3.contour(gamma_range, delta_range, K_grid, levels=[1.0],
                  colors='black', linewidths=2)
ax3.clabel(cs, fmt=r'$K/K_0=1$', fontsize=8)

ax3.plot(params['gamma'], params['delta'], '*', color='black', ms=12, zorder=10)
ax3.annotate('Current\noperating\npoint', xy=(params['gamma'], params['delta']),
             xytext=(params['gamma']+0.5, params['delta']+0.5),
             fontsize=8, arrowprops=dict(arrowstyle='->', lw=0.8))

g_line = np.linspace(0.2, 3.0, 50)
ax3.plot(g_line, g_line, '--', color='white', lw=1.5, alpha=0.7)
ax3.text(2.5, 2.7, r'$\delta = \gamma$', color='white', fontsize=9, rotation=40)

ax3.set_xlabel(r'Writing acceleration $\gamma$')
ax3.set_ylabel(r'Review acceleration $\delta$')
ax3.set_title(r'$K/K_0$ at $t = 20$ yr', fontsize=11, fontweight='bold')

ax3.text(3.0, 0.5, 'Paradox\nregime', color='white', fontsize=10, fontweight='bold', ha='center')
ax3.text(0.8, 2.5, 'Benefit\nregime', color='white', fontsize=10, fontweight='bold', ha='center')

fig3.tight_layout()
fig3.savefig('/home/claude/figure3.png', dpi=300)
print("Figure 3 saved.")

# ===== FIGURE 4: Policy levers (2 panels) =====
fig4, axes4 = plt.subplots(1, 2, figsize=(7.0, 3.5))

# (a) Varying delta
ax = axes4[0]
for d_val, col, lab in [(0.0, C_LIGHTRED, r'$\delta=0$'),
                         (0.5, C_RED, r'$\delta=0.5$ (baseline)'),
                         (1.0, C_ORANGE, r'$\delta=1.0$'),
                         (1.5, C_GRAY, r'$\delta=1.5$'),
                         (2.0, C_GREEN, r'$\delta=2.0$ (critical)'),
                         (2.5, C_BLUE, r'$\delta=2.5$')]:
    p2 = dict(params)
    p2['delta'] = d_val
    sol2 = solve_ivp(model, (0, 20), y0, args=(p2,),
                     t_eval=np.linspace(0, 20, 500), method='RK45',
                     max_step=0.05, rtol=1e-9, atol=1e-11)
    t2 = sol2.t + 2022
    Q2, q2 = sol2.y
    pr2 = np.array([phi_r_func(Qi, p2['Q_c']) for Qi in Q2])
    R2 = p2['R_max'] * (1 + d_val * pr2)
    K2 = R2 * q2
    lw = 2.5 if d_val == 0.5 else 1.5
    ax.plot(t2, K2, color=col, lw=lw, label=lab)

ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel(r'$K(t)/K_0$')
ax.set_ylim(0.3, 1.5)
ax.legend(fontsize=7, loc='lower left')
ax.set_title(r'(a) Effect of review acceleration $\delta$', fontsize=10, fontweight='bold', loc='left')

# (b) Varying q_min
ax = axes4[1]
for qm_val, col, lab in [(0.1, C_LIGHTRED, r'$q_{min}=0.1$'),
                           (0.2, C_RED, r'$q_{min}=0.2$ (baseline)'),
                           (0.3, C_ORANGE, r'$q_{min}=0.3$'),
                           (0.4, C_GRAY, r'$q_{min}=0.4$'),
                           (0.5, C_GREEN, r'$q_{min}=0.5$'),
                           (0.6, C_BLUE, r'$q_{min}=0.6$')]:
    p2 = dict(params)
    p2['q_min'] = qm_val
    sol2 = solve_ivp(model, (0, 20), y0, args=(p2,),
                     t_eval=np.linspace(0, 20, 500), method='RK45',
                     max_step=0.05, rtol=1e-9, atol=1e-11)
    t2 = sol2.t + 2022
    Q2, q2 = sol2.y
    pr2 = np.array([phi_r_func(Qi, p2['Q_c']) for Qi in Q2])
    R2 = p2['R_max'] * (1 + p2['delta'] * pr2)
    K2 = R2 * q2
    lw = 2.5 if qm_val == 0.2 else 1.5
    ax.plot(t2, K2, color=col, lw=lw, label=lab)

ax.axhline(1.0, color='gray', ls='--', lw=0.8, alpha=0.5)
ax.set_xlabel('Year')
ax.set_ylabel(r'$K(t)/K_0$')
ax.set_ylim(0.3, 1.5)
ax.legend(fontsize=7, loc='lower left')
ax.set_title(r'(b) Effect of quality floor $q_{min}$', fontsize=10, fontweight='bold', loc='left')

fig4.tight_layout()
fig4.savefig('/home/claude/figure4.png', dpi=300)
print("Figure 4 saved.")

print("\nAll figures generated successfully!")
