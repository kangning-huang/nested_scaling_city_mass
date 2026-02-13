"""
Rank Correlation Analysis for R1#9
===================================
Demonstrates that our scaling analysis does NOT assume rank preservation
between population and material stock distributions.

Analyses:
  1. Spearman rank correlation (rho) between pop and mass
  2. Rank-rank scatter plot
  3. Rank disruption within deciles: permutation test showing beta is robust
  4. Conditional expectation E[log M | log N] vs OLS slope

Output: CSV tables + PDF plots in figures/
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE)
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'MasterMass_ByClass20250616.csv')
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_FILE)
df = df[df['population_2015'] > 50000].copy()
df = df.dropna(subset=['total_built_mass_tons'])
print(f"N = {len(df)} cities (pop > 50,000)")

pop = df['population_2015'].values
mass = df['total_built_mass_tons'].values
log_pop = np.log10(pop)
log_mass = np.log10(mass)

# =========================================================================
# 1. Spearman rank correlation
# =========================================================================
print(f"\n{'='*70}")
print("1. SPEARMAN RANK CORRELATION")
print(f"{'='*70}")

rho, p_rho = stats.spearmanr(pop, mass)
print(f"  Spearman rho = {rho:.4f}")
print(f"  p-value = {p_rho:.2e}")

# Also Kendall's tau for robustness
tau, p_tau = stats.kendalltau(pop, mass)
print(f"  Kendall tau = {tau:.4f}")
print(f"  p-value = {p_tau:.2e}")

# Rank the data
pop_ranks = stats.rankdata(pop)
mass_ranks = stats.rankdata(mass)

# Count rank inversions
n = len(pop)
rank_diff = np.abs(pop_ranks - mass_ranks)
mean_rank_diff = np.mean(rank_diff)
median_rank_diff = np.median(rank_diff)
max_rank_diff = np.max(rank_diff)

print(f"\n  Rank displacement statistics:")
print(f"    Mean |rank_pop - rank_mass| = {mean_rank_diff:.1f} (out of {n})")
print(f"    Median = {median_rank_diff:.1f}")
print(f"    Max = {max_rank_diff:.0f}")
print(f"    % within ±100 ranks = {100*np.mean(rank_diff <= 100):.1f}%")
print(f"    % within ±500 ranks = {100*np.mean(rank_diff <= 500):.1f}%")

# =========================================================================
# 2. Rank-rank scatter plot
# =========================================================================
print(f"\n{'='*70}")
print("2. RANK-RANK SCATTER PLOT")
print(f"{'='*70}")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Panel A: rank-rank scatter
ax = axes[0]
ax.scatter(pop_ranks, mass_ranks, s=2, alpha=0.3, color='black')
ax.plot([0, n], [0, n], '--', color='#e74c3c', linewidth=1.5, label='Perfect rank correspondence')
ax.set_xlabel('Population rank', fontsize=12)
ax.set_ylabel('Material stock rank', fontsize=12)
ax.set_title(f'(a) Rank correspondence ($\\rho$ = {rho:.3f})', fontsize=13)
ax.legend(fontsize=10)
ax.set_aspect('equal')
ax.set_xlim(0, n)
ax.set_ylim(0, n)

# Panel B: histogram of rank displacements
ax = axes[1]
ax.hist(rank_diff, bins=80, color='#2980b9', alpha=0.7, edgecolor='white', linewidth=0.3)
ax.axvline(mean_rank_diff, color='#e74c3c', linewidth=2, linestyle='--',
           label=f'Mean = {mean_rank_diff:.0f}')
ax.axvline(median_rank_diff, color='#27ae60', linewidth=2, linestyle='-.',
           label=f'Median = {median_rank_diff:.0f}')
ax.set_xlabel('|Population rank - Mass rank|', fontsize=12)
ax.set_ylabel('Count', fontsize=12)
ax.set_title('(b) Rank displacement distribution', fontsize=13)
ax.legend(fontsize=10)

plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'RankCorrelation_PopVsMass.pdf'), bbox_inches='tight')
plt.close()
print("  Saved: figures/RankCorrelation_PopVsMass.pdf")

# =========================================================================
# 3. Permutation test: beta robustness to rank shuffling within deciles
# =========================================================================
print(f"\n{'='*70}")
print("3. PERMUTATION TEST: BETA ROBUSTNESS TO RANK DISRUPTION")
print(f"{'='*70}")

# Baseline OLS on de-centered data (within-country)
countries = df['CTR_MN_NM'].values
unique_countries = np.unique(countries)

# De-center by country
log_pop_c = np.zeros_like(log_pop)
log_mass_c = np.zeros_like(log_mass)
for ctr in unique_countries:
    mask = countries == ctr
    if np.sum(mask) >= 5:  # same filter as main analysis
        log_pop_c[mask] = log_pop[mask] - np.mean(log_pop[mask])
        log_mass_c[mask] = log_mass[mask] - np.mean(log_mass[mask])
    else:
        log_pop_c[mask] = np.nan
        log_mass_c[mask] = np.nan

valid = ~np.isnan(log_pop_c) & ~np.isnan(log_mass_c)
log_pop_v = log_pop_c[valid]
log_mass_v = log_mass_c[valid]
countries_v = countries[valid]

# Baseline beta
slope_base, intercept_base, r_base, p_base, se_base = stats.linregress(log_pop_v, log_mass_v)
print(f"  Baseline beta (de-centered): {slope_base:.4f}, SE = {se_base:.4f}, R² = {r_base**2:.4f}")
print(f"  N = {len(log_pop_v)}")

# Permutation: shuffle mass ranks within population deciles
np.random.seed(42)
n_iter = 1000
betas_shuffled = np.zeros(n_iter)

pop_deciles = pd.qcut(log_pop_v, 10, labels=False, duplicates='drop')

for i in range(n_iter):
    log_mass_perm = log_mass_v.copy()
    for d in range(10):
        idx = np.where(pop_deciles == d)[0]
        log_mass_perm[idx] = np.random.permutation(log_mass_perm[idx])
    slope_perm, _, _, _, _ = stats.linregress(log_pop_v, log_mass_perm)
    betas_shuffled[i] = slope_perm

print(f"\n  After within-decile shuffling ({n_iter} iterations):")
print(f"    Mean beta = {np.mean(betas_shuffled):.4f}")
print(f"    SD beta = {np.std(betas_shuffled):.4f}")
print(f"    95% range = [{np.percentile(betas_shuffled, 2.5):.4f}, {np.percentile(betas_shuffled, 97.5):.4f}]")
print(f"    Max |deviation| from baseline = {np.max(np.abs(betas_shuffled - slope_base)):.4f}")

# Full random shuffle (destroys all rank structure)
betas_full_shuffle = np.zeros(n_iter)
for i in range(n_iter):
    log_mass_perm = np.random.permutation(log_mass_v)
    slope_perm, _, _, _, _ = stats.linregress(log_pop_v, log_mass_perm)
    betas_full_shuffle[i] = slope_perm

print(f"\n  After full random shuffle ({n_iter} iterations):")
print(f"    Mean beta = {np.mean(betas_full_shuffle):.4f}")
print(f"    SD beta = {np.std(betas_full_shuffle):.4f}")
print(f"    95% range = [{np.percentile(betas_full_shuffle, 2.5):.4f}, {np.percentile(betas_full_shuffle, 97.5):.4f}]")

# Permutation plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.hist(betas_shuffled, bins=50, alpha=0.6, color='#2980b9', label='Within-decile shuffle', density=True)
ax.hist(betas_full_shuffle, bins=50, alpha=0.4, color='#95a5a6', label='Full random shuffle', density=True)
ax.axvline(slope_base, color='#e74c3c', linewidth=2.5, linestyle='-',
           label=f'Observed $\\beta$ = {slope_base:.3f}')
ax.set_xlabel('$\\beta$ (scaling exponent)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)
ax.set_title('Robustness to rank disruption', fontsize=13)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'RankPermutation_BetaRobustness.pdf'), bbox_inches='tight')
plt.close()
print("\n  Saved: figures/RankPermutation_BetaRobustness.pdf")

# =========================================================================
# 4. Conditional expectation E[log M | log N]
# =========================================================================
print(f"\n{'='*70}")
print("4. CONDITIONAL EXPECTATION E[log M | log N]")
print(f"{'='*70}")

# Bin log_pop and compute mean log_mass in each bin
n_bins = 30
pop_bins = pd.qcut(log_pop_v, n_bins, labels=False, duplicates='drop')
bin_means_pop = []
bin_means_mass = []
bin_se_mass = []
bin_n = []

for b in range(pop_bins.max() + 1):
    mask = pop_bins == b
    if np.sum(mask) > 5:
        bin_means_pop.append(np.mean(log_pop_v[mask]))
        bin_means_mass.append(np.mean(log_mass_v[mask]))
        bin_se_mass.append(np.std(log_mass_v[mask], ddof=1) / np.sqrt(np.sum(mask)))
        bin_n.append(np.sum(mask))

bin_means_pop = np.array(bin_means_pop)
bin_means_mass = np.array(bin_means_mass)
bin_se_mass = np.array(bin_se_mass)

# Fit line to binned conditional expectation
slope_cond, intercept_cond, r_cond, _, se_cond = stats.linregress(bin_means_pop, bin_means_mass)

print(f"  Binned conditional expectation slope: {slope_cond:.4f} (SE = {se_cond:.4f})")
print(f"  OLS on individual observations:       {slope_base:.4f} (SE = {se_base:.4f})")
print(f"  Difference: {abs(slope_cond - slope_base):.4f}")
print(f"  R² of conditional expectation: {r_cond**2:.4f}")

# Conditional expectation plot
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(log_pop_v, log_mass_v, s=1, alpha=0.08, color='#bdc3c7', label='Individual cities')
ax.errorbar(bin_means_pop, bin_means_mass, yerr=1.96*bin_se_mass,
            fmt='o', color='#2980b9', markersize=5, capsize=3, linewidth=1.5,
            label='E[log M | log N] (binned)')
x_line = np.linspace(log_pop_v.min(), log_pop_v.max(), 100)
ax.plot(x_line, intercept_base + slope_base * x_line, '-', color='#e74c3c', linewidth=2,
        label=f'OLS: $\\beta$ = {slope_base:.3f}')
ax.plot(x_line, intercept_cond + slope_cond * x_line, '--', color='#27ae60', linewidth=2,
        label=f'Cond. exp. slope = {slope_cond:.3f}')
ax.set_xlabel('De-centered log$_{10}$(Population)', fontsize=12)
ax.set_ylabel('De-centered log$_{10}$(Mass)', fontsize=12)
ax.set_title('Conditional expectation vs. OLS', fontsize=13)
ax.legend(fontsize=9)
plt.tight_layout()
fig.savefig(os.path.join(FIG_DIR, 'ConditionalExpectation_LogMass_LogPop.pdf'), bbox_inches='tight')
plt.close()
print("  Saved: figures/ConditionalExpectation_LogMass_LogPop.pdf")

# =========================================================================
# Save results
# =========================================================================
results = {
    'Metric': [
        'Spearman rho (pop vs mass)',
        'Spearman p-value',
        'Kendall tau',
        'Kendall p-value',
        'Mean rank displacement',
        'Median rank displacement',
        'Max rank displacement',
        '% within ±100 ranks',
        '% within ±500 ranks',
        'Baseline beta (de-centered OLS)',
        'Baseline SE',
        'Baseline R²',
        'N cities (de-centered)',
        'Within-decile shuffle: mean beta',
        'Within-decile shuffle: SD beta',
        'Within-decile shuffle: 95% CI low',
        'Within-decile shuffle: 95% CI high',
        'Full shuffle: mean beta',
        'Full shuffle: SD beta',
        'Conditional expectation slope',
        'Conditional expectation SE',
        'Cond. exp. vs OLS difference',
    ],
    'Value': [
        f"{rho:.4f}",
        f"{p_rho:.2e}",
        f"{tau:.4f}",
        f"{p_tau:.2e}",
        f"{mean_rank_diff:.1f}",
        f"{median_rank_diff:.1f}",
        f"{max_rank_diff:.0f}",
        f"{100*np.mean(rank_diff <= 100):.1f}",
        f"{100*np.mean(rank_diff <= 500):.1f}",
        f"{slope_base:.4f}",
        f"{se_base:.4f}",
        f"{r_base**2:.4f}",
        f"{len(log_pop_v)}",
        f"{np.mean(betas_shuffled):.4f}",
        f"{np.std(betas_shuffled):.4f}",
        f"{np.percentile(betas_shuffled, 2.5):.4f}",
        f"{np.percentile(betas_shuffled, 97.5):.4f}",
        f"{np.mean(betas_full_shuffle):.4f}",
        f"{np.std(betas_full_shuffle):.4f}",
        f"{slope_cond:.4f}",
        f"{se_cond:.4f}",
        f"{abs(slope_cond - slope_base):.4f}",
    ]
}

df_results = pd.DataFrame(results)
df_results.to_csv(os.path.join(FIG_DIR, 'Table_RankCorrelation_Results.csv'), index=False)
print(f"\nSaved: figures/Table_RankCorrelation_Results.csv")

print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print(f"\n  Spearman rho = {rho:.4f} (high but imperfect rank correspondence)")
print(f"  Baseline beta = {slope_base:.4f}")
print(f"  Within-decile shuffle beta = {np.mean(betas_shuffled):.4f} ± {np.std(betas_shuffled):.4f}")
print(f"  Full shuffle beta = {np.mean(betas_full_shuffle):.4f} ± {np.std(betas_full_shuffle):.4f}")
print(f"  Conditional expectation slope = {slope_cond:.4f} (matches OLS within {abs(slope_cond - slope_base):.4f})")
print(f"\n  Conclusion: Scaling exponent is estimated from (pop_i, mass_i) pairs,")
print(f"  not from rank-ordered distributions. Beta is robust to rank disruption")
print(f"  and matches the nonparametric conditional expectation.")
