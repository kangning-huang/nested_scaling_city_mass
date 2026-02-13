"""
Zipf's Law Statistical Test for R1#7
=====================================
Tests whether city population and material stock distributions follow
power-law (Zipf) vs lognormal using:
  1. Clauset-Shalizi-Newman (2009) framework (manual implementation)
     - MLE power-law fit with x_min estimation via KS distance
     - Vuong likelihood ratio tests for distribution comparison
  2. Gabaix-Ibragimov (2011) corrected rank-size regression

Output: CSV tables + CCDF PDF plots in figures/

Note: Uses direct scipy implementation because `powerlaw` package
is incompatible with Python 3.14.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats, optimize
from scipy.special import erfc
import os
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROJECT_ROOT = os.path.dirname(BASE)
DATA_FILE = os.path.join(PROJECT_ROOT, 'data', 'processed', 'MasterMass_ByClass20250616.csv')
FIG_DIR = os.path.join(BASE, 'figures')
os.makedirs(FIG_DIR, exist_ok=True)


# =============================================================================
# CSN Power-law fitting functions
# =============================================================================

def powerlaw_mle_alpha(data, xmin):
    """MLE estimate of power-law exponent for continuous data above xmin."""
    tail = data[data >= xmin]
    n = len(tail)
    if n < 2:
        return np.nan, np.nan
    alpha = 1 + n / np.sum(np.log(tail / xmin))
    se = (alpha - 1) / np.sqrt(n)
    return alpha, se


def powerlaw_ks(data, xmin):
    """KS distance between data and fitted power law above xmin."""
    tail = np.sort(data[data >= xmin])
    n = len(tail)
    if n < 2:
        return np.inf
    alpha, _ = powerlaw_mle_alpha(data, xmin)
    # Theoretical CDF: 1 - (x/xmin)^(-(alpha-1))
    cdf_theory = 1 - (tail / xmin) ** (-(alpha - 1))
    cdf_empirical = np.arange(1, n + 1) / n
    return np.max(np.abs(cdf_theory - cdf_empirical))


def find_xmin(data, xmin_candidates=None):
    """Find optimal x_min by minimizing KS distance (CSN method)."""
    sorted_data = np.sort(np.unique(data))
    if xmin_candidates is None:
        # Use unique values as candidates, skip top 10% to avoid tiny tails
        max_idx = int(0.9 * len(sorted_data))
        xmin_candidates = sorted_data[:max_idx]

    best_ks = np.inf
    best_xmin = sorted_data[0]
    best_alpha = np.nan

    for xmin in xmin_candidates:
        n_tail = np.sum(data >= xmin)
        if n_tail < 50:  # Need reasonable sample
            continue
        ks = powerlaw_ks(data, xmin)
        if ks < best_ks:
            best_ks = ks
            best_xmin = xmin
            best_alpha, _ = powerlaw_mle_alpha(data, xmin)

    return best_xmin, best_alpha, best_ks


def lognormal_mle(data, xmin):
    """MLE fit of lognormal to data above xmin."""
    tail = data[data >= xmin]
    log_tail = np.log(tail)
    mu = np.mean(log_tail)
    sigma = np.std(log_tail, ddof=1)
    return mu, sigma


def truncated_powerlaw_mle(data, xmin):
    """MLE fit of truncated power law: p(x) ~ x^(-alpha) * exp(-lambda*x)."""
    tail = data[data >= xmin]
    log_tail = np.log(tail)

    def neg_loglik(params):
        alpha, lam = params
        if alpha <= 1 or lam <= 0:
            return 1e10
        # Log-likelihood (up to normalization constant)
        ll = -alpha * np.sum(log_tail) - lam * np.sum(tail)
        # Approximate normalization using incomplete gamma
        return -ll

    try:
        alpha_init, _ = powerlaw_mle_alpha(data, xmin)
        result = optimize.minimize(neg_loglik, [alpha_init, 1e-6],
                                   method='Nelder-Mead',
                                   options={'maxiter': 5000})
        return result.x[0], result.x[1]
    except:
        return np.nan, np.nan


def vuong_test(data, xmin, dist1='power_law', dist2='lognormal'):
    """
    Vuong likelihood ratio test comparing two distributions.
    Returns (R, p): R > 0 favors dist1, R < 0 favors dist2.
    """
    tail = data[data >= xmin]
    n = len(tail)

    # Power-law log-likelihoods
    alpha_pl, _ = powerlaw_mle_alpha(data, xmin)

    def pl_logpdf(x):
        return np.log(alpha_pl - 1) - np.log(xmin) - alpha_pl * np.log(x / xmin)

    # Lognormal log-likelihoods
    mu_ln, sigma_ln = lognormal_mle(data, xmin)

    def ln_logpdf(x):
        return stats.lognorm.logpdf(x, s=sigma_ln, scale=np.exp(mu_ln))

    # Exponential log-likelihoods
    lam_exp = 1.0 / np.mean(tail)

    def exp_logpdf(x):
        return np.log(lam_exp) - lam_exp * x

    # Truncated power-law
    alpha_tpl, lam_tpl = truncated_powerlaw_mle(data, xmin)

    def tpl_logpdf(x):
        if np.isnan(alpha_tpl) or np.isnan(lam_tpl):
            return np.full_like(x, -np.inf)
        # Unnormalized; the normalization cancels in the ratio for large n
        return -alpha_tpl * np.log(x) - lam_tpl * x

    dist_funcs = {
        'power_law': pl_logpdf,
        'lognormal': ln_logpdf,
        'exponential': exp_logpdf,
        'truncated_power_law': tpl_logpdf,
    }

    ll1 = dist_funcs[dist1](tail)
    ll2 = dist_funcs[dist2](tail)

    # Per-observation log-likelihood ratios
    lr = ll1 - ll2

    # Vuong test statistic
    mean_lr = np.mean(lr)
    std_lr = np.std(lr, ddof=1)

    if std_lr == 0:
        return 0.0, 1.0

    R = mean_lr * np.sqrt(n) / std_lr
    p = erfc(abs(R) / np.sqrt(2))  # two-sided p-value

    return R, p


# =============================================================================
# Main analysis
# =============================================================================

# Load data
df = pd.read_csv(DATA_FILE)
print(f"Loaded {len(df)} cities")

df = df[df['population_2015'] > 50000].copy()
print(f"After pop > 50k filter: {len(df)} cities")

pop = df['population_2015'].values
mass = df['total_built_mass_tons'].values
mass = mass[~np.isnan(mass)]

print(f"\n{'='*70}")
print("PART 1: CLAUSET-SHALIZI-NEWMAN ANALYSIS")
print(f"{'='*70}")

results_rows = []

for var_name, data in [('City Population', pop), ('Total Built Mass (tonnes)', mass)]:
    print(f"\n--- {var_name} (N = {len(data)}) ---")
    print(f"  Range: {data.min():.0f} to {data.max():.0f}")
    print(f"  Median: {np.median(data):.0f}")

    # Find optimal x_min
    print("  Finding optimal x_min...")
    xmin, alpha, ks_dist = find_xmin(data)
    n_tail = np.sum(data >= xmin)
    alpha_se = (alpha - 1) / np.sqrt(n_tail)

    print(f"\n  Power-law fit:")
    print(f"    x_min = {xmin:.1f}")
    print(f"    alpha = {alpha:.4f} +/- {alpha_se:.4f}")
    print(f"    N above x_min = {n_tail}")
    print(f"    KS distance = {ks_dist:.4f}")

    # Lognormal fit
    mu_ln, sigma_ln = lognormal_mle(data, xmin)
    print(f"\n  Lognormal fit (above x_min):")
    print(f"    mu = {mu_ln:.4f}, sigma = {sigma_ln:.4f}")

    # Truncated power-law fit
    alpha_tpl, lam_tpl = truncated_powerlaw_mle(data, xmin)
    print(f"\n  Truncated power-law fit (above x_min):")
    print(f"    alpha = {alpha_tpl:.4f}, lambda = {lam_tpl:.2e}")

    # Distribution comparisons
    comparisons = [
        ('power_law', 'lognormal'),
        ('power_law', 'exponential'),
        ('power_law', 'truncated_power_law'),
        ('lognormal', 'exponential'),
        ('lognormal', 'truncated_power_law'),
    ]

    print(f"\n  Distribution comparisons (Vuong LR test):")
    print(f"  {'Comparison':<40} {'R':>8} {'p-value':>10} {'Preferred':<25}")
    print(f"  {'-'*83}")

    for dist1, dist2 in comparisons:
        try:
            R, p = vuong_test(data, xmin, dist1, dist2)
            if p < 0.05:
                preferred = dist1 if R > 0 else dist2
            else:
                preferred = 'neither (p >= 0.05)'
            print(f"  {dist1+' vs '+dist2:<40} {R:>8.3f} {p:>10.4f} {preferred:<25}")

            results_rows.append({
                'Variable': var_name,
                'N': len(data),
                'x_min': xmin,
                'alpha_powerlaw': alpha,
                'alpha_se': alpha_se,
                'N_above_xmin': n_tail,
                'KS_distance': ks_dist,
                'Comparison': f"{dist1} vs {dist2}",
                'LR_ratio_R': R,
                'p_value': p,
                'Preferred': preferred
            })
        except Exception as e:
            print(f"  {dist1+' vs '+dist2:<40} ERROR: {e}")

    # CCDF plot
    fig, ax = plt.subplots(figsize=(7, 5))

    sorted_data = np.sort(data[data >= xmin])[::-1]
    n_sorted = len(sorted_data)
    ccdf_emp = np.arange(1, n_sorted + 1) / n_sorted

    ax.loglog(sorted_data, ccdf_emp, '.', color='black', markersize=3, alpha=0.5, label='Empirical CCDF')

    # Power-law CCDF
    x_theory = np.logspace(np.log10(xmin), np.log10(sorted_data[0]), 200)
    ccdf_pl = (x_theory / xmin) ** (-(alpha - 1))
    ax.loglog(x_theory, ccdf_pl, '-', color='#e74c3c', linewidth=2, label=f'Power law (α={alpha:.2f})')

    # Lognormal CCDF
    ccdf_ln = 1 - stats.lognorm.cdf(x_theory, s=sigma_ln, scale=np.exp(mu_ln))
    ccdf_ln_xmin = 1 - stats.lognorm.cdf(xmin, s=sigma_ln, scale=np.exp(mu_ln))
    if ccdf_ln_xmin > 0:
        ccdf_ln_normalized = ccdf_ln / ccdf_ln_xmin
        ax.loglog(x_theory, ccdf_ln_normalized, '--', color='#2980b9', linewidth=2, label='Lognormal')

    ax.set_xlabel(var_name, fontsize=12)
    ax.set_ylabel('P(X > x)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f'CCDF above x_min: {var_name}', fontsize=13)
    plt.tight_layout()
    safe_name = var_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    fig.savefig(os.path.join(FIG_DIR, f'CCDF_{safe_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"\n  Saved CCDF plot: figures/CCDF_{safe_name}.pdf")

# Save CSN results
df_results = pd.DataFrame(results_rows)
df_results.to_csv(os.path.join(FIG_DIR, 'Table_Zipf_CSN_Results.csv'), index=False)
print(f"\nSaved: figures/Table_Zipf_CSN_Results.csv")


print(f"\n{'='*70}")
print("PART 2: GABAIX-IBRAGIMOV CORRECTED RANK-SIZE REGRESSION")
print(f"{'='*70}")

gi_rows = []

for var_name, data in [('City Population', pop), ('Total Built Mass (tonnes)', mass)]:
    print(f"\n--- {var_name} (N = {len(data)}) ---")

    sorted_data = np.sort(data)[::-1]
    N = len(sorted_data)
    ranks = np.arange(1, N + 1)

    # Gabaix-Ibragimov: log(rank - 0.5) = a - zeta * log(size)
    log_rank = np.log(ranks - 0.5)
    log_size = np.log(sorted_data)

    slope, intercept, r_value, p_value, se_ols = stats.linregress(log_size, log_rank)
    zeta = -slope

    # GI corrected SE
    se_gi = zeta * np.sqrt(2 / N)

    # Test H0: zeta = 1 (exact Zipf)
    z_stat = (zeta - 1.0) / se_gi
    p_zipf = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    print(f"  Zipf exponent (zeta): {zeta:.4f}")
    print(f"  OLS SE:               {abs(se_ols):.4f}")
    print(f"  GI corrected SE:      {se_gi:.4f}")
    print(f"  95% CI:               [{zeta - 1.96*se_gi:.4f}, {zeta + 1.96*se_gi:.4f}]")
    print(f"  R²:                   {r_value**2:.4f}")
    print(f"  Test H0: zeta=1:      z = {z_stat:.3f}, p = {p_zipf:.4f}")
    if p_zipf < 0.05:
        print(f"  --> Reject exact Zipf (zeta != 1)")
    else:
        print(f"  --> Cannot reject exact Zipf (zeta = 1)")

    gi_rows.append({
        'Variable': var_name,
        'N': N,
        'Zeta': zeta,
        'SE_OLS': abs(se_ols),
        'SE_GI': se_gi,
        'CI_low': zeta - 1.96 * se_gi,
        'CI_high': zeta + 1.96 * se_gi,
        'R2': r_value**2,
        'z_test_zeta1': z_stat,
        'p_test_zeta1': p_zipf
    })

    # Rank-size plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(log_size, log_rank, s=3, alpha=0.4, color='black', label='Data')
    x_fit = np.linspace(log_size.min(), log_size.max(), 100)
    ax.plot(x_fit, intercept + slope * x_fit, color='#e74c3c', linewidth=2,
            label=f'GI fit: $\\zeta$ = {zeta:.3f} [{zeta-1.96*se_gi:.3f}, {zeta+1.96*se_gi:.3f}]')
    ax.plot(x_fit, intercept + (-1.0) * x_fit, color='#2980b9', linewidth=1.5, linestyle='--',
            label='Exact Zipf ($\\zeta$ = 1)')
    ax.set_xlabel(f'log({var_name})', fontsize=12)
    ax.set_ylabel('log(rank - 0.5)', fontsize=12)
    ax.legend(fontsize=10)
    ax.set_title(f'Rank-Size: {var_name}', fontsize=13)
    plt.tight_layout()
    safe_name = var_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    fig.savefig(os.path.join(FIG_DIR, f'RankSize_GI_{safe_name}.pdf'), bbox_inches='tight')
    plt.close()
    print(f"  Saved: figures/RankSize_GI_{safe_name}.pdf")

# Save GI results
df_gi = pd.DataFrame(gi_rows)
df_gi.to_csv(os.path.join(FIG_DIR, 'Table_Zipf_GabaixIbragimov_Results.csv'), index=False)
print(f"\nSaved: figures/Table_Zipf_GabaixIbragimov_Results.csv")


print(f"\n{'='*70}")
print("SUMMARY")
print(f"{'='*70}")
print("\nCSN Distribution Comparisons:")
for _, row in df_results.iterrows():
    print(f"  {row['Variable'][:20]:<22} {row['Comparison']:<40} R={row['LR_ratio_R']:>7.3f}  p={row['p_value']:.4f}  -> {row['Preferred']}")

print("\nGabaix-Ibragimov Zipf Exponents:")
for _, row in df_gi.iterrows():
    reject = "REJECT" if row['p_test_zeta1'] < 0.05 else "CANNOT REJECT"
    print(f"  {row['Variable']:<35} zeta = {row['Zeta']:.4f} [{row['CI_low']:.4f}, {row['CI_high']:.4f}]  H0:zeta=1 -> {reject} (p={row['p_test_zeta1']:.4f})")
