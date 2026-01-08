#!/usr/bin/env python3
"""
Analyze relationship between mobility mass deviation from scaling and network centrality.

Hypothesis: Higher centrality hexagons have more mobility infrastructure serving
through-traffic, leading to POSITIVE deviation from neighborhood-level scaling.
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path

def main():
    # Load merged data
    data_path = Path(__file__).parent.parent.parent.parent / "results" / "merged_mass_centrality.csv"

    print("=" * 70)
    print("MOBILITY DEVIATION vs POPULATION-WEIGHTED CENTRALITY ANALYSIS")
    print("=" * 70)

    df = pd.read_csv(data_path)
    print(f"\nLoaded {len(df)} neighborhoods from {len(df['city_name'].unique())} cities")

    # Filter to rows with valid centrality data
    df_valid = df[df['weighted_centrality'] > 0].copy()
    print(f"Neighborhoods with centrality data: {len(df_valid)}")

    # Key variables
    # deviation_log_mobility: log deviation from expected mobility mass based on population
    # weighted_centrality: population-weighted betweenness centrality

    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST")
    print("=" * 70)
    print("\nHypothesis: Higher centrality → More mobility mass serving through-traffic")
    print("            → POSITIVE deviation from expected mobility mass")
    print("Expected: Positive correlation between centrality and mobility deviation")

    # Global correlation
    print("\n" + "-" * 50)
    print("GLOBAL CORRELATION (all neighborhoods)")
    print("-" * 50)

    x = df_valid['weighted_centrality']
    y = df_valid['deviation_log_mobility']

    # Spearman (rank-based, robust)
    spearman_r, spearman_p = stats.spearmanr(x, y)
    print(f"\nSpearman ρ (raw values): {spearman_r:+.4f}  (p = {spearman_p:.2e})")

    # Pearson with log-transformed centrality
    log_x = np.log10(x + 1)
    pearson_r, pearson_p = stats.pearsonr(log_x, y)
    print(f"Pearson r (log centrality): {pearson_r:+.4f}  (p = {pearson_p:.2e})")

    # Interpret direction
    if spearman_r > 0:
        print("\n→ POSITIVE correlation: Higher centrality = MORE mobility mass than expected")
        print("  This CONTRADICTS the hypothesis that central hexagons serve through-traffic")
        print("  (If true, central hexagons would have LESS deviation, not more)")
    else:
        print("\n→ NEGATIVE correlation: Higher centrality = LESS mobility mass than expected")
        print("  This SUPPORTS the hypothesis (central areas share mobility infrastructure)")

    # Correlation with percentage deviation (more interpretable)
    print("\n" + "-" * 50)
    print("USING PERCENTAGE DEVIATION (more interpretable)")
    print("-" * 50)

    y_pct = df_valid['pct_deviation_mobility']
    spearman_pct, p_pct = stats.spearmanr(x, y_pct)
    print(f"\nSpearman ρ (centrality vs % deviation): {spearman_pct:+.4f}  (p = {p_pct:.2e})")

    # Descriptive stats by centrality quartile
    print("\n" + "-" * 50)
    print("MOBILITY DEVIATION BY CENTRALITY QUARTILE")
    print("-" * 50)

    df_valid['centrality_quartile'] = pd.qcut(df_valid['weighted_centrality'], 4, labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)'])

    quartile_stats = df_valid.groupby('centrality_quartile').agg({
        'deviation_log_mobility': ['mean', 'median', 'std'],
        'pct_deviation_mobility': ['mean', 'median'],
        'weighted_centrality': 'mean'
    }).round(4)

    print("\n{:<12} {:>12} {:>12} {:>15} {:>15}".format(
        "Quartile", "Log Dev Mean", "Log Dev Med", "% Dev Mean", "% Dev Median"))
    print("-" * 70)

    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        q_data = df_valid[df_valid['centrality_quartile'] == q]
        log_mean = q_data['deviation_log_mobility'].mean()
        log_med = q_data['deviation_log_mobility'].median()
        pct_mean = q_data['pct_deviation_mobility'].mean()
        pct_med = q_data['pct_deviation_mobility'].median()
        print(f"{q:<12} {log_mean:>+12.4f} {log_med:>+12.4f} {pct_mean:>+14.1f}% {pct_med:>+14.1f}%")

    # T-test: high vs low centrality
    print("\n" + "-" * 50)
    print("T-TEST: HIGH vs LOW CENTRALITY")
    print("-" * 50)

    high_cent = df_valid[df_valid['centrality_quartile'] == 'Q4 (high)']['deviation_log_mobility']
    low_cent = df_valid[df_valid['centrality_quartile'] == 'Q1 (low)']['deviation_log_mobility']

    t_stat, t_p = stats.ttest_ind(high_cent, low_cent)
    print(f"\nHigh centrality (Q4): mean = {high_cent.mean():+.4f}, n = {len(high_cent)}")
    print(f"Low centrality (Q1):  mean = {low_cent.mean():+.4f}, n = {len(low_cent)}")
    print(f"\nT-statistic: {t_stat:.3f}")
    print(f"P-value: {t_p:.2e}")

    diff = high_cent.mean() - low_cent.mean()
    print(f"\nDifference (high - low): {diff:+.4f}")
    if diff > 0:
        print("→ High centrality hexagons have MORE mobility mass than expected")
    else:
        print("→ High centrality hexagons have LESS mobility mass than expected")

    # City-level correlations
    print("\n" + "=" * 70)
    print("CITY-LEVEL CORRELATIONS")
    print("=" * 70)

    print("\n{:<20} {:>6} {:>12} {:>12} {:>10}".format(
        "City", "N", "Spearman ρ", "p-value", "Sig"))
    print("-" * 65)

    city_results = []
    for city in sorted(df_valid['city_name'].unique()):
        city_data = df_valid[df_valid['city_name'] == city]
        if len(city_data) >= 10:
            r, p = stats.spearmanr(city_data['weighted_centrality'],
                                   city_data['deviation_log_mobility'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{city:<20} {len(city_data):>6} {r:>+12.3f} {p:>12.2e} {sig:>10}")
            city_results.append({'city': city, 'n': len(city_data), 'rho': r, 'p': p})

    # Summary
    city_df = pd.DataFrame(city_results)
    pos_sig = sum((city_df['rho'] > 0) & (city_df['p'] < 0.05))
    neg_sig = sum((city_df['rho'] < 0) & (city_df['p'] < 0.05))

    print(f"\nSignificant positive correlations: {pos_sig}/{len(city_df)}")
    print(f"Significant negative correlations: {neg_sig}/{len(city_df)}")

    # Revised interpretation
    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    print(f"""
Your hypothesis: Higher centrality → more mobility infra for through-traffic
                → POSITIVE deviation (more mobility mass than expected)

Observed: Spearman ρ = {spearman_r:+.4f} (p = {spearman_p:.2e})
""")

    if spearman_r > 0.1 and spearman_p < 0.05:
        print("""FINDING: POSITIVE correlation observed

This SUPPORTS your hypothesis:
- High centrality hexagons have MORE mobility mass than their population predicts
- This excess mobility mass likely serves through-traffic from other areas
- Central locations are "subsidizing" the mobility needs of surrounding areas

Alternative interpretation:
- Central areas may simply have denser, more redundant road networks
- Urban cores may have been over-built with roads historically
""")
    elif spearman_r < -0.1 and spearman_p < 0.05:
        print("""FINDING: NEGATIVE correlation observed

This CONTRADICTS your hypothesis:
- High centrality hexagons have LESS mobility mass than their population predicts
- Central areas share mobility infrastructure efficiently
- Peripheral areas have excess road infrastructure relative to their usage

Possible explanation:
- Central roads are heavily utilized (high capacity utilization)
- Peripheral roads have lower utilization (over-provisioned)
- Network centrality measures traffic THROUGH an area, not infrastructure stock
""")
    else:
        print(f"""FINDING: {'Weak' if abs(spearman_r) < 0.1 else 'Moderate'} correlation (ρ = {spearman_r:+.3f})

The relationship is {'not statistically significant' if spearman_p >= 0.05 else 'statistically significant but weak'}.
""")

    print("=" * 70)


if __name__ == '__main__':
    main()
