#!/usr/bin/env python3
"""
Analyze relationship between mobility mass deviation and resolution 7 network centrality.

Maps res7 centrality to res6 parent hexagons to join with deviation data.

Hypothesis: Higher centrality hexagons have more mobility infrastructure serving
through-traffic, leading to POSITIVE deviation from neighborhood-level scaling.
"""

import pandas as pd
import numpy as np
import json
import h3
from scipy import stats
from pathlib import Path
from collections import defaultdict

def load_res7_centrality(results_dir):
    """Load all res7 centrality files and aggregate to res6 level."""
    centrality_files = list(results_dir.glob("*_res7_centrality_weighted.geojson"))
    print(f"Found {len(centrality_files)} res7 centrality files")

    all_data = []

    for f in centrality_files:
        with open(f) as fp:
            data = json.load(fp)

        city_name = data['properties']['city_name']
        city_id = data['properties']['city_id']

        for feature in data['features']:
            props = feature['properties']
            h3_res7 = props['h3_index']

            # Get res6 parent
            try:
                h3_res6 = h3.cell_to_parent(h3_res7, 6)
            except:
                continue

            all_data.append({
                'h3_res7': h3_res7,
                'h3_res6': h3_res6,
                'city_name': city_name,
                'city_id': city_id,
                'weighted_centrality_res7': props.get('weighted_centrality', 0),
                'betweenness_res7': props.get('betweenness', 0),
                'population_res7': props.get('population', 0)
            })

    df = pd.DataFrame(all_data)
    print(f"Loaded {len(df)} res7 hexagons")
    print(f"Cities: {df['city_name'].unique().tolist()}")

    return df


def aggregate_to_res6(df_res7):
    """Aggregate res7 centrality to res6 level."""
    # For each res6 hex, aggregate its res7 children
    agg = df_res7.groupby(['h3_res6', 'city_name', 'city_id']).agg({
        'weighted_centrality_res7': ['sum', 'mean', 'max'],
        'betweenness_res7': ['mean', 'max'],
        'h3_res7': 'count'
    }).reset_index()

    # Flatten column names
    agg.columns = ['h3_res6', 'city_name', 'city_id',
                   'centrality_sum', 'centrality_mean', 'centrality_max',
                   'betweenness_mean', 'betweenness_max', 'n_res7_children']

    print(f"Aggregated to {len(agg)} res6 hexagons")
    return agg


def main():
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    project_results = script_dir.parent.parent.parent / "results"

    print("=" * 70)
    print("MOBILITY DEVIATION vs RES7 CENTRALITY ANALYSIS")
    print("=" * 70)

    # Load res7 centrality data
    print("\n1. Loading res7 centrality data...")
    df_res7 = load_res7_centrality(results_dir)

    # Aggregate to res6 level
    print("\n2. Aggregating to res6 level...")
    df_cent = aggregate_to_res6(df_res7)

    # Load deviation data
    print("\n3. Loading deviation data...")
    deviation_path = project_results / "neighborhood_deviations_detailed.csv"
    df_dev = pd.read_csv(deviation_path)
    print(f"Loaded {len(df_dev)} neighborhoods with deviation data")

    # Merge on res6 hex
    print("\n4. Merging datasets...")
    df_merged = df_cent.merge(
        df_dev[['hex_id', 'city_id', 'UC_NM_MN', 'population', 'mass_building', 'mass_mobility',
                'deviation_log_building', 'deviation_log_mobility',
                'pct_deviation_building', 'pct_deviation_mobility', 'quadrant']],
        left_on='h3_res6',
        right_on='hex_id',
        how='inner',
        suffixes=('_cent', '_dev')
    )
    print(f"Merged dataset: {len(df_merged)} neighborhoods")
    print(f"Cities in merged data: {df_merged['city_name'].unique().tolist()}")

    # Filter to valid centrality
    df_valid = df_merged[df_merged['centrality_sum'] > 0].copy()
    print(f"With valid centrality: {len(df_valid)}")

    if len(df_valid) < 10:
        print("ERROR: Not enough data for analysis")
        return

    # Analysis
    print("\n" + "=" * 70)
    print("HYPOTHESIS TEST (Resolution 7 Centrality)")
    print("=" * 70)
    print("\nHypothesis: Higher centrality → More mobility mass serving through-traffic")
    print("            → POSITIVE deviation from expected mobility mass")

    # Test with different aggregation methods
    for centrality_var, label in [
        ('centrality_sum', 'Sum of res7 centrality'),
        ('centrality_mean', 'Mean of res7 centrality'),
        ('centrality_max', 'Max of res7 centrality')
    ]:
        print(f"\n{'-'*50}")
        print(f"Using: {label}")
        print(f"{'-'*50}")

        x = df_valid[centrality_var]
        y = df_valid['deviation_log_mobility']

        spearman_r, spearman_p = stats.spearmanr(x, y)
        print(f"Spearman ρ: {spearman_r:+.4f}  (p = {spearman_p:.2e})")

        # Log transform for Pearson
        log_x = np.log10(x + 1)
        pearson_r, pearson_p = stats.pearsonr(log_x, y)
        print(f"Pearson r (log): {pearson_r:+.4f}  (p = {pearson_p:.2e})")

    # Use sum for detailed analysis (most comprehensive)
    print("\n" + "=" * 70)
    print("DETAILED ANALYSIS (using centrality sum)")
    print("=" * 70)

    x = df_valid['centrality_sum']
    y = df_valid['deviation_log_mobility']

    spearman_r, spearman_p = stats.spearmanr(x, y)

    # Quartile analysis
    print("\n" + "-" * 50)
    print("MOBILITY DEVIATION BY CENTRALITY QUARTILE")
    print("-" * 50)

    df_valid['centrality_quartile'] = pd.qcut(
        df_valid['centrality_sum'], 4,
        labels=['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']
    )

    print("\n{:<12} {:>8} {:>12} {:>12} {:>12}".format(
        "Quartile", "N", "Log Dev", "% Dev", "Centrality"))
    print("-" * 60)

    for q in ['Q1 (low)', 'Q2', 'Q3', 'Q4 (high)']:
        q_data = df_valid[df_valid['centrality_quartile'] == q]
        log_dev = q_data['deviation_log_mobility'].mean()
        pct_dev = q_data['pct_deviation_mobility'].mean()
        cent = q_data['centrality_sum'].mean()
        print(f"{q:<12} {len(q_data):>8} {log_dev:>+12.4f} {pct_dev:>+11.1f}% {cent:>12,.0f}")

    # T-test
    print("\n" + "-" * 50)
    print("T-TEST: HIGH vs LOW CENTRALITY")
    print("-" * 50)

    high = df_valid[df_valid['centrality_quartile'] == 'Q4 (high)']['deviation_log_mobility']
    low = df_valid[df_valid['centrality_quartile'] == 'Q1 (low)']['deviation_log_mobility']

    t_stat, t_p = stats.ttest_ind(high, low)
    print(f"\nHigh centrality (Q4): mean = {high.mean():+.4f}, n = {len(high)}")
    print(f"Low centrality (Q1):  mean = {low.mean():+.4f}, n = {len(low)}")
    print(f"\nT-statistic: {t_stat:.3f}")
    print(f"P-value: {t_p:.2e}")
    print(f"Difference: {high.mean() - low.mean():+.4f}")

    # City-level analysis
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
            r, p = stats.spearmanr(city_data['centrality_sum'],
                                   city_data['deviation_log_mobility'])
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            print(f"{city:<20} {len(city_data):>6} {r:>+12.3f} {p:>12.2e} {sig:>10}")
            city_results.append({'city': city, 'n': len(city_data), 'rho': r, 'p': p})

    # Compare with res6 results
    print("\n" + "=" * 70)
    print("COMPARISON: RES6 vs RES7 CENTRALITY")
    print("=" * 70)

    print(f"""
Resolution 6 analysis (from merged_mass_centrality.csv):
  Spearman ρ = -0.175 (p = 1.68e-06)

Resolution 7 analysis (current):
  Spearman ρ = {spearman_r:+.4f} (p = {spearman_p:.2e})
""")

    # Interpretation
    print("=" * 70)
    print("INTERPRETATION")
    print("=" * 70)

    if spearman_r > 0.1 and spearman_p < 0.05:
        direction = "POSITIVE"
        supports = "SUPPORTS"
    elif spearman_r < -0.1 and spearman_p < 0.05:
        direction = "NEGATIVE"
        supports = "CONTRADICTS"
    else:
        direction = "WEAK/NON-SIGNIFICANT"
        supports = "INCONCLUSIVE for"

    print(f"""
Hypothesis: Higher centrality → more through-traffic infrastructure
            → POSITIVE mobility deviation

Result at Resolution 7: {direction} correlation (ρ = {spearman_r:+.4f})

This {supports} the hypothesis.
""")

    # Save results
    output_path = results_dir / "mobility_deviation_centrality_res7_analysis.csv"
    df_valid.to_csv(output_path, index=False)
    print(f"Saved merged data to: {output_path}")


if __name__ == '__main__':
    main()
