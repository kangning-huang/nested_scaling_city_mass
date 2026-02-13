#!/usr/bin/env python3
"""
Calculate 95% CIs for beta coefficients - EXACT REPRODUCTION of R script methodology.

This follows the EXACT methodology from 05_sensitivity_random_datasource.R:
- City level: Uses MasterMass_ByClass20250616.csv with CTR_MN_NM (country name) for random effects
- Neighborhood level: Uses Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv with CTR_MN_ISO

This should produce betas matching individual_datasource_betas.csv
"""

import pandas as pd
import numpy as np
from scipy.stats import linregress
import statsmodels.formula.api as smf
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def fit_mixed_normalize_ols(df, mass_col, pop_col='population_2015',
                            country_col='CTR_MN_ISO', level='city'):
    """
    Fit mixed model, normalize, run OLS - matching R script exactly.
    """
    # Filter positive values
    valid = (df[mass_col] > 0) & (df[pop_col] > 0)
    df_clean = df[valid].copy()

    if len(df_clean) < 10:
        return None

    # Log transform
    df_clean['log_mass'] = np.log10(df_clean[mass_col])
    df_clean['log_pop'] = np.log10(df_clean[pop_col])

    try:
        if level == 'city':
            # City level: simple country random effects
            model = smf.mixedlm("log_mass ~ log_pop", df_clean,
                               groups=df_clean[country_col])
        else:
            # Neighborhood level: nested random effects
            df_clean['Country_City'] = df_clean['CTR_MN_ISO'] + "_" + df_clean['ID_HDC_G0'].astype(str)
            model = smf.mixedlm("log_mass ~ log_pop", df_clean,
                               groups=df_clean['CTR_MN_ISO'],
                               re_formula="1",
                               vc_formula={"City": "0 + C(Country_City)"})

        result = model.fit(reml=False, method='nm')

        # Extract random effects
        random_effects = result.random_effects
        re_dict = {country: re[0] if hasattr(re, '__getitem__') else re
                   for country, re in random_effects.items()}

        # Normalize: remove random effects
        df_clean['random_effect'] = df_clean[country_col].map(re_dict).fillna(0)
        df_clean['log_mass_normalized'] = df_clean['log_mass'] - df_clean['random_effect']

        # Simple OLS on normalized data
        log_pop = df_clean['log_pop'].values
        log_mass_norm = df_clean['log_mass_normalized'].values

        slope, intercept, r_value, p_value, std_err = linregress(log_pop, log_mass_norm)

        # Calculate 95% CI
        ci_lower = slope - 1.96 * std_err
        ci_upper = slope + 1.96 * std_err

        return {
            'beta_ols': slope,
            'std_err': std_err,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'r_squared': r_value**2,
            'n': len(df_clean)
        }

    except Exception as e:
        print(f"  ERROR: {e}")
        return None

def analyze_city_level():
    """
    City-level analysis using MasterMass file with CTR_MN_NM grouping.
    """
    print("\n" + "="*80)
    print("CITY-LEVEL ANALYSIS (MasterMass data)")
    print("="*80)

    # Load MasterMass file
    base_dir = Path(__file__).parent.parent.parent
    data_file = base_dir / "data/processed/MasterMass_ByClass20250616.csv"

    print(f"\nLoading: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} cities")

    # Calculate total mass for each data source
    df['mass_Esch2022'] = df['BuildingMass_Total_Esch2022'] + df['mobility_mass_tons']
    df['mass_Li2022'] = df['BuildingMass_Total_Li2022'] + df['mobility_mass_tons']
    df['mass_Liu2024'] = df['BuildingMass_Total_Liu2024'] + df['mobility_mass_tons']

    # Filter: countries with ≥5 cities (using CTR_MN_NM like R script)
    city_counts = df['CTR_MN_NM'].value_counts()
    valid_countries = city_counts[city_counts >= 5].index
    df = df[df['CTR_MN_NM'].isin(valid_countries)]

    print(f"Filtered to {len(df)} cities in {len(valid_countries)} countries")

    # Run analysis for each data source
    data_sources = ['Esch2022', 'Li2022', 'Liu2024']
    results = []

    for ds in data_sources:
        print(f"\n{ds}:")
        result = fit_mixed_normalize_ols(df, f'mass_{ds}',
                                         country_col='CTR_MN_NM',
                                         level='city')
        if result:
            print(f"  β = {result['beta_ols']:.6f}")
            print(f"  SE = {result['std_err']:.6f}")
            print(f"  95% CI = [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
            print(f"  R² = {result['r_squared']:.6f}")
            print(f"  N = {result['n']}")

            results.append({
                'level': 'City',
                'data_source': ds,
                **result
            })

    return results

def analyze_neighborhood_level():
    """
    Neighborhood-level analysis using H3 Resolution 6 file with CTR_MN_ISO grouping.
    """
    print("\n" + "="*80)
    print("NEIGHBORHOOD-LEVEL ANALYSIS (H3 Resolution 6)")
    print("="*80)

    # Load neighborhood file
    base_dir = Path(__file__).parent.parent.parent
    data_file = base_dir / "data/processed/h3_resolution6/Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"

    print(f"\nLoading: {data_file.name}")
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df)} neighborhoods")

    # Calculate total mass for each data source
    df['mass_Esch2022'] = df['BuildingMass_Total_Esch2022'] + df['mobility_mass_tons']
    df['mass_Li2022'] = df['BuildingMass_Total_Li2022'] + df['mobility_mass_tons']
    df['mass_Liu2024'] = df['BuildingMass_Total_Liu2024'] + df['mobility_mass_tons']

    # Filter: matching R script criteria
    # Countries with ≥3 cities, cities with >50k pop and >3 neighborhoods
    city_stats = df.groupby('ID_HDC_G0').agg({
        'population_2015': 'sum',
        'CTR_MN_ISO': 'first',
        'h3index': 'count'
    }).rename(columns={'h3index': 'n_neighborhoods'})

    city_counts = city_stats.groupby('CTR_MN_ISO').size()
    valid_countries = city_counts[city_counts >= 3].index

    valid_cities = city_stats[
        (city_stats['CTR_MN_ISO'].isin(valid_countries)) &
        (city_stats['population_2015'] > 50000) &
        (city_stats['n_neighborhoods'] > 3)
    ].index

    df = df[df['ID_HDC_G0'].isin(valid_cities)].copy()

    print(f"Filtered to {len(df)} neighborhoods in {len(valid_cities)} cities")

    # Run analysis for each data source
    data_sources = ['Esch2022', 'Li2022', 'Liu2024']
    results = []

    for ds in data_sources:
        print(f"\n{ds}:")
        result = fit_mixed_normalize_ols(df, f'mass_{ds}',
                                         country_col='CTR_MN_ISO',
                                         level='neighborhood')
        if result:
            print(f"  β = {result['beta_ols']:.6f}")
            print(f"  SE = {result['std_err']:.6f}")
            print(f"  95% CI = [{result['ci_lower']:.6f}, {result['ci_upper']:.6f}]")
            print(f"  R² = {result['r_squared']:.6f}")
            print(f"  N = {result['n']}")

            results.append({
                'level': 'Neighborhood',
                'data_source': ds,
                **result
            })

    return results

def main():
    print("="*80)
    print("EXACT REPRODUCTION OF R SCRIPT METHODOLOGY")
    print("="*80)

    # Run both analyses
    city_results = analyze_city_level()
    neigh_results = analyze_neighborhood_level()

    # Combine and save
    all_results = city_results + neigh_results
    results_df = pd.DataFrame(all_results)

    output_file = 'sensitivity_datasource/individual_datasource_betas_with_CI_EXACT.csv'
    results_df.to_csv(output_file, index=False)

    print("\n" + "="*80)
    print("FINAL SUMMARY")
    print("="*80)

    print("\nCity Level:")
    for _, row in results_df[results_df['level'] == 'City'].iterrows():
        print(f"  {row['data_source']}: β = {row['beta_ols']:.4f}, "
              f"95% CI = [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], N = {row['n']}")

    print("\nNeighborhood Level:")
    for _, row in results_df[results_df['level'] == 'Neighborhood'].iterrows():
        print(f"  {row['data_source']}: β = {row['beta_ols']:.4f}, "
              f"95% CI = [{row['ci_lower']:.4f}, {row['ci_upper']:.4f}], N = {row['n']}")

    print(f"\n\nResults saved to: {output_file}")

    # Compare with expected values
    print("\n" + "="*80)
    print("VALIDATION CHECK")
    print("="*80)

    expected = pd.read_csv('sensitivity_datasource/individual_datasource_betas.csv')
    print("\nExpected values from individual_datasource_betas.csv:")
    print(expected.to_string(index=False))

    print("\n\nMy calculated values:")
    comparison = results_df[['level', 'data_source', 'beta_ols']].copy()
    comparison['beta_ols'] = comparison['beta_ols'].round(6)
    print(comparison.to_string(index=False))

if __name__ == "__main__":
    main()
