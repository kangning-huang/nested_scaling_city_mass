#!/usr/bin/env python3
"""
validate_h3_hierarchical_consistency.py - Validate H3 data consistency across resolutions.

This script validates that mass values aggregate correctly across H3 resolution hierarchy:
- Resolution 5 (coarse) → Resolution 6 (medium) → Resolution 7 (fine)
- Each parent hexagon should equal the sum of its ~7 child hexagons

H3 Hierarchy:
- 1 Resolution 5 hexagon ≈ 7 Resolution 6 hexagons
- 1 Resolution 6 hexagon ≈ 7 Resolution 7 hexagons

Usage:
    python scripts/validate_h3_hierarchical_consistency.py
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import h3
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error, r2_score

# Configuration
BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "results" / "validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Data files
DATA_FILES = {
    5: PROCESSED_DIR / "h3_resolution5" / "Fig3_Volume_Pavement_Neighborhood_H3_Resolution5_2025-12-30.csv",
    6: PROCESSED_DIR / "h3_resolution6" / "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv",
    7: PROCESSED_DIR / "h3_resolution7" / "Fig3_Mass_Neighborhood_H3_Resolution7_COMPLETE_2026-01-29.csv"
}

# Mass columns to validate
MASS_COLUMNS = [
    'BuildingMass_Total_Esch2022',
    'BuildingMass_Total_Li2022',
    'BuildingMass_Total_Liu2024',
    'BuildingMass_AverageTotal',
    'total_built_mass_tons'
]


def load_resolution_data(resolution: int) -> pd.DataFrame:
    """Load H3 data for a given resolution."""
    file_path = DATA_FILES[resolution]

    if not file_path.exists():
        raise FileNotFoundError(f"Data file not found: {file_path}")

    print(f"\nLoading Resolution {resolution} data from: {file_path.name}")
    df = pd.read_csv(file_path)

    # Ensure h3index column exists
    if 'h3index' not in df.columns:
        if 'neighborhood_id' in df.columns:
            df['h3index'] = df['neighborhood_id']
        else:
            raise ValueError(f"No h3index column found in Resolution {resolution} data")

    print(f"  Loaded {len(df):,} hexagons")
    print(f"  Columns: {df.columns.tolist()[:10]}...")

    return df


def aggregate_to_parent(df_child: pd.DataFrame, parent_resolution: int,
                       value_columns: list) -> pd.DataFrame:
    """
    Aggregate child hexagons to parent hexagons at coarser resolution.

    Args:
        df_child: DataFrame with child hexagons
        parent_resolution: Target parent resolution (must be < child resolution)
        value_columns: Columns to sum when aggregating

    Returns:
        DataFrame with aggregated values at parent resolution
    """
    print(f"\n  Aggregating to Resolution {parent_resolution}...")

    # Get parent h3index for each child hexagon
    df_agg = df_child.copy()
    df_agg['parent_h3index'] = df_agg['h3index'].apply(
        lambda x: h3.cell_to_parent(x, parent_resolution)
    )

    # Sum values by parent hexagon
    agg_dict = {col: 'sum' for col in value_columns if col in df_agg.columns}
    df_parent = df_agg.groupby('parent_h3index').agg(agg_dict).reset_index()
    df_parent.rename(columns={'parent_h3index': 'h3index'}, inplace=True)

    print(f"  Aggregated {len(df_child):,} child hexagons → {len(df_parent):,} parent hexagons")
    print(f"  Ratio: {len(df_child) / len(df_parent):.2f} children per parent (expected ~7)")

    return df_parent


def compare_datasets(df_actual: pd.DataFrame, df_aggregated: pd.DataFrame,
                    value_columns: list, comparison_name: str) -> pd.DataFrame:
    """
    Compare actual parent values with aggregated child values.

    Args:
        df_actual: DataFrame with actual parent resolution data
        df_aggregated: DataFrame with aggregated child data
        value_columns: Columns to compare
        comparison_name: Name for this comparison (e.g., "Res7→Res6")

    Returns:
        DataFrame with comparison statistics
    """
    print(f"\n{'='*70}")
    print(f"COMPARISON: {comparison_name}")
    print(f"{'='*70}")

    # Merge datasets
    df_merged = df_actual.merge(
        df_aggregated,
        on='h3index',
        how='inner',
        suffixes=('_actual', '_aggregated')
    )

    print(f"\nMatched hexagons: {len(df_merged):,}")
    print(f"  Actual only: {len(df_actual) - len(df_merged):,}")
    print(f"  Aggregated only: {len(df_aggregated) - len(df_merged):,}")

    # Calculate statistics for each column
    results = []

    for col in value_columns:
        actual_col = f"{col}_actual"
        agg_col = f"{col}_aggregated"

        if actual_col not in df_merged.columns or agg_col not in df_merged.columns:
            print(f"\n  Skipping {col}: Column not found in merged data")
            continue

        # Filter out zero/null values for statistics
        mask = (df_merged[actual_col] > 0) & (df_merged[agg_col] > 0)
        df_valid = df_merged[mask]

        if len(df_valid) == 0:
            print(f"\n  Skipping {col}: No valid non-zero pairs")
            continue

        actual_vals = df_valid[actual_col].values
        agg_vals = df_valid[agg_col].values

        # Calculate metrics
        correlation, p_value = pearsonr(actual_vals, agg_vals)
        r2 = r2_score(actual_vals, agg_vals)
        rmse = np.sqrt(mean_squared_error(actual_vals, agg_vals))

        # Relative difference
        rel_diff = (agg_vals - actual_vals) / actual_vals * 100
        median_rel_diff = np.median(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # Absolute difference
        abs_diff = np.abs(agg_vals - actual_vals)
        median_abs_diff = np.median(abs_diff)

        results.append({
            'comparison': comparison_name,
            'column': col,
            'n_hexagons': len(df_valid),
            'correlation': correlation,
            'p_value': p_value,
            'r2': r2,
            'rmse': rmse,
            'median_rel_diff_pct': median_rel_diff,
            'mean_rel_diff_pct': mean_rel_diff,
            'median_abs_diff': median_abs_diff,
            'total_actual': actual_vals.sum(),
            'total_aggregated': agg_vals.sum(),
            'total_diff_pct': (agg_vals.sum() - actual_vals.sum()) / actual_vals.sum() * 100
        })

        print(f"\n  {col}:")
        print(f"    Valid hexagons: {len(df_valid):,}")
        print(f"    Correlation (r): {correlation:.4f} (p={p_value:.2e})")
        print(f"    R²: {r2:.4f}")
        print(f"    RMSE: {rmse:,.0f} tonnes")
        print(f"    Median relative diff: {median_rel_diff:+.2f}%")
        print(f"    Mean relative diff: {mean_rel_diff:+.2f}%")
        print(f"    Total actual: {actual_vals.sum():,.0f} tonnes")
        print(f"    Total aggregated: {agg_vals.sum():,.0f} tonnes")
        print(f"    Total difference: {(agg_vals.sum() - actual_vals.sum()) / actual_vals.sum() * 100:+.2f}%")

    return pd.DataFrame(results), df_merged


def create_validation_plots(df_merged: pd.DataFrame, comparison_name: str,
                            value_columns: list, output_dir: Path):
    """Create scatter plots comparing actual vs aggregated values."""
    print(f"\n  Creating validation plots for {comparison_name}...")

    n_cols = len(value_columns)
    fig, axes = plt.subplots(1, min(n_cols, 3), figsize=(5*min(n_cols, 3), 4))
    if n_cols == 1:
        axes = [axes]

    for idx, col in enumerate(value_columns[:3]):  # Plot first 3 columns
        actual_col = f"{col}_actual"
        agg_col = f"{col}_aggregated"

        if actual_col not in df_merged.columns or agg_col not in df_merged.columns:
            continue

        ax = axes[idx] if n_cols > 1 else axes[0]

        # Filter valid values
        mask = (df_merged[actual_col] > 0) & (df_merged[agg_col] > 0)
        df_plot = df_merged[mask]

        if len(df_plot) == 0:
            continue

        # Log-log scatter plot
        x = np.log10(df_plot[actual_col])
        y = np.log10(df_plot[agg_col])

        ax.scatter(x, y, alpha=0.3, s=10, label='Data')

        # 1:1 line
        min_val = min(x.min(), y.min())
        max_val = max(x.max(), y.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='1:1 line', linewidth=2)

        # Calculate R²
        r2 = r2_score(df_plot[actual_col], df_plot[agg_col])

        ax.set_xlabel(f'log₁₀(Actual) [tonnes]')
        ax.set_ylabel(f'log₁₀(Aggregated) [tonnes]')
        ax.set_title(f'{col.replace("BuildingMass_Total_", "").replace("_", " ")}\nR² = {r2:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    plot_file = output_dir / f"validation_{comparison_name.replace('→', '_to_').replace(' ', '_')}.png"
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"    Saved plot: {plot_file}")
    plt.close()


def main():
    print("="*70)
    print("H3 HIERARCHICAL VALIDATION")
    print("="*70)
    print("\nValidating mass conservation across H3 resolution hierarchy")
    print("Expected relationship: 1 parent hexagon ≈ 7 child hexagons")

    # Load data for all resolutions
    data = {}
    for res in [5, 6, 7]:
        try:
            data[res] = load_resolution_data(res)
        except FileNotFoundError as e:
            print(f"\n  WARNING: {e}")
            print(f"  Skipping Resolution {res}")

    if len(data) < 2:
        print("\nERROR: Need at least 2 resolutions to perform validation")
        sys.exit(1)

    # Determine which mass columns are available
    available_columns = set()
    for df in data.values():
        available_columns.update(df.columns)

    mass_cols_to_validate = [col for col in MASS_COLUMNS if col in available_columns]

    if not mass_cols_to_validate:
        print(f"\nERROR: No mass columns found. Looking for: {MASS_COLUMNS}")
        sys.exit(1)

    print(f"\nValidating columns: {mass_cols_to_validate}")

    # Validation 1: Resolution 7 → Resolution 6
    all_results = []

    if 7 in data and 6 in data:
        print(f"\n{'='*70}")
        print("VALIDATION 1: Resolution 7 → Resolution 6")
        print(f"{'='*70}")

        df_7to6 = aggregate_to_parent(data[7], parent_resolution=6, value_columns=mass_cols_to_validate)
        results_7to6, merged_7to6 = compare_datasets(
            df_actual=data[6],
            df_aggregated=df_7to6,
            value_columns=mass_cols_to_validate,
            comparison_name="Res7→Res6"
        )
        all_results.append(results_7to6)

        # Create plots
        create_validation_plots(merged_7to6, "Res7→Res6", mass_cols_to_validate, OUTPUT_DIR)

    # Validation 2: Resolution 6 → Resolution 5
    if 6 in data and 5 in data:
        print(f"\n{'='*70}")
        print("VALIDATION 2: Resolution 6 → Resolution 5")
        print(f"{'='*70}")

        df_6to5 = aggregate_to_parent(data[6], parent_resolution=5, value_columns=mass_cols_to_validate)
        results_6to5, merged_6to5 = compare_datasets(
            df_actual=data[5],
            df_aggregated=df_6to5,
            value_columns=mass_cols_to_validate,
            comparison_name="Res6→Res5"
        )
        all_results.append(results_6to5)

        # Create plots
        create_validation_plots(merged_6to5, "Res6→Res5", mass_cols_to_validate, OUTPUT_DIR)

    # Validation 3 (optional): Resolution 7 → Resolution 5 (two-step)
    if 7 in data and 5 in data:
        print(f"\n{'='*70}")
        print("VALIDATION 3: Resolution 7 → Resolution 5 (two-step)")
        print(f"{'='*70}")

        df_7to5 = aggregate_to_parent(data[7], parent_resolution=5, value_columns=mass_cols_to_validate)
        results_7to5, merged_7to5 = compare_datasets(
            df_actual=data[5],
            df_aggregated=df_7to5,
            value_columns=mass_cols_to_validate,
            comparison_name="Res7→Res5"
        )
        all_results.append(results_7to5)

        # Create plots
        create_validation_plots(merged_7to5, "Res7→Res5", mass_cols_to_validate, OUTPUT_DIR)

    # Combine all results
    if all_results:
        df_all_results = pd.concat(all_results, ignore_index=True)

        # Save results
        results_file = OUTPUT_DIR / "h3_hierarchical_validation_results.csv"
        df_all_results.to_csv(results_file, index=False)
        print(f"\n{'='*70}")
        print(f"RESULTS SAVED: {results_file}")
        print(f"{'='*70}")

        # Print summary
        print("\nSUMMARY STATISTICS:")
        print("-" * 70)
        summary = df_all_results.groupby('comparison').agg({
            'correlation': 'mean',
            'r2': 'mean',
            'median_rel_diff_pct': 'mean',
            'total_diff_pct': 'mean'
        })
        print(summary.to_string())

        # Check validation thresholds
        print("\n" + "="*70)
        print("VALIDATION ASSESSMENT")
        print("="*70)

        thresholds = {
            'correlation': 0.95,
            'r2': 0.90,
            'median_rel_diff_pct': 10.0,  # ±10%
            'total_diff_pct': 5.0  # ±5%
        }

        passed_all = True
        for idx, row in df_all_results.iterrows():
            passed = True
            issues = []

            if row['correlation'] < thresholds['correlation']:
                passed = False
                issues.append(f"Low correlation: {row['correlation']:.3f}")

            if row['r2'] < thresholds['r2']:
                passed = False
                issues.append(f"Low R²: {row['r2']:.3f}")

            if abs(row['median_rel_diff_pct']) > thresholds['median_rel_diff_pct']:
                passed = False
                issues.append(f"High median difference: {row['median_rel_diff_pct']:.2f}%")

            if abs(row['total_diff_pct']) > thresholds['total_diff_pct']:
                passed = False
                issues.append(f"High total difference: {row['total_diff_pct']:.2f}%")

            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"\n{status}: {row['comparison']} - {row['column']}")
            if not passed:
                for issue in issues:
                    print(f"  - {issue}")
                passed_all = False

        if passed_all:
            print("\n" + "="*70)
            print("✓ ALL VALIDATIONS PASSED")
            print("="*70)
        else:
            print("\n" + "="*70)
            print("✗ SOME VALIDATIONS FAILED - Review issues above")
            print("="*70)


if __name__ == "__main__":
    main()
