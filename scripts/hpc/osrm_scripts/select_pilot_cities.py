#!/usr/bin/env python3
"""
Select a diverse 100-city pilot sample for validating the HPC pipeline.

Selection criteria:
1. 5 cities from each size category (by H3 grid count)
2. Geographic diversity (all continents represented)
3. Include Shanghai (already verified) as baseline

Usage:
    python select_pilot_cities.py --input all_cities_h3_grids.gpkg --output pilot_cities.txt
"""

import geopandas as gpd
import pandas as pd
import json
import argparse
import random
from collections import defaultdict

# Continent mapping based on country
COUNTRY_TO_CONTINENT = {
    # Asia
    "China": "Asia", "India": "Asia", "Japan": "Asia", "Indonesia": "Asia",
    "Pakistan": "Asia", "Bangladesh": "Asia", "Vietnam": "Asia", "Philippines": "Asia",
    "Thailand": "Asia", "Myanmar": "Asia", "South Korea": "Asia", "Malaysia": "Asia",
    "Nepal": "Asia", "Sri Lanka": "Asia", "Cambodia": "Asia", "Taiwan": "Asia",
    "Laos": "Asia", "Mongolia": "Asia", "Singapore": "Asia", "Uzbekistan": "Asia",
    "Kazakhstan": "Asia", "Afghanistan": "Asia", "Iran": "Asia", "Iraq": "Asia",
    "Saudi Arabia": "Asia", "Yemen": "Asia", "Syria": "Asia", "Jordan": "Asia",
    "United Arab Emirates": "Asia", "Israel": "Asia", "Lebanon": "Asia",
    "Palestine": "Asia", "Kuwait": "Asia", "Oman": "Asia", "Qatar": "Asia",
    "Bahrain": "Asia", "Turkey": "Asia", "Azerbaijan": "Asia", "Georgia": "Asia",
    "Armenia": "Asia",

    # Africa
    "Nigeria": "Africa", "Ethiopia": "Africa", "Egypt": "Africa",
    "Democratic Republic of the Congo": "Africa", "South Africa": "Africa",
    "Tanzania": "Africa", "Kenya": "Africa", "Uganda": "Africa", "Algeria": "Africa",
    "Sudan": "Africa", "Morocco": "Africa", "Angola": "Africa", "Mozambique": "Africa",
    "Ghana": "Africa", "Madagascar": "Africa", "Cameroon": "Africa",
    "Ivory Coast": "Africa", "Niger": "Africa", "Burkina Faso": "Africa",
    "Mali": "Africa", "Malawi": "Africa", "Zambia": "Africa", "Senegal": "Africa",
    "Zimbabwe": "Africa", "Chad": "Africa", "Somalia": "Africa", "Guinea": "Africa",
    "Rwanda": "Africa", "Benin": "Africa", "Burundi": "Africa", "Tunisia": "Africa",
    "South Sudan": "Africa", "Togo": "Africa", "Sierra Leone": "Africa",
    "Libya": "Africa", "Central African Republic": "Africa", "Eritrea": "Africa",
    "Republic of the Congo": "Africa", "Liberia": "Africa", "Mauritania": "Africa",
    "Namibia": "Africa", "Botswana": "Africa", "Lesotho": "Africa", "Gambia": "Africa",

    # Europe
    "Russia": "Europe", "Germany": "Europe", "United Kingdom": "Europe",
    "France": "Europe", "Italy": "Europe", "Spain": "Europe", "Ukraine": "Europe",
    "Poland": "Europe", "Romania": "Europe", "Netherlands": "Europe",
    "Belgium": "Europe", "Czech Republic": "Europe", "Greece": "Europe",
    "Portugal": "Europe", "Sweden": "Europe", "Hungary": "Europe", "Belarus": "Europe",
    "Austria": "Europe", "Serbia": "Europe", "Switzerland": "Europe",
    "Bulgaria": "Europe", "Denmark": "Europe", "Finland": "Europe",
    "Slovakia": "Europe", "Norway": "Europe", "Ireland": "Europe", "Croatia": "Europe",

    # North America
    "United States": "North America", "Canada": "North America", "Mexico": "North America",

    # Latin America & Caribbean
    "Brazil": "Latin America", "Colombia": "Latin America", "Argentina": "Latin America",
    "Peru": "Latin America", "Venezuela": "Latin America", "Chile": "Latin America",
    "Ecuador": "Latin America", "Bolivia": "Latin America", "Paraguay": "Latin America",
    "Uruguay": "Latin America", "Cuba": "Latin America", "Haiti": "Latin America",
    "Dominican Republic": "Latin America", "Guatemala": "Latin America",
    "Honduras": "Latin America", "Nicaragua": "Latin America", "Costa Rica": "Latin America",
    "Panama": "Latin America", "Jamaica": "Latin America",

    # Oceania
    "Australia": "Oceania", "New Zealand": "Oceania", "Papua New Guinea": "Oceania",
    "Fiji": "Oceania",
}


def select_pilot_cities(input_path: str, output_path: str, n_cities: int = 100):
    """Select diverse pilot cities across size categories and continents."""

    print(f"Reading {input_path}...")
    gdf = gpd.read_file(input_path)

    # Get unique cities with grid counts
    city_grids = gdf.groupby('ID_HDC_G0').agg({
        'UC_NM_MN': 'first',
        'CTR_MN_NM': 'first',
        'h3index': 'count'
    }).reset_index()
    city_grids.columns = ['city_id', 'city_name', 'country', 'n_grids']

    # Add continent
    city_grids['continent'] = city_grids['country'].map(
        lambda x: COUNTRY_TO_CONTINENT.get(x, 'Unknown')
    )

    # Define size categories
    def categorize(n):
        if n >= 100:
            return 'XLarge'
        elif n >= 51:
            return 'Large'
        elif n >= 6:
            return 'Medium'
        elif n >= 2:
            return 'Small'
        else:
            return 'Tiny'

    city_grids['size_category'] = city_grids['n_grids'].apply(categorize)

    print(f"\nTotal cities: {len(city_grids)}")
    print(f"\nSize distribution:")
    print(city_grids['size_category'].value_counts())
    print(f"\nContinent distribution:")
    print(city_grids['continent'].value_counts())

    # Target allocation
    target_per_category = {
        'XLarge': 5,   # All 35 XLarge cities could be included, but limit to 5
        'Large': 10,
        'Medium': 25,
        'Small': 30,
        'Tiny': 30
    }

    # Specific cities to always include (verified or important)
    must_include = [
        12400,  # Shanghai (verified)
    ]

    # Known large cities to include for validation
    target_xlarge = [
        # Guangzhou, Tokyo, NYC, LA, Jakarta - get their IDs
    ]

    selected = []
    selected_ids = set(must_include)

    # First, add must-include cities
    for city_id in must_include:
        row = city_grids[city_grids['city_id'] == city_id]
        if not row.empty:
            selected.append(row.iloc[0].to_dict())
            print(f"Added must-include: {row.iloc[0]['city_name']} ({row.iloc[0]['country']})")

    # For each size category, select cities ensuring continent diversity
    for category, target_count in target_per_category.items():
        category_cities = city_grids[
            (city_grids['size_category'] == category) &
            (~city_grids['city_id'].isin(selected_ids))
        ]

        if len(category_cities) == 0:
            print(f"  {category}: No cities available")
            continue

        # Group by continent for diversity
        by_continent = category_cities.groupby('continent').apply(
            lambda x: x.sample(min(len(x), max(1, target_count // 6)))
        ).reset_index(drop=True)

        # Sample from grouped result
        n_to_select = min(target_count, len(by_continent))
        sampled = by_continent.sample(n=n_to_select, random_state=42)

        for _, row in sampled.iterrows():
            if row['city_id'] not in selected_ids:
                selected.append(row.to_dict())
                selected_ids.add(row['city_id'])

        print(f"  {category}: Selected {len(sampled)} cities")

    # Ensure we have exactly n_cities
    if len(selected) < n_cities:
        # Add more from any category
        remaining = city_grids[~city_grids['city_id'].isin(selected_ids)]
        additional = remaining.sample(n=min(n_cities - len(selected), len(remaining)), random_state=42)
        for _, row in additional.iterrows():
            selected.append(row.to_dict())
            selected_ids.add(row['city_id'])
    elif len(selected) > n_cities:
        # Trim to target
        selected = selected[:n_cities]

    # Create output dataframe
    pilot_df = pd.DataFrame(selected)
    pilot_df = pilot_df.sort_values('n_grids', ascending=False)

    # Write city IDs to text file
    with open(output_path, 'w') as f:
        for city_id in pilot_df['city_id']:
            f.write(f"{city_id}\n")
    print(f"\nCity IDs written to {output_path}")

    # Write detailed JSON
    json_path = output_path.replace('.txt', '.json')
    pilot_data = {
        "n_cities": len(pilot_df),
        "total_grids": int(pilot_df['n_grids'].sum()),
        "by_category": pilot_df.groupby('size_category').size().to_dict(),
        "by_continent": pilot_df.groupby('continent').size().to_dict(),
        "cities": pilot_df.to_dict('records')
    }
    with open(json_path, 'w') as f:
        json.dump(pilot_data, f, indent=2)
    print(f"Detailed info written to {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("PILOT BATCH SUMMARY")
    print(f"{'='*60}")
    print(f"Total cities: {len(pilot_df)}")
    print(f"Total H3 grids: {pilot_df['n_grids'].sum()}")
    print(f"\nBy size category:")
    for cat in ['XLarge', 'Large', 'Medium', 'Small', 'Tiny']:
        count = len(pilot_df[pilot_df['size_category'] == cat])
        grids = pilot_df[pilot_df['size_category'] == cat]['n_grids'].sum()
        print(f"  {cat}: {count} cities, {grids} grids")

    print(f"\nBy continent:")
    for cont in sorted(pilot_df['continent'].unique()):
        count = len(pilot_df[pilot_df['continent'] == cont])
        print(f"  {cont}: {count} cities")

    print(f"\nLargest cities in pilot:")
    for _, row in pilot_df.head(10).iterrows():
        print(f"  {row['city_name']} ({row['country']}): {row['n_grids']} grids")

    return pilot_df


def main():
    parser = argparse.ArgumentParser(description="Select pilot cities for HPC validation")
    parser.add_argument("--input", "-i", required=True, help="Input GeoPackage file")
    parser.add_argument("--output", "-o", default="pilot_cities.txt", help="Output file for city IDs")
    parser.add_argument("--n-cities", "-n", type=int, default=100, help="Number of cities to select")
    args = parser.parse_args()

    select_pilot_cities(args.input, args.output, args.n_cities)


if __name__ == "__main__":
    main()
