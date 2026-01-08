#!/usr/bin/env python3
"""
Prepare city lists grouped by country for HPC batch processing.

Reads all_cities_h3_grids.gpkg and generates:
1. Text files with city IDs for each country (for SLURM job arrays)
2. Summary statistics by country
3. Country-to-OSM file mapping

Usage:
    python prepare_city_lists.py --input all_cities_h3_grids.gpkg --output-dir city_lists/
"""

import geopandas as gpd
import pandas as pd
import json
import os
import argparse
from collections import defaultdict

# Mapping from country names to Geofabrik OSM file names
COUNTRY_TO_OSM = {
    # Asia
    "China": "asia/china",
    "India": "asia/india",
    "Japan": "asia/japan",
    "Indonesia": "asia/indonesia",
    "Pakistan": "asia/pakistan",
    "Bangladesh": "asia/bangladesh",
    "Vietnam": "asia/vietnam",
    "Philippines": "asia/philippines",
    "Thailand": "asia/thailand",
    "Myanmar": "asia/myanmar",
    "South Korea": "asia/south-korea",
    "Malaysia": "asia/malaysia",
    "Nepal": "asia/nepal",
    "Sri Lanka": "asia/sri-lanka",
    "Cambodia": "asia/cambodia",
    "Taiwan": "asia/taiwan",
    "Laos": "asia/laos",
    "Mongolia": "asia/mongolia",
    "Singapore": "asia/malaysia-singapore-brunei",
    "Uzbekistan": "asia/uzbekistan",
    "Kazakhstan": "asia/kazakhstan",
    "Afghanistan": "asia/afghanistan",
    "Tajikistan": "asia/tajikistan",
    "Kyrgyzstan": "asia/kyrgyzstan",
    "Turkmenistan": "asia/turkmenistan",

    # Middle East
    "Iran": "asia/iran",
    "Iraq": "asia/iraq",
    "Saudi Arabia": "asia/gcc-states",
    "Yemen": "asia/yemen",
    "Syria": "asia/syria",
    "Jordan": "asia/jordan",
    "United Arab Emirates": "asia/gcc-states",
    "Israel": "asia/israel-and-palestine",
    "Lebanon": "asia/lebanon",
    "Palestine": "asia/israel-and-palestine",
    "Kuwait": "asia/gcc-states",
    "Oman": "asia/gcc-states",
    "Qatar": "asia/gcc-states",
    "Bahrain": "asia/gcc-states",

    # Africa
    "Nigeria": "africa/nigeria",
    "Ethiopia": "africa/ethiopia",
    "Egypt": "africa/egypt",
    "Democratic Republic of the Congo": "africa/congo-democratic-republic",
    "South Africa": "africa/south-africa",
    "Tanzania": "africa/tanzania",
    "Kenya": "africa/kenya",
    "Uganda": "africa/uganda",
    "Algeria": "africa/algeria",
    "Sudan": "africa/sudan",
    "Morocco": "africa/morocco",
    "Angola": "africa/angola",
    "Mozambique": "africa/mozambique",
    "Ghana": "africa/ghana",
    "Madagascar": "africa/madagascar",
    "Cameroon": "africa/cameroon",
    "Ivory Coast": "africa/ivory-coast",
    "Niger": "africa/niger",
    "Burkina Faso": "africa/burkina-faso",
    "Mali": "africa/mali",
    "Malawi": "africa/malawi",
    "Zambia": "africa/zambia",
    "Senegal": "africa/senegal",
    "Zimbabwe": "africa/zimbabwe",
    "Chad": "africa/chad",
    "Somalia": "africa/somalia",
    "Guinea": "africa/guinea",
    "Rwanda": "africa/rwanda",
    "Benin": "africa/benin",
    "Burundi": "africa/burundi",
    "Tunisia": "africa/tunisia",
    "South Sudan": "africa/south-sudan",
    "Togo": "africa/togo",
    "Sierra Leone": "africa/sierra-leone",
    "Libya": "africa/libya",
    "Central African Republic": "africa/central-african-republic",
    "Eritrea": "africa/eritrea",
    "Republic of the Congo": "africa/congo-brazzaville",
    "Liberia": "africa/liberia",
    "Mauritania": "africa/mauritania",
    "Namibia": "africa/namibia",
    "Botswana": "africa/botswana",
    "Lesotho": "africa/lesotho",
    "Gambia": "africa/senegal-and-gambia",
    "Guinea-Bissau": "africa/guinea-bissau",
    "Gabon": "africa/gabon",
    "Equatorial Guinea": "africa/equatorial-guinea",
    "Mauritius": "africa/mauritius",
    "Eswatini": "africa/swaziland",
    "Djibouti": "africa/djibouti",
    "Comoros": "africa/comoros",
    "Cape Verde": "africa/cape-verde",
    "Sao Tome and Principe": "africa/sao-tome-and-principe",
    "Seychelles": "africa/seychelles",

    # Europe
    "Russia": "russia",
    "Germany": "europe/germany",
    "United Kingdom": "europe/great-britain",
    "France": "europe/france",
    "Italy": "europe/italy",
    "Spain": "europe/spain",
    "Ukraine": "europe/ukraine",
    "Poland": "europe/poland",
    "Romania": "europe/romania",
    "Netherlands": "europe/netherlands",
    "Belgium": "europe/belgium",
    "Czech Republic": "europe/czech-republic",
    "Greece": "europe/greece",
    "Portugal": "europe/portugal",
    "Sweden": "europe/sweden",
    "Hungary": "europe/hungary",
    "Belarus": "europe/belarus",
    "Austria": "europe/austria",
    "Serbia": "europe/serbia",
    "Switzerland": "europe/switzerland",
    "Bulgaria": "europe/bulgaria",
    "Denmark": "europe/denmark",
    "Finland": "europe/finland",
    "Slovakia": "europe/slovakia",
    "Norway": "europe/norway",
    "Ireland": "europe/ireland-and-northern-ireland",
    "Croatia": "europe/croatia",
    "Moldova": "europe/moldova",
    "Bosnia and Herzegovina": "europe/bosnia-herzegovina",
    "Albania": "europe/albania",
    "Lithuania": "europe/lithuania",
    "North Macedonia": "europe/macedonia",
    "Slovenia": "europe/slovenia",
    "Latvia": "europe/latvia",
    "Estonia": "europe/estonia",
    "Montenegro": "europe/montenegro",
    "Luxembourg": "europe/luxembourg",
    "Malta": "europe/malta",
    "Iceland": "europe/iceland",
    "Kosovo": "europe/kosovo",
    "Turkey": "europe/turkey",
    "Cyprus": "europe/cyprus",
    "Georgia": "europe/georgia",
    "Armenia": "europe/armenia",
    "Azerbaijan": "europe/azerbaijan",

    # North America
    "United States": "north-america/us",  # Will need state-level handling
    "Canada": "north-america/canada",
    "Mexico": "north-america/mexico",

    # Central America & Caribbean
    "Guatemala": "central-america/guatemala",
    "Cuba": "central-america/cuba",
    "Haiti": "central-america/haiti-and-domrep",
    "Dominican Republic": "central-america/haiti-and-domrep",
    "Honduras": "central-america/honduras",
    "Nicaragua": "central-america/nicaragua",
    "El Salvador": "central-america/el-salvador",
    "Costa Rica": "central-america/costa-rica",
    "Panama": "central-america/panama",
    "Jamaica": "central-america/jamaica",
    "Trinidad and Tobago": "central-america/trinidad-and-tobago",
    "Belize": "central-america/belize",

    # South America
    "Brazil": "south-america/brazil",
    "Colombia": "south-america/colombia",
    "Argentina": "south-america/argentina",
    "Peru": "south-america/peru",
    "Venezuela": "south-america/venezuela",
    "Chile": "south-america/chile",
    "Ecuador": "south-america/ecuador",
    "Bolivia": "south-america/bolivia",
    "Paraguay": "south-america/paraguay",
    "Uruguay": "south-america/uruguay",
    "Guyana": "south-america/guyana",
    "Suriname": "south-america/suriname",

    # Oceania
    "Australia": "australia-oceania/australia",
    "New Zealand": "australia-oceania/new-zealand",
    "Papua New Guinea": "australia-oceania/papua-new-guinea",
    "Fiji": "australia-oceania/fiji",
    "Solomon Islands": "australia-oceania/solomon-islands",
    "Vanuatu": "australia-oceania/vanuatu",
    "New Caledonia": "australia-oceania/new-caledonia",
}


def prepare_city_lists(input_path: str, output_dir: str):
    """Generate city lists grouped by country."""

    print(f"Reading {input_path}...")
    gdf = gpd.read_file(input_path)

    # Get unique cities (one row per city)
    cities = gdf.groupby('ID_HDC_G0').first().reset_index()
    print(f"Total cities: {len(cities)}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Group by country
    country_stats = defaultdict(lambda: {"cities": [], "grids": 0})

    for _, row in cities.iterrows():
        city_id = row['ID_HDC_G0']
        country = row['CTR_MN_NM']
        city_name = row['UC_NM_MN']

        # Count grids for this city
        n_grids = len(gdf[gdf['ID_HDC_G0'] == city_id])

        country_stats[country]["cities"].append({
            "id": city_id,
            "name": city_name,
            "grids": n_grids
        })
        country_stats[country]["grids"] += n_grids

    # Write city lists per country
    summary = []
    for country, data in sorted(country_stats.items(), key=lambda x: -len(x[1]["cities"])):
        # Sanitize country name for filename
        safe_name = country.lower().replace(" ", "_").replace("'", "").replace(",", "")
        filename = f"{safe_name}_cities.txt"
        filepath = os.path.join(output_dir, filename)

        # Write city IDs (one per line)
        with open(filepath, 'w') as f:
            for city in data["cities"]:
                f.write(f"{city['id']}\n")

        # Get OSM mapping
        osm_file = COUNTRY_TO_OSM.get(country, "UNKNOWN")

        summary.append({
            "country": country,
            "filename": filename,
            "n_cities": len(data["cities"]),
            "n_grids": data["grids"],
            "avg_grids": round(data["grids"] / len(data["cities"]), 1),
            "osm_path": osm_file,
            "cities": data["cities"]
        })

        print(f"  {country}: {len(data['cities'])} cities, {data['grids']} grids -> {filename}")

    # Write summary JSON
    summary_path = os.path.join(output_dir, "country_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary written to {summary_path}")

    # Write OSM download script
    osm_script_path = os.path.join(output_dir, "download_osm.sh")
    with open(osm_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write("# Download OSM files for all countries\n")
        f.write("# Run on HPC: bash download_osm.sh\n\n")
        f.write("OSM_DIR=/scratch/kh3657/osrm/osm-data\n")
        f.write("mkdir -p $OSM_DIR\n")
        f.write("cd $OSM_DIR\n\n")

        # Collect unique OSM files needed
        osm_files = set()
        for item in summary:
            if item["osm_path"] != "UNKNOWN":
                osm_files.add(item["osm_path"])

        for osm_path in sorted(osm_files):
            url = f"https://download.geofabrik.de/{osm_path}-latest.osm.pbf"
            filename = osm_path.replace("/", "_") + "-latest.osm.pbf"
            f.write(f'echo "Downloading {osm_path}..."\n')
            f.write(f'wget -c -O {filename} {url}\n\n')

    print(f"OSM download script written to {osm_script_path}")

    # Print statistics
    print(f"\n{'='*60}")
    print("STATISTICS")
    print(f"{'='*60}")
    print(f"Total countries: {len(summary)}")
    print(f"Total cities: {sum(s['n_cities'] for s in summary)}")
    print(f"Total H3 grids: {sum(s['n_grids'] for s in summary)}")
    print(f"\nTop 10 countries by city count:")
    for item in summary[:10]:
        print(f"  {item['country']}: {item['n_cities']} cities ({item['n_grids']} grids)")

    # Check for unknown OSM mappings
    unknown = [s['country'] for s in summary if s['osm_path'] == 'UNKNOWN']
    if unknown:
        print(f"\nWARNING: {len(unknown)} countries have unknown OSM mappings:")
        for c in unknown[:10]:
            print(f"  - {c}")

    return summary


def main():
    parser = argparse.ArgumentParser(description="Prepare city lists by country for HPC processing")
    parser.add_argument("--input", "-i", required=True, help="Input GeoPackage file")
    parser.add_argument("--output-dir", "-o", default="city_lists", help="Output directory for city lists")
    args = parser.parse_args()

    prepare_city_lists(args.input, args.output_dir)


if __name__ == "__main__":
    main()
