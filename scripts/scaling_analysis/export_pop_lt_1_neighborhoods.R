# =============================================================================
# Export Neighborhoods with Population < 1 for Manual Review
# =============================================================================
# Purpose: Export all 869 neighborhoods with population < 1 to CSV
#          These are data quality issues (H3 level 6 = ~10km², impossible to have < 1 person)
# Output:  _Revision1/data/neighborhoods_pop_lt_1.csv
# =============================================================================

library(readr)
library(dplyr)

# Set paths
project_root <- rprojroot::find_root(rprojroot::has_file("CLAUDE.md"))
data_file <- file.path(project_root, "data", "processed", "h3_resolution6",
                       "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")
output_file <- file.path(project_root, "_Revision1", "data",
                         "neighborhoods_pop_lt_1.csv")

# Load data
cat("Loading data from:", data_file, "\n")
data <- read_csv(data_file, show_col_types = FALSE)
cat("Total rows:", nrow(data), "\n")

# Filter population < 1
pop_lt_1 <- data %>%
  filter(population_2015 < 1) %>%
  select(
    h3index,
    city_id = ID_HDC_G0,
    city_name = UC_NM_MN,
    country = CTR_MN_NM,
    country_iso = CTR_MN_ISO,
    population_2015,
    total_built_mass_tons,
    BuildingMass_AverageTotal,
    mobility_mass_tons,
    RoadMass_Average,
    OtherPavMass_Average
  ) %>%
  arrange(desc(total_built_mass_tons))

cat("\nNeighborhoods with population < 1:", nrow(pop_lt_1), "\n")

# Summary statistics by country
cat("\n=== Distribution by Country (top 10) ===\n")
country_summary <- pop_lt_1 %>%
  group_by(country) %>%
  summarise(
    count = n(),
    total_mass_million_tons = sum(total_built_mass_tons, na.rm = TRUE) / 1e6
  ) %>%
  arrange(desc(count)) %>%
  head(10)
print(country_summary)

# Summary of extreme cases (high mass, zero pop)
cat("\n=== Extreme Cases (pop=0 but mass > 1M tonnes) ===\n")
extreme_cases <- pop_lt_1 %>%
  filter(population_2015 == 0, total_built_mass_tons > 1e6) %>%
  select(city_name, country, total_built_mass_tons) %>%
  arrange(desc(total_built_mass_tons))
print(extreme_cases)

# Export to CSV
write_csv(pop_lt_1, output_file)
cat("\nExported to:", output_file, "\n")
cat("Rows exported:", nrow(pop_lt_1), "\n")
