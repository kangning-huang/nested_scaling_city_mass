# Neighborhood Scaling Deviation Analysis: Building vs Mobility Mass
# This script calculates deviations from regression lines and classifies neighborhoods
# into quadrants based on whether they are above/below expected values

# Load required libraries
library(pacman)
pacman::p_load(
  Matrix, sf, lme4, readr, dplyr, performance, tibble,
  ggplot2, rprojroot, scales, viridis, cowplot, tidyr
)

# Set working directory to script location
# If running interactively in RStudio, uncomment:
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ============================================================================
# SECTION 1: Load and Preprocess Data
# ============================================================================

cat("Loading H3 hexagon data...\n")

# Load the H3 hexagon grids with both building and mobility mass
data <- readr::read_csv(
  file.path("..", "..", "0.CleanProject_GlobalScaling", "data", "processed",
            "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")
) %>%
  dplyr::rename(
    mass_building = BuildingMass_AverageTotal,
    mass_mobility = mobility_mass_tons,
    mass_total = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM,
    hex_id = h3index
  ) %>%
  dplyr::filter(population > 0) %>%
  dplyr::filter(mass_building > 0) %>%
  dplyr::filter(mass_mobility > 0)

cat(sprintf("Loaded %d neighborhoods\n", nrow(data)))

# ============================================================================
# SECTION 2: Filter Data by Countries and Cities
# ============================================================================

cat("Filtering data...\n")

# Filter countries with at least 3 cities
country_city_counts <- data %>%
  dplyr::group_by(country_iso) %>%
  dplyr::summarize(num_cities = n_distinct(city_id)) %>%
  dplyr::filter(num_cities >= 3)

filtered_data <- data %>%
  dplyr::filter(country_iso %in% country_city_counts$country_iso)

# Filter cities with total population > 50,000
city_population_totals <- filtered_data %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(total_population = sum(population), .groups = "drop") %>%
  dplyr::filter(total_population > 50000)

filtered_data <- filtered_data %>%
  dplyr::filter(city_id %in% city_population_totals$city_id)

# Filter cities with more than 3 neighborhoods
city_neighborhood_counts <- filtered_data %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(num_neighborhoods = n(), .groups = "drop") %>%
  dplyr::filter(num_neighborhoods > 3)

filtered_data <- filtered_data %>%
  dplyr::filter(city_id %in% city_neighborhood_counts$city_id)

cat(sprintf("After filtering: %d neighborhoods in %d cities in %d countries\n",
            nrow(filtered_data),
            n_distinct(filtered_data$city_id),
            n_distinct(filtered_data$country_iso)))

# ============================================================================
# SECTION 3: Prepare Data for Mixed Models
# ============================================================================

# Log-transform variables for both mass types
filtered_data <- filtered_data %>%
  mutate(
    log_population = log10(population),
    log_mass_building = log10(mass_building),
    log_mass_mobility = log10(mass_mobility)
  )

# Create hierarchical factors
filtered_data$Country <- factor(filtered_data$country)
filtered_data$Country_City <- factor(with(filtered_data, paste(Country, city_id, sep = "_")))

# ============================================================================
# SECTION 4: Fit Nested Random-Effects Models
# ============================================================================

cat("\n=== BUILDING MASS MODEL ===\n")
cat("Fitting nested random-effects model for building mass...\n")

# Model for building mass
model_building <- lmer(
  log_mass_building ~ log_population + (1 | Country) + (1 | Country:Country_City),
  data = filtered_data
)

cat("Building Mass Model fitted successfully.\n")

cat("\n=== MOBILITY MASS MODEL ===\n")
cat("Fitting nested random-effects model for mobility mass...\n")

# Model for mobility mass
model_mobility <- lmer(
  log_mass_mobility ~ log_population + (1 | Country) + (1 | Country:Country_City),
  data = filtered_data
)

cat("Mobility Mass Model fitted successfully.\n")

# ============================================================================
# SECTION 5: Extract Fixed and Random Effects
# ============================================================================

cat("\nExtracting fixed and random effects...\n")

# Building mass effects
fixed_intercept_building <- fixef(model_building)["(Intercept)"]
fixed_slope_building <- fixef(model_building)["log_population"]

ranefs_country_building <- ranef(model_building)$Country %>%
  rownames_to_column("Country") %>%
  rename(ranef_Country_building = `(Intercept)`)

ranefs_city_building <- ranef(model_building)$`Country:Country_City` %>%
  rownames_to_column("Country_City") %>%
  rename(ranef_Country_City_building = `(Intercept)`)

# Mobility mass effects
fixed_intercept_mobility <- fixef(model_mobility)["(Intercept)"]
fixed_slope_mobility <- fixef(model_mobility)["log_population"]

ranefs_country_mobility <- ranef(model_mobility)$Country %>%
  rownames_to_column("Country") %>%
  rename(ranef_Country_mobility = `(Intercept)`)

ranefs_city_mobility <- ranef(model_mobility)$`Country:Country_City` %>%
  rownames_to_column("Country_City") %>%
  rename(ranef_Country_City_mobility = `(Intercept)`)

# ============================================================================
# SECTION 6: Calculate Predicted Values and Deviations
# ============================================================================

cat("\nCalculating predicted values and deviations...\n")

# Merge random effects into data
data_with_predictions <- filtered_data %>%
  mutate(Country_City = paste(Country, ':', Country_City, sep = '')) %>%
  left_join(ranefs_country_building, by = "Country") %>%
  left_join(ranefs_city_building, by = "Country_City") %>%
  left_join(ranefs_country_mobility, by = "Country") %>%
  left_join(ranefs_city_mobility, by = "Country_City") %>%
  mutate(
    # Predicted log mass from mixed-effects model (includes all random effects)
    predicted_log_mass_building = fixed_intercept_building +
                                   fixed_slope_building * log_population +
                                   ranef_Country_building +
                                   ranef_Country_City_building,
    predicted_log_mass_mobility = fixed_intercept_mobility +
                                   fixed_slope_mobility * log_population +
                                   ranef_Country_mobility +
                                   ranef_Country_City_mobility,

    # Deviation from predicted (in log space)
    deviation_log_building = log_mass_building - predicted_log_mass_building,
    deviation_log_mobility = log_mass_mobility - predicted_log_mass_mobility,

    # Ratio of observed to predicted (back to linear space)
    ratio_building = 10^deviation_log_building,
    ratio_mobility = 10^deviation_log_mobility,

    # Percentage deviation
    pct_deviation_building = (ratio_building - 1) * 100,
    pct_deviation_mobility = (ratio_mobility - 1) * 100
  )

# ============================================================================
# SECTION 7: Classify Neighborhoods into Quadrants
# ============================================================================

cat("\nClassifying neighborhoods into quadrants...\n")

# Classify based on whether each mass type is above or below predicted
data_with_quadrants <- data_with_predictions %>%
  mutate(
    # Binary classification for each dimension
    building_status = ifelse(deviation_log_building > 0, "High", "Low"),
    mobility_status = ifelse(deviation_log_mobility > 0, "High", "Low"),

    # Quadrant classification
    quadrant = case_when(
      building_status == "High" & mobility_status == "High" ~ "Q1: High Building, High Mobility",
      building_status == "High" & mobility_status == "Low"  ~ "Q2: High Building, Low Mobility",
      building_status == "Low"  & mobility_status == "High" ~ "Q3: Low Building, High Mobility",
      building_status == "Low"  & mobility_status == "Low"  ~ "Q4: Low Building, Low Mobility",
      TRUE ~ "Unclassified"
    ),

    # Numeric quadrant code for easier filtering
    quadrant_code = case_when(
      quadrant == "Q1: High Building, High Mobility" ~ 1,
      quadrant == "Q2: High Building, Low Mobility"  ~ 2,
      quadrant == "Q3: Low Building, High Mobility"  ~ 3,
      quadrant == "Q4: Low Building, Low Mobility"   ~ 4,
      TRUE ~ NA_real_
    )
  )

# Print quadrant summary
cat("\nQuadrant Distribution:\n")
quadrant_summary <- data_with_quadrants %>%
  group_by(quadrant) %>%
  summarize(
    count = n(),
    percentage = n() / nrow(data_with_quadrants) * 100,
    .groups = "drop"
  ) %>%
  arrange(quadrant)

print(quadrant_summary)

# ============================================================================
# SECTION 8: Create Summary Statistics
# ============================================================================

cat("\nCalculating summary statistics by quadrant...\n")

quadrant_stats <- data_with_quadrants %>%
  group_by(quadrant) %>%
  summarize(
    n_neighborhoods = n(),
    mean_population = mean(population, na.rm = TRUE),
    median_population = median(population, na.rm = TRUE),
    mean_deviation_building = mean(pct_deviation_building, na.rm = TRUE),
    mean_deviation_mobility = mean(pct_deviation_mobility, na.rm = TRUE),
    sd_deviation_building = sd(pct_deviation_building, na.rm = TRUE),
    sd_deviation_mobility = sd(pct_deviation_mobility, na.rm = TRUE),
    .groups = "drop"
  )

cat("\nSummary Statistics by Quadrant:\n")
print(quadrant_stats)

# ============================================================================
# SECTION 9: Load Spatial Data (H3 Hexagons)
# ============================================================================

cat("\nLoading spatial data for H3 hexagons...\n")

# Read the H3 hexagon geometries
sf_hexagons <- sf::read_sf(
  file.path("..", "..", "0.CleanProject_GlobalScaling", "data",
            "processed", "all_cities_h3_grids.gpkg")
) %>%
  dplyr::rename(hex_id = h3index)

cat(sprintf("Loaded %d hexagon geometries\n", nrow(sf_hexagons)))

# ============================================================================
# SECTION 10: Join Spatial Data with Deviation Analysis
# ============================================================================

cat("\nJoining deviation analysis with spatial data...\n")

# Join the deviation data with spatial hexagons
# Keep UC_NM_MN from spatial data (sf_hexagons) if it exists
sf_results <- sf_hexagons %>%
  inner_join(
    data_with_quadrants %>%
      select(
        hex_id, city_id, country_iso, country,
        population, mass_building, mass_mobility,
        log_population, log_mass_building, log_mass_mobility,
        predicted_log_mass_building, predicted_log_mass_mobility,
        deviation_log_building, deviation_log_mobility,
        ratio_building, ratio_mobility,
        pct_deviation_building, pct_deviation_mobility,
        building_status, mobility_status,
        quadrant, quadrant_code
      ),
    by = "hex_id"
  )

cat(sprintf("Successfully joined %d neighborhoods with spatial data\n", nrow(sf_results)))

# ============================================================================
# SECTION 11: Export Results
# ============================================================================

cat("\n=== EXPORTING RESULTS ===\n")

# Create results directory if it doesn't exist
results_dir <- file.path("..", "results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
  cat("Created results directory:", results_dir, "\n")
}

# Export spatial data as GeoPackage
output_gpkg <- file.path(results_dir, "neighborhood_deviations_building_v_mobility.gpkg")
sf::st_write(sf_results, output_gpkg, delete_dsn = TRUE)
cat(sprintf("Exported spatial results to: %s\n", output_gpkg))

# Export quadrant summary statistics as CSV
output_summary <- file.path(results_dir, "quadrant_summary_statistics.csv")
write_csv(quadrant_stats, output_summary)
cat(sprintf("Exported quadrant summary to: %s\n", output_summary))

# Export detailed deviation statistics as CSV (non-spatial)
output_detailed <- file.path(results_dir, "neighborhood_deviations_detailed.csv")
data_with_quadrants %>%
  select(
    hex_id, city_id, country_iso, country, UC_NM_MN,
    population, mass_building, mass_mobility,
    predicted_log_mass_building, predicted_log_mass_mobility,
    deviation_log_building, deviation_log_mobility,
    ratio_building, ratio_mobility,
    pct_deviation_building, pct_deviation_mobility,
    building_status, mobility_status,
    quadrant, quadrant_code
  ) %>%
  write_csv(output_detailed)
cat(sprintf("Exported detailed deviations to: %s\n", output_detailed))

# ============================================================================
# SECTION 12: Create Visualization of Quadrant Distribution
# ============================================================================

cat("\nCreating quadrant visualization...\n")

# Create scatter plot showing quadrant classification
quadrant_plot <- ggplot(data_with_quadrants,
                       aes(x = deviation_log_building,
                           y = deviation_log_mobility,
                           color = quadrant)) +
  geom_point(alpha = 0.3, size = 0.5) +
  geom_hline(yintercept = 0, linetype = "dashed", color = "black") +
  geom_vline(xintercept = 0, linetype = "dashed", color = "black") +
  scale_color_manual(
    values = c(
      "Q1: High Building, High Mobility" = "#e41a1c",
      "Q2: High Building, Low Mobility" = "#377eb8",
      "Q3: Low Building, High Mobility" = "#4daf4a",
      "Q4: Low Building, Low Mobility" = "#984ea3"
    ),
    name = "Quadrant"
  ) +
  labs(
    title = "Neighborhood Classification by Building and Mobility Mass Deviations",
    x = "Building Mass Deviation (log scale)",
    y = "Mobility Mass Deviation (log scale)",
    caption = sprintf("n = %d neighborhoods", nrow(data_with_quadrants))
  ) +
  theme_bw() +
  theme(
    legend.position = "right",
    plot.title = element_text(hjust = 0.5, face = "bold", size = 10)
  )

# Export plot
figures_dir <- file.path("..", "figures")
if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
  cat("Created figures directory:", figures_dir, "\n")
}

output_plot <- file.path(figures_dir, "Quadrant_Classification_Building_v_Mobility.pdf")
ggsave(output_plot, quadrant_plot, width = 10, height = 7, units = "in")
cat(sprintf("Exported quadrant plot to: %s\n", output_plot))

# ============================================================================
# SECTION 13: Create Bi-variate Map for Shanghai
# ============================================================================

cat("\nCreating bi-variate map for Shanghai...\n")

# Filter Shanghai data
shanghai_data <- sf_results %>%
  filter(UC_NM_MN == "Shanghai")

if (nrow(shanghai_data) > 0) {
  cat(sprintf("Found %d neighborhoods in Shanghai\n", nrow(shanghai_data)))

  # Create bi-variate classification (3x3 grid)
  # Classify building deviation into 3 categories
  building_breaks <- quantile(shanghai_data$deviation_log_building,
                              probs = c(0, 0.33, 0.67, 1), na.rm = TRUE)
  mobility_breaks <- quantile(shanghai_data$deviation_log_mobility,
                              probs = c(0, 0.33, 0.67, 1), na.rm = TRUE)

  shanghai_data <- shanghai_data %>%
    mutate(
      building_class = case_when(
        deviation_log_building <= building_breaks[2] ~ 1,
        deviation_log_building <= building_breaks[3] ~ 2,
        TRUE ~ 3
      ),
      mobility_class = case_when(
        deviation_log_mobility <= mobility_breaks[2] ~ 1,
        deviation_log_mobility <= mobility_breaks[3] ~ 2,
        TRUE ~ 3
      ),
      bivariate_class = paste0(building_class, "-", mobility_class)
    )

  # Define bi-variate color palette (3x3)
  bivariate_colors <- c(
    "1-1" = "#e8e8e8", "2-1" = "#ace4e4", "3-1" = "#5ac8c8",
    "1-2" = "#dfb0d6", "2-2" = "#a5add3", "3-2" = "#5698b9",
    "1-3" = "#be64ac", "2-3" = "#8c62aa", "3-3" = "#3b4994"
  )

  # Create the bi-variate map
  bivariate_map <- ggplot() +
    ggspatial::annotation_map_tile(
      type = "cartolight",
      zoom = 11,
      cachedir = tempdir()
    ) +
    geom_sf(data = shanghai_data,
            aes(fill = bivariate_class),
            color = NA,
            alpha = 0.7) +
    scale_fill_manual(
      values = bivariate_colors,
      name = "Bi-variate Class",
      guide = "none"
    ) +
    labs(
      title = "Shanghai: Building vs Mobility Mass Deviations",
      subtitle = "Bi-variate classification of neighborhood-level deviations"
    ) +
    theme_void() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
      plot.subtitle = element_text(hjust = 0.5, size = 10)
    )

  # Create legend for bi-variate map
  legend_data <- expand.grid(
    building_class = 1:3,
    mobility_class = 1:3
  ) %>%
    mutate(bivariate_class = paste0(building_class, "-", mobility_class))

  bivariate_legend <- ggplot(legend_data,
                             aes(x = building_class, y = mobility_class,
                                 fill = bivariate_class)) +
    geom_tile(color = "white", size = 0.5) +
    scale_fill_manual(values = bivariate_colors, guide = "none") +
    scale_x_continuous(breaks = 1:3, labels = c("Low", "Med", "High")) +
    scale_y_continuous(breaks = 1:3, labels = c("Low", "Med", "High")) +
    labs(
      x = "Building Mass →",
      y = "Mobility Mass →"
    ) +
    theme_minimal() +
    theme(
      axis.title = element_text(size = 9, face = "bold"),
      axis.text = element_text(size = 8),
      panel.grid = element_blank(),
      plot.background = element_rect(fill = "white", color = "black")
    )

  # Combine map and legend
  bivariate_plot_combined <- ggdraw() +
    draw_plot(bivariate_map) +
    draw_plot(bivariate_legend, x = 0.70, y = 0.70, width = 0.25, height = 0.25)

  # Export bi-variate map
  output_bivariate <- file.path(figures_dir, "Shanghai_Bivariate_Map_Building_v_Mobility.pdf")
  ggsave(output_bivariate, bivariate_plot_combined, width = 10, height = 8, units = "in")
  cat(sprintf("Exported Shanghai bi-variate map to: %s\n", output_bivariate))

  # Also export as PNG for easier viewing
  output_bivariate_png <- file.path(figures_dir, "Shanghai_Bivariate_Map_Building_v_Mobility.png")
  ggsave(output_bivariate_png, bivariate_plot_combined, width = 10, height = 8, units = "in", dpi = 300)
  cat(sprintf("Exported Shanghai bi-variate map (PNG) to: %s\n", output_bivariate_png))

} else {
  cat("Warning: No data found for Shanghai\n")
}

# ============================================================================
# SECTION 13B: Create Bi-variate Maps for Multiple Cities
# ============================================================================

cat("\nCreating bi-variate maps for multiple cities...\n")

# Define cities to map
cities_to_map <- c("New York", "Beijing", "Los Angeles", "Guangzhou",
                   "London", "Tokyo", "Lagos", "Sydney", "Mumbai")

# Define zoom levels for different cities (adjust based on city size)
city_zoom_levels <- c(
  "New York" = 10, "Beijing" = 10, "Los Angeles" = 10, "Guangzhou" = 10,
  "London" = 10, "Tokyo" = 10, "Lagos" = 10, "Sydney" = 10, "Mumbai" = 10
)

# Loop through each city and create bi-variate map
for (city_name in cities_to_map) {
  cat(sprintf("\nProcessing %s...\n", city_name))

  # Filter city data
  city_data <- sf_results %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data), city_name))

    # Create bi-variate classification (3x3 grid)
    building_breaks <- quantile(city_data$deviation_log_building,
                                probs = c(0, 0.33, 0.67, 1), na.rm = TRUE)
    mobility_breaks <- quantile(city_data$deviation_log_mobility,
                                probs = c(0, 0.33, 0.67, 1), na.rm = TRUE)

    city_data <- city_data %>%
      mutate(
        building_class = case_when(
          deviation_log_building <= building_breaks[2] ~ 1,
          deviation_log_building <= building_breaks[3] ~ 2,
          TRUE ~ 3
        ),
        mobility_class = case_when(
          deviation_log_mobility <= mobility_breaks[2] ~ 1,
          deviation_log_mobility <= mobility_breaks[3] ~ 2,
          TRUE ~ 3
        ),
        bivariate_class = paste0(building_class, "-", mobility_class)
      )

    # Create the bi-variate map
    bivariate_map <- ggplot() +
      ggspatial::annotation_map_tile(
        type = "cartolight",
        zoom = city_zoom_levels[city_name],
        cachedir = tempdir()
      ) +
      geom_sf(data = city_data,
              aes(fill = bivariate_class),
              color = NA,
              alpha = 0.7) +
      scale_fill_manual(
        values = bivariate_colors,
        name = "Bi-variate Class",
        guide = "none"
      ) +
      labs(
        title = sprintf("%s: Building vs Mobility Mass Deviations", city_name),
        subtitle = "Bi-variate classification of neighborhood-level deviations"
      ) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
        plot.subtitle = element_text(hjust = 0.5, size = 10)
      )

    # Create legend (reuse legend_data from Shanghai)
    legend_data <- expand.grid(
      building_class = 1:3,
      mobility_class = 1:3
    ) %>%
      mutate(bivariate_class = paste0(building_class, "-", mobility_class))

    bivariate_legend <- ggplot(legend_data,
                               aes(x = building_class, y = mobility_class,
                                   fill = bivariate_class)) +
      geom_tile(color = "white", size = 0.5) +
      scale_fill_manual(values = bivariate_colors, guide = "none") +
      scale_x_continuous(breaks = 1:3, labels = c("Low", "Med", "High")) +
      scale_y_continuous(breaks = 1:3, labels = c("Low", "Med", "High")) +
      labs(
        x = "Building Mass →",
        y = "Mobility Mass →"
      ) +
      theme_minimal() +
      theme(
        axis.title = element_text(size = 9, face = "bold"),
        axis.text = element_text(size = 8),
        panel.grid = element_blank(),
        plot.background = element_rect(fill = "white", color = "black")
      )

    # Combine map and legend
    bivariate_plot_combined <- ggdraw() +
      draw_plot(bivariate_map) +
      draw_plot(bivariate_legend, x = 0.70, y = 0.70, width = 0.25, height = 0.25)

    # Create safe filename (replace spaces)
    safe_city_name <- gsub(" ", "_", city_name)

    # Export bi-variate map
    output_bivariate <- file.path(figures_dir,
                                  sprintf("%s_Bivariate_Map_Building_v_Mobility.pdf", safe_city_name))
    ggsave(output_bivariate, bivariate_plot_combined, width = 10, height = 8, units = "in")
    cat(sprintf("Exported %s bi-variate map to: %s\n", city_name, output_bivariate))

    # Also export as PNG
    output_bivariate_png <- file.path(figures_dir,
                                      sprintf("%s_Bivariate_Map_Building_v_Mobility.png", safe_city_name))
    ggsave(output_bivariate_png, bivariate_plot_combined, width = 10, height = 8, units = "in", dpi = 300)
    cat(sprintf("Exported %s bi-variate map (PNG) to: %s\n", city_name, output_bivariate_png))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

# ============================================================================
# SECTION 14: Create Summary Report
# ============================================================================

cat("\n=== ANALYSIS COMPLETE ===\n")

cat("\nModel Parameters:\n")
cat(sprintf("Building Mass: β = %.3f, Intercept = %.3f\n",
            fixed_slope_building, fixed_intercept_building))
cat(sprintf("Mobility Mass: β = %.3f, Intercept = %.3f\n",
            fixed_slope_mobility, fixed_intercept_mobility))

cat("\nDeviation Statistics:\n")
cat(sprintf("Building Mass - Mean deviation: %.2f%%, SD: %.2f%%\n",
            mean(data_with_quadrants$pct_deviation_building, na.rm = TRUE),
            sd(data_with_quadrants$pct_deviation_building, na.rm = TRUE)))
cat(sprintf("Mobility Mass - Mean deviation: %.2f%%, SD: %.2f%%\n",
            mean(data_with_quadrants$pct_deviation_mobility, na.rm = TRUE),
            sd(data_with_quadrants$pct_deviation_mobility, na.rm = TRUE)))

cat("\nQuadrant Distribution:\n")
for (i in 1:nrow(quadrant_summary)) {
  cat(sprintf("  %s: %d neighborhoods (%.1f%%)\n",
              quadrant_summary$quadrant[i],
              quadrant_summary$count[i],
              quadrant_summary$percentage[i]))
}

cat("\n=== OUTPUT FILES ===\n")
cat("Spatial data (GeoPackage):\n")
cat(sprintf("  %s\n", output_gpkg))
cat("\nStatistics (CSV):\n")
cat(sprintf("  %s\n", output_summary))
cat(sprintf("  %s\n", output_detailed))
cat("\nVisualization (PDF):\n")
cat(sprintf("  %s\n", output_plot))

cat("\n=== END OF SCRIPT ===\n")
