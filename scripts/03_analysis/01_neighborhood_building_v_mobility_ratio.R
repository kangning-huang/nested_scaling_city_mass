# Building vs Mobility Mass Ratio Analysis
# This script calculates absolute ratios between building mass and mobility mass
# at the neighborhood (H3 hexagon) level

# Load required libraries
library(pacman)
pacman::p_load(
  sf, readr, dplyr, ggplot2, scales, viridis, tidyr
)

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
# SECTION 2: Calculate Building to Mobility Mass Ratios
# ============================================================================

cat("\nCalculating building to mobility mass ratios...\n")

# Calculate various ratio metrics
data_with_ratios <- data %>%
  mutate(
    # Absolute ratio: building mass / mobility mass
    ratio_building_to_mobility = mass_building / mass_mobility,

    # Percentage of total built mass
    pct_building = (mass_building / mass_total) * 100,
    pct_mobility = (mass_mobility / mass_total) * 100,

    # Mass per capita
    mass_building_per_capita = mass_building / population,
    mass_mobility_per_capita = mass_mobility / population,

    # Ratio per capita
    ratio_per_capita = mass_building_per_capita / mass_mobility_per_capita
  )

# ============================================================================
# SECTION 3: Summary Statistics
# ============================================================================

cat("\nCalculating summary statistics...\n")

# Global summary statistics
global_stats <- data_with_ratios %>%
  summarize(
    n_neighborhoods = n(),
    n_cities = n_distinct(city_id),
    n_countries = n_distinct(country_iso),

    # Building mass statistics
    mean_building_mass = mean(mass_building, na.rm = TRUE),
    median_building_mass = median(mass_building, na.rm = TRUE),
    sd_building_mass = sd(mass_building, na.rm = TRUE),

    # Mobility mass statistics
    mean_mobility_mass = mean(mass_mobility, na.rm = TRUE),
    median_mobility_mass = median(mass_mobility, na.rm = TRUE),
    sd_mobility_mass = sd(mass_mobility, na.rm = TRUE),

    # Ratio statistics
    mean_ratio = mean(ratio_building_to_mobility, na.rm = TRUE),
    median_ratio = median(ratio_building_to_mobility, na.rm = TRUE),
    sd_ratio = sd(ratio_building_to_mobility, na.rm = TRUE),

    # Quartiles of ratio
    q25_ratio = quantile(ratio_building_to_mobility, 0.25, na.rm = TRUE),
    q75_ratio = quantile(ratio_building_to_mobility, 0.75, na.rm = TRUE)
  )

cat("\nGlobal Summary Statistics:\n")
print(global_stats)

# Country-level summary
country_stats <- data_with_ratios %>%
  group_by(country_iso, country) %>%
  summarize(
    n_neighborhoods = n(),
    n_cities = n_distinct(city_id),
    mean_ratio = mean(ratio_building_to_mobility, na.rm = TRUE),
    median_ratio = median(ratio_building_to_mobility, na.rm = TRUE),
    sd_ratio = sd(ratio_building_to_mobility, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_ratio))

cat("\nTop 10 Countries by Mean Building/Mobility Ratio:\n")
print(head(country_stats, 10))

# City-level summary
city_stats <- data_with_ratios %>%
  group_by(country_iso, country, city_id, UC_NM_MN) %>%
  summarize(
    n_neighborhoods = n(),
    mean_ratio = mean(ratio_building_to_mobility, na.rm = TRUE),
    median_ratio = median(ratio_building_to_mobility, na.rm = TRUE),
    sd_ratio = sd(ratio_building_to_mobility, na.rm = TRUE),
    total_building_mass = sum(mass_building, na.rm = TRUE),
    total_mobility_mass = sum(mass_mobility, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(mean_ratio))

# ============================================================================
# SECTION 4: Load Spatial Data (H3 Hexagons)
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
# SECTION 5: Join Spatial Data with Ratio Analysis
# ============================================================================

cat("\nJoining ratio analysis with spatial data...\n")

# Join the ratio data with spatial hexagons
sf_results <- sf_hexagons %>%
  inner_join(
    data_with_ratios %>%
      select(
        hex_id, city_id, country_iso, country,
        population, mass_building, mass_mobility, mass_total,
        ratio_building_to_mobility,
        pct_building, pct_mobility,
        mass_building_per_capita, mass_mobility_per_capita,
        ratio_per_capita
      ),
    by = "hex_id"
  )

cat(sprintf("Successfully joined %d neighborhoods with spatial data\n", nrow(sf_results)))

# ============================================================================
# SECTION 6: Export Results
# ============================================================================

cat("\n=== EXPORTING RESULTS ===\n")

# Create results directory if it doesn't exist
results_dir <- file.path("..", "results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
  cat("Created results directory:", results_dir, "\n")
}

# Export spatial data as GeoPackage
output_gpkg <- file.path(results_dir, "neighborhood_building_mobility_ratios.gpkg")
sf::st_write(sf_results, output_gpkg, delete_dsn = TRUE)
cat(sprintf("Exported spatial results to: %s\n", output_gpkg))

# Export global summary statistics as CSV
output_global_stats <- file.path(results_dir, "global_ratio_statistics.csv")
write_csv(global_stats, output_global_stats)
cat(sprintf("Exported global statistics to: %s\n", output_global_stats))

# Export country summary statistics as CSV
output_country_stats <- file.path(results_dir, "country_ratio_statistics.csv")
write_csv(country_stats, output_country_stats)
cat(sprintf("Exported country statistics to: %s\n", output_country_stats))

# Export city summary statistics as CSV
output_city_stats <- file.path(results_dir, "city_ratio_statistics.csv")
write_csv(city_stats, output_city_stats)
cat(sprintf("Exported city statistics to: %s\n", output_city_stats))

# Export detailed neighborhood data as CSV (non-spatial)
output_detailed <- file.path(results_dir, "neighborhood_ratios_detailed.csv")
data_with_ratios %>%
  select(
    hex_id, city_id, country_iso, country, UC_NM_MN,
    population, mass_building, mass_mobility, mass_total,
    ratio_building_to_mobility,
    pct_building, pct_mobility,
    mass_building_per_capita, mass_mobility_per_capita,
    ratio_per_capita
  ) %>%
  write_csv(output_detailed)
cat(sprintf("Exported detailed neighborhood ratios to: %s\n", output_detailed))

# ============================================================================
# SECTION 7: Create Maps for Selected Cities
# ============================================================================

cat("\n=== CREATING CITY MAPS ===\n")

# Create figures directory structure
figures_dir <- file.path("..", "figures", "ratio_maps")
if (!dir.exists(figures_dir)) {
  dir.create(figures_dir, recursive = TRUE)
  cat("Created ratio maps directory:", figures_dir, "\n")
} else {
  cat("Using existing ratio maps directory:", figures_dir, "\n")
}

# Define cities to map
cities_to_map <- c("Shanghai", "New York", "Beijing", "Guangzhou",
                   "Rome", "Paris", "Atlanta", "Tokyo",
                   "Addis Ababa", "Bogota", "Mexico City", "Melbourne")

# Define zoom levels for different cities
city_zoom_levels <- c(
  "Shanghai" = 11,
  "New York" = 10,
  "Beijing" = 10,
  "Guangzhou" = 10,
  "Rome" = 10,
  "Paris" = 10,
  "Atlanta" = 10,
  "Tokyo" = 10,
  "Addis Ababa" = 10,
  "Bogota" = 10,
  "Mexico City" = 10,
  "Melbourne" = 10
)

# Check if ggspatial is available for basemaps
if (!require("ggspatial", quietly = TRUE)) {
  cat("Installing ggspatial package for basemaps...\n")
  install.packages("ggspatial")
  library(ggspatial)
}

# Loop through each city and create map
for (city_name in cities_to_map) {
  cat(sprintf("\nProcessing %s...\n", city_name))

  # Filter city data
  city_data <- sf_results %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data), city_name))

    # Create the map
    city_map <- ggplot() +
      ggspatial::annotation_map_tile(
        type = "cartolight",
        zoom = city_zoom_levels[city_name],
        cachedir = tempdir()
      ) +
      geom_sf(data = city_data,
              aes(fill = ratio_building_to_mobility),
              color = NA,
              alpha = 0.8) +
      scale_fill_viridis_c(
        option = "plasma",
        name = "Building/Mobility\nMass Ratio",
        na.value = "grey50",
        trans = "log10",
        guide = guide_colorbar(
          title.position = "top",
          title.hjust = 0.5,
          barwidth = 15,
          barheight = 0.5
        )
      ) +
      labs(
        title = sprintf("%s: Building to Mobility Mass Ratio", city_name),
        subtitle = sprintf("Mean ratio: %.2f, Median ratio: %.2f",
                          mean(city_data$ratio_building_to_mobility, na.rm = TRUE),
                          median(city_data$ratio_building_to_mobility, na.rm = TRUE)),
        caption = sprintf("n = %d neighborhoods", nrow(city_data))
      ) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        plot.caption = element_text(hjust = 0.5, size = 9, color = "gray50"),
        legend.position = "bottom",
        legend.title = element_text(size = 10, face = "bold")
      )

    # Create safe filename (replace spaces)
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_pdf <- file.path(figures_dir,
                           sprintf("%s_Building_Mobility_Ratio_Map.pdf", safe_city_name))
    ggsave(output_pdf, city_map, width = 10, height = 8, units = "in")
    cat(sprintf("Exported %s map (PDF) to: %s\n", city_name, output_pdf))

    # Export as PNG for easier viewing
    output_png <- file.path(figures_dir,
                           sprintf("%s_Building_Mobility_Ratio_Map.png", safe_city_name))
    ggsave(output_png, city_map, width = 10, height = 8, units = "in", dpi = 300)
    cat(sprintf("Exported %s map (PNG) to: %s\n", city_name, output_png))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

# ============================================================================
# SECTION 8: Create Scatter Plots - Population Density vs Ratio
# ============================================================================

cat("\n=== CREATING SCATTER PLOTS ===\n")

# Calculate population density (population per km2)
# H3 resolution 6 hexagon area is approximately 36.129 km²
h3_res6_area_km2 <- 36.129

# Add population density to the spatial results
sf_results <- sf_results %>%
  mutate(
    population_density = population / h3_res6_area_km2
  )

# Also add to non-spatial data
data_with_ratios <- data_with_ratios %>%
  mutate(
    population_density = population / h3_res6_area_km2
  )

# Define cities for scatter plots
cities_for_scatter <- c("Shanghai", "New York", "Beijing",
                        "Rome", "Paris", "Atlanta", "Tokyo",
                        "Addis Ababa", "Bogota", "Mexico City", "Melbourne")

# Loop through each city and create scatter plot
for (city_name in cities_for_scatter) {
  cat(sprintf("\nCreating scatter plot for %s...\n", city_name))

  # Filter city data
  city_data_scatter <- sf_results %>%
    sf::st_drop_geometry() %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data_scatter) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data_scatter), city_name))

    # Filter data to start at minimum density of 10 people/km²
    city_data_scatter <- city_data_scatter %>%
      filter(population_density >= 10)

    # Calculate correlation on log-transformed population density
    cor_result <- cor.test(log10(city_data_scatter$population_density),
                           city_data_scatter$ratio_building_to_mobility,
                           method = "pearson")
    cor_value <- cor_result$estimate
    cor_pvalue <- cor_result$p.value

    # Create bins based on log-transformed population density
    city_data_scatter <- city_data_scatter %>%
      mutate(
        log_pop_density = log10(population_density),
        # Create bins using cut on log scale
        pop_density_bin = cut(log_pop_density,
                              breaks = 10,
                              labels = FALSE,
                              include.lowest = TRUE)
      )

    # Calculate binned averages
    binned_data <- city_data_scatter %>%
      group_by(pop_density_bin) %>%
      summarize(
        pop_density_mean = 10^mean(log_pop_density, na.rm = TRUE),
        ratio_mean = mean(ratio_building_to_mobility, na.rm = TRUE),
        ratio_se = sd(ratio_building_to_mobility, na.rm = TRUE) / sqrt(n()),
        n_neighborhoods = n(),
        .groups = "drop"
      ) %>%
      filter(!is.na(pop_density_mean) & !is.na(ratio_mean))

    # Create binned scatter plot with error bars
    scatter_plot <- ggplot() +
      # Individual points as background (semi-transparent)
      geom_point(data = city_data_scatter,
                aes(x = population_density, y = ratio_building_to_mobility),
                alpha = 0.2, size = 1, color = "gray50") +
      # Binned averages with error bars
      geom_errorbar(data = binned_data,
                   aes(x = pop_density_mean,
                       ymin = ratio_mean - ratio_se,
                       ymax = ratio_mean + ratio_se),
                   width = 0, alpha = 0.5, color = "red", linewidth = 0.5) +
      geom_point(data = binned_data,
                aes(x = pop_density_mean, y = ratio_mean, size = n_neighborhoods),
                color = "red", alpha = 0.8) +
      geom_line(data = binned_data,
               aes(x = pop_density_mean, y = ratio_mean),
               color = "red", linewidth = 1) +
      scale_x_log10(
        name = "Population Density (people/km²)",
        labels = comma,
        limits = c(10, NA)
      ) +
      scale_y_continuous(
        name = "Building/Mobility Mass Ratio",
        labels = comma
      ) +
      scale_size_continuous(
        name = "Neighborhoods\nper bin",
        range = c(2, 8)
      ) +
      labs(
        title = sprintf("%s: Population Density vs Building/Mobility Ratio", city_name),
        subtitle = sprintf("Pearson r = %.3f (p %s 0.001) | n = %d neighborhoods, %d bins",
                          cor_value,
                          ifelse(cor_pvalue < 0.001, "<", "≥"),
                          nrow(city_data_scatter),
                          nrow(binned_data)),
        caption = "Red line: binned averages with standard error bars"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        plot.caption = element_text(hjust = 0.5, size = 9, color = "gray50"),
        legend.position = "right"
      )

    # Create safe filename
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_scatter_pdf <- file.path(figures_dir,
                                   sprintf("%s_PopDensity_vs_Ratio_Scatter.pdf", safe_city_name))
    ggsave(output_scatter_pdf, scatter_plot, width = 10, height = 7, units = "in")
    cat(sprintf("Exported %s scatter plot (PDF) to: %s\n", city_name, output_scatter_pdf))

    # Export as PNG
    output_scatter_png <- file.path(figures_dir,
                                   sprintf("%s_PopDensity_vs_Ratio_Scatter.png", safe_city_name))
    ggsave(output_scatter_png, scatter_plot, width = 10, height = 7, units = "in", dpi = 300)
    cat(sprintf("Exported %s scatter plot (PNG) to: %s\n", city_name, output_scatter_png))

    # Print correlation statistics
    cat(sprintf("  Correlation: r = %.3f, p-value = %.2e\n", cor_value, cor_pvalue))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

# ============================================================================
# SECTION 9: Create Global Distribution Visualization
# ============================================================================

cat("\nCreating global distribution plot...\n")

# Create histogram of ratios (with log scale on x-axis for better visualization)
ratio_histogram <- ggplot(data_with_ratios,
                         aes(x = ratio_building_to_mobility)) +
  geom_histogram(bins = 50, fill = "#2c7bb6", alpha = 0.7, color = "white") +
  geom_vline(xintercept = median(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE),
             linetype = "dashed", color = "red", size = 1) +
  geom_vline(xintercept = mean(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE),
             linetype = "dashed", color = "blue", size = 1) +
  scale_x_log10(
    name = "Building/Mobility Mass Ratio (log scale)",
    breaks = c(0.1, 0.5, 1, 2, 5, 10, 50, 100),
    labels = c("0.1", "0.5", "1", "2", "5", "10", "50", "100")
  ) +
  scale_y_continuous(
    name = "Number of Neighborhoods",
    labels = comma
  ) +
  labs(
    title = "Global Distribution of Building to Mobility Mass Ratios",
    subtitle = sprintf("Red line: median = %.2f | Blue line: mean = %.2f",
                      median(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE),
                      mean(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE)),
    caption = sprintf("n = %s neighborhoods across %d cities in %d countries",
                     comma(nrow(data_with_ratios)),
                     n_distinct(data_with_ratios$city_id),
                     n_distinct(data_with_ratios$country_iso))
  ) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold", size = 12),
    plot.subtitle = element_text(hjust = 0.5, size = 10),
    plot.caption = element_text(hjust = 0.5, size = 8, color = "gray50")
  )

# Export histogram
output_hist <- file.path(figures_dir, "Global_Building_Mobility_Ratio_Distribution.pdf")
ggsave(output_hist, ratio_histogram, width = 10, height = 6, units = "in")
cat(sprintf("Exported ratio distribution plot to: %s\n", output_hist))

# Also export as PNG
output_hist_png <- file.path(figures_dir, "Global_Building_Mobility_Ratio_Distribution.png")
ggsave(output_hist_png, ratio_histogram, width = 10, height = 6, units = "in", dpi = 300)
cat(sprintf("Exported ratio distribution plot (PNG) to: %s\n", output_hist_png))

# ============================================================================
# SECTION 10: Summary Report
# ============================================================================

cat("\n=== ANALYSIS COMPLETE ===\n")

cat("\nGlobal Summary:\n")
cat(sprintf("  Total neighborhoods: %s\n", comma(nrow(data_with_ratios))))
cat(sprintf("  Total cities: %d\n", n_distinct(data_with_ratios$city_id)))
cat(sprintf("  Total countries: %d\n", n_distinct(data_with_ratios$country_iso)))

cat("\nBuilding/Mobility Mass Ratio Statistics:\n")
cat(sprintf("  Mean ratio: %.2f\n", mean(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE)))
cat(sprintf("  Median ratio: %.2f\n", median(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE)))
cat(sprintf("  SD: %.2f\n", sd(data_with_ratios$ratio_building_to_mobility, na.rm = TRUE)))
cat(sprintf("  25th percentile: %.2f\n", quantile(data_with_ratios$ratio_building_to_mobility, 0.25, na.rm = TRUE)))
cat(sprintf("  75th percentile: %.2f\n", quantile(data_with_ratios$ratio_building_to_mobility, 0.75, na.rm = TRUE)))

cat("\n=== OUTPUT FILES ===\n")
cat("Spatial data (GeoPackage):\n")
cat(sprintf("  %s\n", output_gpkg))
cat("\nStatistics (CSV):\n")
cat(sprintf("  %s\n", output_global_stats))
cat(sprintf("  %s\n", output_country_stats))
cat(sprintf("  %s\n", output_city_stats))
cat(sprintf("  %s\n", output_detailed))
cat("\nVisualization (PDF/PNG):\n")
cat(sprintf("  %s\n", output_hist))
cat("\nCity Maps:\n")
for (city_name in cities_to_map) {
  safe_city_name <- gsub(" ", "_", city_name)
  cat(sprintf("  %s\n", file.path(figures_dir, sprintf("%s_Building_Mobility_Ratio_Map.pdf", safe_city_name))))
}
cat("\nScatter Plots:\n")
for (city_name in cities_for_scatter) {
  safe_city_name <- gsub(" ", "_", city_name)
  cat(sprintf("  %s\n", file.path(figures_dir, sprintf("%s_PopDensity_vs_Ratio_Scatter.pdf", safe_city_name))))
}

# ============================================================================
# SECTION 11: Create Bivariate Maps for Selected Cities
# ============================================================================

cat("\n=== CREATING BIVARIATE MAPS ===\n")

# Set CRAN mirror for package installation
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Install biscale package if not available
if (!require("biscale", quietly = TRUE)) {
  cat("Installing biscale package for bivariate mapping...\n")
  install.packages("biscale")
  library(biscale)
}

# Install cowplot if not already loaded
if (!require("cowplot", quietly = TRUE)) {
  cat("Installing cowplot package...\n")
  install.packages("cowplot")
  library(cowplot)
}

# Create bivariate figures directory if it doesn't exist
bivar_figures_dir <- file.path("..", "figures", "bivariate_maps")
if (!dir.exists(bivar_figures_dir)) {
  dir.create(bivar_figures_dir, recursive = TRUE)
  cat("Created bivariate maps directory:", bivar_figures_dir, "\n")
}

# Define cities for bivariate mapping
cities_for_bivariate <- c("Rome", "Paris", "Atlanta", "Tokyo",
                          "Addis Ababa", "Bogota", "Mexico City", "Melbourne")

# Loop through each city and create bivariate map
for (city_name in cities_for_bivariate) {
  cat(sprintf("\nProcessing bivariate map for %s...\n", city_name))

  # Filter city data
  city_data_bivar <- sf_results %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data_bivar) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data_bivar), city_name))

    # Create bivariate classes (3x3 grid) using per capita measures
    city_data_bivar <- biscale::bi_class(
      city_data_bivar,
      x = mass_building_per_capita,
      y = mass_mobility_per_capita,
      style = "quantile",
      dim = 3
    )

    # Create the bivariate map
    bivar_map <- ggplot() +
      geom_sf(
        data = city_data_bivar,
        aes(fill = bi_class),
        color = "white",
        size = 0.1,
        show.legend = FALSE
      ) +
      biscale::bi_scale_fill(pal = "DkViolet", dim = 3) +
      labs(
        title = sprintf("%s: Building vs Mobility Mass (Per Capita)", city_name),
        subtitle = sprintf(
          "Mean: %.1f t/person building, %.1f t/person mobility | n = %d neighborhoods",
          mean(city_data_bivar$mass_building_per_capita, na.rm = TRUE),
          mean(city_data_bivar$mass_mobility_per_capita, na.rm = TRUE),
          nrow(city_data_bivar)
        )
      ) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        plot.margin = margin(10, 10, 10, 10)
      )

    # Create bivariate legend
    bivar_legend <- biscale::bi_legend(
      pal = "DkViolet",
      dim = 3,
      xlab = "Higher Building Mass →",
      ylab = "Higher Mobility Mass →",
      size = 8
    )

    # Combine map and legend
    final_bivar_plot <- cowplot::ggdraw() +
      cowplot::draw_plot(bivar_map, 0, 0, 1, 1) +
      cowplot::draw_plot(bivar_legend, 0.05, 0.05, 0.25, 0.25)

    # Create safe filename
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_bivar_pdf <- file.path(
      bivar_figures_dir,
      sprintf("%s_Bivariate_Building_Mobility_PerCapita.pdf", safe_city_name)
    )
    ggsave(output_bivar_pdf, final_bivar_plot, width = 10, height = 8, units = "in")
    cat(sprintf("Exported %s bivariate map (PDF) to: %s\n", city_name, output_bivar_pdf))

    # Export as PNG
    output_bivar_png <- file.path(
      bivar_figures_dir,
      sprintf("%s_Bivariate_Building_Mobility_PerCapita.png", safe_city_name)
    )
    ggsave(output_bivar_png, final_bivar_plot, width = 10, height = 8, units = "in", dpi = 300)
    cat(sprintf("Exported %s bivariate map (PNG) to: %s\n", city_name, output_bivar_png))

    # Print summary statistics
    cat(sprintf("  Mean building per capita: %.2f t/person\n",
                mean(city_data_bivar$mass_building_per_capita, na.rm = TRUE)))
    cat(sprintf("  Mean mobility per capita: %.2f t/person\n",
                mean(city_data_bivar$mass_mobility_per_capita, na.rm = TRUE)))
    cat(sprintf("  Mean ratio: %.2f\n",
                mean(city_data_bivar$ratio_building_to_mobility, na.rm = TRUE)))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

cat("\n=== BIVARIATE MAPPING COMPLETE ===\n")
cat("\nBivariate maps exported to:", bivar_figures_dir, "\n")

# ============================================================================
# SECTION 12: Create Comparative Bivariate Maps for All Selected Cities
# ============================================================================

cat("\n=== CREATING MULTI-CITY BIVARIATE COMPARISON ===\n")

# Create a multi-panel comparison plot
bivar_panels <- list()

for (city_name in cities_for_bivariate) {
  city_data_bivar <- sf_results %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data_bivar) > 0) {
    # Create bivariate classes
    city_data_bivar <- biscale::bi_class(
      city_data_bivar,
      x = mass_building_per_capita,
      y = mass_mobility_per_capita,
      style = "quantile",
      dim = 3
    )

    # Create simplified map for panel
    panel_map <- ggplot() +
      geom_sf(
        data = city_data_bivar,
        aes(fill = bi_class),
        color = NA,
        show.legend = FALSE
      ) +
      biscale::bi_scale_fill(pal = "DkViolet", dim = 3) +
      labs(title = city_name) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 10),
        plot.margin = margin(5, 5, 5, 5)
      )

    bivar_panels[[city_name]] <- panel_map
  }
}

# Create legend for the multi-panel plot
multi_legend <- biscale::bi_legend(
  pal = "DkViolet",
  dim = 3,
  xlab = "Building Mass →",
  ylab = "Mobility Mass →",
  size = 6
)

# Combine all panels
if (length(bivar_panels) > 0) {
  multi_city_plot <- cowplot::plot_grid(
    plotlist = bivar_panels,
    ncol = 4,
    nrow = 2,
    labels = NULL
  )

  # Add legend to the combined plot
  final_multi_plot <- cowplot::ggdraw() +
    cowplot::draw_plot(multi_city_plot, 0, 0.1, 1, 0.9) +
    cowplot::draw_plot(multi_legend, 0.42, 0.01, 0.16, 0.16) +
    cowplot::draw_label(
      "Bivariate Comparison: Building vs Mobility Mass (Per Capita)",
      x = 0.5, y = 0.98, hjust = 0.5, vjust = 1, size = 14, fontface = "bold"
    )

  # Export multi-city comparison
  output_multi_pdf <- file.path(
    bivar_figures_dir,
    "MultiCity_Bivariate_Comparison.pdf"
  )
  ggsave(output_multi_pdf, final_multi_plot, width = 16, height = 8, units = "in")
  cat(sprintf("Exported multi-city comparison (PDF) to: %s\n", output_multi_pdf))

  output_multi_png <- file.path(
    bivar_figures_dir,
    "MultiCity_Bivariate_Comparison.png"
  )
  ggsave(output_multi_png, final_multi_plot, width = 16, height = 8, units = "in", dpi = 300)
  cat(sprintf("Exported multi-city comparison (PNG) to: %s\n", output_multi_png))
}

# ============================================================================
# SECTION 13: Move Existing Files to ratio_maps Subfolder
# ============================================================================

cat("\n=== ORGANIZING EXISTING FILES ===\n")

# Define old figures directory (parent directory)
old_figures_dir <- file.path("..", "figures")

# List of file patterns to move (if they exist in the parent directory)
file_patterns_to_move <- c(
  "*_Building_Mobility_Ratio_Map.pdf",
  "*_Building_Mobility_Ratio_Map.png",
  "*_PopDensity_vs_Ratio_Scatter.pdf",
  "*_PopDensity_vs_Ratio_Scatter.png",
  "Global_Building_Mobility_Ratio_Distribution.pdf",
  "Global_Building_Mobility_Ratio_Distribution.png"
)

# Move existing files if they are in the wrong location
files_moved <- 0
for (pattern in file_patterns_to_move) {
  # Find matching files in the old directory
  matching_files <- list.files(
    path = old_figures_dir,
    pattern = glob2rx(pattern),
    full.names = TRUE
  )

  # Move each matching file
  for (old_path in matching_files) {
    # Skip if it's already in the ratio_maps subdirectory
    if (grepl("ratio_maps", old_path)) {
      next
    }

    # Get just the filename
    filename <- basename(old_path)
    new_path <- file.path(figures_dir, filename)

    # Move the file
    if (file.exists(old_path) && !file.exists(new_path)) {
      file.rename(old_path, new_path)
      cat(sprintf("Moved: %s -> ratio_maps/%s\n", filename, filename))
      files_moved <- files_moved + 1
    } else if (file.exists(new_path)) {
      cat(sprintf("Already exists in ratio_maps: %s\n", filename))
    }
  }
}

if (files_moved > 0) {
  cat(sprintf("\nMoved %d existing files to ratio_maps subfolder\n", files_moved))
} else {
  cat("\nNo existing files needed to be moved\n")
}

# ============================================================================
# SECTION 14: Create Population Density Maps for Selected Cities
# ============================================================================

cat("\n=== CREATING POPULATION DENSITY MAPS ===\n")

# Create population density figures directory
popdens_figures_dir <- file.path("..", "figures", "pop_dens_maps")
if (!dir.exists(popdens_figures_dir)) {
  dir.create(popdens_figures_dir, recursive = TRUE)
  cat("Created population density maps directory:", popdens_figures_dir, "\n")
}

# Use the same cities as for bivariate mapping
cities_for_popdens <- c("Rome", "Paris", "Atlanta", "Tokyo",
                        "Addis Ababa", "Bogota", "Mexico City", "Melbourne",
                        "Shanghai", "New York", "Beijing", "Guangzhou")

# Loop through each city and create population density map
for (city_name in cities_for_popdens) {
  cat(sprintf("\nProcessing population density map for %s...\n", city_name))

  # Filter city data
  city_data_popdens <- sf_results %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data_popdens) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data_popdens), city_name))

    # Calculate statistics for subtitle
    mean_density <- mean(city_data_popdens$population_density, na.rm = TRUE)
    median_density <- median(city_data_popdens$population_density, na.rm = TRUE)

    # Create the population density map
    popdens_map <- ggplot() +
      geom_sf(
        data = city_data_popdens,
        aes(fill = population_density),
        color = "white",
        size = 0.1,
        show.legend = TRUE
      ) +
      scale_fill_viridis_c(
        option = "viridis",
        name = "Population Density\n(people/km²)",
        trans = "log10",
        labels = comma,
        na.value = "grey90",
        guide = guide_colorbar(
          title.position = "top",
          title.hjust = 0.5,
          barwidth = 15,
          barheight = 0.8
        )
      ) +
      labs(
        title = sprintf("%s: Population Density", city_name),
        subtitle = sprintf(
          "Mean: %s people/km² | Median: %s people/km² | n = %d neighborhoods",
          comma(round(mean_density)),
          comma(round(median_density)),
          nrow(city_data_popdens)
        )
      ) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 10),
        plot.margin = margin(10, 10, 10, 10),
        legend.position = "bottom",
        legend.title = element_text(size = 10, face = "bold")
      )

    # Create safe filename
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_popdens_pdf <- file.path(
      popdens_figures_dir,
      sprintf("%s_Population_Density.pdf", safe_city_name)
    )
    ggsave(output_popdens_pdf, popdens_map, width = 10, height = 8, units = "in")
    cat(sprintf("Exported %s population density map (PDF) to: %s\n", city_name, output_popdens_pdf))

    # Export as PNG
    output_popdens_png <- file.path(
      popdens_figures_dir,
      sprintf("%s_Population_Density.png", safe_city_name)
    )
    ggsave(output_popdens_png, popdens_map, width = 10, height = 8, units = "in", dpi = 300)
    cat(sprintf("Exported %s population density map (PNG) to: %s\n", city_name, output_popdens_png))

    # Print summary statistics
    cat(sprintf("  Mean density: %s people/km²\n", comma(round(mean_density))))
    cat(sprintf("  Median density: %s people/km²\n", comma(round(median_density))))
    cat(sprintf("  Min density: %s people/km²\n", comma(round(min(city_data_popdens$population_density, na.rm = TRUE)))))
    cat(sprintf("  Max density: %s people/km²\n", comma(round(max(city_data_popdens$population_density, na.rm = TRUE)))))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

cat("\n=== POPULATION DENSITY MAPPING COMPLETE ===\n")
cat("\nPopulation density maps exported to:", popdens_figures_dir, "\n")

# ============================================================================
# SECTION 15: Create Log-Scale Ratio Maps for Selected Cities
# ============================================================================

cat("\n=== CREATING LOG-SCALE RATIO MAPS ===\n")

# Loop through each city and create log-scale ratio map
for (city_name in cities_to_map) {
  cat(sprintf("\nProcessing log-scale ratio map for %s...\n", city_name))

  # Filter city data
  city_data <- sf_results %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data), city_name))

    # Create the map with log-scale ratio
    city_map_log <- ggplot() +
      geom_sf(data = city_data,
              aes(fill = ratio_building_to_mobility),
              color = "white",
              size = 0.1,
              alpha = 0.9) +
      scale_fill_viridis_c(
        option = "plasma",
        name = "Building/Mobility\nMass Ratio\n(log scale)",
        na.value = "grey50",
        trans = "log10",
        breaks = c(0.1, 0.5, 1, 2, 5, 10, 50, 100),
        labels = c("0.1", "0.5", "1", "2", "5", "10", "50", "100"),
        guide = guide_colorbar(
          title.position = "top",
          title.hjust = 0.5,
          barwidth = 15,
          barheight = 0.8
        )
      ) +
      labs(
        title = sprintf("%s: Building to Mobility Mass Ratio (Log Scale)", city_name),
        subtitle = sprintf("Mean ratio: %.2f, Median ratio: %.2f",
                          mean(city_data$ratio_building_to_mobility, na.rm = TRUE),
                          median(city_data$ratio_building_to_mobility, na.rm = TRUE)),
        caption = sprintf("n = %d neighborhoods", nrow(city_data))
      ) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        plot.caption = element_text(hjust = 0.5, size = 9, color = "gray50"),
        legend.position = "bottom",
        legend.title = element_text(size = 10, face = "bold")
      )

    # Create safe filename (replace spaces)
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_pdf_log <- file.path(figures_dir,
                               sprintf("%s_Building_Mobility_Ratio_Map_LogScale.pdf", safe_city_name))
    ggsave(output_pdf_log, city_map_log, width = 10, height = 8, units = "in")
    cat(sprintf("Exported %s log-scale ratio map (PDF) to: %s\n", city_name, output_pdf_log))

    # Export as PNG for easier viewing
    output_png_log <- file.path(figures_dir,
                               sprintf("%s_Building_Mobility_Ratio_Map_LogScale.png", safe_city_name))
    ggsave(output_png_log, city_map_log, width = 10, height = 8, units = "in", dpi = 300)
    cat(sprintf("Exported %s log-scale ratio map (PNG) to: %s\n", city_name, output_png_log))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

cat("\n=== LOG-SCALE RATIO MAPPING COMPLETE ===\n")

# ============================================================================
# SECTION 16: Create Log-Scale Scatter Plots - Population Density vs Ratio
# ============================================================================

cat("\n=== CREATING LOG-SCALE SCATTER PLOTS ===\n")

# Loop through each city and create log-scale scatter plot
for (city_name in cities_for_scatter) {
  cat(sprintf("\nCreating log-scale scatter plot for %s...\n", city_name))

  # Filter city data
  city_data_scatter <- sf_results %>%
    sf::st_drop_geometry() %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data_scatter) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data_scatter), city_name))

    # Filter data to start at minimum density of 10 people/km²
    city_data_scatter <- city_data_scatter %>%
      filter(population_density >= 10)

    # Calculate correlation on log-transformed population density and log-transformed ratio
    cor_result <- cor.test(log10(city_data_scatter$population_density),
                           log10(city_data_scatter$ratio_building_to_mobility),
                           method = "pearson")
    cor_value <- cor_result$estimate
    cor_pvalue <- cor_result$p.value

    # Create bins based on log-transformed population density
    city_data_scatter <- city_data_scatter %>%
      mutate(
        log_pop_density = log10(population_density),
        log_ratio = log10(ratio_building_to_mobility),
        # Create bins using cut on log scale
        pop_density_bin = cut(log_pop_density,
                              breaks = 10,
                              labels = FALSE,
                              include.lowest = TRUE)
      )

    # Calculate binned averages
    binned_data <- city_data_scatter %>%
      group_by(pop_density_bin) %>%
      summarize(
        pop_density_mean = 10^mean(log_pop_density, na.rm = TRUE),
        ratio_mean = 10^mean(log_ratio, na.rm = TRUE),
        ratio_se_log = sd(log_ratio, na.rm = TRUE) / sqrt(n()),
        n_neighborhoods = n(),
        .groups = "drop"
      ) %>%
      mutate(
        ratio_lower = 10^(log10(ratio_mean) - ratio_se_log),
        ratio_upper = 10^(log10(ratio_mean) + ratio_se_log)
      ) %>%
      filter(!is.na(pop_density_mean) & !is.na(ratio_mean))

    # Create binned scatter plot with error bars and log-scale ratio
    scatter_plot_log <- ggplot() +
      # Individual points as background (semi-transparent)
      geom_point(data = city_data_scatter,
                aes(x = population_density, y = ratio_building_to_mobility),
                alpha = 0.2, size = 1, color = "gray50") +
      # Binned averages with error bars
      geom_errorbar(data = binned_data,
                   aes(x = pop_density_mean,
                       ymin = ratio_lower,
                       ymax = ratio_upper),
                   width = 0, alpha = 0.5, color = "red", linewidth = 0.5) +
      geom_point(data = binned_data,
                aes(x = pop_density_mean, y = ratio_mean, size = n_neighborhoods),
                color = "red", alpha = 0.8) +
      geom_line(data = binned_data,
               aes(x = pop_density_mean, y = ratio_mean),
               color = "red", linewidth = 1) +
      scale_x_log10(
        name = "Population Density (people/km²)",
        labels = comma,
        limits = c(10, NA)
      ) +
      scale_y_log10(
        name = "Building/Mobility Mass Ratio (log scale)",
        labels = comma,
        breaks = c(0.1, 0.5, 1, 2, 5, 10, 50, 100)
      ) +
      scale_size_continuous(
        name = "Neighborhoods\nper bin",
        range = c(2, 8)
      ) +
      labs(
        title = sprintf("%s: Population Density vs Building/Mobility Ratio (Log Scale)", city_name),
        subtitle = sprintf("Pearson r = %.3f (p %s 0.001) | n = %d neighborhoods, %d bins",
                          cor_value,
                          ifelse(cor_pvalue < 0.001, "<", "≥"),
                          nrow(city_data_scatter),
                          nrow(binned_data)),
        caption = "Red line: binned averages with standard error bars (on log scale)"
      ) +
      theme_bw() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        plot.caption = element_text(hjust = 0.5, size = 9, color = "gray50"),
        legend.position = "right"
      )

    # Create safe filename
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_scatter_pdf_log <- file.path(figures_dir,
                                       sprintf("%s_PopDensity_vs_Ratio_Scatter_LogScale.pdf", safe_city_name))
    ggsave(output_scatter_pdf_log, scatter_plot_log, width = 10, height = 7, units = "in")
    cat(sprintf("Exported %s log-scale scatter plot (PDF) to: %s\n", city_name, output_scatter_pdf_log))

    # Export as PNG
    output_scatter_png_log <- file.path(figures_dir,
                                       sprintf("%s_PopDensity_vs_Ratio_Scatter_LogScale.png", safe_city_name))
    ggsave(output_scatter_png_log, scatter_plot_log, width = 10, height = 7, units = "in", dpi = 300)
    cat(sprintf("Exported %s log-scale scatter plot (PNG) to: %s\n", city_name, output_scatter_png_log))

    # Print correlation statistics
    cat(sprintf("  Correlation (log-log): r = %.3f, p-value = %.2e\n", cor_value, cor_pvalue))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

cat("\n=== LOG-SCALE SCATTER PLOTS COMPLETE ===\n")

cat("\n=== END OF SCRIPT ===\n")
