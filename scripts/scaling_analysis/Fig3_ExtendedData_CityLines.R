# =============================================================================
# Fig3_ExtendedData_CityLines.R
#
# Extended Data Figure: City-level Regression Lines
# Shows all city regression lines (grey) with highlighted cities
# (New York, Shanghai, Paris)
#
# Key filtering:
#   - Cities with total population > 50,000
#   - Cities with >= 10 neighborhoods (more robust estimates)
#   - Countries with >= 5 qualifying cities
#
# This figure complements Fig 3 by visualizing the variation in city-level
# scaling exponents across the global sample.
#
# =============================================================================

rm(list = ls())

# Suppress default Rplots.pdf output when running non-interactively
if (!interactive()) pdf(NULL)

# =============================================================================
# 1. Packages
# =============================================================================
library(pacman)
pacman::p_load(
  readr, dplyr, tibble, ggplot2, scales, viridis,
  cowplot, tidyr, ggrepel, patchwork
)

# =============================================================================
# 2. Load Neighborhood Data
# =============================================================================

data_neighborhood_raw <- readr::read_csv(
  "../../data/processed/h3_resolution6/Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"
) %>%
  dplyr::rename(
    mass_building = BuildingMass_AverageTotal,
    mass_mobility = mobility_mass_tons,
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM,
    city_name = UC_NM_MN
  ) %>%
  dplyr::filter(population >= 1) %>%  # FIX: >= 1 not > 0 (impossible to have < 1 person in ~10km² H3 hex)
  dplyr::filter(mass_avg > 0)

cat("Raw neighborhoods:", nrow(data_neighborhood_raw), "\n")

# =============================================================================
# 3. Filter with >= 10 Neighborhoods Threshold (More Robust)
# =============================================================================

# Step 1: Cities with total population > 50,000
city_population_totals <- data_neighborhood_raw %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(total_population = sum(population), .groups = "drop") %>%
  dplyr::filter(total_population > 50000)

filtered_neighborhoods <- data_neighborhood_raw %>%
  dplyr::filter(city_id %in% city_population_totals$city_id)

# Step 2: Cities with >= 10 neighborhoods (KEY THRESHOLD)
city_neighborhood_counts <- filtered_neighborhoods %>%
  dplyr::group_by(country_iso, city_id, city_name, country) %>%
  dplyr::summarize(num_neighborhoods = n(), .groups = "drop") %>%
  dplyr::filter(num_neighborhoods >= 10)

filtered_neighborhoods <- filtered_neighborhoods %>%
  dplyr::filter(city_id %in% city_neighborhood_counts$city_id)

# Step 3: Countries with >= 5 qualifying cities
country_city_counts <- filtered_neighborhoods %>%
  dplyr::group_by(country_iso) %>%
  dplyr::summarize(num_cities = n_distinct(city_id)) %>%
  dplyr::filter(num_cities >= 5)

filtered_neighborhoods <- filtered_neighborhoods %>%
  dplyr::filter(country_iso %in% country_city_counts$country_iso)

# Log-transform variables
filtered_neighborhoods <- filtered_neighborhoods %>%
  mutate(
    log_population = log10(population),
    log_mass = log10(mass_avg)
  )

cat("\n=== Neighborhood Data (Filtered, >= 10 neighborhoods) ===\n")
cat("Neighborhoods:", nrow(filtered_neighborhoods), "\n")
cat("Cities:", n_distinct(filtered_neighborhoods$city_id), "\n")
cat("Countries:", n_distinct(filtered_neighborhoods$country), "\n")

# =============================================================================
# 4. Compute City-level Slopes
# =============================================================================

city_slopes <- filtered_neighborhoods %>%
  group_by(country_iso, country, city_id, city_name) %>%
  summarize(
    n_neighborhoods = n(),
    mean_log_pop = mean(log_population),
    mean_log_mass = mean(log_mass),
    # Calculate slope via OLS
    slope = {
      mod <- lm(log_mass ~ log_population, data = cur_data())
      coef(mod)[2]
    },
    intercept = {
      mod <- lm(log_mass ~ log_population, data = cur_data())
      coef(mod)[1]
    },
    r_squared = {
      mod <- lm(log_mass ~ log_population, data = cur_data())
      summary(mod)$r.squared
    },
    .groups = "drop"
  )

cat("\n=== City Slope Statistics ===\n")
cat("Number of cities:", nrow(city_slopes), "\n")
cat("Slope range: [", min(city_slopes$slope), ",", max(city_slopes$slope), "]\n")
cat("Negative slopes:", sum(city_slopes$slope < 0), "\n")

# Show slope distribution
cat("\nSlope quantiles:\n")
print(quantile(city_slopes$slope, probs = c(0, 0.05, 0.25, 0.5, 0.75, 0.95, 1)))

# =============================================================================
# 5. Identify Highlighted Cities
# =============================================================================

# Find specific cities for highlighting
highlight_cities <- c("New York", "Shanghai", "Paris")

highlighted <- city_slopes %>%
  filter(city_name %in% highlight_cities) %>%
  arrange(match(city_name, highlight_cities))

cat("\n=== Highlighted Cities ===\n")
print(highlighted %>% select(city_name, country, slope, n_neighborhoods, r_squared))

# Colors for highlighted cities
highlight_colors <- c(
  "New York" = "#E41A1C",    # Red
  "Shanghai" = "#377EB8",    # Blue
  "Paris" = "#4DAF4A"        # Green
)

# =============================================================================
# 6. Main Scatter Plot with City Regression Lines (Fig 2A style - original units)
# =============================================================================

# Get axis limits from actual data range (in original units)
x_range <- range(filtered_neighborhoods$population)
y_range <- range(filtered_neighborhoods$mass_avg)

cat("\nData ranges (original units):\n")
cat("  Population:", x_range[1], "-", x_range[2], "\n")
cat("  Mass:", y_range[1], "-", y_range[2], "tonnes\n")

# Mark highlighted cities in the neighborhood data
filtered_neighborhoods <- filtered_neighborhoods %>%
  mutate(
    is_highlighted = city_name %in% highlight_cities,
    highlight_group = ifelse(is_highlighted, city_name, "Other")
  )

# Create main plot using ORIGINAL UNITS with log10 scales (like Fig 2A)
main_plot <- ggplot(filtered_neighborhoods, aes(x = population, y = mass_avg)) +
  # Background points (all neighborhoods)
  geom_point(
    shape = 21,
    fill = "grey",
    color = "black",
    stroke = 0.1,
    alpha = 0.15,
    size = 1
  ) +
  # Grey regression lines for non-highlighted cities
  stat_smooth(
    data = filter(filtered_neighborhoods, highlight_group == "Other"),
    method = "lm",
    se = FALSE,
    aes(group = city_id),
    color = "grey70",
    linewidth = 0.3,
    alpha = 0.3
  ) +
  # Colored regression lines for highlighted cities
  stat_smooth(
    data = filter(filtered_neighborhoods, is_highlighted),
    method = "lm",
    se = FALSE,
    aes(group = city_id, color = city_name),
    linewidth = 1.0
  ) +
  # Color scale for highlights
  scale_color_manual(
    values = highlight_colors,
    name = "City"
  ) +
  # Log10 scales with proper formatting (like Fig 2A)
  scale_x_log10(
    name = "Neighborhood population",
    labels = scales::trans_format("log10", scales::math_format(10^.x)),
    expand = c(0.02, 0.02)
  ) +
  scale_y_log10(
    name = "Neighborhood built mass (tonnes)",
    labels = scales::trans_format("log10", scales::math_format(10^.x)),
    expand = c(0.02, 0.02)
  ) +
  # Theme (matching Fig 2A)
  theme_bw() +
  theme(
    legend.position = "none",
    axis.ticks = element_line(colour = "black", linewidth = 0.1),
    axis.line = element_line(colour = "black", linewidth = 0.5),
    text = element_text(size = 9),
    axis.text = element_text(size = 8),
    panel.grid.major.y = element_line(linewidth = 0.2, color = "grey90", linetype = 2),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank()
  )

# =============================================================================
# 7. Add Direct Labels for Highlighted Cities
# =============================================================================

# Get label positions from the max population point of each highlighted city
label_data <- filtered_neighborhoods %>%
  filter(is_highlighted) %>%
  group_by(city_name) %>%
  slice_max(population, n = 1) %>%
  ungroup() %>%
  left_join(highlighted %>% select(city_name, slope), by = "city_name") %>%
  mutate(
    label_x = population * 0.9,
    label_y = mass_avg * 1.1,
    label = sprintf("%s (slope=%.2f)", city_name, slope)
  )

main_plot <- main_plot +
  geom_text_repel(
    data = label_data,
    aes(x = label_x, y = label_y, label = label, color = city_name),
    hjust = 0,
    size = 3,
    fontface = "bold",
    segment.color = NA,
    show.legend = FALSE
  )

# =============================================================================
# 8. Slope Histogram Inset
# =============================================================================

mean_slope <- mean(city_slopes$slope)
median_slope <- median(city_slopes$slope)

slope_hist <- ggplot(city_slopes, aes(x = slope)) +
  geom_histogram(bins = 30, fill = "grey70", color = "black", alpha = 0.8, linewidth = 0.2) +
  geom_vline(xintercept = mean_slope, color = "#fc8d62", linewidth = 0.8) +
  geom_vline(xintercept = median_slope, color = "#fc8d62", linewidth = 0.8, linetype = "dashed") +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", linewidth = 0.5) +
  # Mark highlighted cities
  geom_vline(data = highlighted,
             aes(xintercept = slope, color = city_name),
             linewidth = 0.6, linetype = "solid") +
  scale_color_manual(values = highlight_colors, guide = "none") +
  scale_x_continuous(limits = c(0, 1.5)) +
  labs(
    x = expression("City-level slope " * delta),
    y = NULL
  ) +
  theme_bw(base_size = 8) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(2, 2, 2, 2)
  )

# =============================================================================
# 9. Intercept (M0) Histogram Inset
# =============================================================================

# Calculate effective M0 for each city at REFERENCE POPULATION
# Reference: N_ref = 10,000 (10^4) for neighborhood-level analysis
# M0 = expected mass at N = 10,000, not at N = 1
N_REF_NEIGHBORHOOD <- 1e4  # 10,000 people

city_slopes <- city_slopes %>%
  mutate(
    # M0 at reference population: M0_ref = M0_original * N_ref^slope
    # In log space: M0_log_ref = intercept + slope * log10(N_ref)
    M0_log = mean_log_mass - slope * mean_log_pop + slope * log10(N_REF_NEIGHBORHOOD),
    M0 = 10^M0_log
  )

mean_M0 <- 10^mean(city_slopes$M0_log)
median_M0 <- 10^median(city_slopes$M0_log)

M0_hist <- ggplot(city_slopes, aes(x = M0)) +
  geom_histogram(bins = 30, fill = "grey70", color = "black", alpha = 0.8, linewidth = 0.2) +
  geom_vline(xintercept = mean_M0, color = "#fc8d62", linewidth = 0.8) +
  geom_vline(xintercept = median_M0, color = "#fc8d62", linewidth = 0.8, linetype = "dashed") +
  scale_x_log10(
    labels = trans_format("log10", math_format(10^.x))
  ) +
  labs(
    # M0 at N_ref = 10,000 (typical neighborhood population)
    x = expression(M(N[ref] == 10^4) ~ "(tonnes)"),
    y = NULL
  ) +
  theme_bw(base_size = 8) +
  theme(
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    panel.grid = element_blank(),
    plot.margin = margin(2, 2, 2, 2)
  )

# =============================================================================
# 10. Combine Main Plot with Insets
# =============================================================================

final_plot <- main_plot +
  inset_element(slope_hist,
                left = 0.02, bottom = 0.02,
                right = 0.40, top = 0.28) +
  inset_element(M0_hist,
                left = 0.42, bottom = 0.02,
                right = 0.80, top = 0.28)

# =============================================================================
# 11. Save Figure
# =============================================================================

current_date <- Sys.Date()
output_filename <- paste0("../figures/Fig3_ExtendedData_CityLines_", current_date, ".pdf")

ggsave(
  output_filename,
  final_plot,
  width = 8,
  height = 6,
  units = "in",
  dpi = 300
)

cat("\nFigure saved to:", output_filename, "\n")

# =============================================================================
# 12. Export City Slopes Data
# =============================================================================

write_csv(city_slopes, paste0("../data/city_slopes_ge10_neighborhoods_", current_date, ".csv"))

cat("\nCity slopes data exported.\n")

# =============================================================================
# 13. Summary Statistics
# =============================================================================

cat("\n========================================\n")
cat("SUMMARY (Extended Data Figure)\n")
cat("========================================\n")
cat("Filtering criteria:\n")
cat("  - City total population > 50,000\n")
cat("  - Neighborhoods per city >= 10\n")
cat("  - Cities per country >= 5\n")
cat("\nFinal sample:\n")
cat("  Cities:", nrow(city_slopes), "\n")
cat("  Countries:", n_distinct(city_slopes$country), "\n")
cat("  Neighborhoods:", nrow(filtered_neighborhoods), "\n")
cat("\nSlope statistics:\n")
cat("  Min:", round(min(city_slopes$slope), 4), "\n")
cat("  Max:", round(max(city_slopes$slope), 4), "\n")
cat("  Mean:", round(mean_slope, 4), "(solid orange line)\n")
cat("  Median:", round(median_slope, 4), "(dashed orange line)\n")
cat("  SD:", round(sd(city_slopes$slope), 4), "\n")
cat("  Negative slopes:", sum(city_slopes$slope < 0), "\n")
cat("\nM0 statistics (at N_ref = 10,000):\n")
cat("  Reference population: 10,000 (typical neighborhood size)\n")
cat("  Mean M0:", round(mean_M0, 2), "tonnes (solid orange line)\n")
cat("  Median M0:", round(median_M0, 2), "tonnes (dashed orange line)\n")
cat("\nHighlighted cities:\n")
for (i in 1:nrow(highlighted)) {
  cat(sprintf("  %s: β = %.3f, n = %d neighborhoods, R² = %.3f\n",
              highlighted$city_name[i],
              highlighted$slope[i],
              highlighted$n_neighborhoods[i],
              highlighted$r_squared[i]))
}
cat("========================================\n")
