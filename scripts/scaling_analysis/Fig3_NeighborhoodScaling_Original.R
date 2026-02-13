# =============================================================================
# Fig3 Neighborhood Scaling - Original Methodology (Converted from Rmd)
# =============================================================================
# Purpose: Reproduce the ORIGINAL Fig3 analysis from Fig3_NeighborhoodScaling_UpdateMI.Rmd
#          for comparison with revision methodology
#
# Key Filtering Thresholds (ORIGINAL):
#   - population > 0 (filters out zero/negative)
#   - mass_avg > 0
#   - countries with >= 3 cities
#   - cities with total population > 50,000
#   - cities with > 3 neighborhoods (i.e., >= 4) for main analysis
#   - cities with n() > 10 (i.e., >= 11) for city-level slope estimation
#
# Model: Nested mixed-effects model
#   log_mass_avg ~ log_population + (1 | Country) + (1 | Country:Country_City)
#
# =============================================================================

library(pacman)
pacman::p_load(Matrix, lme4, readr, dplyr, performance, tibble, ggplot2,
               rprojroot, scales, tidyr, cowplot)

# =============================================================================
# SETUP
# =============================================================================

project_root <- rprojroot::find_root(rprojroot::has_file("CLAUDE.md"))
data_file <- file.path(project_root, "data", "processed", "h3_resolution6",
                       "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")
output_dir <- file.path(project_root, "_Revision1", "data")

cat("=== Fig3 Original Methodology ===\n")
cat("Data file:", data_file, "\n")

# =============================================================================
# LOAD AND PREPROCESS DATA
# =============================================================================

data <- readr::read_csv(data_file, show_col_types = FALSE) %>%
  dplyr::rename(
    mass_building = BuildingMass_AverageTotal,
    mass_mobility = mobility_mass_tons,
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  dplyr::filter(population >= 1) %>%   # FIX: >= 1 not > 0 (impossible to have < 1 person in ~10km² H3 hex)
  dplyr::filter(mass_avg > 0)

cat("After population > 0 and mass > 0 filter:", nrow(data), "neighborhoods\n")

# =============================================================================
# FILTER: Countries with >= 3 cities
# =============================================================================

country_city_counts <- data %>%
  dplyr::group_by(country_iso) %>%
  dplyr::summarize(num_cities = n_distinct(city_id)) %>%
  dplyr::filter(num_cities >= 3)        # ORIGINAL: >= 3

filtered_data <- data %>%
  dplyr::filter(country_iso %in% country_city_counts$country_iso)

cat("After country filter (>= 3 cities):", nrow(filtered_data), "neighborhoods\n")
cat("Countries retained:", nrow(country_city_counts), "\n")

# =============================================================================
# FILTER: Cities with total population > 50,000
# =============================================================================

city_population_totals <- filtered_data %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(total_population = sum(population), .groups = "drop") %>%
  dplyr::filter(total_population > 50000)  # ORIGINAL: > 50,000

filtered_data <- filtered_data %>%
  dplyr::filter(city_id %in% city_population_totals$city_id)

cat("After city population filter (> 50k):", nrow(filtered_data), "neighborhoods\n")
cat("Cities retained:", n_distinct(filtered_data$city_id), "\n")

# =============================================================================
# FILTER: Cities with > 3 neighborhoods (i.e., >= 4)
# =============================================================================

city_neighborhood_counts <- filtered_data %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(num_neighborhoods = n(), .groups = "drop") %>%
  dplyr::filter(num_neighborhoods > 3)   # ORIGINAL: > 3 (i.e., >= 4)

filtered_data <- filtered_data %>%
  dplyr::filter(city_id %in% city_neighborhood_counts$city_id)

cat("After neighborhood count filter (> 3):", nrow(filtered_data), "neighborhoods\n")
cat("Cities retained:", n_distinct(filtered_data$city_id), "\n")
cat("Countries retained:", n_distinct(filtered_data$country), "\n")

# =============================================================================
# PREPARE DATA FOR MIXED MODEL
# =============================================================================

filtered_data <- filtered_data %>%
  mutate(
    log_population = log10(population),
    log_mass_avg = log10(mass_avg)
  )

# Create hierarchical factors
filtered_data$Country <- factor(filtered_data$country)
filtered_data$Country_City <- factor(with(filtered_data, paste(Country, city_id, sep = "_")))

# =============================================================================
# FIT NESTED RANDOM-EFFECTS MODEL
# =============================================================================

cat("\n=== Fitting Nested Mixed-Effects Model ===\n")

model <- lmer(log_mass_avg ~ log_population + (1 | Country) + (1 | Country:Country_City),
              data = filtered_data)

# Extract fixed effects
fixed_intercept <- fixef(model)["(Intercept)"]
fixed_slope <- fixef(model)["log_population"]

cat("Fixed intercept:", round(fixed_intercept, 4), "\n")
cat("Fixed slope (beta):", round(fixed_slope, 4), "\n")

# Extract random effects
ranefs_country <- ranef(model)$Country %>%
  rownames_to_column("Country") %>%
  rename(ranef_Country = `(Intercept)`)

ranefs_city <- ranef(model)$`Country:Country_City` %>%
  rownames_to_column("Country_City") %>%
  rename(ranef_Country_City = `(Intercept)`)

# =============================================================================
# CALCULATE R-SQUARED
# =============================================================================

r2 <- performance::r2_nakagawa(model)
cat("\nR-squared (marginal - fixed effects only):", round(r2$R2_marginal, 4), "\n")
cat("R-squared (conditional - fixed + random):", round(r2$R2_conditional, 4), "\n")

# =============================================================================
# MERGE RANDOM EFFECTS BACK TO DATA
# =============================================================================

data_merged <- filtered_data %>%
  mutate(Country_City_full = paste(Country, ':', Country_City, sep = '')) %>%
  left_join(ranefs_country, by = "Country") %>%
  left_join(ranefs_city, by = c("Country_City_full" = "Country_City")) %>%
  mutate(
    estimated_log_mass_avg = fixed_intercept + fixed_slope * log_population +
                             ranef_Country + ranef_Country_City
  )

# =============================================================================
# CITY-LEVEL SLOPES: Filter cities with > 10 neighborhoods (i.e., >= 11)
# =============================================================================

cat("\n=== City-Level Slopes (Original: n() > 10) ===\n")

# ORIGINAL threshold: n() > 10 means >= 11 neighborhoods
cities_with_enough_data <- data_merged %>%
  group_by(Country_City) %>%
  filter(n() > 10)                       # ORIGINAL: > 10 (i.e., >= 11)

cat("Cities meeting n() > 10 threshold:", n_distinct(cities_with_enough_data$Country_City), "\n")
cat("Neighborhoods in these cities:", nrow(cities_with_enough_data), "\n")

# Run regression for each city
city_models <- cities_with_enough_data %>%
  group_by(Country_City) %>%
  group_map(~ lm(log_mass_avg ~ log_population, data = .x))

names(city_models) <- unique(cities_with_enough_data$Country_City)

# Extract slopes
city_fixed_effects <- lapply(city_models, function(model) {
  list(
    intercept = coef(model)["(Intercept)"],
    slope = coef(model)["log_population"]
  )
})

city_slope <- do.call(rbind, lapply(names(city_fixed_effects), function(name) {
  data.frame(
    Country_City = name,
    Slope = city_fixed_effects[[name]]$slope
  )
}))

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

cat("\n=== City Slope Distribution (Original Methodology) ===\n")
cat("Number of cities:", nrow(city_slope), "\n")
cat("Mean slope:", round(mean(city_slope$Slope), 4), "\n")
cat("Median slope:", round(median(city_slope$Slope), 4), "\n")
cat("SD:", round(sd(city_slope$Slope), 4), "\n")
cat("Min:", round(min(city_slope$Slope), 4), "\n")
cat("Max:", round(max(city_slope$Slope), 4), "\n")
cat("Number of NEGATIVE slopes:", sum(city_slope$Slope < 0), "\n")
cat("Number of slopes < 1:", sum(city_slope$Slope < 1), "\n")

# Negative slope cities
if (sum(city_slope$Slope < 0) > 0) {
  cat("\n=== Cities with Negative Slopes ===\n")
  negative_cities <- city_slope %>% filter(Slope < 0) %>% arrange(Slope)
  print(negative_cities)
}

# =============================================================================
# EXPORT CITY SLOPES
# =============================================================================

output_file <- file.path(output_dir, "city_slope_original_method.csv")
write_csv(city_slope, output_file)
cat("\nCity slopes exported to:", output_file, "\n")

# =============================================================================
# BOXPLOT OF SLOPES
# =============================================================================

cat("\n=== Creating Slope Boxplot ===\n")

p <- ggplot(city_slope, aes(x = "City", y = Slope)) +
  geom_jitter(pch = 21, width = 0.2, alpha = 0.3) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, fill = "#bf0000") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "red") +
  coord_flip() +
  labs(
    title = "City-Level Scaling Slopes (Original Method)",
    subtitle = paste0("n = ", nrow(city_slope), " cities (threshold: > 10 neighborhoods)"),
    x = "",
    y = expression("Slope " * beta)
  ) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank()
  )

figure_file <- file.path(project_root, "_Revision1", "figures",
                         "Fig3_city_slopes_original_method.pdf")
dir.create(dirname(figure_file), showWarnings = FALSE, recursive = TRUE)
ggsave(figure_file, p, width = 6, height = 3)
cat("Figure saved to:", figure_file, "\n")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n=== FINAL SUMMARY ===\n")
cat("Total neighborhoods analyzed:", nrow(filtered_data), "\n")
cat("Total cities analyzed:", n_distinct(filtered_data$city_id), "\n")
cat("Total countries:", n_distinct(filtered_data$country), "\n")
cat("Global scaling slope (beta):", round(fixed_slope, 4), "\n")
cat("Number of cities with individual slopes:", nrow(city_slope), "\n")
cat("Negative slopes:", sum(city_slope$Slope < 0), "\n")
