# =============================================================================
# Fig3 Neighborhood Scaling - Original Methodology - MULTISCALE COMPARISON
# =============================================================================
# Purpose: Compare the original mixed-effects methodology across H3 resolutions
#          5 (~252 km^2), 6 (~36 km^2), and 7 (~5.2 km^2) hexagons
#
# Methodology per resolution:
#   1. Filter: population >= 1, mass > 0
#   2. Filter: countries with >= 3 cities
#   3. Filter: cities with total population > 50,000
#   4. Filter: cities with > 3 neighborhoods (>= 4)
#   5. Nested mixed-effects model:
#      log_mass ~ log_pop + (1 | Country) + (1 | Country:Country_City)
#   6. Extract random effects -> normalize -> OLS for global beta
#   7. City-level slopes for cities with > 10 neighborhoods
#
# Outputs: _Revision1/figures/
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(Matrix, lme4, readr, dplyr, performance, tibble, ggplot2,
               rprojroot, scales, tidyr, cowplot)

# =============================================================================
# SETUP
# =============================================================================

project_root <- rprojroot::find_root(rprojroot::has_file("CLAUDE.md"))
data_dir <- file.path(project_root, "_Revision1", "data")
figure_dir <- file.path(project_root, "_Revision1", "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

# Resolution-specific mass data files
mass_files <- list(
  "5"  = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution5_2026-02-02.csv"),
  "6"  = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  "7"  = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
)

# Hexagon areas (approximate, km^2)
hex_areas <- c("5" = 252.9, "6" = 36.13, "7" = 5.161)

cat("=== Fig3 Original Methodology - Multiscale Comparison ===\n")

# =============================================================================
# ANALYSIS FUNCTION: Runs original methodology for one resolution
# =============================================================================

run_original_analysis <- function(res_label, data_file) {
  cat(sprintf("\n======== Resolution %s ========\n", res_label))

  # --- Load and preprocess ---
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
    dplyr::filter(population >= 1) %>%
    dplyr::filter(mass_avg > 0)

  cat("After pop >= 1 and mass > 0:", nrow(data), "neighborhoods\n")

  # --- Filter: countries with >= 3 cities ---
  country_city_counts <- data %>%
    dplyr::group_by(country_iso) %>%
    dplyr::summarize(num_cities = n_distinct(city_id)) %>%
    dplyr::filter(num_cities >= 3)

  data <- data %>%
    dplyr::filter(country_iso %in% country_city_counts$country_iso)

  cat("After country filter (>= 3 cities):", nrow(data), "neighborhoods\n")

  # --- Filter: cities with total population > 50,000 ---
  city_pop <- data %>%
    dplyr::group_by(country_iso, city_id) %>%
    dplyr::summarize(total_pop = sum(population), .groups = "drop") %>%
    dplyr::filter(total_pop > 50000)

  data <- data %>%
    dplyr::filter(city_id %in% city_pop$city_id)

  cat("After city pop > 50k:", nrow(data), "neighborhoods\n")

  # --- Filter: cities with > 3 neighborhoods ---
  city_nbhd <- data %>%
    dplyr::group_by(country_iso, city_id) %>%
    dplyr::summarize(n_nbhd = n(), .groups = "drop") %>%
    dplyr::filter(n_nbhd > 3)

  data <- data %>%
    dplyr::filter(city_id %in% city_nbhd$city_id)

  n_neighborhoods <- nrow(data)
  n_cities <- n_distinct(data$city_id)
  n_countries <- n_distinct(data$country)
  cat("Final:", n_neighborhoods, "neighborhoods,", n_cities, "cities,", n_countries, "countries\n")

  # --- Prepare model data ---
  data <- data %>%
    mutate(
      log_population = log10(population),
      log_mass_avg = log10(mass_avg)
    )

  data$Country <- factor(data$country)
  data$Country_City <- factor(paste(data$Country, data$city_id, sep = "_"))

  # --- Fit nested mixed-effects model ---
  model <- lmer(log_mass_avg ~ log_population + (1 | Country) + (1 | Country:Country_City),
                data = data)

  fixed_intercept <- fixef(model)["(Intercept)"]
  fixed_slope <- fixef(model)["log_population"]
  se_slope <- summary(model)$coefficients["log_population", "Std. Error"]

  cat("Fixed slope (beta):", round(fixed_slope, 4), "\n")
  cat("SE:", round(se_slope, 4), "\n")
  cat("95% CI: [", round(fixed_slope - 1.96 * se_slope, 4), ",",
      round(fixed_slope + 1.96 * se_slope, 4), "]\n")

  # R-squared
  r2 <- performance::r2_nakagawa(model)
  cat("R2 marginal:", round(r2$R2_marginal, 4), "\n")
  cat("R2 conditional:", round(r2$R2_conditional, 4), "\n")

  # --- Normalized OLS (Stage 2) ---
  ranefs_country <- ranef(model)$Country %>%
    rownames_to_column("Country") %>%
    rename(ranef_Country = `(Intercept)`)

  ranefs_city <- ranef(model)$`Country:Country_City` %>%
    rownames_to_column("Country_City") %>%
    rename(ranef_Country_City = `(Intercept)`)

  data_norm <- data %>%
    mutate(Country_City_full = paste(Country, ":", Country_City, sep = "")) %>%
    left_join(ranefs_country, by = "Country") %>%
    left_join(ranefs_city, by = c("Country_City_full" = "Country_City")) %>%
    mutate(
      log_mass_normalized = log_mass_avg - ranef_Country - ranef_Country_City
    )

  ols_norm <- lm(log_mass_normalized ~ log_population, data = data_norm)
  ols_summary <- summary(ols_norm)
  beta_ols <- coef(ols_norm)[2]
  se_ols <- ols_summary$coefficients[2, 2]
  r2_ols <- ols_summary$r.squared

  cat("Normalized OLS beta:", round(beta_ols, 4), "SE:", round(se_ols, 4),
      "R2:", round(r2_ols, 4), "\n")

  # --- City-level slopes (> 10 neighborhoods) ---
  cities_enough <- data %>%
    group_by(Country_City) %>%
    filter(n() > 10) %>%
    ungroup()

  city_slopes <- cities_enough %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    group_modify(~ {
      mod <- lm(log_mass_avg ~ log_population, data = .x)
      tibble(
        Slope = coef(mod)[2],
        n_neighborhoods = nrow(.x)
      )
    }) %>%
    ungroup()

  cat("Cities with individual slopes (> 10 nbhd):", nrow(city_slopes), "\n")
  cat("  Mean slope:", round(mean(city_slopes$Slope), 4), "\n")
  cat("  Median slope:", round(median(city_slopes$Slope), 4), "\n")
  cat("  Negative slopes:", sum(city_slopes$Slope < 0), "\n")

  # Return results
  list(
    resolution = res_label,
    n_neighborhoods = n_neighborhoods,
    n_cities = n_cities,
    n_countries = n_countries,
    beta_mixed = fixed_slope,
    se_mixed = se_slope,
    r2_marginal = r2$R2_marginal,
    r2_conditional = r2$R2_conditional,
    beta_ols = beta_ols,
    se_ols = se_ols,
    r2_ols = r2_ols,
    city_slopes = city_slopes %>% mutate(Resolution = res_label)
  )
}

# =============================================================================
# RUN ANALYSIS FOR ALL RESOLUTIONS
# =============================================================================

results <- list()
for (res in c("5", "6", "7")) {
  results[[res]] <- run_original_analysis(res, mass_files[[res]])
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

cat("\n\n=== MULTISCALE COMPARISON TABLE (Original Methodology) ===\n")

summary_table <- bind_rows(lapply(results, function(r) {
  tibble(
    Resolution = r$resolution,
    `Hex Area (km2)` = hex_areas[r$resolution],
    Neighborhoods = r$n_neighborhoods,
    Cities = r$n_cities,
    Countries = r$n_countries,
    `Beta (mixed)` = round(r$beta_mixed, 4),
    SE_mixed = round(r$se_mixed, 4),
    `CI_low (mixed)` = round(r$beta_mixed - 1.96 * r$se_mixed, 4),
    `CI_high (mixed)` = round(r$beta_mixed + 1.96 * r$se_mixed, 4),
    `R2 marginal` = round(r$r2_marginal, 4),
    `R2 conditional` = round(r$r2_conditional, 4),
    `Beta (norm OLS)` = round(r$beta_ols, 4),
    SE_ols = round(r$se_ols, 4),
    `R2 (norm OLS)` = round(r$r2_ols, 4),
    `N cities (slopes)` = nrow(r$city_slopes),
    `Mean city slope` = round(mean(r$city_slopes$Slope), 4),
    `Median city slope` = round(median(r$city_slopes$Slope), 4),
    `Negative slopes` = sum(r$city_slopes$Slope < 0)
  )
}))

print(as.data.frame(summary_table), row.names = FALSE)

write_csv(summary_table, file.path(figure_dir, "Table_Original_Multiscale_Summary.csv"))
cat("\nSummary table exported.\n")

# =============================================================================
# FIGURE 1: Global Beta Comparison (Forest Plot)
# =============================================================================

beta_data <- bind_rows(lapply(results, function(r) {
  tibble(
    Resolution = paste0("Res ", r$resolution, "\n(", hex_areas[r$resolution], " km\u00b2)"),
    res_num = as.integer(r$resolution),
    Beta = r$beta_mixed,
    CI_low = r$beta_mixed - 1.96 * r$se_mixed,
    CI_high = r$beta_mixed + 1.96 * r$se_mixed,
    N = r$n_neighborhoods,
    Method = "Mixed-effects"
  )
}))

beta_data$Resolution <- factor(beta_data$Resolution,
                               levels = beta_data$Resolution[order(beta_data$res_num)])

p_forest <- ggplot(beta_data, aes(x = Beta, y = Resolution)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2, linewidth = 0.8) +
  geom_point(size = 3, color = "#bf0000") +
  geom_text(aes(label = sprintf("%.4f [%.4f, %.4f]", Beta, CI_low, CI_high)),
            vjust = -1.2, size = 3) +
  labs(
    title = "Global Scaling Exponent by H3 Resolution (Original Method)",
    x = expression("Scaling exponent " * beta),
    y = ""
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 11)
  )

ggsave(file.path(figure_dir, "Fig3_Original_Multiscale_ForestPlot.pdf"),
       p_forest, width = 8, height = 3.5)

# =============================================================================
# FIGURE 2: City-Level Slope Boxplots by Resolution
# =============================================================================

all_city_slopes <- bind_rows(lapply(results, function(r) r$city_slopes))
all_city_slopes$Resolution <- factor(
  paste0("Res ", all_city_slopes$Resolution),
  levels = c("Res 5", "Res 6", "Res 7")
)

p_box <- ggplot(all_city_slopes, aes(x = Resolution, y = Slope, fill = Resolution)) +
  geom_jitter(pch = 21, width = 0.2, stroke = 0.3, alpha = 0.15, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "red", alpha = 0.5) +
  scale_fill_manual(values = c("Res 5" = "#66c2a5", "Res 6" = "#fc8d62", "Res 7" = "#8da0cb")) +
  coord_flip() +
  labs(
    title = "City-Level Scaling Slopes by Resolution (Original Method)",
    subtitle = "Cities with > 10 neighborhoods",
    x = "",
    y = expression("Slope " * beta)
  ) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 11)
  )

ggsave(file.path(figure_dir, "Fig3_Original_Multiscale_CitySlopes.pdf"),
       p_box, width = 7, height = 3.5)

# =============================================================================
# FIGURE 3: Combined Panel
# =============================================================================

combined <- cowplot::plot_grid(p_forest, p_box, ncol = 1, labels = c("A", "B"),
                               rel_heights = c(1, 1))

ggsave(file.path(figure_dir, "Fig3_Original_Multiscale_Combined.pdf"),
       combined, width = 8, height = 7)

# Export city slopes
write_csv(all_city_slopes, file.path(figure_dir, "Table_Original_Multiscale_CitySlopes.csv"))

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n========================================\n")
cat("MULTISCALE COMPARISON COMPLETE (Original)\n")
cat("========================================\n")
for (res in c("5", "6", "7")) {
  r <- results[[res]]
  cat(sprintf("Resolution %s: beta=%.4f [%.4f, %.4f], N=%d neighborhoods, %d cities, %d countries\n",
              res, r$beta_mixed,
              r$beta_mixed - 1.96 * r$se_mixed,
              r$beta_mixed + 1.96 * r$se_mixed,
              r$n_neighborhoods, r$n_cities, r$n_countries))
  cat(sprintf("  City slopes: N=%d, mean=%.4f, median=%.4f, negative=%d\n",
              nrow(r$city_slopes), mean(r$city_slopes$Slope),
              median(r$city_slopes$Slope), sum(r$city_slopes$Slope < 0)))
}
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
