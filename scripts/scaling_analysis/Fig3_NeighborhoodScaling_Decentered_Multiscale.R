# =============================================================================
# Fig3 Neighborhood Scaling - Decentered Methodology - MULTISCALE COMPARISON
# =============================================================================
# Purpose: Compare the de-centering approach (Bettencourt & Lobo 2016) across
#          H3 resolutions 5 (~252 km^2), 6 (~36 km^2), and 7 (~5.2 km^2)
#
# Methodology per resolution:
#   Panel A: De-center neighborhoods by city mean -> OLS -> delta (city-level)
#   Panel B: Country-level slopes from Figure 2 city data (resolution-independent)
#   Panel C: Per-city slopes and M0 distributions
#
# Outputs: _Revision1/figures/
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(Matrix, lme4, readr, dplyr, tibble, ggplot2, scales,
               cowplot, tidyr, ggrepel)

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

hex_areas <- c("5" = 252.9, "6" = 36.13, "7" = 5.161)

N_REF_NEIGHBORHOOD <- 1e4  # Reference population for city-level M0
N_REF_COUNTRY <- 1e6       # Reference population for country-level M0

cat("=== Fig3 Decentered Methodology - Multiscale Comparison ===\n")

# =============================================================================
# LOAD FIGURE 2 CITY DATA (resolution-independent)
# =============================================================================

data_city_fig2 <- read.csv(file.path(data_dir, "MasterMass_ByClass20250616.csv"),
                           stringsAsFactors = FALSE) %>%
  dplyr::filter(total_built_mass_tons > 0)

cat("Figure 2 city data:", nrow(data_city_fig2), "cities\n")

# Apply Figure 2 filters: countries with >= 5 cities
city_counts_fig2 <- table(data_city_fig2$CTR_MN_NM)
valid_countries_fig2 <- names(city_counts_fig2[city_counts_fig2 >= 5])

filtered_cities_fig2 <- data_city_fig2 %>%
  dplyr::filter(CTR_MN_NM %in% valid_countries_fig2) %>%
  mutate(
    log_population = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  )

cat("Filtered Figure 2 cities:", nrow(filtered_cities_fig2), "in",
    length(valid_countries_fig2), "countries\n")

# --- Country-level beta from Figure 2 (de-centered) ---
decentered_cities_fig2 <- filtered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  mutate(
    country_mean_log_pop = mean(log_population),
    country_mean_log_mass = mean(log_mass),
    log_pop_centered = log_population - country_mean_log_pop,
    log_mass_centered = log_mass - country_mean_log_mass
  ) %>%
  ungroup()

mod_fig2 <- lm(log_mass_centered ~ log_pop_centered, data = decentered_cities_fig2)
beta_fig2 <- coef(mod_fig2)[2]
se_beta_fig2 <- summary(mod_fig2)$coefficients[2, 2]
r2_beta_fig2 <- summary(mod_fig2)$r.squared

cat(sprintf("Figure 2 beta (de-centered): %.4f [%.4f, %.4f]\n",
            beta_fig2, beta_fig2 - 1.96 * se_beta_fig2, beta_fig2 + 1.96 * se_beta_fig2))

# --- Country-level slopes from Figure 2 ---
country_slopes_fig2 <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  group_modify(~ {
    mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
    tibble(Slope = coef(mod)[2], n_cities = nrow(.x))
  }) %>%
  ungroup()

# Country M0 at reference population
country_M0_fig2 <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  summarize(
    mean_log_pop = mean(log_population),
    mean_log_mass = mean(log_mass),
    .groups = "drop"
  ) %>%
  left_join(country_slopes_fig2 %>% select(CTR_MN_NM, Slope), by = "CTR_MN_NM") %>%
  mutate(
    M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_COUNTRY),
    M0 = 10^M0_log
  )

country_stats_fig2 <- country_slopes_fig2 %>%
  left_join(country_M0_fig2 %>% select(CTR_MN_NM, M0_log, M0), by = "CTR_MN_NM") %>%
  rename(Country = CTR_MN_NM)

cat("Country-level stats:", nrow(country_stats_fig2), "countries\n")

# =============================================================================
# ANALYSIS FUNCTION: Runs decentered analysis for one resolution
# =============================================================================

run_decentered_analysis <- function(res_label, data_file) {
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

  # --- Filter: cities with total population > 50,000 ---
  city_pop <- data %>%
    dplyr::group_by(country_iso, city_id) %>%
    dplyr::summarize(total_pop = sum(population), .groups = "drop") %>%
    dplyr::filter(total_pop > 50000)

  data <- data %>%
    dplyr::filter(city_id %in% city_pop$city_id)

  # --- Filter: cities with >= 10 neighborhoods ---
  city_nbhd <- data %>%
    dplyr::group_by(country_iso, city_id) %>%
    dplyr::summarize(n_nbhd = n(), .groups = "drop") %>%
    dplyr::filter(n_nbhd >= 10)

  data <- data %>%
    dplyr::filter(city_id %in% city_nbhd$city_id)

  # --- Filter: countries with >= 5 qualifying cities ---
  country_city_counts <- data %>%
    dplyr::group_by(country_iso) %>%
    dplyr::summarize(num_cities = n_distinct(city_id)) %>%
    dplyr::filter(num_cities >= 5)

  data <- data %>%
    dplyr::filter(country_iso %in% country_city_counts$country_iso)

  # --- Log transform ---
  data <- data %>%
    mutate(
      log_population = log10(population),
      log_mass_avg = log10(mass_avg)
    )

  data$Country <- factor(data$country)
  data$Country_City <- factor(paste(data$country, data$city_id, sep = "_"))

  n_neighborhoods <- nrow(data)
  n_cities <- n_distinct(data$city_id)
  n_countries <- n_distinct(data$country)
  cat("Final:", n_neighborhoods, "neighborhoods,", n_cities, "cities,", n_countries, "countries\n")

  # --- De-center neighborhoods by city mean ---
  decentered <- data %>%
    group_by(Country_City) %>%
    mutate(
      city_mean_log_pop = mean(log_population),
      city_mean_log_mass = mean(log_mass_avg),
      log_pop_centered = log_population - city_mean_log_pop,
      log_mass_centered = log_mass_avg - city_mean_log_mass
    ) %>%
    ungroup()

  # --- OLS on pooled de-centered data -> delta ---
  mod_delta <- lm(log_mass_centered ~ log_pop_centered, data = decentered)
  delta <- coef(mod_delta)[2]
  se_delta <- summary(mod_delta)$coefficients[2, 2]
  r2_delta <- summary(mod_delta)$r.squared

  cat(sprintf("Delta (city-level): %.4f [%.4f, %.4f], R2=%.4f\n",
              delta, delta - 1.96 * se_delta, delta + 1.96 * se_delta, r2_delta))

  # --- Per-city slopes ---
  city_slopes <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 10) %>%
    group_modify(~ {
      mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
      tibble(Slope = coef(mod)[2], n_neighborhoods = nrow(.x))
    }) %>%
    ungroup()

  # --- Per-city M0 at reference population ---
  city_M0 <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 10) %>%
    summarize(
      mean_log_pop = mean(log_population),
      mean_log_mass = mean(log_mass_avg),
      .groups = "drop"
    ) %>%
    left_join(city_slopes %>% select(Country_City, Slope), by = "Country_City") %>%
    mutate(
      M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_NEIGHBORHOOD),
      M0 = 10^M0_log
    )

  city_stats <- city_slopes %>%
    left_join(city_M0 %>% select(Country_City, M0_log, M0, mean_log_pop, mean_log_mass),
              by = "Country_City")

  cat("Cities with slopes (>= 10 nbhd):", nrow(city_stats), "\n")
  cat("  Mean slope:", round(mean(city_stats$Slope), 4), "\n")
  cat("  Negative slopes:", sum(city_stats$Slope < 0), "\n")

  list(
    resolution = res_label,
    n_neighborhoods = n_neighborhoods,
    n_cities = n_cities,
    n_countries = n_countries,
    delta = delta,
    se_delta = se_delta,
    r2_delta = r2_delta,
    city_stats = city_stats %>% mutate(Resolution = res_label),
    decentered_data = decentered
  )
}

# =============================================================================
# RUN ANALYSIS FOR ALL RESOLUTIONS
# =============================================================================

results <- list()
for (res in c("5", "6", "7")) {
  results[[res]] <- run_decentered_analysis(res, mass_files[[res]])
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

cat("\n\n=== MULTISCALE COMPARISON TABLE (Decentered Methodology) ===\n")

summary_table <- bind_rows(lapply(results, function(r) {
  tibble(
    Resolution = r$resolution,
    `Hex Area (km2)` = hex_areas[r$resolution],
    Neighborhoods = r$n_neighborhoods,
    Cities = r$n_cities,
    Countries = r$n_countries,
    Delta = round(r$delta, 4),
    SE_delta = round(r$se_delta, 4),
    CI_low = round(r$delta - 1.96 * r$se_delta, 4),
    CI_high = round(r$delta + 1.96 * r$se_delta, 4),
    R2_delta = round(r$r2_delta, 4),
    `N cities (slopes)` = nrow(r$city_stats),
    `Mean city slope` = round(mean(r$city_stats$Slope), 4),
    `Median city slope` = round(median(r$city_stats$Slope), 4),
    `Negative slopes` = sum(r$city_stats$Slope < 0)
  )
}))

# Add Figure 2 beta row for context
fig2_row <- tibble(
  Resolution = "Fig2 (city-level)",
  `Hex Area (km2)` = NA_real_,
  Neighborhoods = NA_integer_,
  Cities = as.integer(nrow(filtered_cities_fig2)),
  Countries = as.integer(length(valid_countries_fig2)),
  Delta = round(beta_fig2, 4),
  SE_delta = round(se_beta_fig2, 4),
  CI_low = round(beta_fig2 - 1.96 * se_beta_fig2, 4),
  CI_high = round(beta_fig2 + 1.96 * se_beta_fig2, 4),
  R2_delta = round(r2_beta_fig2, 4),
  `N cities (slopes)` = NA_integer_,
  `Mean city slope` = NA_real_,
  `Median city slope` = NA_real_,
  `Negative slopes` = NA_integer_
)

summary_table_full <- bind_rows(fig2_row, summary_table)
print(as.data.frame(summary_table_full), row.names = FALSE)

write_csv(summary_table_full, file.path(figure_dir, "Table_Decentered_Multiscale_Summary.csv"))
cat("\nSummary table exported.\n")

# =============================================================================
# FIGURE 1: Forest Plot - Delta by Resolution
# =============================================================================

delta_data <- bind_rows(
  tibble(
    Resolution = "Fig 2 (country)",
    res_order = 1,
    Beta = beta_fig2,
    CI_low = beta_fig2 - 1.96 * se_beta_fig2,
    CI_high = beta_fig2 + 1.96 * se_beta_fig2,
    Color = "#fc8d62"
  ),
  bind_rows(lapply(results, function(r) {
    tibble(
      Resolution = paste0("Res ", r$resolution, " (", hex_areas[r$resolution], " km\u00b2)"),
      res_order = as.integer(r$resolution) + 1,
      Beta = r$delta,
      CI_low = r$delta - 1.96 * r$se_delta,
      CI_high = r$delta + 1.96 * r$se_delta,
      Color = "#bf0000"
    )
  }))
)

delta_data$Resolution <- factor(delta_data$Resolution,
                                levels = delta_data$Resolution[order(delta_data$res_order)])

p_forest <- ggplot(delta_data, aes(x = Beta, y = Resolution)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.2, linewidth = 0.8) +
  geom_point(aes(color = Color), size = 3, show.legend = FALSE) +
  scale_color_identity() +
  geom_text(aes(label = sprintf("%.4f [%.4f, %.4f]", Beta, CI_low, CI_high)),
            vjust = -1.2, size = 2.8) +
  labs(
    title = "Scaling Exponent by H3 Resolution (De-centered Method)",
    subtitle = "Orange = country-level beta (Fig 2), Red = city-level delta (Fig 3)",
    x = expression("Scaling exponent"),
    y = ""
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 11)
  )

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_ForestPlot.pdf"),
       p_forest, width = 8, height = 3.5)

# =============================================================================
# FIGURE 2: City-Level Slope Boxplots by Resolution
# =============================================================================

all_city_stats <- bind_rows(lapply(results, function(r) r$city_stats))
all_city_stats$Resolution <- factor(
  paste0("Res ", all_city_stats$Resolution),
  levels = c("Res 5", "Res 6", "Res 7")
)

# Panel B: Slopes
p_slopes <- ggplot(all_city_stats, aes(x = Resolution, y = Slope, fill = Resolution)) +
  geom_jitter(pch = 21, width = 0.2, stroke = 0.3, alpha = 0.15, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "red", alpha = 0.5) +
  scale_fill_manual(values = c("Res 5" = "#66c2a5", "Res 6" = "#fc8d62", "Res 7" = "#8da0cb")) +
  coord_flip() +
  labs(
    title = expression("City-level slope " * delta[i]),
    x = "",
    y = expression("Slope " * delta)
  ) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 10, hjust = 0.5)
  )

# Panel C: M0 distributions
p_M0 <- ggplot(all_city_stats, aes(x = Resolution, y = M0, fill = Resolution)) +
  geom_jitter(pch = 23, width = 0.2, stroke = 0.3, alpha = 0.15, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  scale_fill_manual(values = c("Res 5" = "#66c2a5", "Res 6" = "#fc8d62", "Res 7" = "#8da0cb")) +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  coord_flip() +
  labs(
    title = expression(M(N[ref]) ~ "(tonnes)"),
    x = "",
    y = expression(M[0] ~ "(tonnes)")
  ) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 10, hjust = 0.5)
  )

# Combine
p_box_combined <- cowplot::plot_grid(p_slopes, p_M0, ncol = 1, labels = c("B", "C"))

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_CitySlopes.pdf"),
       p_box_combined, width = 7, height = 5)

# =============================================================================
# FIGURE 3: Scatter Plots (Panel A) - One per Resolution
# =============================================================================

scatter_panels <- list()
for (res in c("5", "6", "7")) {
  r <- results[[res]]
  dec <- r$decentered_data

  scatter_panels[[res]] <- ggplot(dec, aes(x = log_pop_centered, y = log_mass_centered)) +
    geom_point(alpha = 0.03, color = "#8da0cb") +
    geom_abline(slope = r$delta, intercept = 0, color = "#bf0000", lwd = 0.75) +
    geom_abline(slope = beta_fig2, intercept = 0, color = "#fc8d62", lwd = 0.75) +
    geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.5) +
    coord_cartesian(xlim = c(-5, 3), ylim = c(-5, 3)) +
    labs(
      title = paste0("Resolution ", res, " (", hex_areas[res], " km\u00b2)"),
      x = "Log pop (dev. from city mean)",
      y = "Log mass (dev. from city mean)"
    ) +
    annotate("text", x = -4.5, y = 2.5,
             label = sprintf("delta == %.3f", round(r$delta, 3)),
             color = "#bf0000", parse = TRUE, hjust = 0, size = 3) +
    annotate("text", x = -4.5, y = 1.8,
             label = sprintf("beta == %.3f", round(beta_fig2, 3)),
             color = "#fc8d62", parse = TRUE, hjust = 0, size = 3) +
    annotate("text", x = -4.5, y = -4.5,
             label = sprintf("N = %s", format(r$n_neighborhoods, big.mark = ",")),
             hjust = 0, size = 2.5, color = "grey40") +
    theme_bw() +
    theme(
      legend.position = "none",
      panel.grid = element_blank(),
      plot.title = element_text(size = 9)
    )
}

p_scatter_row <- cowplot::plot_grid(
  scatter_panels[["5"]], scatter_panels[["6"]], scatter_panels[["7"]],
  ncol = 3, labels = c("A1", "A2", "A3")
)

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_Scatter.pdf"),
       p_scatter_row, width = 14, height = 4.5)

# =============================================================================
# FIGURE 4: Full Combined Panel
# =============================================================================

full_combined <- cowplot::plot_grid(
  p_scatter_row,
  cowplot::plot_grid(p_forest, p_box_combined, ncol = 2, rel_widths = c(1.2, 1)),
  ncol = 1, rel_heights = c(1, 1.2)
)

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_FullPanel.pdf"),
       full_combined, width = 14, height = 10)

# Export city stats
write_csv(all_city_stats, file.path(figure_dir, "Table_Decentered_Multiscale_CityStats.csv"))

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n========================================\n")
cat("MULTISCALE COMPARISON COMPLETE (Decentered)\n")
cat("========================================\n")
cat(sprintf("Figure 2 beta: %.4f [%.4f, %.4f]\n",
            beta_fig2, beta_fig2 - 1.96 * se_beta_fig2, beta_fig2 + 1.96 * se_beta_fig2))
for (res in c("5", "6", "7")) {
  r <- results[[res]]
  cat(sprintf("Resolution %s: delta=%.4f [%.4f, %.4f], N=%d neighborhoods, %d cities\n",
              res, r$delta,
              r$delta - 1.96 * r$se_delta,
              r$delta + 1.96 * r$se_delta,
              r$n_neighborhoods, r$n_cities))
  cat(sprintf("  City slopes: N=%d, mean=%.4f, median=%.4f, negative=%d\n",
              nrow(r$city_stats), mean(r$city_stats$Slope),
              median(r$city_stats$Slope), sum(r$city_stats$Slope < 0)))
}
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
