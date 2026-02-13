# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - MULTISCALE - NO FILTERING
# =============================================================================
# Purpose: Same de-centering approach as the filtered version, but with NO
#          sample filtering, to show the effect of filtering on results.
#
# Differences from filtered version:
#   - No population threshold (only pop >= 1 and mass > 0 for log transform)
#   - No minimum neighborhoods per city
#   - No minimum cities per country
#   - No city total population threshold
#   - City slopes computed for ALL cities with >= 2 neighborhoods
#
# Outputs compared side-by-side with filtered results.
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

mass_files <- list(
  "5" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution5_2026-02-02.csv"),
  "6" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  "7" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
)

hex_areas <- c("5" = 252.9, "6" = 36.13, "7" = 5.161)

N_REF_NEIGHBORHOOD <- 1e4
N_REF_COUNTRY <- 1e6

cat("=== Fig3 Decentered - Multiscale - NO FILTERING ===\n")

# =============================================================================
# FIGURE 2 CITY DATA (resolution-independent, also unfiltered)
# =============================================================================

data_city_fig2 <- read.csv(file.path(data_dir, "MasterMass_ByClass20250616.csv"),
                           stringsAsFactors = FALSE) %>%
  dplyr::filter(total_built_mass_tons > 0) %>%
  mutate(
    log_population = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  )

cat("Figure 2 city data (no country filter):", nrow(data_city_fig2), "cities\n")

# De-center cities by country (no minimum city count per country)
decentered_cities_fig2 <- data_city_fig2 %>%
  group_by(CTR_MN_NM) %>%
  mutate(
    country_mean_log_pop = mean(log_population),
    country_mean_log_mass = mean(log_mass),
    log_pop_centered = log_population - country_mean_log_pop,
    log_mass_centered = log_mass - country_mean_log_mass
  ) %>%
  ungroup()

mod_fig2_nf <- lm(log_mass_centered ~ log_pop_centered, data = decentered_cities_fig2)
beta_fig2_nf <- coef(mod_fig2_nf)[2]
se_beta_fig2_nf <- summary(mod_fig2_nf)$coefficients[2, 2]
r2_beta_fig2_nf <- summary(mod_fig2_nf)$r.squared

cat(sprintf("Figure 2 beta (no filter): %.4f [%.4f, %.4f], R2=%.4f, N=%d cities, %d countries\n",
            beta_fig2_nf,
            beta_fig2_nf - 1.96 * se_beta_fig2_nf,
            beta_fig2_nf + 1.96 * se_beta_fig2_nf,
            r2_beta_fig2_nf,
            nrow(decentered_cities_fig2),
            n_distinct(decentered_cities_fig2$CTR_MN_NM)))

# Country slopes (no minimum)
country_slopes_fig2_nf <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 2) %>%  # need at least 2 to fit a line
  group_modify(~ {
    mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
    tibble(Slope = coef(mod)[2], n_cities = nrow(.x))
  }) %>%
  ungroup()

country_M0_fig2_nf <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 2) %>%
  summarize(
    mean_log_pop = mean(log_population),
    mean_log_mass = mean(log_mass),
    .groups = "drop"
  ) %>%
  left_join(country_slopes_fig2_nf %>% select(CTR_MN_NM, Slope), by = "CTR_MN_NM") %>%
  mutate(
    M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_COUNTRY),
    M0 = 10^M0_log
  )

country_stats_fig2_nf <- country_slopes_fig2_nf %>%
  left_join(country_M0_fig2_nf %>% select(CTR_MN_NM, M0_log, M0), by = "CTR_MN_NM") %>%
  rename(Country = CTR_MN_NM)

cat("Country slopes (no filter):", nrow(country_stats_fig2_nf), "countries\n")

# =============================================================================
# ANALYSIS FUNCTION: No filtering
# =============================================================================

run_decentered_nofilter <- function(res_label, data_file) {
  cat(sprintf("\n======== Resolution %s (NO FILTER) ========\n", res_label))

  # Only require pop >= 1 and mass > 0 (needed for log transform)
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
  cat("Total:", n_neighborhoods, "neighborhoods,", n_cities, "cities,", n_countries, "countries\n")

  # --- De-center neighborhoods by city mean ---
  # Need at least 2 neighborhoods per city to compute a city mean meaningfully
  decentered <- data %>%
    group_by(Country_City) %>%
    filter(n() >= 2) %>%
    mutate(
      city_mean_log_pop = mean(log_population),
      city_mean_log_mass = mean(log_mass_avg),
      log_pop_centered = log_population - city_mean_log_pop,
      log_mass_centered = log_mass_avg - city_mean_log_mass
    ) %>%
    ungroup()

  n_neighborhoods_dc <- nrow(decentered)
  n_cities_dc <- n_distinct(decentered$city_id)
  n_countries_dc <- n_distinct(decentered$country)
  cat("After requiring >= 2 nbhd/city:", n_neighborhoods_dc, "neighborhoods,",
      n_cities_dc, "cities,", n_countries_dc, "countries\n")

  # --- OLS on pooled de-centered data -> delta ---
  mod_delta <- lm(log_mass_centered ~ log_pop_centered, data = decentered)
  delta <- coef(mod_delta)[2]
  se_delta <- summary(mod_delta)$coefficients[2, 2]
  r2_delta <- summary(mod_delta)$r.squared

  cat(sprintf("Delta: %.4f [%.4f, %.4f], R2=%.4f\n",
              delta, delta - 1.96 * se_delta, delta + 1.96 * se_delta, r2_delta))

  # --- Per-city slopes (>= 2 neighborhoods, no other filter) ---
  city_slopes <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 2) %>%
    group_modify(~ {
      mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
      tibble(Slope = coef(mod)[2], n_neighborhoods = nrow(.x))
    }) %>%
    ungroup()

  # --- Per-city M0 ---
  city_M0 <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 2) %>%
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

  cat("Cities with slopes:", nrow(city_stats), "\n")
  cat("  Mean slope:", round(mean(city_stats$Slope), 4), "\n")
  cat("  Median slope:", round(median(city_stats$Slope), 4), "\n")
  cat("  Negative slopes:", sum(city_stats$Slope < 0), "\n")

  list(
    resolution = res_label,
    n_neighborhoods = n_neighborhoods_dc,
    n_cities = n_cities_dc,
    n_countries = n_countries_dc,
    delta = delta,
    se_delta = se_delta,
    r2_delta = r2_delta,
    city_stats = city_stats %>% mutate(Resolution = res_label),
    decentered_data = decentered
  )
}

# =============================================================================
# RUN FOR ALL RESOLUTIONS
# =============================================================================

results_nf <- list()
for (res in c("5", "6", "7")) {
  results_nf[[res]] <- run_decentered_nofilter(res, mass_files[[res]])
}

# =============================================================================
# LOAD FILTERED RESULTS FOR COMPARISON
# =============================================================================

filtered_summary_file <- file.path(figure_dir, "Table_Decentered_Multiscale_Summary.csv")
if (file.exists(filtered_summary_file)) {
  filtered_summary <- read_csv(filtered_summary_file, show_col_types = FALSE)
  cat("\nLoaded filtered results for comparison.\n")
} else {
  cat("\nWARNING: Filtered summary not found. Run Decentered_Multiscale.R first.\n")
  filtered_summary <- NULL
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

cat("\n\n=== MULTISCALE COMPARISON - NO FILTERING ===\n")

summary_nf <- bind_rows(lapply(results_nf, function(r) {
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

fig2_row_nf <- tibble(
  Resolution = "Fig2 (no filter)",
  `Hex Area (km2)` = NA_real_,
  Neighborhoods = NA_integer_,
  Cities = as.integer(nrow(decentered_cities_fig2)),
  Countries = as.integer(n_distinct(decentered_cities_fig2$CTR_MN_NM)),
  Delta = round(beta_fig2_nf, 4),
  SE_delta = round(se_beta_fig2_nf, 4),
  CI_low = round(beta_fig2_nf - 1.96 * se_beta_fig2_nf, 4),
  CI_high = round(beta_fig2_nf + 1.96 * se_beta_fig2_nf, 4),
  R2_delta = round(r2_beta_fig2_nf, 4),
  `N cities (slopes)` = NA_integer_,
  `Mean city slope` = NA_real_,
  `Median city slope` = NA_real_,
  `Negative slopes` = NA_integer_
)

summary_nf_full <- bind_rows(fig2_row_nf, summary_nf)
print(as.data.frame(summary_nf_full), row.names = FALSE)

write_csv(summary_nf_full, file.path(figure_dir, "Table_Decentered_Multiscale_NoFilter_Summary.csv"))

# =============================================================================
# SIDE-BY-SIDE COMPARISON TABLE: Filtered vs Unfiltered
# =============================================================================

if (!is.null(filtered_summary)) {
  cat("\n=== FILTERED vs UNFILTERED COMPARISON ===\n")
  cat(sprintf("%-20s %12s %12s %12s %12s\n",
              "", "Delta(filt)", "Delta(nofilt)", "N(filt)", "N(nofilt)"))
  cat(paste(rep("-", 68), collapse = ""), "\n")

  for (res in c("5", "6", "7")) {
    filt_row <- filtered_summary %>% filter(Resolution == res)
    nf_row <- summary_nf %>% filter(Resolution == res)
    if (nrow(filt_row) > 0 && nrow(nf_row) > 0) {
      cat(sprintf("Resolution %-9s %12s %12s %12s %12s\n",
                  res,
                  sprintf("%.4f", filt_row$Delta),
                  sprintf("%.4f", nf_row$Delta),
                  format(filt_row$Neighborhoods, big.mark = ","),
                  format(nf_row$Neighborhoods, big.mark = ",")))
    }
  }
}

# =============================================================================
# FIGURE 1: Forest Plot - Filtered vs Unfiltered
# =============================================================================

# Build comparison data
delta_comparison <- bind_rows(
  # Filtered results
  bind_rows(lapply(c("5", "6", "7"), function(res) {
    if (!is.null(filtered_summary)) {
      row <- filtered_summary %>% filter(Resolution == res)
      if (nrow(row) > 0) {
        return(tibble(
          Resolution = paste0("Res ", res),
          res_num = as.integer(res),
          Beta = row$Delta,
          CI_low = row$CI_low,
          CI_high = row$CI_high,
          Filter = "Filtered"
        ))
      }
    }
    return(NULL)
  })),
  # Unfiltered results
  bind_rows(lapply(results_nf, function(r) {
    tibble(
      Resolution = paste0("Res ", r$resolution),
      res_num = as.integer(r$resolution),
      Beta = r$delta,
      CI_low = r$delta - 1.96 * r$se_delta,
      CI_high = r$delta + 1.96 * r$se_delta,
      Filter = "No filter"
    )
  }))
)

delta_comparison$Resolution <- factor(delta_comparison$Resolution,
                                      levels = c("Res 5", "Res 6", "Res 7"))
delta_comparison$Filter <- factor(delta_comparison$Filter,
                                  levels = c("Filtered", "No filter"))

p_forest <- ggplot(delta_comparison, aes(x = Beta, y = Resolution, color = Filter)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                 height = 0.2, linewidth = 0.8,
                 position = position_dodge(width = 0.5)) +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  geom_text(aes(label = sprintf("%.4f", Beta)),
            vjust = -1.2, size = 2.8,
            position = position_dodge(width = 0.5), show.legend = FALSE) +
  scale_color_manual(values = c("Filtered" = "#bf0000", "No filter" = "#4393c3")) +
  labs(
    title = "De-centered Delta: Filtered vs Unfiltered",
    x = expression("Scaling exponent " * delta),
    y = "",
    color = ""
  ) +
  theme_bw() +
  theme(
    panel.grid.minor = element_blank(),
    legend.position = "top",
    plot.title = element_text(size = 11)
  )

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_FilterComparison_Forest.pdf"),
       p_forest, width = 8, height = 4)

# =============================================================================
# FIGURE 2: City Slope Boxplots - Filtered vs Unfiltered
# =============================================================================

# Load filtered city stats if available
filtered_city_file <- file.path(figure_dir, "Table_Decentered_Multiscale_CityStats.csv")
if (file.exists(filtered_city_file)) {
  filtered_city_stats <- read_csv(filtered_city_file, show_col_types = FALSE) %>%
    mutate(Filter = "Filtered")
} else {
  filtered_city_stats <- tibble()
}

all_city_nf <- bind_rows(lapply(results_nf, function(r) r$city_stats)) %>%
  mutate(
    Filter = "No filter",
    Resolution = paste0("Res ", Resolution)
  )

if (nrow(filtered_city_stats) > 0) {
  combined_city <- bind_rows(
    filtered_city_stats %>%
      select(Resolution, Slope, Filter),
    all_city_nf %>%
      select(Resolution, Slope, Filter)
  )
} else {
  combined_city <- all_city_nf %>% select(Resolution, Slope, Filter)
}

combined_city$Resolution <- factor(combined_city$Resolution,
                                   levels = c("Res 5", "Res 6", "Res 7"))
combined_city$Filter <- factor(combined_city$Filter,
                               levels = c("Filtered", "No filter"))

p_slopes <- ggplot(combined_city, aes(x = interaction(Filter, Resolution, sep = "\n"),
                                      y = Slope, fill = Filter)) +
  geom_jitter(pch = 21, width = 0.2, stroke = 0.2, alpha = 0.1, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.4, show.legend = FALSE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "red", alpha = 0.5) +
  scale_fill_manual(values = c("Filtered" = "#bf0000", "No filter" = "#4393c3")) +
  coord_flip() +
  labs(
    title = "City-Level Slopes: Filtered vs Unfiltered",
    x = "",
    y = expression("Slope " * delta[i])
  ) +
  theme_bw() +
  theme(
    panel.grid.major.y = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(size = 11)
  )

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_FilterComparison_CitySlopes.pdf"),
       p_slopes, width = 8, height = 5)

# =============================================================================
# FIGURE 3: Combined Panel
# =============================================================================

combined_panel <- cowplot::plot_grid(p_forest, p_slopes, ncol = 1,
                                     labels = c("A", "B"), rel_heights = c(1, 1.2))

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_FilterComparison_Combined.pdf"),
       combined_panel, width = 8, height = 9)

# Export
write_csv(all_city_nf, file.path(figure_dir, "Table_Decentered_Multiscale_NoFilter_CityStats.csv"))

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n========================================\n")
cat("FILTER COMPARISON COMPLETE\n")
cat("========================================\n")
cat(sprintf("%-12s %10s %10s %10s %10s\n",
            "Resolution", "Filt delta", "NF delta", "Filt N", "NF N"))
cat(paste(rep("-", 52), collapse = ""), "\n")

if (!is.null(filtered_summary)) {
  for (res in c("5", "6", "7")) {
    filt <- filtered_summary %>% filter(Resolution == res)
    nf <- summary_nf %>% filter(Resolution == res)
    if (nrow(filt) > 0 && nrow(nf) > 0) {
      cat(sprintf("Res %-8s %10.4f %10.4f %10s %10s\n",
                  res, filt$Delta, nf$Delta,
                  format(filt$Neighborhoods, big.mark = ","),
                  format(nf$Neighborhoods, big.mark = ",")))
    }
  }
}

cat("\nUnfiltered details:\n")
for (res in c("5", "6", "7")) {
  r <- results_nf[[res]]
  cat(sprintf("  Res %s: delta=%.4f [%.4f, %.4f], %d neighborhoods, %d cities, %d countries\n",
              res, r$delta,
              r$delta - 1.96 * r$se_delta,
              r$delta + 1.96 * r$se_delta,
              r$n_neighborhoods, r$n_cities, r$n_countries))
  cat(sprintf("    City slopes: N=%d, mean=%.4f, median=%.4f, negative=%d\n",
              nrow(r$city_stats), mean(r$city_stats$Slope),
              median(r$city_stats$Slope), sum(r$city_stats$Slope < 0)))
}
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
