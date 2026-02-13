# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - MULTISCALE - R6 FILTER APPLIED TO ALL
# =============================================================================
# Purpose: Apply the EXACT same city and country filter set determined from
#          Resolution 6 to Resolutions 5 and 7, so that cross-resolution
#          differences are due ONLY to spatial grain, not sample composition.
#
# Approach:
#   1. Run the standard filtering pipeline on Resolution 6 data to identify
#      the qualifying city_ids and country_isos.
#   2. Apply that same set of city_ids and country_isos to Resolutions 5 and 7.
#   3. Run de-centering analysis identically on all three.
#
# Outputs compared side-by-side with the per-resolution filtered results.
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

cat("=== Fig3 Decentered - Multiscale - R6 FILTER ===\n")

# =============================================================================
# FIGURE 2 CITY DATA (same as standard filtered version)
# =============================================================================

data_city_fig2 <- read.csv(file.path(data_dir, "MasterMass_ByClass20250616.csv"),
                           stringsAsFactors = FALSE) %>%
  dplyr::filter(total_built_mass_tons > 0)

city_counts_fig2 <- table(data_city_fig2$CTR_MN_NM)
valid_countries_fig2 <- names(city_counts_fig2[city_counts_fig2 >= 5])

filtered_cities_fig2 <- data_city_fig2 %>%
  dplyr::filter(CTR_MN_NM %in% valid_countries_fig2) %>%
  mutate(
    log_population = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  )

# De-center and compute beta
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

cat(sprintf("Figure 2 beta: %.4f [%.4f, %.4f]\n",
            beta_fig2, beta_fig2 - 1.96 * se_beta_fig2, beta_fig2 + 1.96 * se_beta_fig2))

# Country slopes from Figure 2
country_slopes_fig2 <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  group_modify(~ {
    mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
    tibble(Slope = coef(mod)[2], n_cities = nrow(.x))
  }) %>%
  ungroup()

country_M0_fig2 <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  summarize(mean_log_pop = mean(log_population),
            mean_log_mass = mean(log_mass), .groups = "drop") %>%
  left_join(country_slopes_fig2 %>% select(CTR_MN_NM, Slope), by = "CTR_MN_NM") %>%
  mutate(M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_COUNTRY),
         M0 = 10^M0_log)

country_stats_fig2 <- country_slopes_fig2 %>%
  left_join(country_M0_fig2 %>% select(CTR_MN_NM, M0_log, M0), by = "CTR_MN_NM") %>%
  rename(Country = CTR_MN_NM)

# =============================================================================
# STEP 1: Determine the R6 filter set
# =============================================================================

cat("\n=== Determining R6 filter set ===\n")

data_r6 <- readr::read_csv(mass_files[["6"]], show_col_types = FALSE) %>%
  dplyr::rename(
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  dplyr::filter(population >= 1, mass_avg > 0)

cat("R6 after pop >= 1, mass > 0:", nrow(data_r6), "\n")

# Cities with total pop > 50,000
city_pop_r6 <- data_r6 %>%
  group_by(country_iso, city_id) %>%
  summarize(total_pop = sum(population), .groups = "drop") %>%
  filter(total_pop > 50000)

data_r6 <- data_r6 %>% filter(city_id %in% city_pop_r6$city_id)

# Cities with >= 10 neighborhoods
city_nbhd_r6 <- data_r6 %>%
  group_by(country_iso, city_id) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)

data_r6 <- data_r6 %>% filter(city_id %in% city_nbhd_r6$city_id)

# Countries with >= 5 qualifying cities
country_city_r6 <- data_r6 %>%
  group_by(country_iso) %>%
  summarize(num_cities = n_distinct(city_id)) %>%
  filter(num_cities >= 5)

data_r6 <- data_r6 %>% filter(country_iso %in% country_city_r6$country_iso)

# Extract the R6 filter set
r6_city_ids <- unique(data_r6$city_id)
r6_country_isos <- unique(data_r6$country_iso)

cat("R6 filter set:", length(r6_city_ids), "cities in", length(r6_country_isos), "countries\n")
cat("R6 neighborhoods:", nrow(data_r6), "\n")

# =============================================================================
# STEP 2: Apply R6 filter to all resolutions and run de-centering
# =============================================================================

run_decentered_r6filter <- function(res_label, data_file,
                                    filter_city_ids, filter_country_isos) {
  cat(sprintf("\n======== Resolution %s (R6 FILTER) ========\n", res_label))

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
    dplyr::filter(population >= 1, mass_avg > 0)

  cat("After pop >= 1, mass > 0:", nrow(data), "\n")

  # Apply the R6 city and country filter
  data <- data %>%
    dplyr::filter(city_id %in% filter_city_ids,
                  country_iso %in% filter_country_isos)

  cat("After R6 filter:", nrow(data), "neighborhoods\n")

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
  cat("Sample:", n_neighborhoods, "neighborhoods,", n_cities, "cities,", n_countries, "countries\n")

  # --- De-center by city mean ---
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

  # --- OLS -> delta ---
  mod_delta <- lm(log_mass_centered ~ log_pop_centered, data = decentered)
  delta <- coef(mod_delta)[2]
  se_delta <- summary(mod_delta)$coefficients[2, 2]
  r2_delta <- summary(mod_delta)$r.squared

  cat(sprintf("Delta: %.4f [%.4f, %.4f], R2=%.4f\n",
              delta, delta - 1.96 * se_delta, delta + 1.96 * se_delta, r2_delta))

  # --- Per-city slopes (>= 10 neighborhoods, matching R6 threshold) ---
  city_slopes <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 10) %>%
    group_modify(~ {
      mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
      tibble(Slope = coef(mod)[2], n_neighborhoods = nrow(.x))
    }) %>%
    ungroup()

  # --- Per-city M0 ---
  city_M0 <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 10) %>%
    summarize(mean_log_pop = mean(log_population),
              mean_log_mass = mean(log_mass_avg), .groups = "drop") %>%
    left_join(city_slopes %>% select(Country_City, Slope), by = "Country_City") %>%
    mutate(M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_NEIGHBORHOOD),
           M0 = 10^M0_log)

  city_stats <- city_slopes %>%
    left_join(city_M0 %>% select(Country_City, M0_log, M0, mean_log_pop, mean_log_mass),
              by = "Country_City")

  cat("Cities with slopes (>= 10 nbhd):", nrow(city_stats), "\n")
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

results <- list()
for (res in c("5", "6", "7")) {
  results[[res]] <- run_decentered_r6filter(res, mass_files[[res]],
                                             r6_city_ids, r6_country_isos)
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

cat("\n\n=== MULTISCALE COMPARISON - R6 FILTER APPLIED TO ALL ===\n")

summary_r6f <- bind_rows(lapply(results, function(r) {
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

print(as.data.frame(summary_r6f), row.names = FALSE)
write_csv(summary_r6f, file.path(figure_dir, "Table_Decentered_Multiscale_R6Filter_Summary.csv"))

# =============================================================================
# LOAD PER-RESOLUTION FILTERED RESULTS FOR COMPARISON
# =============================================================================

filtered_summary_file <- file.path(figure_dir, "Table_Decentered_Multiscale_Summary.csv")
filtered_summary <- NULL
if (file.exists(filtered_summary_file)) {
  filtered_summary <- read_csv(filtered_summary_file, show_col_types = FALSE) %>%
    filter(!grepl("Fig2", Resolution))
}

# =============================================================================
# SIDE-BY-SIDE COMPARISON
# =============================================================================

cat("\n=== R6-FILTER vs PER-RESOLUTION FILTER ===\n")
cat(sprintf("%-12s %12s %12s %12s %12s\n",
            "", "Delta(R6f)", "Delta(own)", "N(R6f)", "N(own)"))
cat(paste(rep("-", 60), collapse = ""), "\n")

if (!is.null(filtered_summary)) {
  for (res in c("5", "6", "7")) {
    own <- filtered_summary %>% filter(Resolution == res)
    r6f <- summary_r6f %>% filter(Resolution == res)
    if (nrow(own) > 0 && nrow(r6f) > 0) {
      cat(sprintf("Res %-8s %12.4f %12.4f %12s %12s\n",
                  res,
                  r6f$Delta, own$Delta,
                  format(r6f$Neighborhoods, big.mark = ","),
                  format(own$Neighborhoods, big.mark = ",")))
    }
  }
}

# =============================================================================
# FIGURE 1: Forest Plot - R6 Filter vs Per-Resolution Filter
# =============================================================================

delta_comparison <- bind_rows(
  # R6 filter results
  bind_rows(lapply(results, function(r) {
    tibble(
      Resolution = paste0("Res ", r$resolution),
      res_num = as.integer(r$resolution),
      Beta = r$delta,
      CI_low = r$delta - 1.96 * r$se_delta,
      CI_high = r$delta + 1.96 * r$se_delta,
      Filter = "R6 filter"
    )
  })),
  # Per-resolution filter results
  if (!is.null(filtered_summary)) {
    filtered_summary %>%
      mutate(
        Resolution = paste0("Res ", Resolution),
        res_num = as.integer(gsub("Res ", "", Resolution)),
        Beta = Delta,
        Filter = "Own filter"
      ) %>%
      select(Resolution, res_num, Beta, CI_low, CI_high, Filter)
  }
)

delta_comparison$Resolution <- factor(delta_comparison$Resolution,
                                      levels = c("Res 5", "Res 6", "Res 7"))
delta_comparison$Filter <- factor(delta_comparison$Filter,
                                  levels = c("Own filter", "R6 filter"))

p_forest <- ggplot(delta_comparison, aes(x = Beta, y = Resolution, color = Filter)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                 height = 0.2, linewidth = 0.8,
                 position = position_dodge(width = 0.5)) +
  geom_point(size = 3, position = position_dodge(width = 0.5)) +
  geom_text(aes(label = sprintf("%.4f", Beta)),
            vjust = -1.2, size = 2.8,
            position = position_dodge(width = 0.5), show.legend = FALSE) +
  scale_color_manual(values = c("Own filter" = "#bf0000", "R6 filter" = "#1b9e77")) +
  labs(
    title = "De-centered Delta: Per-Resolution Filter vs R6 Filter",
    subtitle = "R6 filter = same cities & countries across all resolutions",
    x = expression("Scaling exponent " * delta),
    y = "",
    color = ""
  ) +
  theme_bw() +
  theme(panel.grid.minor = element_blank(), legend.position = "top",
        plot.title = element_text(size = 11))

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_R6Filter_Forest.pdf"),
       p_forest, width = 8, height = 4)

# =============================================================================
# FIGURE 2: City Slope Boxplots
# =============================================================================

all_city_r6f <- bind_rows(lapply(results, function(r) r$city_stats)) %>%
  mutate(Resolution = paste0("Res ", Resolution))

all_city_r6f$Resolution <- factor(all_city_r6f$Resolution,
                                  levels = c("Res 5", "Res 6", "Res 7"))

p_slopes <- ggplot(all_city_r6f, aes(x = Resolution, y = Slope, fill = Resolution)) +
  geom_jitter(pch = 21, width = 0.2, stroke = 0.3, alpha = 0.15, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
  geom_hline(yintercept = 0, linetype = "dotted", color = "red", alpha = 0.5) +
  scale_fill_manual(values = c("Res 5" = "#66c2a5", "Res 6" = "#fc8d62", "Res 7" = "#8da0cb")) +
  coord_flip() +
  labs(
    title = "City-Level Slopes (R6 Filter, >= 10 neighborhoods)",
    x = "",
    y = expression("Slope " * delta[i])
  ) +
  theme_bw() +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        plot.title = element_text(size = 11))

# M0 boxplot
p_M0 <- ggplot(all_city_r6f, aes(x = Resolution, y = M0, fill = Resolution)) +
  geom_jitter(pch = 23, width = 0.2, stroke = 0.3, alpha = 0.15, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  scale_fill_manual(values = c("Res 5" = "#66c2a5", "Res 6" = "#fc8d62", "Res 7" = "#8da0cb")) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x))) +
  coord_flip() +
  labs(
    title = expression(M(N[ref]) ~ "(tonnes, R6 filter)"),
    x = "",
    y = expression(M[0] ~ "(tonnes)")
  ) +
  theme_bw() +
  theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
        plot.title = element_text(size = 10, hjust = 0.5))

p_box <- cowplot::plot_grid(p_slopes, p_M0, ncol = 1, labels = c("B", "C"))

# =============================================================================
# FIGURE 3: Scatter Panels
# =============================================================================

scatter_panels <- list()
for (res in c("5", "6", "7")) {
  r <- results[[res]]
  dec <- r$decentered_data

  scatter_panels[[res]] <- ggplot(dec, aes(x = log_pop_centered, y = log_mass_centered)) +
    geom_point(alpha = 0.03, color = "#8da0cb") +
    geom_abline(slope = r$delta, intercept = 0, color = "#bf0000", lwd = 0.75) +
    geom_abline(slope = beta_fig2, intercept = 0, color = "#fc8d62", lwd = 0.75) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", lwd = 0.5) +
    coord_cartesian(xlim = c(-5, 3), ylim = c(-5, 3)) +
    labs(
      title = paste0("Res ", res, " (", hex_areas[res], " km\u00b2)"),
      x = "Log pop (dev. from city mean)",
      y = "Log mass (dev. from city mean)"
    ) +
    annotate("text", x = -4.5, y = 2.5,
             label = sprintf("delta == %.3f", round(r$delta, 3)),
             color = "#bf0000", parse = TRUE, hjust = 0, size = 3) +
    annotate("text", x = -4.5, y = 1.8,
             label = sprintf("95%% CI: [%.3f, %.3f]",
                             round(r$delta - 1.96 * r$se_delta, 3),
                             round(r$delta + 1.96 * r$se_delta, 3)),
             color = "#bf0000", hjust = 0, size = 2.5) +
    annotate("text", x = -4.5, y = 1.2,
             label = sprintf("beta == %.3f", round(beta_fig2, 3)),
             color = "#fc8d62", parse = TRUE, hjust = 0, size = 3) +
    annotate("text", x = -4.5, y = -4.5,
             label = sprintf("N = %s", format(r$n_neighborhoods, big.mark = ",")),
             hjust = 0, size = 2.5, color = "grey40") +
    theme_bw() +
    theme(legend.position = "none", panel.grid = element_blank(),
          plot.title = element_text(size = 9))
}

p_scatter <- cowplot::plot_grid(scatter_panels[["5"]], scatter_panels[["6"]],
                                scatter_panels[["7"]], ncol = 3,
                                labels = c("A1", "A2", "A3"))

# =============================================================================
# FIGURE 4: Full Combined
# =============================================================================

full_combined <- cowplot::plot_grid(
  p_scatter,
  cowplot::plot_grid(p_forest, p_box, ncol = 2, rel_widths = c(1.2, 1)),
  ncol = 1, rel_heights = c(1, 1.2)
)

ggsave(file.path(figure_dir, "Fig3_Decentered_Multiscale_R6Filter_FullPanel.pdf"),
       full_combined, width = 14, height = 10)

# Export city stats
write_csv(all_city_r6f, file.path(figure_dir, "Table_Decentered_Multiscale_R6Filter_CityStats.csv"))

# =============================================================================
# FINAL SUMMARY
# =============================================================================

cat("\n========================================\n")
cat("R6 FILTER COMPARISON COMPLETE\n")
cat("========================================\n")
cat("R6 filter set:", length(r6_city_ids), "cities,", length(r6_country_isos), "countries\n\n")

cat(sprintf("%-12s %10s %10s %10s %10s %10s\n",
            "Resolution", "Delta", "CI_low", "CI_high", "N nbhd", "N cities"))
cat(paste(rep("-", 62), collapse = ""), "\n")
for (res in c("5", "6", "7")) {
  r <- results[[res]]
  cat(sprintf("Res %-8s %10.4f %10.4f %10.4f %10s %10d\n",
              res, r$delta,
              r$delta - 1.96 * r$se_delta,
              r$delta + 1.96 * r$se_delta,
              format(r$n_neighborhoods, big.mark = ","),
              r$n_cities))
}

if (!is.null(filtered_summary)) {
  cat("\nPer-resolution filter (for comparison):\n")
  cat(sprintf("%-12s %10s %10s %10s %10s %10s\n",
              "Resolution", "Delta", "CI_low", "CI_high", "N nbhd", "N cities"))
  cat(paste(rep("-", 62), collapse = ""), "\n")
  for (i in 1:nrow(filtered_summary)) {
    r <- filtered_summary[i, ]
    cat(sprintf("Res %-8s %10.4f %10.4f %10.4f %10s %10d\n",
                r$Resolution, r$Delta, r$CI_low, r$CI_high,
                format(r$Neighborhoods, big.mark = ","), r$Cities))
  }
}
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
