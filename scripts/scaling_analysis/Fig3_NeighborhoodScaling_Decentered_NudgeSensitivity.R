# =============================================================================
# Fig3 Neighborhood Scaling - De-centered - NUDGE SENSITIVITY ANALYSIS
# =============================================================================
# Purpose: Test whether scaling results are sensitive to the particular
#          positions of H3 hexagons by running the same de-centering analysis
#          on four nudged hexagon grids at two resolutions:
#            - Resolution 6: shifted 1.0 km N/S/E/W
#            - Resolution 7: shifted 0.4 km N/S/E/W
#
# Input: Preprocessed nudged CSV files with mass, population, and city
#        assignments from nudged grids.
#
# Approach:
#   1. Load original Resolution 6 data and run standard filtering + analysis.
#   2. For each R6 nudge direction, apply SAME filtering pipeline and R6 filter
#      set, then run de-centering analysis.
#   3. Load original Resolution 7 data and apply R6 filter set.
#   4. For each R7 nudge direction, apply R6 filter set and run analysis.
#   5. Compare delta (scaling exponent) across nudge directions, resolutions.
#
# Key: If results are robust to hexagon positioning, delta values should
#      be similar across all nudge directions within each resolution.
#
# NOTE: All datasets use total_built_mass_tons (building + road + pavement).
#       R7 nudge road/pavement mass transferred from original R7 (~82% overlap).
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
r7_data_dir <- file.path(project_root, "data", "processed", "h3_resolution7")
figure_dir <- file.path(project_root, "_Revision1", "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

# Nudge directions to analyze
nudge_directions <- c("east", "north", "south", "west")

# Original R6 data for comparison
original_r6_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")

# R6 Nudge data files
nudge_r6_files <- setNames(
  file.path(data_dir, paste0("nudge_", nudge_directions, "_r6_processed_2026-02-06.csv")),
  nudge_directions
)

# Original R7 data
original_r7_file <- file.path(r7_data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")

# R7 Nudge data files
nudge_r7_files <- setNames(
  file.path(r7_data_dir, "nudged",
            paste0("nudge_", nudge_directions, "_r7_processed_2026-02-09.csv")),
  nudge_directions
)

N_REF_NEIGHBORHOOD <- 1e4
N_REF_COUNTRY <- 1e6

cat("=== Fig3 De-centered - Nudge Sensitivity Analysis (R6 + R7) ===\n")
cat("Directions:", paste(nudge_directions, collapse = ", "), "\n\n")

# =============================================================================
# FIGURE 2 CITY DATA (same as other scripts)
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

# =============================================================================
# STEP 1: Run original R6 analysis to establish filter set and baseline
# =============================================================================

cat("\n=== Determining R6 filter set from ORIGINAL data ===\n")

data_r6_orig <- readr::read_csv(original_r6_file, show_col_types = FALSE) %>%
  dplyr::rename(
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  dplyr::filter(population >= 1, mass_avg > 0)

cat("R6 original after pop >= 1, mass > 0:", nrow(data_r6_orig), "\n")

# Cities with total pop > 50,000
city_pop_r6 <- data_r6_orig %>%
  group_by(country_iso, city_id) %>%
  summarize(total_pop = sum(population), .groups = "drop") %>%
  filter(total_pop > 50000)

data_r6_orig <- data_r6_orig %>% filter(city_id %in% city_pop_r6$city_id)

# Cities with >= 10 neighborhoods
city_nbhd_r6 <- data_r6_orig %>%
  group_by(country_iso, city_id) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)

data_r6_orig <- data_r6_orig %>% filter(city_id %in% city_nbhd_r6$city_id)

# Countries with >= 5 qualifying cities
country_city_r6 <- data_r6_orig %>%
  group_by(country_iso) %>%
  summarize(num_cities = n_distinct(city_id)) %>%
  filter(num_cities >= 5)

data_r6_orig <- data_r6_orig %>% filter(country_iso %in% country_city_r6$country_iso)

# Extract the R6 filter set
r6_city_ids <- unique(data_r6_orig$city_id)
r6_country_isos <- unique(data_r6_orig$country_iso)

cat("R6 filter set:", length(r6_city_ids), "cities in", length(r6_country_isos), "countries\n")

# =============================================================================
# SHARED ANALYSIS FUNCTION
# =============================================================================

run_decentered_analysis <- function(data, label) {
  cat(sprintf("\n======== %s ========\n", label))

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

  # De-center by city mean
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

  # OLS -> delta
  mod_delta <- lm(log_mass_centered ~ log_pop_centered, data = decentered)
  delta <- coef(mod_delta)[2]
  se_delta <- summary(mod_delta)$coefficients[2, 2]
  r2_delta <- summary(mod_delta)$r.squared

  cat(sprintf("Delta: %.4f [%.4f, %.4f], R2=%.4f\n",
              delta, delta - 1.96 * se_delta, delta + 1.96 * se_delta, r2_delta))

  # Per-city slopes (>= 10 neighborhoods)
  city_slopes <- decentered %>%
    group_by(Country_City, country, city_id, UC_NM_MN) %>%
    filter(n() >= 10) %>%
    group_modify(~ {
      mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
      tibble(Slope = coef(mod)[2], n_neighborhoods = nrow(.x))
    }) %>%
    ungroup()

  # Per-city M0
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

  list(
    label = label,
    n_neighborhoods = n_neighborhoods_dc,
    n_cities = n_cities_dc,
    n_countries = n_countries_dc,
    delta = delta,
    se_delta = se_delta,
    r2_delta = r2_delta,
    city_stats = city_stats %>% mutate(Source = label),
    decentered_data = decentered
  )
}

# =============================================================================
# STEP 2: Run de-centering on ORIGINAL R6 data (baseline)
# =============================================================================

# Run on original R6 data (baseline, using building + pavement mass only)
result_r6_original <- run_decentered_analysis(data_r6_orig, "Original R6")

# =============================================================================
# STEP 3: Run de-centering on each R6 nudge direction
# =============================================================================

results_r6_nudge <- list()

for (dir in nudge_directions) {
  cat(sprintf("\n--- Loading R6 nudge_%s ---\n", dir))

  nudge_data <- readr::read_csv(nudge_r6_files[[dir]], show_col_types = FALSE) %>%
    dplyr::rename(
      mass_avg = total_built_mass_tons,
      population = population_2015,
      city_id = ID_HDC_G0,
      country_iso = CTR_MN_ISO,
      country = CTR_MN_NM
    ) %>%
    dplyr::filter(population >= 1, mass_avg > 0)

  cat(sprintf("  After pop >= 1, mass > 0: %d rows\n", nrow(nudge_data)))

  # Apply R6 filter set (same cities and countries as original)
  nudge_data <- nudge_data %>%
    dplyr::filter(city_id %in% r6_city_ids,
                  country_iso %in% r6_country_isos)

  cat(sprintf("  After R6 filter: %d rows\n", nrow(nudge_data)))

  label <- paste0("R6 Nudge ", tools::toTitleCase(dir))
  results_r6_nudge[[dir]] <- run_decentered_analysis(nudge_data, label)
}

# Combine R6 results
all_r6_results <- c(list(original = result_r6_original), results_r6_nudge)

# =============================================================================
# STEP 4: Load original R7 data and apply R6 filter set
# =============================================================================

cat("\n\n=== RESOLUTION 7 ANALYSIS ===\n")

data_r7_orig <- readr::read_csv(original_r7_file, show_col_types = FALSE) %>%
  dplyr::rename(
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  dplyr::filter(population >= 1, mass_avg > 0)

cat("R7 original after pop >= 1, mass > 0:", nrow(data_r7_orig), "\n")

# Apply R6 filter set
data_r7_orig <- data_r7_orig %>%
  dplyr::filter(city_id %in% r6_city_ids,
                country_iso %in% r6_country_isos)

cat("R7 original after R6 filter:", nrow(data_r7_orig), "\n")

result_r7_original <- run_decentered_analysis(data_r7_orig, "Original R7")

# =============================================================================
# STEP 5: Run de-centering on each R7 nudge direction
# =============================================================================

results_r7_nudge <- list()

for (dir in nudge_directions) {
  cat(sprintf("\n--- Loading R7 nudge_%s ---\n", dir))

  nudge_data <- readr::read_csv(nudge_r7_files[[dir]], show_col_types = FALSE) %>%
    dplyr::rename(
      mass_avg = total_built_mass_tons,
      population = population_2015,
      city_id = ID_HDC_G0,
      country_iso = CTR_MN_ISO,
      country = CTR_MN_NM
    ) %>%
    dplyr::filter(population >= 1, mass_avg > 0)

  cat(sprintf("  After pop >= 1, mass > 0: %d rows\n", nrow(nudge_data)))

  # Apply R6 filter set
  nudge_data <- nudge_data %>%
    dplyr::filter(city_id %in% r6_city_ids,
                  country_iso %in% r6_country_isos)

  cat(sprintf("  After R6 filter: %d rows\n", nrow(nudge_data)))

  label <- paste0("R7 Nudge ", tools::toTitleCase(dir))
  results_r7_nudge[[dir]] <- run_decentered_analysis(nudge_data, label)
}

# Combine R7 results
all_r7_results <- c(list(original = result_r7_original), results_r7_nudge)

# =============================================================================
# R6 SUMMARY TABLE
# =============================================================================

cat("\n\n=== R6 NUDGE SENSITIVITY - COMPARISON TABLE ===\n")

make_summary_df <- function(results_list) {
  bind_rows(lapply(results_list, function(r) {
    tibble(
      Source = r$label,
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
      `Median city slope` = round(median(r$city_stats$Slope), 4)
    )
  }))
}

summary_r6 <- make_summary_df(all_r6_results)
print(as.data.frame(summary_r6), row.names = FALSE)
write_csv(summary_r6, file.path(figure_dir, "Table_Decentered_NudgeSensitivity_R6_Summary.csv"))

# R6 delta range and max deviation
deltas_r6 <- summary_r6$Delta
delta_range_r6 <- max(deltas_r6) - min(deltas_r6)
delta_orig_r6 <- summary_r6$Delta[summary_r6$Source == "Original R6"]
max_deviation_r6 <- max(abs(deltas_r6[-1] - delta_orig_r6))

cat(sprintf("\nR6 Delta range: %.4f\n", delta_range_r6))
cat(sprintf("R6 Max deviation from original: %.4f\n", max_deviation_r6))
cat(sprintf("R6 Original delta: %.4f\n", delta_orig_r6))

# =============================================================================
# R7 SUMMARY TABLE
# =============================================================================

cat("\n\n=== R7 NUDGE SENSITIVITY - COMPARISON TABLE ===\n")

summary_r7 <- make_summary_df(all_r7_results)
print(as.data.frame(summary_r7), row.names = FALSE)
write_csv(summary_r7, file.path(figure_dir, "Table_Decentered_NudgeSensitivity_R7_Summary.csv"))

# R7 delta range and max deviation
deltas_r7 <- summary_r7$Delta
delta_range_r7 <- max(deltas_r7) - min(deltas_r7)
delta_orig_r7 <- summary_r7$Delta[summary_r7$Source == "Original R7"]
max_deviation_r7 <- max(abs(deltas_r7[-1] - delta_orig_r7))

cat(sprintf("\nR7 Delta range: %.4f\n", delta_range_r7))
cat(sprintf("R7 Max deviation from original: %.4f\n", max_deviation_r7))
cat(sprintf("R7 Original delta: %.4f\n", delta_orig_r7))

# Combined summary (both resolutions)
summary_combined <- bind_rows(
  summary_r6 %>% mutate(Resolution = "R6"),
  summary_r7 %>% mutate(Resolution = "R7")
)
write_csv(summary_combined, file.path(figure_dir, "Table_Decentered_NudgeSensitivity_Combined_Summary.csv"))

# =============================================================================
# HELPER: build forest plot for a set of results
# =============================================================================

make_forest_plot <- function(results_list, resolution_label, nudge_dist,
                             delta_orig_val) {
  fd <- bind_rows(lapply(results_list, function(r) {
    tibble(
      Source = r$label,
      Delta  = r$delta,
      CI_low = r$delta - 1.96 * r$se_delta,
      CI_high = r$delta + 1.96 * r$se_delta
    )
  }))

  fd$Source <- factor(fd$Source, levels = rev(fd$Source))
  fd$Type <- ifelse(grepl("Original", fd$Source), "Original", "Nudged")

  ggplot(fd, aes(x = Delta, y = Source, color = Type)) +
    geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
    geom_vline(xintercept = delta_orig_val, linetype = "dotted",
               color = "#1b9e77", alpha = 0.7) +
    geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                   height = 0.25, linewidth = 0.8) +
    geom_point(size = 3) +
    geom_text(aes(label = sprintf("%.4f", Delta)),
              vjust = -1.2, size = 3, show.legend = FALSE) +
    scale_color_manual(values = c("Original" = "#1b9e77", "Nudged" = "#d95f02")) +
    labs(
      title = paste0("Scaling Exponent Sensitivity (", resolution_label, ")"),
      subtitle = paste0("Hexagons nudged ", nudge_dist, " in each cardinal direction"),
      x = expression("Scaling exponent " * delta),
      y = "",
      color = ""
    ) +
    theme_bw() +
    theme(panel.grid.minor = element_blank(),
          legend.position = "top",
          plot.title = element_text(size = 12),
          plot.subtitle = element_text(size = 9, color = "grey40"))
}

# =============================================================================
# HELPER: build scatter panels for a set of results
# =============================================================================

make_scatter_panels <- function(results_list) {
  panels <- list()
  for (i in seq_along(results_list)) {
    r <- results_list[[i]]
    dec <- r$decentered_data

    point_color <- if (grepl("Original", r$label)) "#8da0cb" else "#fc8d62"
    line_color  <- if (grepl("Original", r$label)) "#1b9e77" else "#d95f02"

    panels[[i]] <- ggplot(dec, aes(x = log_pop_centered, y = log_mass_centered)) +
      geom_point(alpha = 0.03, color = point_color) +
      geom_abline(slope = r$delta, intercept = 0, color = line_color, lwd = 0.75) +
      geom_abline(slope = 1, intercept = 0, linetype = "dashed", lwd = 0.5) +
      coord_cartesian(xlim = c(-5, 3), ylim = c(-5, 3)) +
      labs(
        title = r$label,
        x = "Log pop (dev. from city mean)",
        y = "Log mass (dev. from city mean)"
      ) +
      annotate("text", x = -4.5, y = 2.5,
               label = sprintf("delta == %.3f", round(r$delta, 3)),
               color = line_color, parse = TRUE, hjust = 0, size = 3) +
      annotate("text", x = -4.5, y = 1.8,
               label = sprintf("95%% CI: [%.3f, %.3f]",
                               round(r$delta - 1.96 * r$se_delta, 3),
                               round(r$delta + 1.96 * r$se_delta, 3)),
               color = line_color, hjust = 0, size = 2.5) +
      annotate("text", x = -4.5, y = -4.5,
               label = sprintf("N = %s", format(r$n_neighborhoods, big.mark = ",")),
               hjust = 0, size = 2.5, color = "grey40") +
      theme_bw() +
      theme(legend.position = "none", panel.grid = element_blank(),
            plot.title = element_text(size = 9))
  }
  panels
}

# =============================================================================
# HELPER: build city slope & M0 boxplots
# =============================================================================

make_slope_boxplot <- function(city_stats_df, source_levels, fill_values) {
  city_stats_df$Source <- factor(city_stats_df$Source, levels = source_levels)

  ggplot(city_stats_df, aes(x = Source, y = Slope, fill = Source)) +
    geom_jitter(pch = 21, width = 0.2, stroke = 0.3, alpha = 0.1,
                show.legend = FALSE) +
    geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5,
                 show.legend = FALSE) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "black") +
    geom_hline(yintercept = 0, linetype = "dotted", color = "red", alpha = 0.5) +
    scale_fill_manual(values = fill_values) +
    coord_flip() +
    labs(title = "City-Level Slopes (>= 10 neighborhoods)", x = "",
         y = expression("Slope " * delta[i])) +
    theme_bw() +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_text(size = 11))
}

make_M0_boxplot <- function(city_stats_df, source_levels, fill_values) {
  city_stats_df$Source <- factor(city_stats_df$Source, levels = source_levels)

  ggplot(city_stats_df, aes(x = Source, y = M0, fill = Source)) +
    geom_jitter(pch = 23, width = 0.2, stroke = 0.3, alpha = 0.1,
                show.legend = FALSE) +
    geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5,
                 show.legend = FALSE) +
    scale_fill_manual(values = fill_values) +
    scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                  labels = trans_format("log10", math_format(10^.x))) +
    coord_flip() +
    labs(title = expression(M(N[ref]) ~ "(tonnes)"), x = "",
         y = expression(M[0] ~ "(tonnes)")) +
    theme_bw() +
    theme(panel.grid.major.y = element_blank(), panel.grid.minor = element_blank(),
          plot.title = element_text(size = 10, hjust = 0.5))
}

# =============================================================================
# R6 FIGURES
# =============================================================================

r6_fill <- c("Original R6" = "#1b9e77",
             "R6 Nudge East" = "#d95f02", "R6 Nudge North" = "#7570b3",
             "R6 Nudge South" = "#e7298a", "R6 Nudge West" = "#66a61e")
r6_source_levels <- c("Original R6", "R6 Nudge East", "R6 Nudge North",
                       "R6 Nudge South", "R6 Nudge West")

# Forest plot
p_forest_r6 <- make_forest_plot(all_r6_results, "Resolution 6", "1.0 km",
                                 delta_orig_r6)
ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_R6_Forest.pdf"),
       p_forest_r6, width = 8, height = 4.5)

# Scatter panels
scatter_r6 <- make_scatter_panels(all_r6_results)
if (length(scatter_r6) == 5) {
  p_scatter_r6_top <- cowplot::plot_grid(
    scatter_r6[[1]], scatter_r6[[2]], scatter_r6[[3]],
    ncol = 3, labels = c("A", "B", "C")
  )
  p_scatter_r6_bot <- cowplot::plot_grid(
    scatter_r6[[4]], scatter_r6[[5]],
    ncol = 2, labels = c("D", "E")
  )
  p_scatter_r6 <- cowplot::plot_grid(p_scatter_r6_top, p_scatter_r6_bot,
                                      ncol = 1, rel_heights = c(1, 1))
}

# City stats
all_r6_city_stats <- bind_rows(lapply(all_r6_results, function(r) r$city_stats))

p_slopes_r6 <- make_slope_boxplot(all_r6_city_stats, r6_source_levels, r6_fill)
p_M0_r6 <- make_M0_boxplot(all_r6_city_stats, r6_source_levels, r6_fill)

# Combined R6 figure
p_box_r6 <- cowplot::plot_grid(p_slopes_r6, p_M0_r6, ncol = 1, labels = c("F", "G"))
full_r6 <- cowplot::plot_grid(
  p_scatter_r6,
  cowplot::plot_grid(p_forest_r6, p_box_r6, ncol = 2, rel_widths = c(1.2, 1)),
  ncol = 1, rel_heights = c(1.2, 1)
)
ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_R6_FullPanel.pdf"),
       full_r6, width = 16, height = 14)

write_csv(all_r6_city_stats, file.path(figure_dir, "Table_NudgeSensitivity_R6_CityStats.csv"))

# =============================================================================
# R7 FIGURES
# =============================================================================

r7_fill <- c("Original R7" = "#1b9e77",
             "R7 Nudge East" = "#d95f02", "R7 Nudge North" = "#7570b3",
             "R7 Nudge South" = "#e7298a", "R7 Nudge West" = "#66a61e")
r7_source_levels <- c("Original R7", "R7 Nudge East", "R7 Nudge North",
                       "R7 Nudge South", "R7 Nudge West")

# Forest plot
p_forest_r7 <- make_forest_plot(all_r7_results, "Resolution 7", "0.4 km",
                                 delta_orig_r7)
ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_R7_Forest.pdf"),
       p_forest_r7, width = 8, height = 4.5)

# Scatter panels
scatter_r7 <- make_scatter_panels(all_r7_results)
if (length(scatter_r7) == 5) {
  p_scatter_r7_top <- cowplot::plot_grid(
    scatter_r7[[1]], scatter_r7[[2]], scatter_r7[[3]],
    ncol = 3, labels = c("A", "B", "C")
  )
  p_scatter_r7_bot <- cowplot::plot_grid(
    scatter_r7[[4]], scatter_r7[[5]],
    ncol = 2, labels = c("D", "E")
  )
  p_scatter_r7 <- cowplot::plot_grid(p_scatter_r7_top, p_scatter_r7_bot,
                                      ncol = 1, rel_heights = c(1, 1))
}

# City stats
all_r7_city_stats <- bind_rows(lapply(all_r7_results, function(r) r$city_stats))

p_slopes_r7 <- make_slope_boxplot(all_r7_city_stats, r7_source_levels, r7_fill)
p_M0_r7 <- make_M0_boxplot(all_r7_city_stats, r7_source_levels, r7_fill)

# Combined R7 figure
p_box_r7 <- cowplot::plot_grid(p_slopes_r7, p_M0_r7, ncol = 1, labels = c("F", "G"))
full_r7 <- cowplot::plot_grid(
  p_scatter_r7,
  cowplot::plot_grid(p_forest_r7, p_box_r7, ncol = 2, rel_widths = c(1.2, 1)),
  ncol = 1, rel_heights = c(1.2, 1)
)
ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_R7_FullPanel.pdf"),
       full_r7, width = 16, height = 14)

write_csv(all_r7_city_stats, file.path(figure_dir, "Table_NudgeSensitivity_R7_CityStats.csv"))

# =============================================================================
# CROSS-RESOLUTION FOREST PLOT (R6 + R7 side by side)
# =============================================================================

cross_res_data <- bind_rows(
  bind_rows(lapply(all_r6_results, function(r) {
    tibble(Source = r$label, Delta = r$delta,
           CI_low = r$delta - 1.96 * r$se_delta,
           CI_high = r$delta + 1.96 * r$se_delta,
           Resolution = "R6")
  })),
  bind_rows(lapply(all_r7_results, function(r) {
    tibble(Source = r$label, Delta = r$delta,
           CI_low = r$delta - 1.96 * r$se_delta,
           CI_high = r$delta + 1.96 * r$se_delta,
           Resolution = "R7")
  }))
)

cross_res_data$Type <- ifelse(grepl("Original", cross_res_data$Source),
                               "Original", "Nudged")
cross_res_data$Source <- factor(cross_res_data$Source,
                                 levels = rev(c(r6_source_levels, r7_source_levels)))

p_forest_combined <- ggplot(cross_res_data,
                             aes(x = Delta, y = Source, color = Type, shape = Resolution)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_vline(xintercept = delta_orig_r6, linetype = "dotted",
             color = "#1b9e77", alpha = 0.5) +
  geom_vline(xintercept = delta_orig_r7, linetype = "dotted",
             color = "#4575b4", alpha = 0.5) +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high),
                 height = 0.25, linewidth = 0.7) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.4f", Delta)),
            vjust = -1.2, size = 2.5, show.legend = FALSE) +
  scale_color_manual(values = c("Original" = "#1b9e77", "Nudged" = "#d95f02")) +
  scale_shape_manual(values = c("R6" = 16, "R7" = 17)) +
  labs(
    title = "Nudge Sensitivity Across Resolutions",
    subtitle = "R6: 1.0 km nudge | R7: 0.4 km nudge",
    x = expression("Scaling exponent " * delta),
    y = "", color = "", shape = ""
  ) +
  theme_bw() +
  theme(panel.grid.minor = element_blank(),
        legend.position = "top",
        plot.title = element_text(size = 12),
        plot.subtitle = element_text(size = 9, color = "grey40"))

ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_CrossResolution_Forest.pdf"),
       p_forest_combined, width = 9, height = 7)

# =============================================================================
# BACKWARD COMPATIBILITY: also save R6 figures under old names
# =============================================================================

ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_Forest.pdf"),
       p_forest_r6, width = 8, height = 4.5)
ggsave(file.path(figure_dir, "Fig3_NudgeSensitivity_FullPanel.pdf"),
       full_r6, width = 16, height = 14)
write_csv(summary_r6, file.path(figure_dir, "Table_Decentered_NudgeSensitivity_Summary.csv"))
write_csv(all_r6_city_stats, file.path(figure_dir, "Table_NudgeSensitivity_CityStats.csv"))

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print_summary_block <- function(results_list, res_label, delta_range, max_dev) {
  cat(sprintf("\n--- %s ---\n", res_label))
  cat(sprintf("%-20s %10s %10s %10s %10s %10s\n",
              "Source", "Delta", "CI_low", "CI_high", "N nbhd", "N cities"))
  cat(paste(rep("-", 70), collapse = ""), "\n")
  for (i in seq_along(results_list)) {
    r <- results_list[[i]]
    cat(sprintf("%-20s %10.4f %10.4f %10.4f %10s %10d\n",
                r$label, r$delta,
                r$delta - 1.96 * r$se_delta,
                r$delta + 1.96 * r$se_delta,
                format(r$n_neighborhoods, big.mark = ","),
                r$n_cities))
  }
  cat(sprintf("Delta range: %.4f (max deviation from original: %.4f)\n",
              delta_range, max_dev))

  if (max_dev < 0.01) {
    cat("  -> ROBUST (<0.01 deviation)\n")
  } else if (max_dev < 0.02) {
    cat("  -> MINOR sensitivity (<0.02 deviation)\n")
  } else {
    cat("  -> NOTABLE sensitivity (>=0.02 deviation)\n")
  }
}

cat("\n========================================\n")
cat("NUDGE SENSITIVITY ANALYSIS COMPLETE\n")
cat("========================================\n")
cat("R6 filter set:", length(r6_city_ids), "cities,", length(r6_country_isos), "countries\n")

print_summary_block(all_r6_results, "Resolution 6 (1.0 km nudge)",
                     delta_range_r6, max_deviation_r6)
print_summary_block(all_r7_results, "Resolution 7 (0.4 km nudge)",
                     delta_range_r7, max_deviation_r7)

cat(sprintf("\n--- Cross-Resolution ---\n"))
cat(sprintf("R6 original delta: %.4f\n", delta_orig_r6))
cat(sprintf("R7 original delta: %.4f\n", delta_orig_r7))
cat(sprintf("R6 nudge range: [%.4f, %.4f]\n", min(deltas_r6), max(deltas_r6)))
cat(sprintf("R7 nudge range: [%.4f, %.4f]\n", min(deltas_r7), max(deltas_r7)))

cat("\n========================================\n")
cat("Outputs in:", figure_dir, "\n")
