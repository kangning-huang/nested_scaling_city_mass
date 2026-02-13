# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - R7 Per-Resolution - Nudge Sensitivity
# =============================================================================
# Tests whether R7 scaling results are sensitive to H3 hexagon positioning
# by running de-centering analysis on four nudged hexagon grids.
#
# Per-resolution filter at R7 (determined from ORIGINAL R7, applied to nudges):
#   - pop >= 1, mass > 0
#   - City total pop > 50,000
#   - >= 10 neighborhoods per city
#   - >= 5 qualifying cities per country
#
# Nudge directions: +0.4 km East, North, South, West
#
# Output: figures/Table_Decentered_R7_PerRes_NudgeSensitivity_Summary.csv
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(readr, dplyr, tibble, ggplot2, scales, cowplot)

# =============================================================================
# SETUP
# =============================================================================

rev_dir <- normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
project_root <- normalizePath(file.path(rev_dir, ".."), mustWork = TRUE)
data_dir <- file.path(rev_dir, "data")
r7_data_dir <- file.path(project_root, "data", "processed", "h3_resolution7")
figure_dir <- file.path(rev_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

nudge_directions <- c("east", "north", "south", "west")

original_r7_file <- file.path(r7_data_dir,
  "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")

nudge_r7_files <- setNames(
  file.path(r7_data_dir, "nudged",
            paste0("nudge_", nudge_directions, "_r7_processed_2026-02-09.csv")),
  nudge_directions
)

cat("=== Fig3 De-centered - R7 Per-Resolution Filter - Nudge Sensitivity ===\n")
cat("Directions:", paste(nudge_directions, collapse = ", "), "\n\n")

# =============================================================================
# STEP 1: Load original R7 data and apply per-resolution filter
# =============================================================================

cat("=== Loading original R7 data and applying per-resolution filter ===\n")

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

# Cities with total pop > 50,000
city_pop <- data_r7_orig %>%
  group_by(country_iso, city_id) %>%
  summarize(total_pop = sum(population), .groups = "drop") %>%
  filter(total_pop > 50000)
data_r7_orig <- data_r7_orig %>% filter(city_id %in% city_pop$city_id)

# Cities with >= 10 neighborhoods
city_nbhd <- data_r7_orig %>%
  group_by(country_iso, city_id) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)
data_r7_orig <- data_r7_orig %>% filter(city_id %in% city_nbhd$city_id)

# Countries with >= 5 qualifying cities
country_city <- data_r7_orig %>%
  group_by(country_iso) %>%
  summarize(num_cities = n_distinct(city_id)) %>%
  filter(num_cities >= 5)
data_r7_orig <- data_r7_orig %>% filter(country_iso %in% country_city$country_iso)

r7_city_ids <- unique(data_r7_orig$city_id)
r7_country_isos <- unique(data_r7_orig$country_iso)

cat("R7 per-resolution filter:", length(r7_city_ids), "cities in",
    length(r7_country_isos), "countries\n\n")

# =============================================================================
# SHARED ANALYSIS FUNCTION
# =============================================================================

run_decentered_analysis <- function(data, label) {
  cat(sprintf("\n--- %s ---\n", label))

  data <- data %>%
    mutate(
      log_population = log10(population),
      log_mass_avg = log10(mass_avg)
    )

  data$Country_City <- factor(paste(data$country, data$city_id, sep = "_"))

  # De-center by city mean
  decentered <- data %>%
    group_by(Country_City) %>%
    filter(n() >= 2) %>%
    mutate(
      log_pop_centered = log_population - mean(log_population),
      log_mass_centered = log_mass_avg - mean(log_mass_avg)
    ) %>%
    ungroup()

  mod <- lm(log_mass_centered ~ log_pop_centered, data = decentered)
  s <- summary(mod)
  delta <- coef(mod)[2]
  se <- s$coefficients[2, 2]

  # Per-city slopes (>= 10 neighborhoods)
  city_slopes <- decentered %>%
    group_by(Country_City) %>%
    filter(n() >= 10) %>%
    group_modify(~ {
      mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
      tibble(Slope = coef(mod)[2], n_neighborhoods = nrow(.x))
    }) %>%
    ungroup()

  cat(sprintf("  Delta: %.4f [%.4f, %.4f], R2=%.4f, N=%d nbhd, %d cities, %d countries\n",
              delta, delta - 1.96 * se, delta + 1.96 * se, s$r.squared,
              nrow(decentered), n_distinct(decentered$city_id),
              n_distinct(decentered$country)))
  cat(sprintf("  Cities with slopes: %d, Mean: %.4f, Median: %.4f\n",
              nrow(city_slopes), mean(city_slopes$Slope), median(city_slopes$Slope)))

  list(
    label = label,
    n_neighborhoods = nrow(decentered),
    n_cities = n_distinct(decentered$city_id),
    n_countries = n_distinct(decentered$country),
    delta = delta, se_delta = se, r2_delta = s$r.squared,
    mean_city_slope = mean(city_slopes$Slope),
    median_city_slope = median(city_slopes$Slope),
    n_cities_slopes = nrow(city_slopes)
  )
}

# =============================================================================
# STEP 2: Run on original R7
# =============================================================================

result_original <- run_decentered_analysis(data_r7_orig, "Original R7")

# =============================================================================
# STEP 3: Run on each nudge direction (apply same R7 filter set)
# =============================================================================

results_nudge <- list()

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

  # Apply R7 per-resolution filter set (from original)
  nudge_data <- nudge_data %>%
    dplyr::filter(city_id %in% r7_city_ids,
                  country_iso %in% r7_country_isos)

  cat(sprintf("  After R7 filter: %d rows\n", nrow(nudge_data)))

  label <- paste0("R7 Nudge ", tools::toTitleCase(dir))
  results_nudge[[dir]] <- run_decentered_analysis(nudge_data, label)
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

all_results <- c(list(original = result_original), results_nudge)

summary_df <- bind_rows(lapply(all_results, function(r) {
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
    `N cities (slopes)` = r$n_cities_slopes,
    `Mean city slope` = round(r$mean_city_slope, 4),
    `Median city slope` = round(r$median_city_slope, 4)
  )
}))

cat("\n\n=== R7 PER-RESOLUTION NUDGE SENSITIVITY - COMPARISON TABLE ===\n")
print(as.data.frame(summary_df), row.names = FALSE)

outfile <- file.path(figure_dir, "Table_Decentered_R7_PerRes_NudgeSensitivity_Summary.csv")
write_csv(summary_df, outfile)
cat("\nSaved:", outfile, "\n")

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================

deltas <- summary_df$Delta
delta_range <- max(deltas) - min(deltas)
delta_orig <- summary_df$Delta[summary_df$Source == "Original R7"]
max_deviation <- max(abs(deltas[-1] - delta_orig))

cat("\n========================================\n")
cat("R7 PER-RESOLUTION NUDGE SENSITIVITY\n")
cat("========================================\n")
cat(sprintf("  Cities: %d, Countries: %d\n", length(r7_city_ids), length(r7_country_isos)))
cat(sprintf("  Original delta: %.4f\n", delta_orig))
cat(sprintf("  Delta range: %.4f\n", delta_range))
cat(sprintf("  Max deviation from original: %.4f\n", max_deviation))

if (max_deviation < 0.01) {
  cat("  -> ROBUST (<0.01 deviation)\n")
} else if (max_deviation < 0.02) {
  cat("  -> MINOR sensitivity (<0.02 deviation)\n")
} else {
  cat("  -> NOTABLE sensitivity (>=0.02 deviation)\n")
}
cat("========================================\n")
