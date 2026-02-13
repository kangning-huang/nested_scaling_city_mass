# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - R7 Per-Resolution - Source Sensitivity
# =============================================================================
# Tests robustness of R7 neighborhood-level scaling exponent (delta) to building
# volume data source choice: Esch2022, Li2022, Liu2024.
#
# Per-resolution filter at R7 (applied independently, not R6 filter):
#   - pop >= 1, mass > 0
#   - City total pop > 50,000
#   - >= 10 neighborhoods per city
#   - >= 5 qualifying cities per country
#
# Analyses:
#   A. Baseline (equal-weighted average of 3 sources)
#   B. Individual sources (Esch2022, Li2022, Liu2024)
#   C. Random source selection (100 iterations)
#
# Output: figures/Table_Decentered_R7_PerRes_Source_Sensitivity_*.csv
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(readr, dplyr, tibble, tidyr, ggplot2, scales, patchwork)

set.seed(42)

# =============================================================================
# SETUP
# =============================================================================

rev_dir <- normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
data_dir <- file.path(rev_dir, "data")
figure_dir <- file.path(rev_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

N_ITERATIONS <- 100

mass_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
sources <- c("Esch2022", "Li2022", "Liu2024")

cat("=== Fig3 De-centered - R7 Per-Resolution Filter - Source Sensitivity ===\n\n")

# =============================================================================
# STEP 1: Load R7 data
# =============================================================================

cat("=== Loading Resolution 7 data ===\n")
df <- readr::read_csv(mass_file, show_col_types = FALSE)
cat("  Loaded:", nrow(df), "neighborhoods\n")

# Compute per-source total mass
for (src in sources) {
  bldg_col <- paste0("BuildingMass_Total_", src)
  total_col <- paste0("total_", src)
  df[[total_col]] <- df[[bldg_col]] + df$mobility_mass_tons
}

# =============================================================================
# STEP 2: Apply per-resolution filter at R7
# =============================================================================

cat("\n=== Applying per-resolution filter at R7 ===\n")

d7 <- df %>%
  filter(population_2015 >= 1, total_built_mass_tons > 0)

# Cities with total pop > 50,000
city_pop <- d7 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(total_pop = sum(population_2015), .groups = "drop") %>%
  filter(total_pop > 50000)
d7 <- d7 %>% filter(ID_HDC_G0 %in% city_pop$ID_HDC_G0)

# Cities with >= 10 neighborhoods
city_nbhd <- d7 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)
d7 <- d7 %>% filter(ID_HDC_G0 %in% city_nbhd$ID_HDC_G0)

# Countries with >= 5 qualifying cities
country_city <- d7 %>%
  group_by(CTR_MN_ISO) %>%
  summarize(num_cities = n_distinct(ID_HDC_G0)) %>%
  filter(num_cities >= 5)
d7 <- d7 %>% filter(CTR_MN_ISO %in% country_city$CTR_MN_ISO)

r7_city_ids <- unique(d7$ID_HDC_G0)
r7_country_isos <- unique(d7$CTR_MN_ISO)
cat("R7 per-resolution filter:", length(r7_city_ids), "cities,",
    length(r7_country_isos), "countries\n\n")

# =============================================================================
# DE-CENTERED OLS FUNCTION
# =============================================================================

run_decentered_nbhd <- function(df, mass_col) {
  d <- df %>%
    filter(population_2015 >= 1, .data[[mass_col]] > 0) %>%
    filter(ID_HDC_G0 %in% r7_city_ids,
           CTR_MN_ISO %in% r7_country_isos) %>%
    mutate(
      log_pop = log10(population_2015),
      log_mass = log10(.data[[mass_col]]),
      Country_City = paste(CTR_MN_NM, ID_HDC_G0, sep = "_")
    )

  dc <- d %>%
    group_by(Country_City) %>%
    filter(n() >= 2) %>%
    mutate(
      log_pop_centered = log_pop - mean(log_pop),
      log_mass_centered = log_mass - mean(log_mass)
    ) %>%
    ungroup()

  mod <- lm(log_mass_centered ~ log_pop_centered, data = dc)
  s <- summary(mod)
  delta <- coef(mod)[2]
  se <- s$coefficients[2, 2]

  list(
    delta = delta, se = se, r2 = s$r.squared,
    ci_low = delta - 1.96 * se, ci_high = delta + 1.96 * se,
    n_neighborhoods = nrow(dc),
    n_cities = n_distinct(dc$ID_HDC_G0),
    n_countries = n_distinct(dc$CTR_MN_NM)
  )
}

# =============================================================================
# STEP 3: Run analyses
# =============================================================================

# A. Baseline
cat("--- Baseline ---\n")
res_baseline <- run_decentered_nbhd(df, "total_built_mass_tons")
cat(sprintf("  delta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d, cities = %d, countries = %d\n",
            res_baseline$delta, res_baseline$ci_low, res_baseline$ci_high,
            res_baseline$r2, res_baseline$n_neighborhoods,
            res_baseline$n_cities, res_baseline$n_countries))

# B. Individual sources
cat("--- Individual sources ---\n")
res_sources <- list()
for (src in sources) {
  total_col <- paste0("total_", src)
  res_sources[[src]] <- run_decentered_nbhd(df, total_col)
  cat(sprintf("  %-12s delta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d\n",
              src, res_sources[[src]]$delta,
              res_sources[[src]]$ci_low, res_sources[[src]]$ci_high,
              res_sources[[src]]$r2, res_sources[[src]]$n_neighborhoods))
}

# C. Random source selection
cat(sprintf("--- Random (%d iterations) ---\n", N_ITERATIONS))

work_df <- df %>%
  filter(population_2015 >= 1, total_built_mass_tons > 0,
         ID_HDC_G0 %in% r7_city_ids, CTR_MN_ISO %in% r7_country_isos) %>%
  mutate(Country_City = paste(CTR_MN_NM, ID_HDC_G0, sep = "_"))

bldg_matrix <- as.matrix(work_df[, paste0("BuildingMass_Total_", sources)])
mobility <- work_df$mobility_mass_tons

# Pre-compute which sources are available (> 0) per neighborhood
available_mask <- bldg_matrix > 0  # logical matrix: N x 3

iter_results <- vector("list", N_ITERATIONS)
for (iter in seq_len(N_ITERATIONS)) {
  # For each neighborhood, randomly pick one source from those with data
  source_idx <- integer(nrow(work_df))
  for (i in seq_len(nrow(work_df))) {
    avail <- which(available_mask[i, ])
    if (length(avail) == 0L) {
      source_idx[i] <- NA_integer_
    } else if (length(avail) == 1L) {
      source_idx[i] <- avail
    } else {
      source_idx[i] <- sample(avail, 1L)
    }
  }
  random_bldg <- bldg_matrix[cbind(seq_len(nrow(work_df)), source_idx)]
  random_bldg[is.na(source_idx)] <- 0
  random_total <- random_bldg + mobility

  tmp <- work_df
  tmp$random_total <- random_total

  d <- tmp %>%
    filter(random_total > 0) %>%
    mutate(log_pop = log10(population_2015), log_mass = log10(random_total))

  dc <- d %>%
    group_by(Country_City) %>%
    filter(n() >= 2) %>%
    mutate(
      log_pop_centered = log_pop - mean(log_pop),
      log_mass_centered = log_mass - mean(log_mass)
    ) %>%
    ungroup()

  mod <- lm(log_mass_centered ~ log_pop_centered, data = dc)
  s <- summary(mod)

  iter_results[[iter]] <- tibble(
    iteration = iter,
    delta = coef(mod)[2],
    se = s$coefficients[2, 2],
    r2 = s$r.squared,
    n_neighborhoods = nrow(dc)
  )
}

iter_df <- bind_rows(iter_results)

cat(sprintf("  Delta: mean = %.4f, sd = %.4f, 95%% range = [%.4f, %.4f]\n",
            mean(iter_df$delta), sd(iter_df$delta),
            quantile(iter_df$delta, 0.025), quantile(iter_df$delta, 0.975)))

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_rows <- list()

summary_rows[[1]] <- tibble(
  Resolution = 7, Hex_km2 = 5.161,
  Approach = "Baseline (Average)", Delta = res_baseline$delta,
  SE = res_baseline$se, CI_low = res_baseline$ci_low,
  CI_high = res_baseline$ci_high, R2 = res_baseline$r2,
  N_neighborhoods = res_baseline$n_neighborhoods,
  N_cities = res_baseline$n_cities, N_countries = res_baseline$n_countries)

for (i in seq_along(sources)) {
  src <- sources[i]
  r <- res_sources[[src]]
  summary_rows[[i + 1]] <- tibble(
    Resolution = 7, Hex_km2 = 5.161,
    Approach = src, Delta = r$delta, SE = r$se,
    CI_low = r$ci_low, CI_high = r$ci_high, R2 = r$r2,
    N_neighborhoods = r$n_neighborhoods,
    N_cities = r$n_cities, N_countries = r$n_countries)
}

summary_rows[[5]] <- tibble(
  Resolution = 7, Hex_km2 = 5.161,
  Approach = "Random (mean)", Delta = mean(iter_df$delta),
  SE = sd(iter_df$delta),
  CI_low = quantile(iter_df$delta, 0.025),
  CI_high = quantile(iter_df$delta, 0.975),
  R2 = mean(iter_df$r2),
  N_neighborhoods = round(mean(iter_df$n_neighborhoods)),
  N_cities = res_baseline$n_cities,
  N_countries = res_baseline$n_countries)

summary_df <- bind_rows(summary_rows)

cat("\n=== SUMMARY TABLE ===\n")
print(as.data.frame(summary_df), row.names = FALSE)

outfile_long <- file.path(figure_dir,
  paste0("Table_Decentered_R7_PerRes_Source_Sensitivity_Long_", Sys.Date(), ".csv"))
write_csv(summary_df, outfile_long)
cat("\nSaved:", outfile_long, "\n")

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================

source_deltas <- sapply(res_sources, function(r) unname(r$delta))
source_range <- diff(range(source_deltas))

cat("\n========================================\n")
cat("R7 PER-RESOLUTION FILTER SOURCE SENSITIVITY\n")
cat("========================================\n")
cat(sprintf("  Cities: %d, Countries: %d\n", length(r7_city_ids), length(r7_country_isos)))
cat(sprintf("  Baseline delta: %.4f\n", res_baseline$delta))
cat(sprintf("  Individual sources: Esch=%.4f, Li=%.4f, Liu=%.4f\n",
            source_deltas[["Esch2022"]], source_deltas[["Li2022"]], source_deltas[["Liu2024"]]))
cat(sprintf("  Source range: %.4f (%.2f%% of mean)\n",
            source_range, source_range / mean(source_deltas) * 100))
cat(sprintf("  Random mean: %.4f, sd: %.4f\n", mean(iter_df$delta), sd(iter_df$delta)))
cat(sprintf("  All sublinear: %s\n",
            ifelse(all(source_deltas < 1) & res_baseline$delta < 1, "YES", "NO")))
cat("========================================\n")
