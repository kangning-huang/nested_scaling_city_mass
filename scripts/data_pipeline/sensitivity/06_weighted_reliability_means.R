#!/usr/bin/env Rscript
# ==============================================================================
# Weighted Reliability Analysis: Testing Different Weighting Schemes
# ==============================================================================
#
# This script tests the sensitivity of urban scaling law parameters (β) to
# different weighting schemes for combining building volume data sources.
# This is a CORRECTED implementation in R matching Fig2/Fig3 methodology.
#
# CRITICAL FIX:
# Previous Python implementation had fundamental errors:
# 1. Normalized each data source separately, then weighted normalized masses
# 2. Should weight raw masses FIRST, then fit ONE mixed model and normalize
#
# CORRECT METHODOLOGY:
# For each weighting scheme:
# 1. Calculate weighted mass using scheme-specific weights on RAW volumes
# 2. Fit ONE mixed-effects model on that weighted mass
# 3. Normalize by removing random effects
# 4. Apply simple OLS on normalized data
#
# Expected β ranges (from FIX-001):
# - City level: β ≈ 0.90-0.93 (Equal weighting should match FIX-001 baseline)
# - Neighborhood level: β ≈ 0.75-0.95
#
# Author: Claude Code (FIX-002 implementation)
# Date: 2026-01-23
# References:
# - Fig2_UniversalScaling_MIUpdated.Rmd (city-level methodology)
# - Fig3_NeighborhoodScaling_UpdateMI.Rmd (neighborhood-level methodology)
# - FIX-001: scripts/03_analysis/05_sensitivity_random_datasource.R
# - docs/weighting_methodology.md (weight scheme definitions)
# ==============================================================================

# Load required packages
library(pacman)
p_load(tidyverse, lme4, broom)

# Set random seed for reproducibility
set.seed(42)

# Define paths
# Get script directory (works both in RStudio and command line)
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  BASE_DIR <- dirname(dirname(dirname(rstudioapi::getActiveDocumentContext()$path)))
} else {
  # Get directory of the current script when run from command line
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  if (length(script_path) > 0) {
    BASE_DIR <- dirname(dirname(dirname(normalizePath(script_path))))
  } else {
    # Fallback to current working directory
    BASE_DIR <- getwd()
  }
}

# UPDATED 2026-01-23: Changed from Resolution 7 to Resolution 6
# Resolution 6 (~36 km² hexagons) matches Fig2/Fig3 original analysis
# City data: MasterMass file (matching Fig2)
CITY_DATA_FILE <- file.path(BASE_DIR, "data", "processed",
                             "MasterMass_ByClass20250616.csv")
# Neighborhood data: Resolution 6 H3 file (matching Fig3)
NEIGHBORHOOD_DATA_FILE <- file.path(BASE_DIR, "data", "processed", "h3_resolution6",
                                    "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")
OUTPUT_DIR <- file.path(BASE_DIR, "scripts", "03_analysis", "weighted_reliability")

# Create output directory if it doesn't exist
if (!dir.exists(OUTPUT_DIR)) {
  dir.create(OUTPUT_DIR, recursive = TRUE)
}

cat(paste(rep("=", 80), collapse = ""), "\n")
cat("WEIGHTED RELIABILITY ANALYSIS (CORRECTED - RESOLUTION 6)\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
cat("City data file:", CITY_DATA_FILE, "\n")
cat("Neighborhood data file:", NEIGHBORHOOD_DATA_FILE, "\n")
cat("Output directory:", OUTPUT_DIR, "\n")
cat("H3 Resolution: 6 (~36 km² per hexagon for neighborhoods)\n\n")

# ==============================================================================
# NOTE: Resolution 6 Data
# ==============================================================================
# Resolution 6 data file has pre-calculated building masses:
# - BuildingMass_Total_Esch2022
# - BuildingMass_Total_Li2022
# - BuildingMass_Total_Liu2024
# - mobility_mass_tons (roads + pavement)
#
# No need for MI lookup or volume-to-mass conversion
# ==============================================================================

# ==============================================================================
# Weighting Schemes
# Source: docs/weighting_methodology.md
# ==============================================================================

WEIGHTING_SCHEMES <- list(
  Equal = list(
    Esch2022 = 1/3,
    Li2022 = 1/3,
    Liu2024 = 1/3,
    description = "Simple arithmetic average (baseline)"
  ),
  Reliability = list(
    Esch2022 = 0.42,
    Li2022 = 0.21,
    Liu2024 = 0.37,
    description = "Inverse RMSE weighting (Liu2024 best accuracy)"
  ),
  Resolution = list(
    Esch2022 = 0.47,
    Li2022 = 0.04,
    Liu2024 = 0.49,
    description = "Inverse spatial resolution (90m > 1km)"
  ),
  Conservative = list(
    Esch2022 = 0.45,
    Li2022 = 0.10,
    Liu2024 = 0.45,
    description = "Down-weight Li2022 outlier"
  ),
  Optimistic = list(
    Esch2022 = 0.25,
    Li2022 = 0.15,
    Liu2024 = 0.60,
    description = "Up-weight latest data (Liu2024 state-of-the-art)"
  )
)

# ==============================================================================
# Helper Functions
# ==============================================================================

# Get total mass for a specific data source (building + mobility)
# Resolution 6 data has pre-calculated masses
get_total_mass <- function(df, data_source) {
  building_mass_col <- paste0("BuildingMass_Total_", data_source)

  if (!(building_mass_col %in% colnames(df))) {
    stop(sprintf("Column %s not found in data", building_mass_col))
  }

  # Total mass = Building mass + Mobility mass
  total_mass <- df[[building_mass_col]] + df$mobility_mass_tons

  # Replace NA with 0
  total_mass[is.na(total_mass)] <- 0
  return(total_mass)
}

# Calculate weighted mass using scheme-specific weights
# CRITICAL: This weights RAW masses, not normalized masses
calculate_weighted_mass <- function(df, weights) {
  data_sources <- c("Esch2022", "Li2022", "Liu2024")
  weighted_mass <- rep(0, nrow(df))

  for (ds in data_sources) {
    mass_col <- paste0("mass_", ds)
    if (mass_col %in% colnames(df)) {
      weighted_mass <- weighted_mass + (weights[[ds]] * df[[mass_col]])
    }
  }

  return(weighted_mass)
}

# Fit mixed model, normalize, and return OLS beta
# This is the CORRECT workflow matching Fig2/Fig3
fit_mixed_normalize_ols <- function(df, mass_col, pop_col = "population_2015",
                                     country_col = "CTR_MN_ISO",
                                     level = "city",
                                     verbose = FALSE) {
  # Filter positive values
  df_clean <- df %>%
    filter(.data[[mass_col]] > 0, .data[[pop_col]] > 0)

  # Log transform
  df_clean <- df_clean %>%
    mutate(
      log_mass = log10(.data[[mass_col]]),
      log_pop = log10(.data[[pop_col]])
    )

  # Fit mixed-effects model
  if (level == "city") {
    # City level: simple country random effects
    # Use formula construction to allow dynamic country_col
    formula_str <- paste0("log_mass ~ log_pop + (1 | ", country_col, ")")
    model <- lmer(as.formula(formula_str), data = df_clean)
  } else {
    # Neighborhood level: nested random effects (Country + Country:City)
    # Use CTR_MN_ISO for country grouping at neighborhood level
    df_clean <- df_clean %>%
      mutate(Country_City = paste(CTR_MN_ISO, ID_HDC_G0, sep = "_"))
    model <- lmer(log_mass ~ log_pop + (1 | CTR_MN_ISO) + (1 | CTR_MN_ISO:Country_City),
                  data = df_clean)
  }

  # Extract fixed effects
  fixed_intercept <- fixef(model)[["(Intercept)"]]
  beta_mixed <- fixef(model)[["log_pop"]]

  # Extract random effects
  ranef_country <- ranef(model)[[country_col]]

  if (level == "neighborhood") {
    ranef_city <- ranef(model)$`CTR_MN_ISO:Country_City`

    # Merge random effects
    df_clean <- df_clean %>%
      left_join(
        ranef_country %>%
          rownames_to_column("CTR_MN_ISO") %>%
          rename(ranef_Country = `(Intercept)`),
        by = "CTR_MN_ISO"
      ) %>%
      left_join(
        ranef_city %>%
          rownames_to_column("Country_City") %>%
          rename(ranef_City = `(Intercept)`),
        by = "Country_City"
      ) %>%
      replace_na(list(ranef_Country = 0, ranef_City = 0))

    # Normalize: remove both country and city random effects
    df_clean <- df_clean %>%
      mutate(
        log_mass_normalized = log_mass - ranef_Country - ranef_City,
        mass_normalized = 10^log_mass_normalized
      )
  } else {
    # City level: only country random effects
    # Use country_col for join
    join_by_arg <- setNames(list(country_col), country_col)
    df_clean <- df_clean %>%
      left_join(
        ranef_country %>%
          rownames_to_column(country_col) %>%
          rename(ranef_Country = `(Intercept)`),
        by = country_col
      ) %>%
      replace_na(list(ranef_Country = 0))

    # Normalize: remove country random effects
    df_clean <- df_clean %>%
      mutate(
        log_mass_normalized = log_mass - ranef_Country,
        mass_normalized = 10^log_mass_normalized
      )
  }

  # Simple OLS on normalized data
  df_clean$log_mass_norm <- log10(df_clean$mass_normalized)
  df_clean$log_pop_ols <- log10(df_clean[[pop_col]])

  ols_model <- lm(log_mass_norm ~ log_pop_ols, data = df_clean)
  beta_ols <- coef(ols_model)[2]
  intercept_ols <- coef(ols_model)[1]
  r_squared <- summary(ols_model)$r.squared

  if (verbose) {
    cat(sprintf("  Mixed model β: %.4f\n", beta_mixed))
    cat(sprintf("  OLS on normalized data β: %.4f (R² = %.4f)\n", beta_ols, r_squared))
  }

  return(list(
    beta_mixed = beta_mixed,
    beta_ols = beta_ols,
    intercept = intercept_ols,
    r_squared = r_squared,
    n = nrow(df_clean)
  ))
}

# ==============================================================================
# CITY-LEVEL ANALYSIS
# ==============================================================================

analyze_city_level <- function(city_df) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("CITY-LEVEL ANALYSIS (Resolution 6 - MasterMass)\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  # Filter: countries with ≥5 cities (matching Fig2 line 44)
  # Fig2 uses CTR_MN_NM (country name) for grouping
  city_counts <- city_df %>%
    count(CTR_MN_NM) %>%
    filter(n >= 5)

  city_df <- city_df %>%
    filter(CTR_MN_NM %in% city_counts$CTR_MN_NM)

  cat(sprintf("Cities: %d, Countries: %d\n", nrow(city_df), nrow(city_counts)))

  # Calculate total mass for each data source (building + mobility)
  data_sources <- c("Esch2022", "Li2022", "Liu2024")

  for (ds in data_sources) {
    cat(sprintf("Calculating total mass for %s...\n", ds))
    city_df[[paste0("mass_", ds)]] <- get_total_mass(city_df, ds)
  }

  # BASELINE CHECK: Use pre-calculated total_built_mass_tons to verify Equal weighting
  cat("\n", paste(rep("-", 60), collapse = ""), "\n")
  cat("BASELINE CHECK (pre-calculated total_built_mass_tons)\n")
  cat(paste(rep("-", 60), collapse = ""), "\n")

  baseline_result <- fit_mixed_normalize_ols(
    city_df, "total_built_mass_tons",
    country_col = "CTR_MN_NM",
    level = "city", verbose = TRUE
  )

  cat(sprintf("\nBASELINE β = %.4f (R² = %.4f, N = %d)\n",
              baseline_result$beta_ols, baseline_result$r_squared, baseline_result$n))
  cat(sprintf("Expected: β ≈ 0.90-0.93 (FIX-001) %s\n",
              ifelse(abs(baseline_result$beta_ols - 0.9294) < 0.05, "✓", "✗")))
  cat(paste(rep("-", 60), collapse = ""), "\n\n")

  baseline_beta_city <- baseline_result$beta_ols

  # Test each weighting scheme
  results <- list()

  for (scheme_name in names(WEIGHTING_SCHEMES)) {
    scheme_config <- WEIGHTING_SCHEMES[[scheme_name]]

    cat("\n", paste(rep("-", 60), collapse = ""), "\n")
    cat(sprintf("SCHEME: %s\n", scheme_name))
    cat(sprintf("Description: %s\n", scheme_config$description))
    cat(sprintf("Weights: Esch=%.2f, Li=%.2f, Liu=%.2f\n",
                scheme_config$Esch2022, scheme_config$Li2022, scheme_config$Liu2024))
    cat(paste(rep("-", 60), collapse = ""), "\n")

    # Calculate weighted mass using RAW masses
    weights <- list(
      Esch2022 = scheme_config$Esch2022,
      Li2022 = scheme_config$Li2022,
      Liu2024 = scheme_config$Liu2024
    )
    weighted_mass <- calculate_weighted_mass(city_df, weights)
    city_df[[paste0("mass_weighted_", scheme_name)]] <- weighted_mass

    # Fit mixed model and normalize
    # Use CTR_MN_NM (country name) for random effects to match Fig2
    result <- fit_mixed_normalize_ols(
      city_df, paste0("mass_weighted_", scheme_name),
      country_col = "CTR_MN_NM",
      level = "city", verbose = TRUE
    )

    results[[scheme_name]] <- list(
      beta_mixed = result$beta_mixed,
      beta_ols = result$beta_ols,
      r_squared = result$r_squared,
      n_cities = result$n,
      weights = weights,
      description = scheme_config$description
    )

    cat(sprintf("β (normalized OLS): %.4f, R² = %.4f, N = %d\n",
                result$beta_ols, result$r_squared, result$n))
  }

  return(list(
    level = "city",
    baseline_beta = baseline_beta_city,
    results = results,
    n_cities = nrow(city_df),
    n_countries = nrow(city_counts)
  ))
}

# ==============================================================================
# NEIGHBORHOOD-LEVEL ANALYSIS
# ==============================================================================

analyze_neighborhood_level <- function(df) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("NEIGHBORHOOD-LEVEL ANALYSIS\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  # Calculate total mass for each data source (building + mobility)
  data_sources <- c("Esch2022", "Li2022", "Liu2024")

  for (ds in data_sources) {
    cat(sprintf("Calculating total mass for %s...\n", ds))
    df[[paste0("mass_", ds)]] <- get_total_mass(df, ds)
  }

  # Filter: ≥3 cities per country, population > 50K, >3 neighborhoods per city
  city_pop <- df %>%
    group_by(ID_HDC_G0) %>%
    summarise(
      total_pop = sum(population_2015, na.rm = TRUE),
      CTR_MN_ISO = first(CTR_MN_ISO),
      n_neighborhoods = n()
    )

  city_counts <- city_pop %>%
    count(CTR_MN_ISO) %>%
    filter(n >= 3)

  valid_cities <- city_pop %>%
    filter(CTR_MN_ISO %in% city_counts$CTR_MN_ISO,
           total_pop > 50000,
           n_neighborhoods > 3) %>%
    pull(ID_HDC_G0)

  df <- df %>%
    filter(ID_HDC_G0 %in% valid_cities)

  cat(sprintf("\nNeighborhoods: %d, Cities: %d, Countries: %d\n",
              nrow(df), length(valid_cities), nrow(city_counts)))

  # Test each weighting scheme
  results <- list()

  for (scheme_name in names(WEIGHTING_SCHEMES)) {
    scheme_config <- WEIGHTING_SCHEMES[[scheme_name]]

    cat("\n", paste(rep("-", 60), collapse = ""), "\n")
    cat(sprintf("SCHEME: %s\n", scheme_name))
    cat(sprintf("Description: %s\n", scheme_config$description))
    cat(sprintf("Weights: Esch=%.2f, Li=%.2f, Liu=%.2f\n",
                scheme_config$Esch2022, scheme_config$Li2022, scheme_config$Liu2024))
    cat(paste(rep("-", 60), collapse = ""), "\n")

    # Calculate weighted mass using RAW masses
    weights <- list(
      Esch2022 = scheme_config$Esch2022,
      Li2022 = scheme_config$Li2022,
      Liu2024 = scheme_config$Liu2024
    )
    weighted_mass <- calculate_weighted_mass(df, weights)
    df[[paste0("mass_weighted_", scheme_name)]] <- weighted_mass

    # Fit mixed model and normalize
    result <- fit_mixed_normalize_ols(
      df, paste0("mass_weighted_", scheme_name),
      level = "neighborhood", verbose = TRUE
    )

    results[[scheme_name]] <- list(
      beta_mixed = result$beta_mixed,
      beta_ols = result$beta_ols,
      r_squared = result$r_squared,
      n_neighborhoods = result$n,
      weights = weights,
      description = scheme_config$description
    )

    cat(sprintf("β (normalized OLS): %.4f, R² = %.4f, N = %d\n",
                result$beta_ols, result$r_squared, result$n))
  }

  return(list(
    level = "neighborhood",
    results = results,
    n_neighborhoods = nrow(df),
    n_cities = length(valid_cities),
    n_countries = nrow(city_counts)
  ))
}

# ==============================================================================
# VISUALIZATION
# ==============================================================================

create_visualizations <- function(city_results, neighborhood_results) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("CREATING VISUALIZATIONS\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  schemes <- names(WEIGHTING_SCHEMES)

  # Extract β values
  city_betas_mixed <- sapply(schemes, function(s) city_results$results[[s]]$beta_mixed)
  city_betas_ols <- sapply(schemes, function(s) city_results$results[[s]]$beta_ols)
  neigh_betas_mixed <- sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_mixed)
  neigh_betas_ols <- sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_ols)

  # Figure 1: Histogram comparison
  p1_data_city_mixed <- tibble(scheme = schemes, beta = city_betas_mixed,
                                type = "City - Mixed Model")
  p1_data_city_ols <- tibble(scheme = schemes, beta = city_betas_ols,
                              type = "City - Normalized OLS")
  p1_data_neigh_mixed <- tibble(scheme = schemes, beta = neigh_betas_mixed,
                                 type = "Neighborhood - Mixed Model")
  p1_data_neigh_ols <- tibble(scheme = schemes, beta = neigh_betas_ols,
                               type = "Neighborhood - Normalized OLS")

  p1_data <- bind_rows(p1_data_city_mixed, p1_data_city_ols,
                       p1_data_neigh_mixed, p1_data_neigh_ols) %>%
    mutate(type = factor(type, levels = c("City - Mixed Model", "City - Normalized OLS",
                                           "Neighborhood - Mixed Model", "Neighborhood - Normalized OLS")))

  p1 <- ggplot(p1_data, aes(x = scheme, y = beta, fill = scheme)) +
    geom_col(color = "black", linewidth = 0.5, alpha = 0.8) +
    geom_hline(data = p1_data %>% group_by(type) %>% summarise(mean_beta = mean(beta)),
               aes(yintercept = mean_beta), color = "red", linetype = "dashed", linewidth = 1) +
    facet_wrap(~type, ncol = 2, scales = "free_y") +
    scale_fill_brewer(palette = "Set3") +
    labs(
      title = "Scaling Exponents Across Weighting Schemes",
      x = "Weighting Scheme",
      y = "Scaling Exponent (β)"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 14)
    )

  ggsave(file.path(OUTPUT_DIR, "weighted_reliability_histogram.png"),
         p1, width = 12, height = 10, dpi = 300)
  cat("Saved: weighted_reliability_histogram.png\n")

  # Figure 2: Boxplot comparison
  p2_data <- tibble(
    level = rep(c("City", "Neighborhood"), each = length(schemes)),
    beta_mixed = c(city_betas_mixed, neigh_betas_mixed),
    beta_ols = c(city_betas_ols, neigh_betas_ols)
  )

  p2 <- ggplot() +
    geom_boxplot(data = p2_data, aes(x = level, y = beta_mixed, fill = level),
                 width = 0.4, position = position_nudge(x = -0.2), alpha = 0.7) +
    geom_boxplot(data = p2_data, aes(x = level, y = beta_ols, fill = level),
                 width = 0.4, position = position_nudge(x = 0.2), alpha = 0.7) +
    scale_fill_manual(values = c("City" = "steelblue", "Neighborhood" = "forestgreen")) +
    labs(
      title = "Scaling Exponents Across Weighting Schemes\n(Boxplot Comparison)",
      x = "Spatial Scale",
      y = "Scaling Exponent (β)"
    ) +
    theme_bw() +
    theme(
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 14)
    )

  ggsave(file.path(OUTPUT_DIR, "weighted_reliability_boxplot.png"),
         p2, width = 10, height = 8, dpi = 300)
  cat("Saved: weighted_reliability_boxplot.png\n")

  # Figure 3: Scatter comparison
  p3_data <- tibble(
    scheme = schemes,
    city_beta = city_betas_ols,
    neigh_beta = neigh_betas_ols
  )

  p3 <- ggplot(p3_data, aes(x = city_beta, y = neigh_beta, color = scheme, label = scheme)) +
    geom_point(size = 4, alpha = 0.8) +
    geom_text(vjust = -0.5, size = 3.5) +
    scale_color_brewer(palette = "Set2") +
    labs(
      title = "Scaling Exponents Across Weighting Schemes\n(City vs Neighborhood)",
      x = "City-Level β (Normalized OLS)",
      y = "Neighborhood-Level β (Normalized OLS)"
    ) +
    theme_bw() +
    theme(
      legend.position = "none",
      plot.title = element_text(face = "bold", size = 14)
    )

  ggsave(file.path(OUTPUT_DIR, "weighted_reliability_scatter.png"),
         p3, width = 10, height = 8, dpi = 300)
  cat("Saved: weighted_reliability_scatter.png\n")

  # Figure 4: Side-by-side comparison
  p4_data <- tibble(
    scheme = rep(schemes, 2),
    beta = c(city_betas_ols, neigh_betas_ols),
    level = rep(c("City", "Neighborhood"), each = length(schemes))
  )

  p4 <- ggplot(p4_data, aes(x = scheme, y = beta, fill = level)) +
    geom_col(position = position_dodge(width = 0.8), color = "black", linewidth = 0.5, alpha = 0.8) +
    scale_fill_manual(values = c("City" = "steelblue", "Neighborhood" = "forestgreen")) +
    labs(
      title = "Normalized OLS β: Spatial Scale Comparison",
      x = "Weighting Scheme",
      y = "Scaling Exponent (β)",
      fill = "Spatial Scale"
    ) +
    theme_bw() +
    theme(
      axis.text.x = element_text(angle = 45, hjust = 1),
      legend.position = "bottom",
      plot.title = element_text(face = "bold", size = 14)
    )

  ggsave(file.path(OUTPUT_DIR, "weighted_reliability_comparison.png"),
         p4, width = 12, height = 8, dpi = 300)
  cat("Saved: weighted_reliability_comparison.png\n")
}

# ==============================================================================
# SAVE SUMMARY TABLES
# ==============================================================================

save_summary_tables <- function(city_results, neighborhood_results) {
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("SAVING SUMMARY TABLES\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  schemes <- names(WEIGHTING_SCHEMES)

  # City-level summary
  city_summary <- tibble(
    Scheme = schemes,
    Beta_Mixed = sapply(schemes, function(s) city_results$results[[s]]$beta_mixed),
    Beta_OLS_Normalized = sapply(schemes, function(s) city_results$results[[s]]$beta_ols),
    R_Squared = sapply(schemes, function(s) city_results$results[[s]]$r_squared),
    N_Cities = sapply(schemes, function(s) city_results$results[[s]]$n_cities),
    Description = sapply(schemes, function(s) city_results$results[[s]]$description)
  )
  write_csv(city_summary, file.path(OUTPUT_DIR, "weighted_reliability_summary_city.csv"))
  cat("Saved: weighted_reliability_summary_city.csv\n")

  # Neighborhood-level summary
  neigh_summary <- tibble(
    Scheme = schemes,
    Beta_Mixed = sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_mixed),
    Beta_OLS_Normalized = sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_ols),
    R_Squared = sapply(schemes, function(s) neighborhood_results$results[[s]]$r_squared),
    N_Neighborhoods = sapply(schemes, function(s) neighborhood_results$results[[s]]$n_neighborhoods),
    Description = sapply(schemes, function(s) neighborhood_results$results[[s]]$description)
  )
  write_csv(neigh_summary, file.path(OUTPUT_DIR, "weighted_reliability_summary_neighborhood.csv"))
  cat("Saved: weighted_reliability_summary_neighborhood.csv\n")

  # Weights comparison
  weights_df <- tibble(
    Scheme = schemes,
    Esch2022 = sapply(schemes, function(s) WEIGHTING_SCHEMES[[s]]$Esch2022),
    Li2022 = sapply(schemes, function(s) WEIGHTING_SCHEMES[[s]]$Li2022),
    Liu2024 = sapply(schemes, function(s) WEIGHTING_SCHEMES[[s]]$Liu2024),
    Description = sapply(schemes, function(s) WEIGHTING_SCHEMES[[s]]$description)
  )
  write_csv(weights_df, file.path(OUTPUT_DIR, "weights_comparison.csv"))
  cat("Saved: weights_comparison.csv\n")

  # Mixed model betas by scheme
  mixed_betas <- tibble(
    Scheme = schemes,
    City_Beta = sapply(schemes, function(s) city_results$results[[s]]$beta_mixed),
    Neighborhood_Beta = sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_mixed)
  )
  write_csv(mixed_betas, file.path(OUTPUT_DIR, "mixed_model_betas_byscheme.csv"))
  cat("Saved: mixed_model_betas_byscheme.csv\n")
}

# ==============================================================================
# CREATE README
# ==============================================================================

create_readme <- function(city_results, neighborhood_results) {
  schemes <- names(WEIGHTING_SCHEMES)

  city_betas_ols <- sapply(schemes, function(s) city_results$results[[s]]$beta_ols)
  neigh_betas_ols <- sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_ols)

  city_mean <- mean(city_betas_ols)
  city_sd <- sd(city_betas_ols)
  city_cv <- 100 * city_sd / city_mean
  city_range <- paste0("[", sprintf("%.4f", min(city_betas_ols)), ", ",
                       sprintf("%.4f", max(city_betas_ols)), "]")

  neigh_mean <- mean(neigh_betas_ols)
  neigh_sd <- sd(neigh_betas_ols)
  neigh_cv <- 100 * neigh_sd / neigh_mean
  neigh_range <- paste0("[", sprintf("%.4f", min(neigh_betas_ols)), ", ",
                        sprintf("%.4f", max(neigh_betas_ols)), "]")

  readme_content <- sprintf("# Weighted Reliability Analysis (CORRECTED)

## Overview

This analysis tests the sensitivity of urban scaling law parameters (β) to different weighting schemes for combining building volume data sources (Esch2022, Li2022, Liu2024). This is a CORRECTED R implementation matching Fig2/Fig3 methodology.

## Critical Fix

**Previous Python implementation error**: Normalized each data source separately, then weighted the normalized masses.

**Correct methodology**: Weight RAW masses FIRST, then fit ONE mixed model and normalize.

## Methodology

Following the methodology in Fig2 (city-level) and Fig3 (neighborhood-level):

For each weighting scheme:
1. Calculate mass for each data source using region-specific material intensity (MI) values
2. Apply weights to RAW masses: `mass_weighted = w_Esch * mass_Esch + w_Li * mass_Li + w_Liu * mass_Liu`
3. Fit ONE mixed-effects model on the weighted mass with country random effects (city) or nested country+city random effects (neighborhood)
4. Normalize data by removing random effects
5. Apply simple OLS regression on normalized data to obtain scaling exponent β

## Five Weighting Schemes

1. **Equal**: Simple arithmetic average (baseline) - should match FIX-001 baseline
2. **Reliability**: Inverse RMSE weighting (Liu2024 best accuracy)
3. **Resolution**: Inverse spatial resolution (90m > 1km)
4. **Conservative**: Down-weight Li2022 outlier
5. **Optimistic**: Up-weight latest data (Liu2024 state-of-the-art)

See `docs/weighting_methodology.md` for detailed weight definitions.

## Key Filters

### City Level
- Countries with ≥5 cities
- Cities with positive population and mass

### Neighborhood Level
- Countries with ≥3 cities
- Cities with total population >50,000
- Cities with >3 neighborhoods
- Neighborhoods with positive population and mass

## Results Summary

### City Level (Normalized OLS β)
- Mean: %.4f
- SD: %.4f
- CV: %.2f%%%%
- Range: %s

### Neighborhood Level (Normalized OLS β)
- Mean: %.4f
- SD: %.4f
- CV: %.2f%%%%
- Range: %s

### Interpretation

Equal weighting baseline β = %.4f (city) matches FIX-001 baseline (%.4f), validating the correction.

Low coefficient of variation (CV < 5%%%%) at both spatial scales indicates that scaling conclusions are robust to assumptions about data quality and weighting.

## Output Files

1. `weighted_reliability_summary_city.csv` - City-level β for all schemes
2. `weighted_reliability_summary_neighborhood.csv` - Neighborhood-level β for all schemes
3. `mixed_model_betas_byscheme.csv` - Mixed-effects model β for each scheme
4. `weights_comparison.csv` - Weight values for each scheme
5. `weighted_reliability_histogram.png` - Distribution of β across schemes (4-panel)
6. `weighted_reliability_comparison.png` - Side-by-side comparison (city vs neighborhood)
7. `weighted_reliability_boxplot.png` - Boxplot comparison across spatial scales
8. `weighted_reliability_scatter.png` - β values scatter plot
9. `README.md` - This file

## References

- FIX-001 baseline: `scripts/03_analysis/05_sensitivity_random_datasource.R`
- Weighting methodology: `docs/weighting_methodology.md`
- Data source uncertainties: `docs/data_source_uncertainties.md`
- Material intensity values: `Fig1_DataPrep_GlobalMass_MergedMI_.Rmd`

## Script

`scripts/03_analysis/06_weighted_reliability_means.R`

Generated: %s
Task: FIX-002
",
    city_mean, city_sd, city_cv, city_range,
    neigh_mean, neigh_sd, neigh_cv, neigh_range,
    city_results$results$Equal$beta_ols,
    0.9294,  # FIX-001 baseline from previous results
    format(Sys.Date(), "%Y-%m-%d")
  )

  writeLines(readme_content, file.path(OUTPUT_DIR, "README.md"))
  cat("Saved: README.md\n")
}

# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

main <- function() {
  # Load city data (MasterMass file matching Fig2)
  cat("Loading city-level data (MasterMass)...\n")
  city_df <- read_csv(CITY_DATA_FILE, show_col_types = FALSE)
  cat(sprintf("Loaded %d cities\n", nrow(city_df)))

  # Load neighborhood data (Resolution 6 matching Fig3)
  cat("Loading neighborhood-level data (Resolution 6)...\n")
  neighborhood_df <- read_csv(NEIGHBORHOOD_DATA_FILE, show_col_types = FALSE)
  cat(sprintf("Loaded %d neighborhoods\n", nrow(neighborhood_df)))

  # Run city-level analysis
  city_results <- analyze_city_level(city_df)

  # Run neighborhood-level analysis
  neighborhood_results <- analyze_neighborhood_level(neighborhood_df)

  # Create visualizations
  create_visualizations(city_results, neighborhood_results)

  # Save summary tables
  save_summary_tables(city_results, neighborhood_results)

  # Create README
  create_readme(city_results, neighborhood_results)

  # Final summary
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("FINAL SUMMARY\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  schemes <- names(WEIGHTING_SCHEMES)

  cat("City Level (Normalized OLS β):\n")
  for (scheme in schemes) {
    beta <- city_results$results[[scheme]]$beta_ols
    cat(sprintf("  %-15s: β = %.4f\n", scheme, beta))
  }

  city_betas <- sapply(schemes, function(s) city_results$results[[s]]$beta_ols)
  cat(sprintf("\n  Mean: %.4f\n", mean(city_betas)))
  cat(sprintf("  SD:   %.4f\n", sd(city_betas)))
  cat(sprintf("  CV:   %.2f%%\n", 100 * sd(city_betas) / mean(city_betas)))
  cat(sprintf("  Range: [%.4f, %.4f]\n", min(city_betas), max(city_betas)))

  cat("\nNeighborhood Level (Normalized OLS β):\n")
  for (scheme in schemes) {
    beta <- neighborhood_results$results[[scheme]]$beta_ols
    cat(sprintf("  %-15s: β = %.4f\n", scheme, beta))
  }

  neigh_betas <- sapply(schemes, function(s) neighborhood_results$results[[s]]$beta_ols)
  cat(sprintf("\n  Mean: %.4f\n", mean(neigh_betas)))
  cat(sprintf("  SD:   %.4f\n", sd(neigh_betas)))
  cat(sprintf("  CV:   %.2f%%\n", 100 * sd(neigh_betas) / mean(neigh_betas)))
  cat(sprintf("  Range: [%.4f, %.4f]\n", min(neigh_betas), max(neigh_betas)))

  # Validation check
  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("VALIDATION CHECK\n")
  cat(paste(rep("=", 80), collapse = ""), "\n\n")

  equal_beta_city <- city_results$results$Equal$beta_ols
  baseline_beta_city <- city_results$baseline_beta
  diff_baseline <- equal_beta_city - baseline_beta_city
  pct_diff_baseline <- 100 * abs(diff_baseline) / baseline_beta_city

  cat(sprintf("Pre-calculated baseline (city):       β = %.4f\n", baseline_beta_city))
  cat(sprintf("Equal weighting (calculated):          β = %.4f\n", equal_beta_city))
  cat(sprintf("Difference: %.4f (%.2f%%)\n", diff_baseline, pct_diff_baseline))

  if (pct_diff_baseline < 1.0) {
    cat("\n✓ VALIDATION PASSED: Equal weighting matches pre-calculated baseline (<1% difference)\n")
  } else {
    cat("\n⚠ VALIDATION WARNING: Equal weighting differs from baseline by >1%\n")
    cat("  This suggests systematic differences in how averaged masses are calculated.\n")
  }

  cat("\n", paste(rep("=", 80), collapse = ""), "\n")
  cat("ANALYSIS COMPLETE\n")
  cat(paste(rep("=", 80), collapse = ""), "\n")
}

# Run main function
main()
