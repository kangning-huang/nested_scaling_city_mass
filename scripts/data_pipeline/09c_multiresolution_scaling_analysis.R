# Multi-Resolution Scaling Analysis: Comparing H3 Resolutions 5, 6, 7, 8
# This script compares scaling coefficients across different spatial resolutions
# using both multi-source averaging and Esch2022-only building mass
#
# Author: Generated for NYU China Grant project
# Date: 2024-12-30

# Load required libraries
library(pacman)
pacman::p_load(lme4, dplyr, ggplot2, readr, tibble, performance, scales, tidyr, cowplot)

# Define base path (works both in RStudio and command line)
base_path <- tryCatch({
  dirname(rstudioapi::getActiveDocumentContext()$path)
}, error = function(e) {
  if (interactive()) {
    getwd()
  } else {
    args <- commandArgs(trailingOnly = FALSE)
    file_arg <- grep("--file=", args, value = TRUE)
    if (length(file_arg) > 0) {
      dirname(normalizePath(sub("--file=", "", file_arg)))
    } else {
      getwd()
    }
  }
})
if (is.null(base_path) || base_path == "" || base_path == ".") {
  base_path <- getwd()
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

#' Run scaling analysis on a dataset
#' @param data Data frame with required columns
#' @param resolution_name Name for reporting
#' @param mass_column Column to use for mass (default: total_built_mass_tons)
run_scaling_analysis <- function(data, resolution_name, mass_column = "total_built_mass_tons") {
  cat("\n========================================\n")
  cat("Running analysis for:", resolution_name, "\n")
  cat("Mass column:", mass_column, "\n")
  cat("========================================\n")

  # Rename columns to standard names
  data <- data %>%
    dplyr::rename(
      population = population_2015,
      city_id = ID_HDC_G0,
      country_iso = CTR_MN_ISO,
      country = CTR_MN_NM
    )

  # Handle the mass column dynamically
  if (mass_column == "BuildingMass_Total_Esch2022") {
    data <- data %>%
      dplyr::rename(mass_avg = BuildingMass_Total_Esch2022)
  } else {
    data <- data %>%
      dplyr::rename(mass_avg = total_built_mass_tons)
  }

  # Filter valid data
  data <- data %>%
    dplyr::filter(population > 0) %>%
    dplyr::filter(!is.na(mass_avg) & mass_avg > 0)

  cat("Initial rows:", nrow(data), "\n")

  # Filter countries with >= 3 cities
  country_city_counts <- data %>%
    dplyr::group_by(country_iso) %>%
    dplyr::summarize(num_cities = n_distinct(city_id), .groups = "drop") %>%
    dplyr::filter(num_cities >= 3)

  filtered_data <- data %>%
    dplyr::filter(country_iso %in% country_city_counts$country_iso)

  # Filter cities with total population > 50,000
  city_population_totals <- filtered_data %>%
    dplyr::group_by(country_iso, city_id) %>%
    dplyr::summarize(total_population = sum(population), .groups = "drop") %>%
    dplyr::filter(total_population > 50000)

  filtered_data <- filtered_data %>%
    dplyr::filter(city_id %in% city_population_totals$city_id)

  # Filter cities with > 3 neighborhoods
  city_neighborhood_counts <- filtered_data %>%
    dplyr::group_by(country_iso, city_id) %>%
    dplyr::summarize(num_neighborhoods = n(), .groups = "drop") %>%
    dplyr::filter(num_neighborhoods > 3)

  filtered_data <- filtered_data %>%
    dplyr::filter(city_id %in% city_neighborhood_counts$city_id)

  if (nrow(filtered_data) < 100) {
    cat("WARNING: Only", nrow(filtered_data), "rows after filtering - too few for analysis\n")
    return(NULL)
  }

  cat("Filtered rows:", nrow(filtered_data), "\n")
  cat("Number of countries:", n_distinct(filtered_data$country_iso), "\n")
  cat("Number of cities:", n_distinct(filtered_data$city_id), "\n")

  # Log-transform variables
  filtered_data <- filtered_data %>%
    mutate(
      log_population = log10(population),
      log_mass_avg = log10(mass_avg)
    )

  # Create hierarchical factors
  filtered_data$Country <- factor(filtered_data$country)
  filtered_data$Country_City <- factor(with(filtered_data, paste(Country, city_id, sep = "_")))

  # Fit the nested random-effects model
  model <- lmer(log_mass_avg ~ log_population + (1 | Country) + (1 | Country:Country_City),
                data = filtered_data)

  # Calculate R-squared
  r2 <- performance::r2_nakagawa(model)

  # Extract fixed effects
  fixed_intercept <- fixef(model)["(Intercept)"]
  fixed_slope <- fixef(model)["log_population"]

  cat("\nFixed Effects:\n")
  cat("  Intercept:", round(fixed_intercept, 4), "\n")
  cat("  Slope (beta):", round(fixed_slope, 4), "\n")

  # Get confidence intervals for fixed effects
  ci <- confint(model, parm = "beta_", method = "Wald")
  cat("  95% CI for slope:", round(ci["log_population", 1], 4), "to", round(ci["log_population", 2], 4), "\n")

  # Return results
  list(
    model = model,
    fixed_intercept = fixed_intercept,
    fixed_slope = fixed_slope,
    slope_ci = ci["log_population", ],
    r2 = r2,
    n_obs = nrow(filtered_data),
    n_countries = n_distinct(filtered_data$country_iso),
    n_cities = n_distinct(filtered_data$city_id),
    data = filtered_data
  )
}

#' Load data for a given resolution
#' @param resolution H3 resolution level (5, 6, 7, or 8)
load_resolution_data <- function(resolution) {
  # Find the most recent file for this resolution
  res_dir <- file.path(base_path, "..", "data", "processed", paste0("h3_resolution", resolution))

  if (!dir.exists(res_dir)) {
    cat("Directory not found:", res_dir, "\n")
    return(NULL)
  }

  # Find mass files
  mass_files <- list.files(res_dir, pattern = "Fig3_Mass_Neighborhood.*\\.csv$", full.names = TRUE)

  if (length(mass_files) == 0) {
    cat("No mass data files found for resolution", resolution, "\n")
    return(NULL)
  }

  # Use the most recent file
  latest_file <- mass_files[order(file.info(mass_files)$mtime, decreasing = TRUE)[1]]
  cat("Loading:", basename(latest_file), "\n")

  readr::read_csv(latest_file, show_col_types = FALSE)
}

# =============================================================================
# MAIN ANALYSIS
# =============================================================================

cat("\n")
cat("================================================================================\n")
cat("MULTI-RESOLUTION SCALING ANALYSIS\n")
cat("================================================================================\n")

# Define H3 resolution parameters
h3_params <- data.frame(
  resolution = c(5, 6, 7, 8),
  area_km2 = c(252.9, 36.1, 5.2, 0.74),
  edge_km = c(8.5, 3.2, 1.2, 0.46)
)

# Load data for each resolution
cat("\n--- Loading Data ---\n")
data_list <- list()
for (res in h3_params$resolution) {
  data_list[[as.character(res)]] <- load_resolution_data(res)
}

# =============================================================================
# ANALYSIS 1: Multi-source mass (resolutions that have all data sources)
# =============================================================================
cat("\n")
cat("================================================================================\n")
cat("ANALYSIS 1: Multi-Source Building Mass (Esch2022 + Liu2024 + Li2022 average)\n")
cat("================================================================================\n")

results_multisource <- list()
for (res in names(data_list)) {
  if (!is.null(data_list[[res]])) {
    # Check if multi-source columns exist
    if ("total_built_mass_tons" %in% names(data_list[[res]])) {
      area <- h3_params$area_km2[h3_params$resolution == as.integer(res)]
      name <- paste0("Resolution ", res, " (~", area, " km2)")
      results_multisource[[res]] <- run_scaling_analysis(data_list[[res]], name, "total_built_mass_tons")
    }
  }
}

# =============================================================================
# ANALYSIS 2: Esch2022-only mass (all resolutions including 8)
# =============================================================================
cat("\n")
cat("================================================================================\n")
cat("ANALYSIS 2: Esch2022-Only Building Mass (90m resolution data)\n")
cat("================================================================================\n")

results_esch2022 <- list()
for (res in names(data_list)) {
  if (!is.null(data_list[[res]])) {
    # Check if Esch2022 column exists
    if ("BuildingMass_Total_Esch2022" %in% names(data_list[[res]])) {
      area <- h3_params$area_km2[h3_params$resolution == as.integer(res)]
      name <- paste0("Resolution ", res, " (~", area, " km2) - Esch2022 only")
      results_esch2022[[res]] <- run_scaling_analysis(data_list[[res]], name, "BuildingMass_Total_Esch2022")
    }
  }
}

# =============================================================================
# COMPARISON TABLE
# =============================================================================
cat("\n")
cat("================================================================================\n")
cat("COMPARISON OF SCALING COEFFICIENTS\n")
cat("================================================================================\n")

# Build comparison table
build_comparison_table <- function(results, analysis_type) {
  if (length(results) == 0) return(NULL)

  do.call(rbind, lapply(names(results), function(res) {
    r <- results[[res]]
    if (is.null(r)) return(NULL)
    area <- h3_params$area_km2[h3_params$resolution == as.integer(res)]
    data.frame(
      Analysis = analysis_type,
      Resolution = as.integer(res),
      Area_km2 = area,
      Slope_Beta = r$fixed_slope,
      Slope_CI_Lower = r$slope_ci[1],
      Slope_CI_Upper = r$slope_ci[2],
      Intercept = r$fixed_intercept,
      N_Observations = r$n_obs,
      N_Cities = r$n_cities,
      R2_Marginal = r$r2$R2_marginal,
      R2_Conditional = r$r2$R2_conditional
    )
  }))
}

table_multisource <- build_comparison_table(results_multisource, "Multi-source")
table_esch2022 <- build_comparison_table(results_esch2022, "Esch2022-only")

comparison_table <- rbind(table_multisource, table_esch2022)
if (!is.null(comparison_table)) {
  comparison_table <- comparison_table[order(comparison_table$Analysis, comparison_table$Resolution), ]
  print(comparison_table)
}

# =============================================================================
# VISUALIZATION
# =============================================================================
cat("\n--- Creating Visualizations ---\n")

# Prepare data for plotting
if (!is.null(comparison_table) && nrow(comparison_table) > 0) {

  # Create slope comparison plot
  comparison_table$Label <- paste0("Res ", comparison_table$Resolution, "\n(", comparison_table$Area_km2, " km2)")

  p_slope <- ggplot(comparison_table, aes(x = Label, y = Slope_Beta, color = Analysis, shape = Analysis)) +
    geom_point(size = 4, position = position_dodge(width = 0.3)) +
    geom_errorbar(aes(ymin = Slope_CI_Lower, ymax = Slope_CI_Upper),
                  width = 0.2, position = position_dodge(width = 0.3)) +
    geom_hline(yintercept = 1, linetype = "dashed", color = "black", alpha = 0.7) +
    scale_color_manual(values = c("Multi-source" = "#fc8d62", "Esch2022-only" = "#8da0cb")) +
    labs(
      title = "Scaling Exponent Across H3 Resolutions",
      subtitle = "log10(Mass) = beta * log10(Population) + intercept",
      y = "Scaling Exponent (beta)",
      x = "H3 Resolution"
    ) +
    ylim(0.5, 1.1) +
    theme_bw() +
    theme(
      plot.title = element_text(hjust = 0.5, face = "bold"),
      plot.subtitle = element_text(hjust = 0.5),
      legend.position = "bottom"
    )

  # Create scatter plots for each resolution (Esch2022-only for consistency)
  scatter_plots <- list()
  for (res in names(results_esch2022)) {
    r <- results_esch2022[[res]]
    if (!is.null(r)) {
      area <- h3_params$area_km2[h3_params$resolution == as.integer(res)]
      p <- ggplot(r$data, aes(x = 10^log_population, y = 10^log_mass_avg)) +
        geom_point(alpha = 0.03, color = "#8da0cb") +
        geom_smooth(method = "lm", color = "#bf0000", alpha = 0.7, lwd = 0.75) +
        geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
        scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                      labels = trans_format("log10", math_format(10^.x)),
                      limits = c(1, 10^6.5)) +
        scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                      labels = trans_format("log10", math_format(10^.x)),
                      limits = c(10, 10^8)) +
        labs(
          title = paste0("Res ", res, " (beta = ", round(r$fixed_slope, 3), ")"),
          x = "Population",
          y = "Building Mass (tonnes)"
        ) +
        theme_bw() +
        theme(panel.grid.minor = element_blank())

      scatter_plots[[res]] <- p
    }
  }

  # Combine plots
  n_plots <- length(scatter_plots)
  if (n_plots > 0) {
    scatter_grid <- do.call(plot_grid, c(scatter_plots, list(ncol = min(n_plots, 2))))

    combined_plot <- plot_grid(
      scatter_grid,
      p_slope,
      ncol = 1,
      rel_heights = c(1, 0.5),
      labels = c("", LETTERS[n_plots + 1])
    )

    # Save figure
    today_date <- Sys.Date()
    output_filename <- file.path(base_path, "..", "figures",
                                 paste0("Fig3_MultiResolution_Scaling_", today_date, ".pdf"))

    ggsave(output_filename, combined_plot, width = 10, height = 10, units = "in")
    cat("Figure saved to:", output_filename, "\n")
  }
}

# Save comparison table
if (!is.null(comparison_table)) {
  today_date <- Sys.Date()
  output_csv <- file.path(base_path, "..", "data", "processed",
                          paste0("multiresolution_scaling_comparison_", today_date, ".csv"))
  write_csv(comparison_table, output_csv)
  cat("Comparison table saved to:", output_csv, "\n")
}

# =============================================================================
# SUMMARY
# =============================================================================
cat("\n")
cat("================================================================================\n")
cat("SUMMARY\n")
cat("================================================================================\n")

if (!is.null(comparison_table)) {
  # Check if scaling is invariant across resolutions
  esch_only <- comparison_table[comparison_table$Analysis == "Esch2022-only", ]
  if (nrow(esch_only) > 1) {
    slope_range <- range(esch_only$Slope_Beta)
    cat("\nEsch2022-only analysis:\n")
    cat("  Resolutions analyzed:", paste(esch_only$Resolution, collapse = ", "), "\n")
    cat("  Slope range:", round(slope_range[1], 4), "to", round(slope_range[2], 4), "\n")
    cat("  Slope variation:", round(diff(slope_range), 4), "\n")

    # Check CI overlap
    all_overlap <- TRUE
    for (i in 1:(nrow(esch_only)-1)) {
      for (j in (i+1):nrow(esch_only)) {
        overlap <- (esch_only$Slope_CI_Upper[i] >= esch_only$Slope_CI_Lower[j]) &&
                   (esch_only$Slope_CI_Upper[j] >= esch_only$Slope_CI_Lower[i])
        if (!overlap) all_overlap <- FALSE
      }
    }
    cat("  All 95% CIs overlap:", all_overlap, "\n")

    if (all_overlap) {
      cat("\n  => SCALING IS INVARIANT across spatial resolutions!\n")
    } else {
      cat("\n  => Significant differences in scaling across resolutions.\n")
    }
  }
}

cat("\nAnalysis complete.\n")
