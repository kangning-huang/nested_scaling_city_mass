# Compare Scaling Coefficients Between H3 Resolution 6 and Resolution 7
# This script runs the nested mixed-effects model on both resolutions
# and compares the scaling coefficients

# Load required libraries
library(pacman)
pacman::p_load(lme4, dplyr, ggplot2, readr, tibble, performance, scales, tidyr, cowplot)

# Define base path (works both in RStudio and command line)
base_path <- tryCatch({
  dirname(rstudioapi::getActiveDocumentContext()$path)
}, error = function(e) {
  # If not in RStudio, use the script's location or current directory
  if (interactive()) {
    getwd()
  } else {
    # When run via Rscript, get the script directory
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

# Function to run scaling analysis on a dataset
run_scaling_analysis <- function(data, resolution_name) {
  cat("\n========================================\n")
  cat("Running analysis for:", resolution_name, "\n")
  cat("========================================\n")

  # Rename columns to match expected names
  data <- data %>%
    dplyr::rename(
      mass_building = BuildingMass_AverageTotal,
      mass_mobility = mobility_mass_tons,
      mass_avg = total_built_mass_tons,
      population = population_2015,
      city_id = ID_HDC_G0,
      country_iso = CTR_MN_ISO,
      country = CTR_MN_NM
    ) %>%
    dplyr::filter(population > 0) %>%
    dplyr::filter(mass_avg > 0)

  cat("Initial rows:", nrow(data), "\n")

  # Filter countries with >= 3 cities
  country_city_counts <- data %>%
    dplyr::group_by(country_iso) %>%
    dplyr::summarize(num_cities = n_distinct(city_id)) %>%
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

  # Print model summary
  cat("\nModel Summary:\n")
  print(summary(model))

  # Calculate R-squared
  r2 <- performance::r2_nakagawa(model)
  cat("\nR-squared (Nakagawa):\n")
  print(r2)

  # Extract fixed effects
  fixed_intercept <- fixef(model)["(Intercept)"]
  fixed_slope <- fixef(model)["log_population"]

  cat("\nFixed Effects:\n")
  cat("  Intercept:", fixed_intercept, "\n")
  cat("  Slope (β):", fixed_slope, "\n")

  # Get confidence intervals for fixed effects
  ci <- confint(model, parm = "beta_", method = "Wald")
  cat("\n95% CI for slope:", ci["log_population", 1], "to", ci["log_population", 2], "\n")

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

# ========================================
# Load Resolution 6 Data
# ========================================
cat("\nLoading Resolution 6 data...\n")
data_res6 <- readr::read_csv(
  file.path(base_path, "..", "data", "processed", "h3_resolution6",
            "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  show_col_types = FALSE
)

# ========================================
# Load Resolution 7 Data
# ========================================
cat("\nLoading Resolution 7 data...\n")
data_res7 <- readr::read_csv(
  file.path(base_path, "..", "data", "processed", "h3_resolution7",
            "Fig3_Mass_Neighborhood_H3_Resolution7_2025-12-29.csv"),
  show_col_types = FALSE
)

# ========================================
# Run analyses
# ========================================
results_res6 <- run_scaling_analysis(data_res6, "Resolution 6 (~36 km²)")
results_res7 <- run_scaling_analysis(data_res7, "Resolution 7 (~5.2 km²)")

# ========================================
# Compare Results
# ========================================
cat("\n\n========================================\n")
cat("COMPARISON OF SCALING COEFFICIENTS\n")
cat("========================================\n\n")

comparison_table <- data.frame(
  Resolution = c("H3 Resolution 6 (~36 km²)", "H3 Resolution 7 (~5.2 km²)"),
  Slope_Beta = c(results_res6$fixed_slope, results_res7$fixed_slope),
  Intercept = c(results_res6$fixed_intercept, results_res7$fixed_intercept),
  Slope_CI_Lower = c(results_res6$slope_ci[1], results_res7$slope_ci[1]),
  Slope_CI_Upper = c(results_res6$slope_ci[2], results_res7$slope_ci[2]),
  N_Observations = c(results_res6$n_obs, results_res7$n_obs),
  N_Countries = c(results_res6$n_countries, results_res7$n_countries),
  N_Cities = c(results_res6$n_cities, results_res7$n_cities),
  R2_Marginal = c(results_res6$r2$R2_marginal, results_res7$r2$R2_marginal),
  R2_Conditional = c(results_res6$r2$R2_conditional, results_res7$r2$R2_conditional)
)

print(comparison_table)

# ========================================
# Statistical test for slope difference
# ========================================
cat("\n\nDifference in scaling exponent (β):\n")
cat("Resolution 6 β:", round(results_res6$fixed_slope, 4), "\n")
cat("Resolution 7 β:", round(results_res7$fixed_slope, 4), "\n")
cat("Difference:", round(results_res7$fixed_slope - results_res6$fixed_slope, 4), "\n")

# Check if CIs overlap
ci_overlap <- (results_res6$slope_ci[2] >= results_res7$slope_ci[1]) &&
              (results_res7$slope_ci[2] >= results_res6$slope_ci[1])
cat("\n95% Confidence Intervals overlap:", ci_overlap, "\n")

if (ci_overlap) {
  cat("=> The scaling coefficients are NOT significantly different between resolutions.\n")
} else {
  cat("=> The scaling coefficients ARE significantly different between resolutions.\n")
}

# ========================================
# Visualization
# ========================================

# Create comparison plot of slopes with CI
slope_comparison <- data.frame(
  Resolution = c("Resolution 6\n(~36 km²)", "Resolution 7\n(~5.2 km²)"),
  Slope = c(results_res6$fixed_slope, results_res7$fixed_slope),
  CI_Lower = c(results_res6$slope_ci[1], results_res7$slope_ci[1]),
  CI_Upper = c(results_res6$slope_ci[2], results_res7$slope_ci[2])
)

p_slope <- ggplot(slope_comparison, aes(x = Resolution, y = Slope)) +
  geom_point(size = 4, color = "#bf0000") +
  geom_errorbar(aes(ymin = CI_Lower, ymax = CI_Upper), width = 0.2, color = "#bf0000") +
  geom_hline(yintercept = 1, linetype = "dashed", color = "black", alpha = 0.7) +
  labs(
    title = "Scaling Exponent (β) Comparison",
    subtitle = "log₁₀(M) = β · log₁₀(N) + intercept",
    y = "Scaling Exponent (β)",
    x = ""
  ) +
  annotate("text", x = 2.3, y = 1, label = "Linear scaling", hjust = 0, vjust = -0.5, size = 3) +
  ylim(0.5, 1.1) +
  theme_bw() +
  theme(
    plot.title = element_text(hjust = 0.5, face = "bold"),
    plot.subtitle = element_text(hjust = 0.5)
  )

# Scatter plots for both resolutions
p_res6 <- ggplot(results_res6$data, aes(x = 10^log_population, y = 10^log_mass_avg)) +
  geom_point(alpha = 0.03, color = "#8da0cb") +
  geom_smooth(method = "lm", color = "#bf0000", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)), limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)), limits = c(1, 10^8)) +
  labs(
    title = paste0("Resolution 6 (β = ", round(results_res6$fixed_slope, 3), ")"),
    x = "Population (N)",
    y = "Total built mass (M, tonnes)"
  ) +
  theme_bw() +
  theme(panel.grid.minor = element_blank())

p_res7 <- ggplot(results_res7$data, aes(x = 10^log_population, y = 10^log_mass_avg)) +
  geom_point(alpha = 0.03, color = "#8da0cb") +
  geom_smooth(method = "lm", color = "#bf0000", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)), limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)), limits = c(1, 10^8)) +
  labs(
    title = paste0("Resolution 7 (β = ", round(results_res7$fixed_slope, 3), ")"),
    x = "Population (N)",
    y = "Total built mass (M, tonnes)"
  ) +
  theme_bw() +
  theme(panel.grid.minor = element_blank())

# Combine plots
combined_plot <- plot_grid(
  plot_grid(p_res6, p_res7, ncol = 2, labels = c("A", "B")),
  p_slope,
  ncol = 1,
  rel_heights = c(1, 0.6),
  labels = c("", "C")
)

# Save the comparison figure
today_date <- Sys.Date()
output_filename <- file.path(base_path, "..", "figures",
                             paste0("Fig3_Resolution_Comparison_", today_date, ".pdf"))

ggsave(output_filename, combined_plot, width = 10, height = 8, units = "in")
cat("\n\nFigure saved to:", output_filename, "\n")

# Display the plot
print(combined_plot)

# Save comparison results to CSV
output_csv <- file.path(base_path, "..", "data", "processed", "h3_resolution7",
                        paste0("scaling_comparison_res6_vs_res7_", today_date, ".csv"))
write_csv(comparison_table, output_csv)
cat("Comparison table saved to:", output_csv, "\n")
