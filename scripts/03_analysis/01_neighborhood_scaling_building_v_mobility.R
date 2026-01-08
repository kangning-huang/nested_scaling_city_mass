# Neighborhood Scaling Analysis: Building vs Mobility Mass
# This script conducts parallel scaling analyses for building mass and mobility mass
# to compare their scaling patterns across cities and neighborhoods

# Load required libraries
library(pacman)
pacman::p_load(
  Matrix, sf, lme4, readr, dplyr, performance, tibble,
  ggplot2, rprojroot, scales, viridis, cowplot, tidyr,
  ggspatial, moments, poweRlaw
)

# Set working directory to script location
# If running interactively in RStudio, uncomment:
# setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

# ============================================================================
# SECTION 1: Load and Preprocess Data
# ============================================================================

cat("Loading H3 hexagon data...\n")

# Load the H3 hexagon grids with both building and mobility mass
data <- readr::read_csv(
  file.path("..", "..", "0.CleanProject_GlobalScaling", "data", "processed",
            "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv")
) %>%
  dplyr::rename(
    mass_building = BuildingMass_AverageTotal,
    mass_mobility = mobility_mass_tons,
    mass_total = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  dplyr::filter(population > 0) %>%
  dplyr::filter(mass_building > 0) %>%
  dplyr::filter(mass_mobility > 0)

cat(sprintf("Loaded %d neighborhoods\n", nrow(data)))

# ============================================================================
# SECTION 2: Filter Data by Countries and Cities
# ============================================================================

cat("Filtering data...\n")

# Filter countries with at least 3 cities
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

# Filter cities with more than 3 neighborhoods
city_neighborhood_counts <- filtered_data %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(num_neighborhoods = n(), .groups = "drop") %>%
  dplyr::filter(num_neighborhoods > 3)

filtered_data <- filtered_data %>%
  dplyr::filter(city_id %in% city_neighborhood_counts$city_id)

cat(sprintf("After filtering: %d neighborhoods in %d cities in %d countries\n",
            nrow(filtered_data),
            n_distinct(filtered_data$city_id),
            n_distinct(filtered_data$country_iso)))

# ============================================================================
# SECTION 3: Prepare Data for Mixed Models
# ============================================================================

# Log-transform variables for both mass types
filtered_data <- filtered_data %>%
  mutate(
    log_population = log10(population),
    log_mass_building = log10(mass_building),
    log_mass_mobility = log10(mass_mobility)
  )

# Create hierarchical factors
filtered_data$Country <- factor(filtered_data$country)
filtered_data$Country_City <- factor(with(filtered_data, paste(Country, city_id, sep = "_")))

# ============================================================================
# SECTION 4: Nested Random-Effects Models
# ============================================================================

cat("\n=== BUILDING MASS MODEL ===\n")
cat("Fitting nested random-effects model for building mass...\n")

# Model for building mass
model_building <- lmer(
  log_mass_building ~ log_population + (1 | Country) + (1 | Country:Country_City),
  data = filtered_data
)

cat("\nBuilding Mass Model Summary:\n")
print(summary(model_building))

# Calculate R-squared for building model
r2_building <- performance::r2_nakagawa(model_building)
cat(sprintf("\nBuilding Mass R²: Conditional = %.3f, Marginal = %.3f\n",
            r2_building$R2_conditional, r2_building$R2_marginal))

cat("\n=== MOBILITY MASS MODEL ===\n")
cat("Fitting nested random-effects model for mobility mass...\n")

# Model for mobility mass
model_mobility <- lmer(
  log_mass_mobility ~ log_population + (1 | Country) + (1 | Country:Country_City),
  data = filtered_data
)

cat("\nMobility Mass Model Summary:\n")
print(summary(model_mobility))

# Calculate R-squared for mobility model
r2_mobility <- performance::r2_nakagawa(model_mobility)
cat(sprintf("\nMobility Mass R²: Conditional = %.3f, Marginal = %.3f\n",
            r2_mobility$R2_conditional, r2_mobility$R2_marginal))

# ============================================================================
# SECTION 5: Extract Fixed and Random Effects
# ============================================================================

# Building mass effects
fixed_intercept_building <- fixef(model_building)["(Intercept)"]
fixed_slope_building <- fixef(model_building)["log_population"]

ranefs_country_building <- ranef(model_building)$Country %>%
  rownames_to_column("Country") %>%
  rename(ranef_Country_building = `(Intercept)`)

ranefs_city_building <- ranef(model_building)$`Country:Country_City` %>%
  rownames_to_column("Country_City") %>%
  rename(ranef_Country_City_building = `(Intercept)`)

# Mobility mass effects
fixed_intercept_mobility <- fixef(model_mobility)["(Intercept)"]
fixed_slope_mobility <- fixef(model_mobility)["log_population"]

ranefs_country_mobility <- ranef(model_mobility)$Country %>%
  rownames_to_column("Country") %>%
  rename(ranef_Country_mobility = `(Intercept)`)

ranefs_city_mobility <- ranef(model_mobility)$`Country:Country_City` %>%
  rownames_to_column("Country_City") %>%
  rename(ranef_Country_City_mobility = `(Intercept)`)

# ============================================================================
# SECTION 6: Calculate Normalized Mass for Both Types
# ============================================================================

# Merge random effects into data
data_merged <- filtered_data %>%
  mutate(Country_City = paste(Country, ':', Country_City, sep = '')) %>%
  left_join(ranefs_country_building, by = "Country") %>%
  left_join(ranefs_city_building, by = "Country_City") %>%
  left_join(ranefs_country_mobility, by = "Country") %>%
  left_join(ranefs_city_mobility, by = "Country_City") %>%
  mutate(
    # Normalized building mass
    normalized_log_mass_building = log_mass_building -
      (fixed_intercept_building + ranef_Country_building + ranef_Country_City_building),
    # Normalized mobility mass
    normalized_log_mass_mobility = log_mass_mobility -
      (fixed_intercept_mobility + ranef_Country_mobility + ranef_Country_City_mobility)
  )

# ============================================================================
# SECTION 7: Calculate City-Level Slopes and Intercepts
# ============================================================================

cat("\nCalculating city-level regression parameters...\n")

# Filter cities with more than 10 neighborhoods
cities_with_enough_data <- data_merged %>%
  group_by(Country_City) %>%
  filter(n() > 10)

# Run regressions for building mass
city_models_building <- cities_with_enough_data %>%
  group_by(Country_City) %>%
  group_map(~ lm(log_mass_building ~ log_population, data = .x))
names(city_models_building) <- unique(cities_with_enough_data$Country_City)

city_params_building <- do.call(rbind, lapply(names(city_models_building), function(name) {
  model <- city_models_building[[name]]
  data.frame(
    Country_City = name,
    Slope_building = coef(model)["log_population"],
    Intercept_building = coef(model)["(Intercept)"]
  )
}))

# Run regressions for mobility mass
city_models_mobility <- cities_with_enough_data %>%
  group_by(Country_City) %>%
  group_map(~ lm(log_mass_mobility ~ log_population, data = .x))
names(city_models_mobility) <- unique(cities_with_enough_data$Country_City)

city_params_mobility <- do.call(rbind, lapply(names(city_models_mobility), function(name) {
  model <- city_models_mobility[[name]]
  data.frame(
    Country_City = name,
    Slope_mobility = coef(model)["log_population"],
    Intercept_mobility = coef(model)["(Intercept)"]
  )
}))

# Combine parameters
city_params_combined <- city_params_building %>%
  left_join(city_params_mobility, by = "Country_City")

# Export city-level parameters
output_file <- file.path("..", "results", "city_params_building_v_mobility.csv")
write_csv(city_params_combined, output_file)
cat(sprintf("Exported city parameters to: %s\n", output_file))

# ============================================================================
# SECTION 8: Prepare Data for Comparative Visualization
# ============================================================================

# Create summary statistics for boxplots
# Building mass - country level
country_data_building <- city_params_combined %>%
  mutate(Type = "Country", Mass_Type = "Building") %>%
  select(Slope = Slope_building, Intercept = Intercept_building, Type, Mass_Type)

# Building mass - city level
city_data_building <- city_params_combined %>%
  mutate(Type = "City", Mass_Type = "Building") %>%
  select(Slope = Slope_building, Intercept = Intercept_building, Type, Mass_Type)

# Mobility mass - country level
country_data_mobility <- city_params_combined %>%
  mutate(Type = "Country", Mass_Type = "Mobility") %>%
  select(Slope = Slope_mobility, Intercept = Intercept_mobility, Type, Mass_Type)

# Mobility mass - city level
city_data_mobility <- city_params_combined %>%
  mutate(Type = "City", Mass_Type = "Mobility") %>%
  select(Slope = Slope_mobility, Intercept = Intercept_mobility, Type, Mass_Type)

# Combine all data
combined_data <- bind_rows(
  country_data_building,
  city_data_building,
  country_data_mobility,
  city_data_mobility
)

# ============================================================================
# SECTION 9: Create Comparative Scatter Plots
# ============================================================================

cat("\nCreating scatter plots...\n")

# Scatter plot for building mass
scatter_building <- ggplot(data_merged,
                          aes(x = 10^log_population, y = 10^normalized_log_mass_building)) +
  geom_point(alpha = 0.05, color = "#8da0cb") +
  geom_smooth(method = "lm", color = "#bf0000", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^5)) +
  labs(
    title = "Building Mass",
    x = "Population (N)",
    y = "Normalized building mass (t)"
  ) +
  annotate("text", x = 10, y = 10^4.5,
           label = sprintf("β = %.2f", fixed_slope_building),
           color = "#bf0000", hjust = 0, size = 4) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Scatter plot for mobility mass
scatter_mobility <- ggplot(data_merged,
                          aes(x = 10^log_population, y = 10^normalized_log_mass_mobility)) +
  geom_point(alpha = 0.05, color = "#8da0cb") +
  geom_smooth(method = "lm", color = "#bf0000", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^5)) +
  labs(
    title = "Mobility Mass",
    x = "Population (N)",
    y = "Normalized mobility mass (t)"
  ) +
  annotate("text", x = 10, y = 10^4.5,
           label = sprintf("β = %.2f", fixed_slope_mobility),
           color = "#bf0000", hjust = 0, size = 4) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Combine scatter plots
scatter_combined <- plot_grid(
  scatter_building, scatter_mobility,
  labels = c("A", "B"),
  ncol = 2
)

# ============================================================================
# SECTION 9B: Create Scatter Plots with Bivariate Palette Colors
# ============================================================================

cat("\nCreating scatter plots with bivariate palette colors...\n")

# Scatter plot for building mass (using bivariate "high building, low mobility" color)
scatter_building_bivar <- ggplot(data_merged,
                          aes(x = 10^log_population, y = 10^normalized_log_mass_building)) +
  geom_point(alpha = 0.05, color = "#4885c1") +  # High building, low mobility color
  geom_smooth(method = "lm", color = "#4885c1", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^5)) +
  labs(
    title = "Building Mass",
    x = "Population (N)",
    y = "Normalized building mass (t)"
  ) +
  annotate("text", x = 10, y = 10^4.5,
           label = sprintf("β = %.2f", fixed_slope_building),
           color = "#4885c1", hjust = 0, size = 4) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Scatter plot for mobility mass (using bivariate "low building, high mobility" color)
scatter_mobility_bivar <- ggplot(data_merged,
                          aes(x = 10^log_population, y = 10^normalized_log_mass_mobility)) +
  geom_point(alpha = 0.05, color = "#ae3a4e") +  # Low building, high mobility color
  geom_smooth(method = "lm", color = "#ae3a4e", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^5)) +
  labs(
    title = "Mobility Mass",
    x = "Population (N)",
    y = "Normalized mobility mass (t)"
  ) +
  annotate("text", x = 10, y = 10^4.5,
           label = sprintf("β = %.2f", fixed_slope_mobility),
           color = "#ae3a4e", hjust = 0, size = 4) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# Combine bivariate-colored scatter plots
scatter_combined_bivar <- plot_grid(
  scatter_building_bivar, scatter_mobility_bivar,
  labels = c("A", "B"),
  ncol = 2
)

# ============================================================================
# SECTION 9C: Create Combined Scatter Plot (Both Mass Types in Same Plot)
# ============================================================================

cat("\nCreating combined scatter plot with both mass types...\n")

# Prepare data in long format for combined plotting
data_combined_plot <- data_merged %>%
  select(log_population, normalized_log_mass_building, normalized_log_mass_mobility) %>%
  pivot_longer(
    cols = c(normalized_log_mass_building, normalized_log_mass_mobility),
    names_to = "mass_type",
    values_to = "normalized_log_mass"
  ) %>%
  mutate(
    mass_type_label = ifelse(mass_type == "normalized_log_mass_building",
                             "Building Mass",
                             "Mobility Mass")
  )

# Create combined scatter plot
scatter_combined_overlay <- ggplot(data_combined_plot,
                                   aes(x = 10^log_population,
                                       y = 10^normalized_log_mass,
                                       color = mass_type_label)) +
  geom_point(alpha = 0.02, size = 1) +  # Increased transparency
  geom_smooth(method = "lm", alpha = 0.5, lwd = 1, se = FALSE) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_x_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^6.5)) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(1, 10^5)) +
  scale_color_manual(
    values = c("Building Mass" = "#4885c1", "Mobility Mass" = "#ae3a4e"),
    name = ""
  ) +
  labs(
    title = "Building vs Mobility Mass Scaling",
    x = "Population (N)",
    y = "Normalized mass (t)"
  ) +
  annotate("text", x = 10, y = 10^4.7,
           label = sprintf("Building: β = %.2f", fixed_slope_building),
           color = "#4885c1", hjust = 0, size = 4) +
  annotate("text", x = 10, y = 10^4.4,
           label = sprintf("Mobility: β = %.2f", fixed_slope_mobility),
           color = "#ae3a4e", hjust = 0, size = 4) +
  theme_bw() +
  theme(
    legend.position = c(0.85, 0.15),
    legend.background = element_rect(fill = "white", color = "gray80"),
    legend.title = element_blank(),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold")
  )

# ============================================================================
# SECTION 10: Create Comparative Boxplots
# ============================================================================

cat("Creating boxplots...\n")

# Reshape data for plotting
combined_long <- combined_data %>%
  pivot_longer(cols = c("Slope", "Intercept"),
               names_to = "variable",
               values_to = "value")

# Reorder factors
combined_long$variable <- factor(combined_long$variable, levels = c("Slope", "Intercept"))
combined_long$Mass_Type <- factor(combined_long$Mass_Type, levels = c("Building", "Mobility"))

# Create slope boxplot
slope_plot <- ggplot(combined_long %>% filter(variable == "Slope"),
                    aes(x = Type, y = value, fill = Mass_Type)) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA,
               position = position_dodge(0.8)) +
  geom_point(pch = 21, stroke = 0.3, alpha = 0.3,
             position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8)) +
  coord_flip() +
  scale_fill_manual(values = c("Building" = "#66c2a5", "Mobility" = "#fc8d62"),
                   name = "Mass Type") +
  scale_x_discrete(labels = c("City" = "Across-\nneighborhood",
                             "Country" = "Across-city")) +
  labs(x = "", y = "", title = expression("Slope " * beta)) +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  )

# Create intercept boxplot (log scale)
intercept_plot <- ggplot(combined_long %>% filter(variable == "Intercept"),
                        aes(x = Type, y = 10^value, fill = Mass_Type)) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA,
               position = position_dodge(0.8)) +
  geom_point(pch = 23, stroke = 0.3, alpha = 0.3,
             position = position_jitterdodge(jitter.width = 0.1, dodge.width = 0.8)) +
  coord_flip() +
  scale_fill_manual(values = c("Building" = "#66c2a5", "Mobility" = "#fc8d62"),
                   name = "Mass Type") +
  scale_x_discrete(labels = c("City" = "Across-\nneighborhood",
                             "Country" = "Across-city")) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
                labels = trans_format("log10", math_format(10^.x)),
                limits = c(10^2, 10^5)) +
  labs(x = "", y = "", title = expression(Intercept ~ M[0])) +
  theme_bw() +
  theme(
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    plot.title = element_text(hjust = 0.5, face = "bold"),
    legend.position = "bottom"
  )

# Combine boxplots
boxplot_combined <- plot_grid(
  slope_plot, intercept_plot,
  labels = c("C", "D"),
  ncol = 1,
  align = 'v'
)

# ============================================================================
# SECTION 11: Create Final Combined Figure
# ============================================================================

cat("Creating final combined figure...\n")

# Combine all panels
final_plot <- plot_grid(
  scatter_combined, boxplot_combined,
  ncol = 2,
  rel_widths = c(2, 1)
)

# ============================================================================
# SECTION 12: Export Results and Figures
# ============================================================================

cat("\nExporting figures...\n")

# Create output directory if it doesn't exist
output_dir <- file.path("..", "figures")
if (!dir.exists(output_dir)) {
  dir.create(output_dir, recursive = TRUE)
  cat("Created output directory:", output_dir, "\n")
}

# Get today's date
today_date <- Sys.Date()

# Export final combined figure
output_filename <- file.path(output_dir,
                            paste0("Fig_NeighborhoodScaling_Building_v_Mobility_",
                                   today_date, ".pdf"))
ggsave(output_filename, final_plot, width = 12, height = 6, units = "in")
cat(sprintf("Exported combined figure to: %s\n", output_filename))

# Export individual scatter plots
output_filename <- file.path(output_dir,
                            paste0("ScatterPlots_Building_v_Mobility_",
                                   today_date, ".pdf"))
ggsave(output_filename, scatter_combined, width = 10, height = 5, units = "in")
cat(sprintf("Exported scatter plots to: %s\n", output_filename))

# Export bivariate-colored scatter plots
output_filename <- file.path(output_dir,
                            paste0("ScatterPlots_Building_v_Mobility_BivarColors_",
                                   today_date, ".pdf"))
ggsave(output_filename, scatter_combined_bivar, width = 10, height = 5, units = "in")
cat(sprintf("Exported bivariate-colored scatter plots to: %s\n", output_filename))

# Export combined overlay scatter plot
output_filename <- file.path(output_dir,
                            paste0("ScatterPlot_Building_v_Mobility_Combined_",
                                   today_date, ".pdf"))
ggsave(output_filename, scatter_combined_overlay, width = 8, height = 6, units = "in")
cat(sprintf("Exported combined overlay scatter plot to: %s\n", output_filename))

# Export boxplots
output_filename <- file.path(output_dir,
                            paste0("Boxplots_Building_v_Mobility_",
                                   today_date, ".pdf"))
ggsave(output_filename, boxplot_combined, width = 6, height = 8, units = "in")
cat(sprintf("Exported boxplots to: %s\n", output_filename))

# ============================================================================
# SECTION 13: Export Model Summary Statistics
# ============================================================================

cat("\nExporting model summary statistics...\n")

# Create results directory if it doesn't exist
results_dir <- file.path("..", "results")
if (!dir.exists(results_dir)) {
  dir.create(results_dir, recursive = TRUE)
  cat("Created results directory:", results_dir, "\n")
}

# Compile model comparison summary
model_summary <- data.frame(
  Mass_Type = c("Building", "Mobility"),
  Fixed_Intercept = c(fixed_intercept_building, fixed_intercept_mobility),
  Fixed_Slope = c(fixed_slope_building, fixed_slope_mobility),
  R2_Conditional = c(r2_building$R2_conditional, r2_mobility$R2_conditional),
  R2_Marginal = c(r2_building$R2_marginal, r2_mobility$R2_marginal)
)

output_file <- file.path(results_dir, "model_summary_building_v_mobility.csv")
write_csv(model_summary, output_file)
cat(sprintf("Exported model summary to: %s\n", output_file))

# Print final summary
cat("\n=== ANALYSIS COMPLETE ===\n")
cat("\nModel Comparison Summary:\n")
print(model_summary)

cat("\nSlope Comparison:\n")
cat(sprintf("Building Mass Slope: %.3f\n", fixed_slope_building))
cat(sprintf("Mobility Mass Slope: %.3f\n", fixed_slope_mobility))
cat(sprintf("Difference: %.3f\n", fixed_slope_building - fixed_slope_mobility))

# ============================================================================
# SECTION 14: Create Bivariate Maps for Selected Cities
# ============================================================================

cat("\n=== CREATING BIVARIATE MAPS ===\n")

# Set CRAN mirror for package installation
options(repos = c(CRAN = "https://cloud.r-project.org/"))

# Install biscale package if not available
if (!require("biscale", quietly = TRUE)) {
  cat("Installing biscale package for bivariate mapping...\n")
  install.packages("biscale")
  library(biscale)
}

# Install cowplot if not already loaded
if (!require("cowplot", quietly = TRUE)) {
  cat("Installing cowplot package...\n")
  install.packages("cowplot")
  library(cowplot)
}

# Create figures directory if it doesn't exist
bivar_figures_dir <- file.path("..", "figures", "bivariate_maps")
if (!dir.exists(bivar_figures_dir)) {
  dir.create(bivar_figures_dir, recursive = TRUE)
  cat("Created bivariate maps directory:", bivar_figures_dir, "\n")
}

# Define cities for bivariate mapping
cities_for_bivariate <- c("Rome", "Paris", "Atlanta", "Tokyo",
                          "Addis Ababa", "Bogota", "Mexico City", "Melbourne")

# Load spatial data for neighborhoods
cat("\nLoading spatial data for bivariate mapping...\n")

# Read the H3 hexagon geometries
sf_hexagons_bivar <- sf::read_sf(
  file.path("..", "..", "0.CleanProject_GlobalScaling", "data",
            "processed", "all_cities_h3_grids.gpkg")
)

# Prepare data for joining - keep all necessary columns including h3index
# First check what columns are available
cat("Checking available columns in data_merged...\n")
cat(paste("Columns:", paste(colnames(data_merged)[1:min(20, ncol(data_merged))], collapse = ", "), "...\n"))

# Select columns, handling the h3index column name
data_for_join <- data_merged

# Ensure we have the h3index column
if (!"h3index" %in% colnames(data_for_join)) {
  # Check if it's named differently in the original data
  if ("h3index" %in% colnames(filtered_data)) {
    # This shouldn't happen, but just in case
    cat("Warning: h3index not in data_merged, using filtered_data\n")
    data_for_join <- filtered_data %>%
      left_join(
        data_merged %>%
          select(
            -any_of(colnames(filtered_data)[colnames(filtered_data) != "h3index"])
          ),
        by = intersect(colnames(filtered_data), colnames(data_merged))
      )
  }
}

# Convert to character for joining
data_for_join <- data_for_join %>%
  mutate(h3index = as.character(h3index))

# Check column name in spatial data
if ("h3index" %in% colnames(sf_hexagons_bivar)) {
  join_col <- "h3index"
} else if ("hex_id" %in% colnames(sf_hexagons_bivar)) {
  sf_hexagons_bivar <- sf_hexagons_bivar %>%
    dplyr::rename(h3index = hex_id)
  join_col <- "h3index"
} else {
  # Use the first non-geometry column as join key
  join_col <- colnames(sf_hexagons_bivar)[1]
  data_for_join <- data_for_join %>%
    dplyr::rename(!!join_col := h3index)
}

# Join with the data_merged to get both mass types
sf_bivariate <- sf_hexagons_bivar %>%
  inner_join(data_for_join, by = join_col)

cat(sprintf("Loaded %d neighborhoods with spatial data\n", nrow(sf_bivariate)))
cat("Columns in sf_bivariate:", paste(colnames(sf_bivariate)[1:min(15, ncol(sf_bivariate))], collapse = ", "), "...\n")

# Clean up duplicate columns - use the .y version from data_merged
if ("UC_NM_MN.y" %in% colnames(sf_bivariate)) {
  sf_bivariate <- sf_bivariate %>%
    select(-UC_NM_MN.x) %>%
    rename(UC_NM_MN = UC_NM_MN.y)
}

# Loop through each city and create bivariate map
for (city_name in cities_for_bivariate) {
  cat(sprintf("\nProcessing bivariate map for %s...\n", city_name))

  # Filter city data
  city_data_bivar <- sf_bivariate %>%
    filter(UC_NM_MN == city_name)

  if (nrow(city_data_bivar) > 0) {
    cat(sprintf("Found %d neighborhoods in %s\n", nrow(city_data_bivar), city_name))

    # Calculate per capita mass for better comparison
    city_data_bivar <- city_data_bivar %>%
      mutate(
        building_per_capita = mass_building / population,
        mobility_per_capita = mass_mobility / population
      )

    # Create bivariate classes (3x3 grid)
    city_data_bivar <- biscale::bi_class(
      city_data_bivar,
      x = building_per_capita,
      y = mobility_per_capita,
      style = "quantile",
      dim = 3
    )

    # Create the bivariate map
    bivar_map <- ggplot() +
      geom_sf(
        data = city_data_bivar,
        aes(fill = bi_class),
        color = "white",
        size = 0.1,
        show.legend = FALSE
      ) +
      biscale::bi_scale_fill(pal = "DkViolet", dim = 3) +
      labs(
        title = sprintf("%s: Building vs Mobility Mass (Per Capita)", city_name),
        subtitle = sprintf("n = %d neighborhoods", nrow(city_data_bivar))
      ) +
      theme_void() +
      theme(
        plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
        plot.subtitle = element_text(hjust = 0.5, size = 11),
        plot.margin = margin(10, 10, 10, 10)
      )

    # Create bivariate legend
    bivar_legend <- biscale::bi_legend(
      pal = "DkViolet",
      dim = 3,
      xlab = "Higher Building Mass →",
      ylab = "Higher Mobility Mass →",
      size = 8
    )

    # Combine map and legend
    final_bivar_plot <- cowplot::ggdraw() +
      cowplot::draw_plot(bivar_map, 0, 0, 1, 1) +
      cowplot::draw_plot(bivar_legend, 0.05, 0.05, 0.25, 0.25)

    # Create safe filename
    safe_city_name <- gsub(" ", "_", city_name)

    # Export as PDF
    output_bivar_pdf <- file.path(bivar_figures_dir,
                                  sprintf("%s_Bivariate_Building_Mobility.pdf", safe_city_name))
    ggsave(output_bivar_pdf, final_bivar_plot, width = 10, height = 8, units = "in")
    cat(sprintf("Exported %s bivariate map (PDF) to: %s\n", city_name, output_bivar_pdf))

    # Export as PNG
    output_bivar_png <- file.path(bivar_figures_dir,
                                  sprintf("%s_Bivariate_Building_Mobility.png", safe_city_name))
    ggsave(output_bivar_png, final_bivar_plot, width = 10, height = 8, units = "in", dpi = 300)
    cat(sprintf("Exported %s bivariate map (PNG) to: %s\n", city_name, output_bivar_png))

    # Print summary statistics
    cat(sprintf("  Mean building per capita: %.2f t/person\n",
                mean(city_data_bivar$building_per_capita, na.rm = TRUE)))
    cat(sprintf("  Mean mobility per capita: %.2f t/person\n",
                mean(city_data_bivar$mobility_per_capita, na.rm = TRUE)))

  } else {
    cat(sprintf("Warning: No data found for %s\n", city_name))
  }
}

cat("\n=== BIVARIATE MAPPING COMPLETE ===\n")
cat("\nBivariate maps exported to:", bivar_figures_dir, "\n")

cat("\n=== END OF SCRIPT ===\n")
