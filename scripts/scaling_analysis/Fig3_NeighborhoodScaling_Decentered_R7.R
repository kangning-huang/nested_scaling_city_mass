#! /usr/bin/env Rscript
# =============================================================================
# Fig3_NeighborhoodScaling_Decentered_R7.R
#
# Resolution-7 version of the baseline Figure 3 (de-centered) workflow.
# Mirrors scripts/Fig3_NeighborhoodScaling_Decentered.R but swaps in the
# resolution-7 neighborhood dataset and grid for Panel A.
#
# Includes subway infrastructure mass added to both neighborhood-level and
# city-level mass totals (from subway_mass_by_hexagon_resolution7.csv).
# =============================================================================

rm(list = ls())

if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(
  sf, lme4, readr, dplyr, tibble, ggplot2, scales, viridis,
  cowplot, tidyr, ggrepel
)

file_args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_args) == 0) getwd() else dirname(normalizePath(sub("^--file=", "", file_args[1])))
rev_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
data_dir <- file.path(rev_dir, "data")
figures_dir <- file.path(rev_dir, "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

neighborhood_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
city_file <- file.path(data_dir, "MasterMass_ByClass20250616.csv")
grid_file <- file.path(data_dir, "all_cities_h3_grids_resolution7.gpkg")
results_dir <- file.path(rev_dir, "results")
subway_file <- file.path(results_dir, "subway_mass_by_hexagon_resolution7.csv")

# --- Load subway mass data ---
cat("Loading subway mass data...\n")
subway_hex <- readr::read_csv(subway_file, show_col_types = FALSE)
cat("Subway hexagons loaded:", nrow(subway_hex), "\n")

data_neighborhood_raw <- readr::read_csv(neighborhood_file) %>%
  dplyr::rename(
    mass_building = BuildingMass_AverageTotal,
    mass_mobility = mobility_mass_tons,
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  # Add subway mass to neighborhood mass
  dplyr::left_join(
    subway_hex %>% dplyr::select(city_id, h3index, subway_mass_tonnes),
    by = c("city_id" = "city_id", "h3index" = "h3index")
  ) %>%
  dplyr::mutate(
    subway_mass_tonnes = dplyr::coalesce(subway_mass_tonnes, 0),
    mass_avg = mass_avg + subway_mass_tonnes
  ) %>%
  dplyr::filter(population >= 1) %>%
  dplyr::filter(mass_avg > 0)

cat("Raw neighborhoods (R7):", nrow(data_neighborhood_raw), "\n")

# Add subway mass to city-level data
subway_city <- subway_hex %>%
  dplyr::group_by(city_id) %>%
  dplyr::summarize(subway_mass_city = sum(subway_mass_tonnes, na.rm = TRUE), .groups = "drop")

data_city_fig2 <- read.csv(city_file, stringsAsFactors = FALSE) %>%
  dplyr::left_join(subway_city, by = c("ID_HDC_G0" = "city_id")) %>%
  dplyr::mutate(
    subway_mass_city = dplyr::coalesce(subway_mass_city, 0),
    total_built_mass_tons = total_built_mass_tons + subway_mass_city
  ) %>%
  dplyr::filter(total_built_mass_tons > 0)

cat("Raw cities (Figure 2 data):", nrow(data_city_fig2), "\n")

city_population_totals <- data_neighborhood_raw %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(total_population = sum(population), .groups = "drop") %>%
  dplyr::filter(total_population > 50000)

filtered_neighborhoods <- data_neighborhood_raw %>%
  dplyr::filter(city_id %in% city_population_totals$city_id)

city_neighborhood_counts <- filtered_neighborhoods %>%
  dplyr::group_by(country_iso, city_id) %>%
  dplyr::summarize(num_neighborhoods = n(), .groups = "drop") %>%
  dplyr::filter(num_neighborhoods >= 10)

filtered_neighborhoods <- filtered_neighborhoods %>%
  dplyr::filter(city_id %in% city_neighborhood_counts$city_id)

country_city_counts <- filtered_neighborhoods %>%
  dplyr::group_by(country_iso) %>%
  dplyr::summarize(num_cities = n_distinct(city_id)) %>%
  dplyr::filter(num_cities >= 5)

filtered_neighborhoods <- filtered_neighborhoods %>%
  dplyr::filter(country_iso %in% country_city_counts$country_iso)

filtered_neighborhoods <- filtered_neighborhoods %>%
  mutate(
    log_population = log10(population),
    log_mass_avg = log10(mass_avg)
  )

filtered_neighborhoods$Country <- factor(filtered_neighborhoods$country)
filtered_neighborhoods$Country_City <- factor(with(filtered_neighborhoods, paste(country, city_id, sep = "_")))

cat("\n=== Neighborhood Data (Filtered, R7) ===\n")
cat("Neighborhoods:", nrow(filtered_neighborhoods), "\n")
cat("Cities:", n_distinct(filtered_neighborhoods$city_id), "\n")
cat("Countries:", n_distinct(filtered_neighborhoods$country), "\n")

city_counts_fig2 <- table(data_city_fig2$CTR_MN_NM)
valid_countries_fig2 <- names(city_counts_fig2[city_counts_fig2 >= 5])

filtered_cities_fig2 <- data_city_fig2 %>%
  dplyr::filter(CTR_MN_NM %in% valid_countries_fig2) %>%
  mutate(
    log_population = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  )

cat("\n=== City Data (Figure 2, Filtered) ===\n")
cat("Cities:", nrow(filtered_cities_fig2), "\n")
cat("Countries:", length(valid_countries_fig2), "\n")

decentered_neighborhoods <- filtered_neighborhoods %>%
  group_by(Country_City) %>%
  mutate(
    city_mean_log_pop = mean(log_population),
    city_mean_log_mass = mean(log_mass_avg),
    log_pop_centered = log_population - city_mean_log_pop,
    log_mass_centered = log_mass_avg - city_mean_log_mass
  ) %>%
  ungroup()

mod_city_level <- lm(log_mass_centered ~ log_pop_centered, data = decentered_neighborhoods)
summary_city <- summary(mod_city_level)

delta <- coef(mod_city_level)[2]
se_delta <- summary_city$coefficients[2, 2]
r2_city_level <- summary_city$r.squared
n_neighborhoods <- nrow(decentered_neighborhoods)

cat("\n=== City-level De-centered OLS (delta, R7) - COMPUTED ===\n")
cat("Delta (slope):", round(delta, 4), "\n")
cat("95% CI: [", round(delta - 1.96 * se_delta, 4), ",", round(delta + 1.96 * se_delta, 4), "]\n")
cat("SE:", round(se_delta, 4), "\n")
cat("R-squared:", round(r2_city_level, 3), "\n")
cat("N neighborhoods:", n_neighborhoods, "\n")

beta <- 0.90

cat("\n=== Country-level Beta (from Figure 2) - HARDCODED ===\n")
cat("Beta (slope):", beta, "\n")

city_slopes <- decentered_neighborhoods %>%
  group_by(Country_City, country, city_id, UC_NM_MN) %>%
  filter(n() >= 10) %>%
  group_modify(~ {
    mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
    tibble(
      Slope = coef(mod)[2],
      n_neighborhoods = nrow(.x)
    )
  }) %>%
  ungroup()

N_REF_NEIGHBORHOOD <- 1e4

city_M0 <- decentered_neighborhoods %>%
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
    by = "Country_City"
  )

cat("\nNumber of cities with sufficient data for city-level slopes:", nrow(city_stats), "\n")

decentered_cities_fig2 <- filtered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  mutate(
    country_mean_log_pop = mean(log_population),
    country_mean_log_mass = mean(log_mass),
    log_pop_centered = log_population - country_mean_log_pop,
    log_mass_centered = log_mass - country_mean_log_mass
  ) %>%
  ungroup()

country_slopes <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  group_modify(~ {
    mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
    tibble(
      Slope = coef(mod)[2],
      n_cities = nrow(.x)
    )
  }) %>%
  ungroup()

N_REF_COUNTRY <- 1e6

country_M0 <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  summarize(
    mean_log_pop = mean(log_population),
    mean_log_mass = mean(log_mass),
    .groups = "drop"
  ) %>%
  left_join(country_slopes %>% select(CTR_MN_NM, Slope), by = "CTR_MN_NM") %>%
  mutate(
    M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_COUNTRY),
    M0 = 10^M0_log
  )

country_stats <- country_slopes %>%
  left_join(country_M0 %>% select(CTR_MN_NM, M0_log, M0, mean_log_pop, mean_log_mass),
    by = "CTR_MN_NM"
  ) %>%
  rename(Country = CTR_MN_NM)

cat("\n=== Country-level Stats (from Figure 2 city data) ===\n")
cat("Number of countries:", nrow(country_stats), "\n")
cat("Total cities across countries:", sum(country_stats$n_cities), "\n")

write_csv(country_stats, file.path(data_dir, "country_slope_decentered_fig3_R7.csv"))
write_csv(city_stats, file.path(data_dir, "city_slope_decentered_fig3_R7.csv"))

cat("\nExported: country_slope_decentered_fig3_R7.csv, city_slope_decentered_fig3_R7.csv\n")

decentered_neighborhoods <- decentered_neighborhoods %>%
  mutate(is_newyork = (UC_NM_MN == "New York"))

sf_cities <- sf::read_sf(grid_file) %>%
  dplyr::rename(hex_id = h3index)

sf_newyork <- sf_cities %>% dplyr::filter(UC_NM_MN == "New York")

data_newyork <- decentered_neighborhoods %>%
  filter(is_newyork, population > 10)

sf_newyork <- sf_newyork %>%
  left_join(data_newyork, by = c("hex_id" = "h3index"))

ny_boundary <- st_union(sf_newyork)
max_hex <- sf_newyork %>% filter(log_population == max(log_population, na.rm = TRUE))

map_inset <- ggplot() +
  geom_sf(data = sf_newyork, aes(fill = pmax(log_population, 3)), color = NA, alpha = 0.6) +
  geom_sf(data = ny_boundary, color = "#fc8d62", fill = NA, linewidth = 1) +
  geom_sf(data = max_hex, color = "#bf0000", fill = NA, linewidth = 1) +
  scale_fill_viridis_c(option = "viridis", guide = "none") +
  labs(title = "New York") +
  theme_void() +
  theme(legend.position = "none")

scatter_plot <- ggplot(decentered_neighborhoods,
  aes(x = log_pop_centered, y = log_mass_centered)
) +
  geom_point(alpha = 0.03, color = "#8da0cb") +
  geom_point(
    data = data_newyork,
    aes(x = log_pop_centered, y = log_mass_centered, color = pmax(log_population, 3))
  ) +
  geom_abline(slope = delta, intercept = 0, color = "#bf0000", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = beta, intercept = 0, color = "#fc8d62", alpha = 0.7, lwd = 0.75) +
  geom_abline(slope = 1, intercept = 0, color = "black", linetype = "dashed", lwd = 0.75) +
  scale_color_viridis_c(option = "viridis", guide = "none") +
  coord_cartesian(xlim = c(-5, 3), ylim = c(-4.5, 2)) +
  labs(
    x = "Log population (deviation from city mean)",
    y = "Log mass (deviation from city mean)"
  ) +
  theme_bw() +
  theme(
    legend.position = "none",
    panel.grid.major.x = element_blank(),
    panel.grid.minor.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank()
  ) +
  annotate(
    "text",
    x = -4.8,
    y = 1.7,
    label = sprintf("Delta*log[10](M[ij]) == %.2f * Delta*log[10](N[ij])", round(beta, 2)),
    color = "#fc8d62", parse = TRUE, hjust = 0, size = 3
  ) +
  annotate(
    "text",
    x = -4.8,
    y = 1.3,
    label = sprintf("Delta*log[10](M[ijk]) == %.2f * Delta*log[10](N[ijk])", round(delta, 2)),
    color = "#bf0000", parse = TRUE, hjust = 0, size = 3
  )

panel_A <- ggdraw() +
  draw_plot(scatter_plot) +
  draw_plot(map_inset, x = 0.6, y = 0.15, width = 0.35, height = 0.35)

country_slope_data <- country_stats %>%
  select(Slope) %>%
  mutate(Type = "Country-level")

city_slope_data <- city_stats %>%
  select(Slope) %>%
  mutate(Type = "City-level")

combined_slope_data <- bind_rows(country_slope_data, city_slope_data)
combined_slope_data$Type <- factor(combined_slope_data$Type, levels = c("City-level", "Country-level"))

country_75th <- quantile(country_stats$Slope, 0.75, na.rm = TRUE)
highlight_country <- country_stats %>%
  filter(Slope > country_75th) %>%
  slice_max(Slope, n = 1)

city_25th <- quantile(city_stats$Slope, 0.25, na.rm = TRUE)
city_75th <- quantile(city_stats$Slope, 0.75, na.rm = TRUE)
highlight_city <- city_stats %>%
  filter(Slope < city_25th | Slope > city_75th) %>%
  slice_max(abs(Slope - median(city_stats$Slope, na.rm = TRUE)), n = 1)

slope_plot <- ggplot(combined_slope_data, aes(x = Type, y = Slope, fill = Type)) +
  geom_jitter(pch = 21, width = 0.2, stroke = 0.3, alpha = 0.3, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  geom_point(data = data.frame(Type = "Country-level", Slope = highlight_country$Slope),
    aes(x = Type, y = Slope),
    pch = 21, fill = "#fc8d62", size = 4, stroke = 1
  ) +
  geom_text_repel(
    data = data.frame(Type = "Country-level", Slope = highlight_country$Slope, label = highlight_country$Country),
    aes(x = Type, y = Slope, label = label),
    nudge_x = 0.3, size = 2.5, segment.size = 0.3
  ) +
  geom_point(data = data.frame(Type = "City-level", Slope = highlight_city$Slope),
    aes(x = Type, y = Slope),
    pch = 21, fill = "#bf0000", size = 4, stroke = 1
  ) +
  geom_text_repel(
    data = data.frame(Type = "City-level", Slope = highlight_city$Slope, label = highlight_city$UC_NM_MN),
    aes(x = Type, y = Slope, label = label),
    nudge_x = 0.3, size = 2.5, segment.size = 0.3
  ) +
  coord_flip() +
  scale_fill_manual(values = c("City-level" = "#bf0000", "Country-level" = "#fc8d62")) +
  labs(x = "", y = "") +
  ggtitle(expression("Slope " * beta)) +
  theme_bw() +
  theme(
    panel.spacing = unit(0, "npc"),
    strip.background = element_blank(),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )

country_M0_data <- country_stats %>%
  select(M0) %>%
  mutate(Type = "Country-level")

city_M0_data <- city_stats %>%
  select(M0) %>%
  mutate(Type = "City-level")

combined_M0_data <- bind_rows(country_M0_data, city_M0_data)
combined_M0_data$Type <- factor(combined_M0_data$Type, levels = c("City-level", "Country-level"))

highlight_country_M0 <- country_stats %>%
  filter(M0 > quantile(M0, 0.75, na.rm = TRUE)) %>%
  slice_max(M0, n = 1)

city_M0_25th <- quantile(city_stats$M0, 0.25, na.rm = TRUE)
city_M0_75th <- quantile(city_stats$M0, 0.75, na.rm = TRUE)
highlight_city_M0 <- city_stats %>%
  filter(M0 < city_M0_25th | M0 > city_M0_75th) %>%
  slice_max(abs(log10(M0) - median(log10(city_stats$M0), na.rm = TRUE)), n = 1)

intercept_plot <- ggplot(combined_M0_data, aes(x = Type, y = M0, fill = Type)) +
  geom_jitter(pch = 23, width = 0.2, stroke = 0.3, alpha = 0.3, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, color = "grey25", lwd = 0.5, show.legend = FALSE) +
  geom_point(data = data.frame(Type = "Country-level", M0 = highlight_country_M0$M0),
    aes(x = Type, y = M0),
    pch = 23, fill = "#fc8d62", size = 4, stroke = 1
  ) +
  geom_text_repel(
    data = data.frame(Type = "Country-level", M0 = highlight_country_M0$M0, label = highlight_country_M0$Country),
    aes(x = Type, y = M0, label = label),
    nudge_x = 0.3, size = 2.5, segment.size = 0.3
  ) +
  geom_point(data = data.frame(Type = "City-level", M0 = highlight_city_M0$M0),
    aes(x = Type, y = M0),
    pch = 23, fill = "#bf0000", size = 4, stroke = 1
  ) +
  geom_text_repel(
    data = data.frame(Type = "City-level", M0 = highlight_city_M0$M0, label = highlight_city_M0$UC_NM_MN),
    aes(x = Type, y = M0, label = label),
    nudge_x = 0.3, size = 2.5, segment.size = 0.3
  ) +
  coord_flip() +
  scale_fill_manual(values = c("City-level" = "#bf0000", "Country-level" = "#fc8d62")) +
  scale_y_log10(
    breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  labs(x = "", y = "") +
  ggtitle(expression(M(N[ref]) ~ "(tonnes)")) +
  theme_bw() +
  theme(
    panel.spacing = unit(0, "npc"),
    panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(),
    strip.background = element_blank(),
    strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5)
  )

box_plot <- cowplot::plot_grid(slope_plot, intercept_plot, nrow = 2, align = "h", rel_widths = c(1, 1), labels = c("B", "C"))

combined_plot_horizontal <- plot_grid(panel_A, box_plot, labels = c("A", ""), ncol = 2, nrow = 1, rel_widths = c(4 * 1.4, 4 / 1.1))

current_date <- Sys.Date()
output_filename <- file.path(figures_dir, paste0("Fig3_Decentered_R7_", current_date, ".pdf"))

ggsave(output_filename, combined_plot_horizontal, width = (4 * 1.4) + (4 / 1.1), height = 4, units = "in")

cat("\nFigure saved to:", output_filename, "\n")

cat("\n========================================\n")
cat("SUMMARY FOR REVISION LETTER (Figure 3, R7)\n")
cat("========================================\n")
cat("De-centering approach (Bettencourt & Lobo 2016):\n")
cat("  - City-level (delta): Neighborhoods de-centered by city mean\n")
cat("  - Country-level (beta): HARDCODED from Figure 2 (not recomputed)\n")
cat("\nResults:\n")
cat("  City-level delta (COMPUTED):", round(delta, 4), "\n")
cat("    95% CI: [", round(delta - 1.96 * se_delta, 4), ",", round(delta + 1.96 * se_delta, 4), "]\n")
cat("    R-squared:", round(r2_city_level, 3), "\n")
cat("    N neighborhoods:", n_neighborhoods, "\n")
cat("    N cities:", n_distinct(decentered_neighborhoods$city_id), "\n")
cat("\n  Country-level beta (HARDCODED):", beta, "\n")
cat("    (From Figure 2 de-centered analysis)\n")
cat("\nPanel B/C Statistics:\n")
cat("  Country-level (from Figure 2 city data):\n")
cat("    N countries:", nrow(country_stats), "\n")
cat("    Total cities:", sum(country_stats$n_cities), "\n")
cat("  City-level (from neighborhood data, >=10 neighborhoods):\n")
cat("    N cities:", nrow(city_stats), "\n")
cat("    N countries:", n_distinct(city_stats$country), "\n")
cat("\nM0 REFERENCE POPULATIONS (Panel C):\n")
cat("  - Country-level: N_ref = 1,000,000 (typical city population)\n")
cat("  - City-level: N_ref = 10,000 (typical neighborhood population)\n")
cat("  - M0 = expected mass at reference population, not at N=1\n")
cat("  - This makes M0 comparable across entities with different pop centroids\n")
cat("\nFIXES APPLIED:\n")
cat("  1. Panel A axes: x=[-5,3], y=[-4.5,2]\n")
cat("  2. Beta hardcoded as 0.90 from Figure 2\n")
cat("  3. Country-level boxplots use Figure 2's city data\n")
cat("  4. City-level threshold: >=10 neighborhoods\n")
cat("  5. Population filter: >= 1\n")
cat("  6. M0 at reference population (not N=1)\n")
cat("========================================\n")
