#!/usr/bin/env Rscript
# =============================================================================
# Fig3_R7_city_candidates.R
#
# Find top 5 megacities with slopes closest to global delta (0.7132),
# then generate 5 variations of Fig3_Decentered_R7 with each city as inset.
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(
  sf, lme4, readr, dplyr, tibble, ggplot2, scales, viridis,
  cowplot, tidyr, ggspatial, ggrepel
)

# --- Paths ---
file_args <- grep("^--file=", commandArgs(trailingOnly = FALSE), value = TRUE)
script_dir <- if (length(file_args) == 0) getwd() else dirname(normalizePath(sub("^--file=", "", file_args[1])))
rev_dir <- normalizePath(file.path(script_dir, ".."), mustWork = TRUE)
data_dir <- file.path(rev_dir, "data")
figures_dir <- file.path(rev_dir, "figures")
dir.create(figures_dir, showWarnings = FALSE, recursive = TRUE)

neighborhood_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
city_file <- file.path(data_dir, "MasterMass_ByClass20250616.csv")

# Copy grid file locally for speed
grid_file_gdrive <- file.path(data_dir, "all_cities_h3_grids_resolution7.gpkg")
grid_file_local <- file.path(tempdir(), "all_cities_h3_grids_resolution7.gpkg")
if (!file.exists(grid_file_local)) {
  cat("Copying grid file to local temp...\n")
  file.copy(grid_file_gdrive, grid_file_local)
}

# --- Load & Filter Data (same as main script) ---
data_neighborhood_raw <- readr::read_csv(neighborhood_file, show_col_types = FALSE) %>%
  dplyr::rename(
    mass_building = BuildingMass_Total_Esch2022,
    mass_mobility = mobility_mass_tons,
    mass_avg = total_built_mass_tons,
    population = population_2015,
    city_id = ID_HDC_G0,
    country_iso = CTR_MN_ISO,
    country = CTR_MN_NM
  ) %>%
  dplyr::filter(population >= 1, mass_avg > 0)

city_population_totals <- data_neighborhood_raw %>%
  group_by(country_iso, city_id) %>%
  summarize(total_population = sum(population), .groups = "drop") %>%
  filter(total_population > 50000)

filtered_neighborhoods <- data_neighborhood_raw %>%
  filter(city_id %in% city_population_totals$city_id)

city_neighborhood_counts <- filtered_neighborhoods %>%
  group_by(country_iso, city_id) %>%
  summarize(num_neighborhoods = n(), .groups = "drop") %>%
  filter(num_neighborhoods >= 10)

filtered_neighborhoods <- filtered_neighborhoods %>%
  filter(city_id %in% city_neighborhood_counts$city_id)

country_city_counts <- filtered_neighborhoods %>%
  group_by(country_iso) %>%
  summarize(num_cities = n_distinct(city_id)) %>%
  filter(num_cities >= 5)

filtered_neighborhoods <- filtered_neighborhoods %>%
  filter(country_iso %in% country_city_counts$country_iso) %>%
  mutate(
    log_population = log10(population),
    log_mass_avg = log10(mass_avg)
  )

filtered_neighborhoods$Country <- factor(filtered_neighborhoods$country)
filtered_neighborhoods$Country_City <- factor(
  with(filtered_neighborhoods, paste(country, city_id, sep = "_"))
)

cat("Filtered neighborhoods:", nrow(filtered_neighborhoods), "\n")
cat("Filtered cities:", n_distinct(filtered_neighborhoods$city_id), "\n")

# --- De-center ---
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
delta <- coef(mod_city_level)[2]
se_delta <- summary(mod_city_level)$coefficients[2, 2]
r2_city_level <- summary(mod_city_level)$r.squared
beta <- 0.90

cat("\nGlobal delta:", round(delta, 4), "\n")

# --- Compute city-level slopes ---
city_slopes <- decentered_neighborhoods %>%
  group_by(Country_City, country, city_id, UC_NM_MN) %>%
  filter(n() >= 10) %>%
  group_modify(~ {
    mod <- lm(log_mass_centered ~ log_pop_centered, data = .x)
    tibble(
      Slope = coef(mod)[2],
      R2 = summary(mod)$r.squared,
      n_neighborhoods = nrow(.x)
    )
  }) %>%
  ungroup()

# --- Find top 5 megacities closest to delta ---
# "Megacity" = cities with many neighborhoods (>=200 at R7, proxy for large population)
mega_candidates <- city_slopes %>%
  filter(n_neighborhoods >= 200) %>%
  mutate(slope_diff = abs(Slope - delta)) %>%
  arrange(slope_diff)

cat("\n=== Top 20 Megacity Candidates (>=200 neighborhoods R7) ===\n")
print(as.data.frame(
  mega_candidates %>%
    head(20) %>%
    select(UC_NM_MN, country, Slope, slope_diff, R2, n_neighborhoods)
), row.names = FALSE)

# Pick top 5
top5 <- mega_candidates %>% head(5)

cat("\n=== SELECTED TOP 5 ===\n")
for (i in seq_len(nrow(top5))) {
  cat(sprintf(
    "%d. %s (%s) - slope=%.4f, diff=%.4f, n=%d, R2=%.3f\n",
    i, top5$UC_NM_MN[i], top5$country[i],
    top5$Slope[i], top5$slope_diff[i],
    top5$n_neighborhoods[i], top5$R2[i]
  ))
}

# --- Load grid for map insets ---
cat("\nLoading grid file...\n")
sf_cities <- sf::read_sf(grid_file_local) %>%
  dplyr::rename(hex_id = h3index)
cat("Grid loaded:", nrow(sf_cities), "hexagons\n")

# --- Also load city-level data for panels B/C (same as main script) ---
data_city_fig2 <- read.csv(city_file, stringsAsFactors = FALSE) %>%
  filter(total_built_mass_tons > 0)

city_counts_fig2 <- table(data_city_fig2$CTR_MN_NM)
valid_countries_fig2 <- names(city_counts_fig2[city_counts_fig2 >= 5])
filtered_cities_fig2 <- data_city_fig2 %>%
  filter(CTR_MN_NM %in% valid_countries_fig2) %>%
  mutate(
    log_population = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  )

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
    tibble(Slope = coef(mod)[2], n_cities = nrow(.x))
  }) %>%
  ungroup()

N_REF_COUNTRY <- 1e6
country_M0 <- decentered_cities_fig2 %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  summarize(mean_log_pop = mean(log_population), mean_log_mass = mean(log_mass), .groups = "drop") %>%
  left_join(country_slopes %>% select(CTR_MN_NM, Slope), by = "CTR_MN_NM") %>%
  mutate(M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_COUNTRY), M0 = 10^M0_log)

country_stats <- country_slopes %>%
  left_join(country_M0 %>% select(CTR_MN_NM, M0_log, M0, mean_log_pop, mean_log_mass), by = "CTR_MN_NM") %>%
  rename(Country = CTR_MN_NM)

N_REF_NEIGHBORHOOD <- 1e4
city_M0 <- decentered_neighborhoods %>%
  group_by(Country_City, country, city_id, UC_NM_MN) %>%
  filter(n() >= 10) %>%
  summarize(mean_log_pop = mean(log_population), mean_log_mass = mean(log_mass_avg), .groups = "drop") %>%
  left_join(city_slopes %>% select(Country_City, Slope), by = "Country_City") %>%
  mutate(M0_log = mean_log_mass - Slope * mean_log_pop + Slope * log10(N_REF_NEIGHBORHOOD), M0 = 10^M0_log)

city_stats <- city_slopes %>%
  left_join(city_M0 %>% select(Country_City, M0_log, M0, mean_log_pop, mean_log_mass), by = "Country_City")

# --- Build Panels B and C (shared across all 5 figures) ---
country_slope_data <- country_stats %>% select(Slope) %>% mutate(Type = "Country-level")
city_slope_data <- city_stats %>% select(Slope) %>% mutate(Type = "City-level")
combined_slope_data <- bind_rows(country_slope_data, city_slope_data)
combined_slope_data$Type <- factor(combined_slope_data$Type, levels = c("City-level", "Country-level"))

country_75th <- quantile(country_stats$Slope, 0.75, na.rm = TRUE)
highlight_country <- country_stats %>% filter(Slope > country_75th) %>% slice_max(Slope, n = 1)

city_25th <- quantile(city_stats$Slope, 0.25, na.rm = TRUE)
city_75th <- quantile(city_stats$Slope, 0.75, na.rm = TRUE)
highlight_city <- city_stats %>%
  filter(Slope < city_25th | Slope > city_75th) %>%
  slice_max(abs(Slope - median(city_stats$Slope, na.rm = TRUE)), n = 1)

slope_plot <- ggplot(combined_slope_data, aes(x = Type, y = Slope, fill = Type)) +
  geom_jitter(pch = 21, width = 0.2, stroke = 0.3, alpha = 0.3, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, lwd = 0.5, show.legend = FALSE) +
  geom_point(data = data.frame(Type = "Country-level", Slope = highlight_country$Slope),
    aes(x = Type, y = Slope), pch = 21, fill = "#fc8d62", size = 4, stroke = 1) +
  geom_text_repel(
    data = data.frame(Type = "Country-level", Slope = highlight_country$Slope, label = highlight_country$Country),
    aes(x = Type, y = Slope, label = label), nudge_x = 0.3, size = 2.5, segment.size = 0.3) +
  geom_point(data = data.frame(Type = "City-level", Slope = highlight_city$Slope),
    aes(x = Type, y = Slope), pch = 21, fill = "#bf0000", size = 4, stroke = 1) +
  geom_text_repel(
    data = data.frame(Type = "City-level", Slope = highlight_city$Slope, label = highlight_city$UC_NM_MN),
    aes(x = Type, y = Slope, label = label), nudge_x = 0.3, size = 2.5, segment.size = 0.3) +
  coord_flip() +
  scale_fill_manual(values = c("City-level" = "#bf0000", "Country-level" = "#fc8d62")) +
  labs(x = "", y = "") +
  ggtitle(expression("Slope " * beta)) +
  theme_bw() +
  theme(
    panel.spacing = unit(0, "npc"), strip.background = element_blank(),
    panel.grid.major.x = element_blank(), panel.grid.major.y = element_blank(),
    panel.grid.minor.y = element_blank(), strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5))

country_M0_data <- country_stats %>% select(M0) %>% mutate(Type = "Country-level")
city_M0_data <- city_stats %>% select(M0) %>% mutate(Type = "City-level")
combined_M0_data <- bind_rows(country_M0_data, city_M0_data)
combined_M0_data$Type <- factor(combined_M0_data$Type, levels = c("City-level", "Country-level"))

highlight_country_M0 <- country_stats %>%
  filter(M0 > quantile(M0, 0.75, na.rm = TRUE)) %>% slice_max(M0, n = 1)
city_M0_25th <- quantile(city_stats$M0, 0.25, na.rm = TRUE)
city_M0_75th <- quantile(city_stats$M0, 0.75, na.rm = TRUE)
highlight_city_M0 <- city_stats %>%
  filter(M0 < city_M0_25th | M0 > city_M0_75th) %>%
  slice_max(abs(log10(M0) - median(log10(city_stats$M0), na.rm = TRUE)), n = 1)

intercept_plot <- ggplot(combined_M0_data, aes(x = Type, y = M0, fill = Type)) +
  geom_jitter(pch = 23, width = 0.2, stroke = 0.3, alpha = 0.3, show.legend = FALSE) +
  geom_boxplot(width = 0.6, alpha = 0.7, outlier.shape = NA, color = "grey25", lwd = 0.5, show.legend = FALSE) +
  geom_point(data = data.frame(Type = "Country-level", M0 = highlight_country_M0$M0),
    aes(x = Type, y = M0), pch = 23, fill = "#fc8d62", size = 4, stroke = 1) +
  geom_text_repel(
    data = data.frame(Type = "Country-level", M0 = highlight_country_M0$M0, label = highlight_country_M0$Country),
    aes(x = Type, y = M0, label = label), nudge_x = 0.3, size = 2.5, segment.size = 0.3) +
  geom_point(data = data.frame(Type = "City-level", M0 = highlight_city_M0$M0),
    aes(x = Type, y = M0), pch = 23, fill = "#bf0000", size = 4, stroke = 1) +
  geom_text_repel(
    data = data.frame(Type = "City-level", M0 = highlight_city_M0$M0, label = highlight_city_M0$UC_NM_MN),
    aes(x = Type, y = M0, label = label), nudge_x = 0.3, size = 2.5, segment.size = 0.3) +
  coord_flip() +
  scale_fill_manual(values = c("City-level" = "#bf0000", "Country-level" = "#fc8d62")) +
  scale_y_log10(breaks = trans_breaks("log10", function(x) 10^x),
    labels = trans_format("log10", math_format(10^.x))) +
  labs(x = "", y = "") +
  ggtitle(expression(M(N[ref]) ~ "(tonnes)")) +
  theme_bw() +
  theme(
    panel.spacing = unit(0, "npc"), panel.grid.major.x = element_blank(),
    panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank(),
    strip.background = element_blank(), strip.text = element_text(face = "bold"),
    plot.title = element_text(hjust = 0.5))

box_plot <- cowplot::plot_grid(slope_plot, intercept_plot, nrow = 2, align = "h",
  rel_widths = c(1, 1), labels = c("B", "C"))

# --- Generate 5 figure variations ---
for (i in seq_len(nrow(top5))) {
  city_name <- top5$UC_NM_MN[i]
  city_slope <- top5$Slope[i]
  city_n <- top5$n_neighborhoods[i]

  cat(sprintf("\n--- Generating figure %d/5: %s (slope=%.4f, n=%d) ---\n",
    i, city_name, city_slope, city_n))

  # Get city data for scatter highlight
  data_city_inset <- decentered_neighborhoods %>%
    filter(UC_NM_MN == city_name, population > 10)

  # Get city grid for map inset
  sf_city_inset <- sf_cities %>% filter(UC_NM_MN == city_name)

  sf_city_inset <- sf_city_inset %>%
    left_join(data_city_inset, by = c("hex_id" = "h3index"))

  city_boundary <- st_union(sf_city_inset)
  max_hex <- sf_city_inset %>% filter(log_population == max(log_population, na.rm = TRUE))

  # Determine appropriate zoom level based on city area
  city_area_km2 <- as.numeric(st_area(city_boundary)) / 1e6
  zoom_level <- if (city_area_km2 > 5000) 9 else if (city_area_km2 > 2000) 10 else if (city_area_km2 > 500) 11 else 12
  cat(sprintf("  City area: %.0f km2, zoom: %d\n", city_area_km2, zoom_level))

  map_inset <- ggplot() +
    annotation_map_tile(type = "cartolight", zoom = zoom_level) +
    geom_sf(data = sf_city_inset, aes(fill = pmax(log_population, 3)), color = NA, alpha = 0.6) +
    geom_sf(data = city_boundary, color = "#fc8d62", fill = NA, linewidth = 1) +
    geom_sf(data = max_hex, color = "#bf0000", fill = NA, linewidth = 1) +
    scale_fill_viridis_c(option = "viridis", guide = "none") +
    labs(title = city_name) +
    theme_void() +
    theme(legend.position = "none")

  scatter_plot <- ggplot(decentered_neighborhoods,
    aes(x = log_pop_centered, y = log_mass_centered)) +
    geom_point(alpha = 0.03, color = "#8da0cb") +
    geom_point(data = data_city_inset,
      aes(x = log_pop_centered, y = log_mass_centered, color = pmax(log_population, 3))) +
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
      panel.grid.major.x = element_blank(), panel.grid.minor.x = element_blank(),
      panel.grid.major.y = element_blank(), panel.grid.minor.y = element_blank()
    ) +
    annotate("text", x = -4.8, y = 1.7,
      label = sprintf("Delta*log[10](M[ij]) == %.2f * Delta*log[10](N[ij])", round(beta, 2)),
      color = "#fc8d62", parse = TRUE, hjust = 0, size = 3) +
    annotate("text", x = -4.8, y = 1.3,
      label = sprintf("Delta*log[10](M[ijk]) == %.2f * Delta*log[10](N[ijk])", round(delta, 2)),
      color = "#bf0000", parse = TRUE, hjust = 0, size = 3)

  panel_A <- ggdraw() +
    draw_plot(scatter_plot) +
    draw_plot(map_inset, x = 0.6, y = 0.15, width = 0.35, height = 0.35)

  combined_plot <- plot_grid(panel_A, box_plot, labels = c("A", ""), ncol = 2, nrow = 1,
    rel_widths = c(4 * 1.4, 4 / 1.1))

  # Clean city name for filename
  city_clean <- gsub("[^A-Za-z0-9]", "_", city_name)
  output_file <- file.path(figures_dir, sprintf("Fig3_Decentered_R7_%s_%s.pdf",
    city_clean, Sys.Date()))

  ggsave(output_file, combined_plot, width = (4 * 1.4) + (4 / 1.1), height = 4, units = "in")
  cat(sprintf("  Saved: %s\n", basename(output_file)))
}

cat("\n=== ALL 5 FIGURES GENERATED ===\n")
