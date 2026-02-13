# =============================================================================
# Fig3_NeighborhoodScaling_Decentered_R7_Weighted_Source.R
#
# Neighborhood-level (H3 Resolution 7) scaling analysis using region-specific
# reliability-weighted building volume source averaging.
#
# Methodology:
#   1. Map each neighborhood to one of 13 world regions via CTR_MN_ISO
#   2. Apply region-specific reliability weights to building mass sources
#   3. Weighted average building mass = sum(w_i * mass_i) / sum(w_i)
#      (only using sources where data > 0 for that row)
#   4. Total mass = weighted_bldg + mobility_mass_tons
#   5. Apply per-resolution filter:
#      - pop >= 1, mass > 0
#      - City total pop > 50,000
#      - >= 10 neighborhoods per city
#      - >= 5 qualifying cities per country
#   6. De-center log(pop) and log(mass) by city mean
#   7. OLS on pooled de-centered data -> delta
#
# Comparison:
#   A. Region-weighted average (new)
#   B. Equal-weighted average (baseline)
#   C. Individual sources (Esch2022, Li2022, Liu2024)
#
# Output: CSV summary + PDF figure to _Revision1/figures/
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

mass_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
sources <- c("Esch2022", "Li2022", "Liu2024")

cat("=== Fig3 De-centered - R7 Region-Weighted Source Averaging (Neighborhood) ===\n\n")

# =============================================================================
# REGION-SPECIFIC RELIABILITY WEIGHTS
# =============================================================================
# Raw weights (1=low, 3=high confidence):
#   | World Region              | Esch2022 | Li2022 | Liu2024 |
#   |---------------------------|----------|--------|---------|
#   | North America             | 2        | 3      | 3       |
#   | Europe (W/C)              | 2        | 3      | 3       |
#   | Europe (East)             | 2        | 2      | 2       |
#   | East Asia (China)         | 2        | 1      | 3       |
#   | East Asia (Japan/Korea)   | 1        | 2      | 2       |
#   | South Asia                | 2        | 1      | 1       |
#   | Southeast Asia            | 2        | 2      | 1       |
#   | Central Asia              | 1        | 1      | 1       |
#   | Middle East / N. Africa   | 2        | 2      | 1       |
#   | Sub-Saharan Africa        | 2        | 2      | 1       |
#   | Latin America             | 1        | 2      | 1       |
#   | Oceania / Australia       | 1        | 3      | 2       |
#   | Russia / North Asia       | 1        | 1      | 2       |
#
# Weights are normalized to sum to 1 within each region.
# =============================================================================

region_weights <- tribble(
  ~region,                    ~w_Esch2022, ~w_Li2022, ~w_Liu2024,
  "North America",            2,           3,         3,
  "Europe (W/C)",             2,           3,         3,
  "Europe (East)",            2,           2,         2,
  "East Asia (China)",        2,           1,         3,
  "East Asia (Japan/Korea)",  1,           2,         2,
  "South Asia",               2,           1,         1,
  "Southeast Asia",           2,           2,         1,
  "Central Asia",             1,           1,         1,
  "Middle East / N. Africa",  2,           2,         1,
  "Sub-Saharan Africa",       2,           2,         1,
  "Latin America",            1,           2,         1,
  "Oceania / Australia",      1,           3,         2,
  "Russia / North Asia",      1,           1,         2
)

# Normalize each row so weights sum to 1
region_weights <- region_weights %>%
  mutate(
    w_total = w_Esch2022 + w_Li2022 + w_Liu2024,
    w_Esch2022 = w_Esch2022 / w_total,
    w_Li2022 = w_Li2022 / w_total,
    w_Liu2024 = w_Liu2024 / w_total
  ) %>%
  select(-w_total)

cat("Region weights (normalized):\n")
print(as.data.frame(region_weights), row.names = FALSE)

# =============================================================================
# COUNTRY ISO -> WORLD REGION MAPPING
# =============================================================================
# Comprehensive mapping for all 172 ISO codes in the datasets.
# Unmapped countries default to equal weights (1/3, 1/3, 1/3).
# =============================================================================

iso_to_region <- c(
  # --- North America ---
  "USA" = "North America",
  "CAN" = "North America",

  # --- Europe (W/C): Western, Northern, and Southern Europe ---
  "DEU" = "Europe (W/C)",
  "FRA" = "Europe (W/C)",
  "GBR" = "Europe (W/C)",
  "NLD" = "Europe (W/C)",
  "BEL" = "Europe (W/C)",
  "AUT" = "Europe (W/C)",
  "CHE" = "Europe (W/C)",
  "LUX" = "Europe (W/C)",
  "IRL" = "Europe (W/C)",
  "ESP" = "Europe (W/C)",
  "ITA" = "Europe (W/C)",
  "PRT" = "Europe (W/C)",
  "GRC" = "Europe (W/C)",
  "DNK" = "Europe (W/C)",
  "SWE" = "Europe (W/C)",
  "NOR" = "Europe (W/C)",
  "FIN" = "Europe (W/C)",
  "MLT" = "Europe (W/C)",
  "CYP" = "Europe (W/C)",
  "SVN" = "Europe (W/C)",
  "HRV" = "Europe (W/C)",
  "EST" = "Europe (W/C)",
  "LVA" = "Europe (W/C)",
  "LTU" = "Europe (W/C)",
  "POL" = "Europe (W/C)",
  "CZE" = "Europe (W/C)",
  "SVK" = "Europe (W/C)",
  "ISR" = "Europe (W/C)",
  "JEY" = "Europe (W/C)",

  # --- Europe (East) ---
  "ROU" = "Europe (East)",
  "BGR" = "Europe (East)",
  "HUN" = "Europe (East)",
  "SRB" = "Europe (East)",
  "BIH" = "Europe (East)",
  "ALB" = "Europe (East)",
  "MKD" = "Europe (East)",
  "MDA" = "Europe (East)",
  "UKR" = "Europe (East)",
  "BLR" = "Europe (East)",
  "MNE" = "Europe (East)",
  "XKO" = "Europe (East)",
  "GEO" = "Europe (East)",
  "ARM" = "Europe (East)",
  "AZE" = "Europe (East)",

  # --- East Asia (China) ---
  "CHN" = "East Asia (China)",
  "TWN" = "East Asia (China)",

  # --- East Asia (Japan/Korea) ---
  "JPN" = "East Asia (Japan/Korea)",
  "KOR" = "East Asia (Japan/Korea)",
  "PRK" = "East Asia (Japan/Korea)",

  # --- South Asia ---
  "IND" = "South Asia",
  "PAK" = "South Asia",
  "BGD" = "South Asia",
  "LKA" = "South Asia",
  "NPL" = "South Asia",
  "AFG" = "South Asia",
  "BTN" = "South Asia",

  # --- Southeast Asia ---
  "THA" = "Southeast Asia",
  "VNM" = "Southeast Asia",
  "IDN" = "Southeast Asia",
  "PHL" = "Southeast Asia",
  "MYS" = "Southeast Asia",
  "MMR" = "Southeast Asia",
  "SGP" = "Southeast Asia",
  "KHM" = "Southeast Asia",
  "LAO" = "Southeast Asia",
  "TLS" = "Southeast Asia",
  "BRN" = "Southeast Asia",

  # --- Central Asia ---
  "KAZ" = "Central Asia",
  "UZB" = "Central Asia",
  "TKM" = "Central Asia",
  "KGZ" = "Central Asia",
  "TJK" = "Central Asia",
  "MNG" = "Central Asia",

  # --- Middle East / N. Africa ---
  "SAU" = "Middle East / N. Africa",
  "IRN" = "Middle East / N. Africa",
  "IRQ" = "Middle East / N. Africa",
  "EGY" = "Middle East / N. Africa",
  "MAR" = "Middle East / N. Africa",
  "TUN" = "Middle East / N. Africa",
  "DZA" = "Middle East / N. Africa",
  "LBY" = "Middle East / N. Africa",
  "ARE" = "Middle East / N. Africa",
  "KWT" = "Middle East / N. Africa",
  "QAT" = "Middle East / N. Africa",
  "BHR" = "Middle East / N. Africa",
  "OMN" = "Middle East / N. Africa",
  "JOR" = "Middle East / N. Africa",
  "LBN" = "Middle East / N. Africa",
  "SYR" = "Middle East / N. Africa",
  "YEM" = "Middle East / N. Africa",
  "PSE" = "Middle East / N. Africa",
  "TUR" = "Middle East / N. Africa",
  "SDN" = "Middle East / N. Africa",
  "ESH" = "Middle East / N. Africa",
  "MRT" = "Middle East / N. Africa",

  # --- Sub-Saharan Africa ---
  "NGA" = "Sub-Saharan Africa",
  "KEN" = "Sub-Saharan Africa",
  "ZAF" = "Sub-Saharan Africa",
  "GHA" = "Sub-Saharan Africa",
  "TZA" = "Sub-Saharan Africa",
  "ETH" = "Sub-Saharan Africa",
  "AGO" = "Sub-Saharan Africa",
  "CMR" = "Sub-Saharan Africa",
  "COD" = "Sub-Saharan Africa",
  "COG" = "Sub-Saharan Africa",
  "CIV" = "Sub-Saharan Africa",
  "SEN" = "Sub-Saharan Africa",
  "MLI" = "Sub-Saharan Africa",
  "BFA" = "Sub-Saharan Africa",
  "NER" = "Sub-Saharan Africa",
  "TCD" = "Sub-Saharan Africa",
  "CAF" = "Sub-Saharan Africa",
  "BEN" = "Sub-Saharan Africa",
  "TGO" = "Sub-Saharan Africa",
  "GIN" = "Sub-Saharan Africa",
  "SLE" = "Sub-Saharan Africa",
  "LBR" = "Sub-Saharan Africa",
  "GAB" = "Sub-Saharan Africa",
  "SOM" = "Sub-Saharan Africa",
  "MDG" = "Sub-Saharan Africa",
  "MOZ" = "Sub-Saharan Africa",
  "ZMB" = "Sub-Saharan Africa",
  "ZWE" = "Sub-Saharan Africa",
  "MWI" = "Sub-Saharan Africa",
  "RWA" = "Sub-Saharan Africa",
  "BDI" = "Sub-Saharan Africa",
  "UGA" = "Sub-Saharan Africa",
  "NAM" = "Sub-Saharan Africa",
  "MUS" = "Sub-Saharan Africa",
  "GMB" = "Sub-Saharan Africa",
  "GNB" = "Sub-Saharan Africa",
  "GNQ" = "Sub-Saharan Africa",
  "ERI" = "Sub-Saharan Africa",
  "SSD" = "Sub-Saharan Africa",
  "LSO" = "Sub-Saharan Africa",
  "SWZ" = "Sub-Saharan Africa",
  "BWA" = "Sub-Saharan Africa",
  "COM" = "Sub-Saharan Africa",

  # --- Latin America ---
  "BRA" = "Latin America",
  "MEX" = "Latin America",
  "ARG" = "Latin America",
  "COL" = "Latin America",
  "PER" = "Latin America",
  "CHL" = "Latin America",
  "VEN" = "Latin America",
  "ECU" = "Latin America",
  "BOL" = "Latin America",
  "PRY" = "Latin America",
  "URY" = "Latin America",
  "SUR" = "Latin America",
  "GUY" = "Latin America",
  "GTM" = "Latin America",
  "HND" = "Latin America",
  "SLV" = "Latin America",
  "NIC" = "Latin America",
  "CRI" = "Latin America",
  "PAN" = "Latin America",
  "CUB" = "Latin America",
  "DOM" = "Latin America",
  "HTI" = "Latin America",
  "JAM" = "Latin America",
  "TTO" = "Latin America",
  "BHS" = "Latin America",
  "BRB" = "Latin America",
  "CUW" = "Latin America",
  "PRI" = "Latin America",
  "BLZ" = "Latin America",

  # --- Oceania / Australia ---
  "AUS" = "Oceania / Australia",
  "NZL" = "Oceania / Australia",

  # --- Russia / North Asia ---
  "RUS" = "Russia / North Asia"
)

# =============================================================================
# LOAD DATA
# =============================================================================

cat("Loading R7 neighborhood data...\n")
df <- readr::read_csv(mass_file, show_col_types = FALSE)
cat("  Loaded:", nrow(df), "neighborhoods\n")

# Map each neighborhood to a world region
df$world_region <- iso_to_region[df$CTR_MN_ISO]

# Check for unmapped countries
unmapped <- df %>% filter(is.na(world_region)) %>% pull(CTR_MN_ISO) %>% unique()
if (length(unmapped) > 0) {
  cat("WARNING: Unmapped ISOs (will use equal weights):", paste(unmapped, collapse = ", "), "\n")
  df$world_region[is.na(df$world_region)] <- "UNMAPPED"
}

# Region distribution
cat("\nNeighborhoods per region:\n")
region_counts <- sort(table(df$world_region), decreasing = TRUE)
for (i in seq_along(region_counts)) {
  cat(sprintf("  %-30s %d\n", names(region_counts)[i], region_counts[i]))
}

# =============================================================================
# COMPUTE REGION-WEIGHTED BUILDING MASS
# =============================================================================

# Merge region weights into data
df <- df %>%
  left_join(region_weights, by = c("world_region" = "region"))

# For unmapped regions, use equal weights
df$w_Esch2022[is.na(df$w_Esch2022)] <- 1/3
df$w_Li2022[is.na(df$w_Li2022)] <- 1/3
df$w_Liu2024[is.na(df$w_Liu2024)] <- 1/3

# Compute weighted building mass:
# For each row, only use sources with data > 0
df <- df %>%
  mutate(
    # Source masses
    m_esch = BuildingMass_Total_Esch2022,
    m_li   = BuildingMass_Total_Li2022,
    m_liu  = BuildingMass_Total_Liu2024,

    # Indicator for available sources (mass > 0)
    has_esch = as.numeric(m_esch > 0),
    has_li   = as.numeric(m_li > 0),
    has_liu  = as.numeric(m_liu > 0),

    # Weighted numerator (only from available sources)
    weighted_num = (w_Esch2022 * m_esch * has_esch +
                    w_Li2022   * m_li   * has_li +
                    w_Liu2024  * m_liu  * has_liu),

    # Weighted denominator (sum of weights for available sources)
    weighted_den = (w_Esch2022 * has_esch +
                    w_Li2022   * has_li +
                    w_Liu2024  * has_liu),

    # Weighted average building mass
    weighted_bldg_mass = ifelse(weighted_den > 0,
                                weighted_num / weighted_den,
                                0),

    # Total mass = weighted building + mobility
    total_weighted = weighted_bldg_mass + mobility_mass_tons
  )

# Also compute per-source totals for comparison
for (src in sources) {
  bldg_col <- paste0("BuildingMass_Total_", src)
  total_col <- paste0("total_", src)
  df[[total_col]] <- df[[bldg_col]] + df$mobility_mass_tons
}

cat("\nWeighted mass computed for all neighborhoods.\n")

# =============================================================================
# APPLY PER-RESOLUTION FILTER (R7)
# =============================================================================
# Uses baseline (equal-weighted) mass for filtering to keep city/country
# sets consistent across all approaches.
# =============================================================================

cat("\n=== Applying per-resolution filter at R7 ===\n")

d7 <- df %>%
  filter(population_2015 >= 1, total_built_mass_tons > 0)
cat(sprintf("  After pop>=1, mass>0: %d neighborhoods\n", nrow(d7)))

# Cities with total pop > 50,000
city_pop <- d7 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(total_pop = sum(population_2015), .groups = "drop") %>%
  filter(total_pop > 50000)
d7 <- d7 %>% filter(ID_HDC_G0 %in% city_pop$ID_HDC_G0)
cat(sprintf("  After city pop>50k: %d neighborhoods\n", nrow(d7)))

# Cities with >= 10 neighborhoods
city_nbhd <- d7 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)
d7 <- d7 %>% filter(ID_HDC_G0 %in% city_nbhd$ID_HDC_G0)
cat(sprintf("  After >=10 nbhd/city: %d neighborhoods\n", nrow(d7)))

# Countries with >= 5 qualifying cities
country_city <- d7 %>%
  group_by(CTR_MN_ISO) %>%
  summarize(num_cities = n_distinct(ID_HDC_G0)) %>%
  filter(num_cities >= 5)
d7 <- d7 %>% filter(CTR_MN_ISO %in% country_city$CTR_MN_ISO)
cat(sprintf("  After >=5 cities/country: %d neighborhoods\n", nrow(d7)))

r7_city_ids <- unique(d7$ID_HDC_G0)
r7_country_isos <- unique(d7$CTR_MN_ISO)
cat(sprintf("\nR7 filter: %d cities, %d countries, %d neighborhoods\n",
            length(r7_city_ids), length(r7_country_isos), nrow(d7)))

# =============================================================================
# DE-CENTERED OLS FUNCTION (Neighborhood: de-center by city)
# =============================================================================

run_decentered_nbhd <- function(df_full, mass_col) {
  d <- df_full %>%
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
    n_countries = n_distinct(dc$CTR_MN_NM),
    data = dc
  )
}

# =============================================================================
# A. REGION-WEIGHTED AVERAGE (new approach)
# =============================================================================

cat("\n--- Region-Weighted Average ---\n")
res_weighted <- run_decentered_nbhd(df, "total_weighted")
cat(sprintf("  delta = %.4f [%.4f, %.4f], SE = %.5f, R2 = %.4f, N = %d, cities = %d, countries = %d\n",
            res_weighted$delta, res_weighted$ci_low, res_weighted$ci_high,
            res_weighted$se, res_weighted$r2,
            res_weighted$n_neighborhoods, res_weighted$n_cities,
            res_weighted$n_countries))

# =============================================================================
# B. EQUAL-WEIGHTED BASELINE
# =============================================================================

cat("\n--- Equal-Weighted Baseline ---\n")
res_baseline <- run_decentered_nbhd(df, "total_built_mass_tons")
cat(sprintf("  delta = %.4f [%.4f, %.4f], SE = %.5f, R2 = %.4f, N = %d, cities = %d, countries = %d\n",
            res_baseline$delta, res_baseline$ci_low, res_baseline$ci_high,
            res_baseline$se, res_baseline$r2,
            res_baseline$n_neighborhoods, res_baseline$n_cities,
            res_baseline$n_countries))

# =============================================================================
# C. INDIVIDUAL SOURCES
# =============================================================================

cat("\n--- Individual Sources ---\n")
res_sources <- list()
for (src in sources) {
  total_col <- paste0("total_", src)
  res_sources[[src]] <- run_decentered_nbhd(df, total_col)
  cat(sprintf("  %-12s delta = %.4f [%.4f, %.4f], SE = %.5f, R2 = %.4f, N = %d\n",
              src, res_sources[[src]]$delta,
              res_sources[[src]]$ci_low, res_sources[[src]]$ci_high,
              res_sources[[src]]$se, res_sources[[src]]$r2,
              res_sources[[src]]$n_neighborhoods))
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

source_labels <- c("Esch2022" = "Esch2022 (WSF3D)",
                    "Li2022" = "Li2022",
                    "Liu2024" = "Liu2024 (GUS3D)")

summary_table <- bind_rows(
  tibble(Resolution = 7, Hex_km2 = 5.161,
         Approach = "Region-Weighted Avg", Delta = res_weighted$delta,
         SE = res_weighted$se, CI_low = res_weighted$ci_low,
         CI_high = res_weighted$ci_high, R2 = res_weighted$r2,
         N_neighborhoods = res_weighted$n_neighborhoods,
         N_cities = res_weighted$n_cities, N_countries = res_weighted$n_countries),
  tibble(Resolution = 7, Hex_km2 = 5.161,
         Approach = "Equal-Weighted Avg (Baseline)", Delta = res_baseline$delta,
         SE = res_baseline$se, CI_low = res_baseline$ci_low,
         CI_high = res_baseline$ci_high, R2 = res_baseline$r2,
         N_neighborhoods = res_baseline$n_neighborhoods,
         N_cities = res_baseline$n_cities, N_countries = res_baseline$n_countries),
  bind_rows(lapply(sources, function(src) {
    r <- res_sources[[src]]
    tibble(Resolution = 7, Hex_km2 = 5.161,
           Approach = src, Delta = r$delta, SE = r$se,
           CI_low = r$ci_low, CI_high = r$ci_high, R2 = r$r2,
           N_neighborhoods = r$n_neighborhoods,
           N_cities = r$n_cities, N_countries = r$n_countries)
  }))
)

cat("\n=== SUMMARY TABLE ===\n")
print(as.data.frame(summary_table), row.names = FALSE)

outfile_csv <- file.path(figure_dir,
  paste0("Table_Decentered_R7_Weighted_Source_Neighborhood_", Sys.Date(), ".csv"))
write_csv(summary_table, outfile_csv)
cat("\nSaved:", outfile_csv, "\n")

# =============================================================================
# FIGURES
# =============================================================================

cat("\n=== Generating figures ===\n")

# --- Panel A: Forest plot comparing all approaches ---
forest_df <- bind_rows(
  tibble(Label = "Region-Weighted Avg", Beta = res_weighted$delta,
         CI_low = res_weighted$ci_low, CI_high = res_weighted$ci_high,
         Type = "Weighted"),
  tibble(Label = "Equal-Weighted Avg", Beta = res_baseline$delta,
         CI_low = res_baseline$ci_low, CI_high = res_baseline$ci_high,
         Type = "Baseline"),
  bind_rows(lapply(sources, function(src) {
    r <- res_sources[[src]]
    tibble(Label = source_labels[src], Beta = r$delta,
           CI_low = r$ci_low, CI_high = r$ci_high, Type = "Individual")
  }))
)

forest_df$Label <- factor(forest_df$Label, levels = rev(forest_df$Label))

p_forest <- ggplot(forest_df, aes(x = Beta, y = Label, color = Type)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.3,
                 linewidth = 0.7) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.4f", Beta)), vjust = -1, size = 2.8,
            show.legend = FALSE) +
  scale_color_manual(values = c("Weighted" = "#e7298a",
                                "Baseline" = "#fc8d62",
                                "Individual" = "#8da0cb")) +
  labs(
    title = "Neighborhood-Level (R7) Scaling Exponent: Region-Weighted vs Other Approaches",
    x = expression("Scaling exponent " * delta),
    y = NULL,
    color = NULL
  ) +
  theme_bw(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        legend.position = "bottom")

# --- Panel B: De-centered scatter for weighted vs baseline ---
scatter_list <- list(
  "Region-Weighted" = list(res = res_weighted, col = "#e7298a"),
  "Equal-Weighted" = list(res = res_baseline, col = "#fc8d62")
)

# Common axis limits (subsample for efficiency)
set.seed(42)
sample_idx_w <- sample(nrow(res_weighted$data), min(5000, nrow(res_weighted$data)))
sample_idx_b <- sample(nrow(res_baseline$data), min(5000, nrow(res_baseline$data)))
all_x <- c(res_weighted$data$log_pop_centered[sample_idx_w],
            res_baseline$data$log_pop_centered[sample_idx_b])
all_y <- c(res_weighted$data$log_mass_centered[sample_idx_w],
            res_baseline$data$log_mass_centered[sample_idx_b])
x_lim <- range(all_x, na.rm = TRUE) * c(1.05, 1.05)
y_lim <- range(all_y, na.rm = TRUE) * c(1.05, 1.05)

scatter_panels <- list()
for (nm in names(scatter_list)) {
  r <- scatter_list[[nm]]$res
  clr <- scatter_list[[nm]]$col

  # Subsample for plotting
  set.seed(42)
  plot_data <- r$data
  if (nrow(plot_data) > 10000) {
    plot_data <- plot_data[sample(nrow(plot_data), 10000), ]
  }

  scatter_panels[[nm]] <- ggplot(plot_data, aes(x = log_pop_centered,
                                                 y = log_mass_centered)) +
    geom_point(shape = 21, fill = "#8da0cb", color = "black",
               stroke = 0.1, alpha = 0.15, size = 0.5) +
    geom_abline(slope = r$delta, intercept = 0, color = clr, linewidth = 0.6) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.4) +
    coord_cartesian(xlim = x_lim, ylim = y_lim) +
    annotate("text", x = x_lim[1] + 0.1 * diff(x_lim),
             y = y_lim[2] - 0.05 * diff(y_lim),
             label = sprintf("delta == %.4f", r$delta),
             color = clr, parse = TRUE, hjust = 0, size = 3) +
    annotate("text", x = x_lim[1] + 0.1 * diff(x_lim),
             y = y_lim[2] - 0.12 * diff(y_lim),
             label = sprintf("R^2 == %.4f", r$r2),
             parse = TRUE, hjust = 0, size = 2.8, color = "grey40") +
    annotate("text", x = x_lim[2] - 0.02 * diff(x_lim),
             y = y_lim[1] + 0.03 * diff(y_lim),
             label = sprintf("N = %s", format(r$n_neighborhoods, big.mark = ",")),
             hjust = 1, size = 2.5, color = "grey50") +
    ggtitle(nm) +
    xlab("Log pop (dev. from city mean)") +
    ylab("Log mass (dev. from city mean)") +
    theme_bw(base_size = 9) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(size = 9, face = "bold", hjust = 0.5)
    )
}

p_scatter <- scatter_panels[["Region-Weighted"]] | scatter_panels[["Equal-Weighted"]]

# Assemble full figure
final_figure <- p_scatter / p_forest +
  plot_layout(heights = c(1, 0.7))

outfile_pdf <- file.path(figure_dir,
  paste0("Fig3_Decentered_R7_Weighted_Source_Neighborhood_", Sys.Date(), ".pdf"))
ggsave(outfile_pdf, final_figure, width = 24, height = 20, units = "cm", dpi = 300)
cat("Figure saved:", outfile_pdf, "\n")

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================

cat("\n========================================\n")
cat("ROBUSTNESS ASSESSMENT: Region-Weighted vs Equal-Weighted (Neighborhood R7)\n")
cat("========================================\n")

delta_diff <- res_weighted$delta - res_baseline$delta
cat(sprintf("Region-weighted delta: %.4f [%.4f, %.4f]\n",
            res_weighted$delta, res_weighted$ci_low, res_weighted$ci_high))
cat(sprintf("Equal-weighted delta:  %.4f [%.4f, %.4f]\n",
            res_baseline$delta, res_baseline$ci_low, res_baseline$ci_high))
cat(sprintf("Difference:            %.5f (%.3f%%)\n",
            delta_diff, abs(delta_diff) / res_baseline$delta * 100))

# Check if CIs overlap
ci_overlap <- !(res_weighted$ci_high < res_baseline$ci_low |
                res_weighted$ci_low > res_baseline$ci_high)
cat(sprintf("95%% CIs overlap:       %s\n", ifelse(ci_overlap, "YES", "NO")))

cat(sprintf("Both sublinear:        %s\n",
            ifelse(res_weighted$delta < 1 & res_baseline$delta < 1, "YES", "NO")))

# Individual source range
source_deltas <- sapply(res_sources, function(r) unname(r$delta))
cat(sprintf("\nIndividual source deltas: Esch=%.4f, Li=%.4f, Liu=%.4f\n",
            source_deltas[["Esch2022"]], source_deltas[["Li2022"]],
            source_deltas[["Liu2024"]]))
cat(sprintf("Source range: %.4f (%.2f%% of mean)\n",
            diff(range(source_deltas)),
            diff(range(source_deltas)) / mean(source_deltas) * 100))

cat("\nConclusion: ")
if (ci_overlap && abs(delta_diff) < 0.02) {
  cat("ROBUST - region-specific weighting does not materially change scaling exponent\n")
} else if (ci_overlap) {
  cat("MINOR SENSITIVITY - small but measurable shift, CIs still overlap\n")
} else {
  cat("SENSITIVE - region-specific weighting significantly affects scaling exponent\n")
}
cat("========================================\n")

# =============================================================================
# REGION-LEVEL BREAKDOWN
# =============================================================================

cat("\n=== Region-Level Breakdown ===\n")

region_summary <- d7 %>%
  group_by(world_region) %>%
  summarize(
    n_neighborhoods = n(),
    n_cities = n_distinct(ID_HDC_G0),
    n_countries = n_distinct(CTR_MN_ISO),
    mean_w_esch = mean(w_Esch2022, na.rm = TRUE),
    mean_w_li = mean(w_Li2022, na.rm = TRUE),
    mean_w_liu = mean(w_Liu2024, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(desc(n_neighborhoods))

cat(sprintf("%-30s %8s %6s %5s   w_Esch  w_Li  w_Liu\n",
            "Region", "Nbhd", "Cities", "Cntrs"))
for (i in seq_len(nrow(region_summary))) {
  row <- region_summary[i, ]
  cat(sprintf("%-30s %8d %6d %5d   %.3f  %.3f  %.3f\n",
              row$world_region, row$n_neighborhoods, row$n_cities,
              row$n_countries, row$mean_w_esch, row$mean_w_li, row$mean_w_liu))
}

cat("\nOutputs saved to:", figure_dir, "\n")
cat("Done.\n")
