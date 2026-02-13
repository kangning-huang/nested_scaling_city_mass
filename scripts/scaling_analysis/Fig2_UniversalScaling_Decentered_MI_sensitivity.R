# =============================================================================
# Fig2_UniversalScaling_Decentered_MI_sensitivity.R
#
# Compare 3 MI approaches for building mass estimation, then run de-centered
# scaling analysis (Bettencourt & Lobo 2016) on each.
#
# Three MI approaches:
#   1. Global Average (Haberl "Rest of the world" MI for all cities)
#   2. Baseline (Haberl 5-region MI, the current MasterMass approach)
#   3. Fishman 2024 RASMI (32-region, read from Excel)
#
# Key: Only building MI changes. Road + other pavement mass stays constant.
#
# IMPORTANT: Fishman MI values are per floor-area (kg/m2), while Haberl MI
# values are per building volume (kg/m3). We divide Fishman values by 3
# (assuming 3m average floor height) to convert to kg/m3.
#
# Author: Claude Code
# Date: 2026-02-02
# =============================================================================

rm(list = ls())

# Suppress default Rplots.pdf output when running non-interactively
if (!interactive()) pdf(NULL)

# =============================================================================
# 1. Packages
# =============================================================================
library(pacman)
p_load(tidyverse, readxl, patchwork, scales, broom, ggrepel, RColorBrewer)

# =============================================================================
# 2. Paths
# =============================================================================
# Determine base directory
if (interactive() && requireNamespace("rstudioapi", quietly = TRUE)) {
  SCRIPT_DIR <- dirname(rstudioapi::getActiveDocumentContext()$path)
} else {
  args <- commandArgs(trailingOnly = FALSE)
  script_path <- sub("--file=", "", args[grep("--file=", args)])
  if (length(script_path) > 0) {
    SCRIPT_DIR <- dirname(normalizePath(script_path))
  } else {
    SCRIPT_DIR <- getwd()
  }
}

# _Revision1 root
REV_DIR <- dirname(SCRIPT_DIR)
# Project root (0.CleanProject_GlobalScaling)
PROJ_DIR <- dirname(REV_DIR)

cat("Script directory:", SCRIPT_DIR, "\n")
cat("Revision directory:", REV_DIR, "\n")
cat("Project directory:", PROJ_DIR, "\n\n")

# =============================================================================
# 3. Load Data
# =============================================================================

# Baseline MasterMass (for mobility_mass_tons and baseline total)
master_path <- file.path(PROJ_DIR, "data", "processed", "MasterMass_ByClass20250616.csv")
DF_MasterMass <- read.csv(master_path, stringsAsFactors = FALSE)
cat("Loaded MasterMass:", nrow(DF_MasterMass), "cities\n")

# Merged building/road/pavement volumes (for raw volumes by class)
merged_path <- file.path(REV_DIR, "data", "merged_building_road_otherpavement.csv")
DF_Merged <- read.csv(merged_path, stringsAsFactors = FALSE)
cat("Loaded merged data:", nrow(DF_Merged), "cities\n")

# Join to get mobility_mass_tons alongside raw volumes
# MasterMass has 3757 cities; merged has 13135. All master IDs are in merged.
DF_work <- DF_Merged %>%
  inner_join(
    DF_MasterMass %>% select(ID_HDC_G0, mobility_mass_tons, total_built_mass_tons,
                             BuildingMass_AverageTotal),
    by = "ID_HDC_G0"
  )
cat("Joined dataset:", nrow(DF_work), "cities\n\n")

# =============================================================================
# 4. Approach 1: Global Average MI (Haberl "Rest of the World")
# =============================================================================
# These are the "Rest of the world" MI values from Haberl et al. (kg/m3)
# For LW and HR, only "Rest of the world" values exist outside North America
MI_GLOBAL <- c(RS = 299.45, RM = 394.77, NR = 367.81, LW = 154.20, HR = 329.98)

cat("=== Approach 1: Global Average MI (Haberl ROW) ===\n")
cat("MI values (kg/m3):\n")
print(MI_GLOBAL)

# Calculate building mass for each source
calc_building_mass_global <- function(df, source) {
  classes <- c("LW", "RS", "RM", "HR", "NR")
  mass <- rep(0, nrow(df))
  for (cls in classes) {
    vol_col <- paste0("vol_", source, "_", cls)
    if (vol_col %in% colnames(df)) {
      vol <- df[[vol_col]]
      vol[is.na(vol)] <- 0
      mass <- mass + vol * MI_GLOBAL[cls] / 1000  # kg/m3 * m3 / 1000 = tonnes
    }
  }
  return(mass)
}

DF_work$BldgMass_GlobalAvg_Esch <- calc_building_mass_global(DF_work, "Esch2022")
DF_work$BldgMass_GlobalAvg_Li   <- calc_building_mass_global(DF_work, "Li2022")
DF_work$BldgMass_GlobalAvg_Liu  <- calc_building_mass_global(DF_work, "Liu2024")

# Average across non-zero sources (vectorized)
m_esch <- DF_work$BldgMass_GlobalAvg_Esch
m_li   <- DF_work$BldgMass_GlobalAvg_Li
m_liu  <- DF_work$BldgMass_GlobalAvg_Liu
n_pos <- (m_esch > 0) + (m_li > 0) + (m_liu > 0)
DF_work$BldgMass_GlobalAvg <- ifelse(n_pos > 0, (m_esch + m_li + m_liu) / n_pos, 0)

DF_work$total_GlobalAvg <- DF_work$BldgMass_GlobalAvg + DF_work$mobility_mass_tons
cat("Approach 1 computed. Non-zero cities:", sum(DF_work$total_GlobalAvg > 0, na.rm = TRUE), "\n\n")

# =============================================================================
# 5. Approach 2: Baseline (Haberl 5-Region MI)
# =============================================================================
# The baseline is already in MasterMass as total_built_mass_tons
# We just use that directly.
DF_work$total_Baseline <- DF_work$total_built_mass_tons
cat("=== Approach 2: Baseline (Haberl 5-Region MI) ===\n")
cat("Using existing MasterMass total_built_mass_tons directly.\n")
cat("Non-zero cities:", sum(DF_work$total_Baseline > 0, na.rm = TRUE), "\n\n")

# =============================================================================
# 6. Country ISO -> R5_32 Region Mapping (for Fishman 2024)
# =============================================================================
# Official SSP 32-region (R5_32) mapping from IIASA MESSAGE-GLOBIOM
# Source: github.com/iiasa/message-ix-models  data/node/R32.yaml
# Saved as CSV lookup table for reproducibility

r5_32_lookup_path <- file.path(REV_DIR, "data", "R32_ISO_to_region_lookup.csv")
r5_32_lookup <- read.csv(r5_32_lookup_path, stringsAsFactors = FALSE)
cat("Loaded R5_32 lookup:", nrow(r5_32_lookup), "ISO codes ->",
    length(unique(r5_32_lookup$R5_32)), "regions\n")

# Assign R5_32 regions to working dataframe via join
DF_work <- DF_work %>%
  left_join(r5_32_lookup %>% select(ISO3, R5_32), by = c("CTR_MN_ISO" = "ISO3"))

# Report unmapped ISOs
unmapped <- unique(DF_work$CTR_MN_ISO[is.na(DF_work$R5_32)])
if (length(unmapped) > 0) {
  cat("WARNING: Unmapped ISO codes (will use global fallback):", paste(unmapped, collapse = ", "), "\n")
} else {
  cat("All ISO codes mapped to R5_32 regions.\n")
}
cat("Cities per R5_32 region:\n")
print(sort(table(DF_work$R5_32), decreasing = TRUE))

# =============================================================================
# 7. Approach 3: Fishman 2024 RASMI MI
# =============================================================================
cat("\n=== Approach 3: Fishman 2024 RASMI ===\n")

# 7a. Read EO-to-RASMI lookup
lookup_path <- file.path(REV_DIR, "data", "MI_values_Fishman2024", "EO_to_RASMI_lookup.csv")
eo_lookup <- read.csv(lookup_path, stringsAsFactors = FALSE)
cat("EO-to-RASMI lookup:", nrow(eo_lookup), "regions\n")

# Reshape lookup to long format: (R5_32, EO_class) -> RASMI_code
eo_lookup_long <- eo_lookup %>%
  pivot_longer(cols = starts_with("EO_"),
               names_to = "EO_class",
               values_to = "RASMI_code") %>%
  mutate(EO_class = sub("^EO_", "", EO_class))
cat("Lookup entries:", nrow(eo_lookup_long), "\n")

# 7b. Parse RASMI codes into (function, structure) for Fishman DB
# RASMI codes: RSF_RC -> (RS, C), RSF_MAS -> (RS, M), RSF_TIM -> (RS, T),
#              RMF_RC -> (RM, C), RMF_MAS -> (RM, M),
#              NR_RC  -> (NR, C), NR_STL  -> (NR, S), NR_MAS  -> (NR, M)
rasmi_to_func_struct <- tribble(
  ~RASMI_code, ~func, ~struct,
  "RSF_RC",  "RS", "C",
  "RSF_MAS", "RS", "M",
  "RSF_TIM", "RS", "T",
  "RMF_RC",  "RM", "C",
  "RMF_MAS", "RM", "M",
  "NR_RC",   "NR", "C",
  "NR_STL",  "NR", "S",
  "NR_MAS",  "NR", "M"
)

eo_lookup_long <- eo_lookup_long %>%
  left_join(rasmi_to_func_struct, by = "RASMI_code")

# 7c. Read all 8 material sheets and compute total MI per (function, structure, R5_32)
mi_excel_path <- file.path(REV_DIR, "data", "MI_values_Fishman2024", "MI_data_20230905.xlsx")
materials <- c("concrete", "brick", "wood", "steel", "glass", "plastics", "aluminum", "copper")

cat("Reading Fishman MI data from", length(materials), "material sheets...\n")

# For each material sheet, compute median MI at iteration 0 for each (function, structure, R5_32)
mi_by_material <- list()

for (mat in materials) {
  df_mat <- read_excel(mi_excel_path, sheet = mat)

  # Filter to iteration 0 (primary data); fall back to lowest available iteration
  df_mat_clean <- df_mat %>%
    filter(!is.na(.data[["function"]]), !is.na(structure), !is.na(R5_32)) %>%
    rename(func = `function`, mi_value = !!mat)

  # For each (func, struct, R5_32), prefer iteration 0, else lowest available
  df_median <- df_mat_clean %>%
    group_by(func, structure, R5_32) %>%
    arrange(increment_iterations) %>%
    filter(increment_iterations == min(increment_iterations)) %>%
    summarise(
      median_mi = median(mi_value, na.rm = TRUE),
      n_obs = sum(!is.na(mi_value)),
      .groups = "drop"
    ) %>%
    rename(!!paste0("mi_", mat) := median_mi,
           !!paste0("n_", mat) := n_obs)

  mi_by_material[[mat]] <- df_median
}

# Merge all materials into one table
mi_total <- mi_by_material[[1]] %>% select(func, structure, R5_32, starts_with("mi_"))

for (i in 2:length(materials)) {
  mi_total <- mi_total %>%
    full_join(
      mi_by_material[[i]] %>% select(func, structure, R5_32, starts_with("mi_")),
      by = c("func", "structure", "R5_32")
    )
}

# Sum across materials (treating NA as 0)
mi_cols <- paste0("mi_", materials)
mi_total$total_MI_kgm2 <- rowSums(mi_total[, mi_cols], na.rm = TRUE)

# Convert from kg/m2 (floor area) to kg/m3 (volume) by dividing by 3m floor height
mi_total$total_MI_kgm3 <- mi_total$total_MI_kgm2 / 3.0

cat("Fishman MI lookup computed:", nrow(mi_total), "entries\n")

# Print some example values
cat("\nExample Fishman MI values (kg/m3, after floor-to-volume conversion):\n")
examples <- mi_total %>%
  filter((func == "NR" & structure == "C" & R5_32 == "ASIA_CHN") |
         (func == "RS" & structure == "T" & R5_32 == "OECD_USA") |
         (func == "RM" & structure == "C" & R5_32 == "OECD_EU15") |
         (func == "RS" & structure == "M" & R5_32 == "ASIA_IDN"))
print(examples %>% select(func, structure, R5_32, total_MI_kgm2, total_MI_kgm3))

# 7d. Build final lookup: (R5_32, EO_class) -> MI in kg/m3
#
# Strategy for missing combos:
#   1. Look up (target_R5_32, func, struct) in Fishman MI data
#   2. If missing, try same (func, struct) across all regions and use median
#   3. If still missing, fall back to Haberl global average for that EO class

fishman_lookup <- eo_lookup_long %>%
  left_join(
    mi_total %>% select(func, structure, R5_32, total_MI_kgm3),
    by = c("func" = "func", "struct" = "structure", "R5_32" = "R5_32")
  )

# For regions in lookup but not in Fishman data (12 of 32), find proxy
# Proxy mapping: use closest available region
proxy_regions <- c(
  "ASIA_OAS-CPA" = "ASIA_OAS-M",
  "ASIA_PAK"     = "ASIA_IND",
  "LAM_LAM-L"    = "LAM_LAM-M",
  "LAM_MEX"      = "LAM_LAM-M",
  "MAF_NAF"      = "MAF_MEA-M",
  "MAF_SAF"      = "MAF_SSA-L",
  "MAF_SSA-M"    = "MAF_SSA-L",
  "OECD_EEU"     = "OECD_EU12-H",
  "OECD_EU12-M"  = "OECD_EU12-H",
  "REF_CAS"      = "OECD_TUR",
  "REF_EEU-FSU"  = "OECD_EU12-H",
  "REF_RUS"      = "OECD_EU12-H"
)

# Fill in missing MI values using proxy regions
for (i in seq_len(nrow(fishman_lookup))) {
  if (is.na(fishman_lookup$total_MI_kgm3[i])) {
    region <- fishman_lookup$R5_32[i]
    func_val <- fishman_lookup$func[i]
    struct_val <- fishman_lookup$struct[i]

    # Try proxy region
    if (region %in% names(proxy_regions)) {
      proxy <- proxy_regions[region]
      proxy_mi <- mi_total$total_MI_kgm3[
        mi_total$func == func_val &
        mi_total$structure == struct_val &
        mi_total$R5_32 == proxy
      ]
      if (length(proxy_mi) > 0 && !is.na(proxy_mi[1])) {
        fishman_lookup$total_MI_kgm3[i] <- proxy_mi[1]
        next
      }
    }

    # Try global median for this (func, struct)
    global_median <- median(
      mi_total$total_MI_kgm3[mi_total$func == func_val & mi_total$structure == struct_val],
      na.rm = TRUE
    )
    if (!is.na(global_median)) {
      fishman_lookup$total_MI_kgm3[i] <- global_median
    } else {
      # Ultimate fallback: Haberl global average
      fishman_lookup$total_MI_kgm3[i] <- MI_GLOBAL[fishman_lookup$EO_class[i]]
    }
  }
}

# Report coverage
n_filled <- sum(!is.na(fishman_lookup$total_MI_kgm3))
cat("\nFishman lookup coverage:", n_filled, "/", nrow(fishman_lookup),
    "entries filled\n")

# 7e. Apply Fishman MI to each city
#
# For each city, look up its R5_32 region, then for each EO class get the MI,
# compute building mass = volume * MI / 1000

calc_building_mass_fishman <- function(df, source, fishman_lkp, mi_global_fallback) {
  classes <- c("LW", "RS", "RM", "HR", "NR")
  mass <- rep(0, nrow(df))

  for (cls in classes) {
    vol_col <- paste0("vol_", source, "_", cls)
    if (!(vol_col %in% colnames(df))) next

    vol <- df[[vol_col]]
    vol[is.na(vol)] <- 0

    # Look up MI for each city's region
    mi_values <- rep(mi_global_fallback[cls], nrow(df))

    for (j in seq_len(nrow(df))) {
      r5 <- df$R5_32[j]
      if (is.na(r5)) next

      mi_match <- fishman_lkp$total_MI_kgm3[
        fishman_lkp$R5_32 == r5 & fishman_lkp$EO_class == cls
      ]
      if (length(mi_match) > 0 && !is.na(mi_match[1])) {
        mi_values[j] <- mi_match[1]
      }
    }

    mass <- mass + vol * mi_values / 1000
  }

  return(mass)
}

# Vectorized version for efficiency
calc_building_mass_fishman_vec <- function(df, source, fishman_lkp, mi_global_fallback) {
  classes <- c("LW", "RS", "RM", "HR", "NR")
  mass <- rep(0, nrow(df))

  # Pre-build a fast lookup: key = paste(R5_32, EO_class) -> MI value
  lkp_key <- paste(fishman_lkp$R5_32, fishman_lkp$EO_class, sep = "||")
  lkp_val <- fishman_lkp$total_MI_kgm3
  names(lkp_val) <- lkp_key

  for (cls in classes) {
    vol_col <- paste0("vol_", source, "_", cls)
    if (!(vol_col %in% colnames(df))) next

    vol <- df[[vol_col]]
    vol[is.na(vol)] <- 0

    keys <- paste(df$R5_32, cls, sep = "||")
    mi_values <- lkp_val[keys]
    mi_values[is.na(mi_values)] <- mi_global_fallback[cls]

    mass <- mass + vol * as.numeric(mi_values) / 1000
  }

  return(mass)
}

cat("Calculating Fishman building mass...\n")
DF_work$BldgMass_Fishman_Esch <- calc_building_mass_fishman_vec(DF_work, "Esch2022", fishman_lookup, MI_GLOBAL)
DF_work$BldgMass_Fishman_Li   <- calc_building_mass_fishman_vec(DF_work, "Li2022", fishman_lookup, MI_GLOBAL)
DF_work$BldgMass_Fishman_Liu  <- calc_building_mass_fishman_vec(DF_work, "Liu2024", fishman_lookup, MI_GLOBAL)

# Average across non-zero sources (vectorized)
m_esch <- DF_work$BldgMass_Fishman_Esch
m_li   <- DF_work$BldgMass_Fishman_Li
m_liu  <- DF_work$BldgMass_Fishman_Liu
n_pos <- (m_esch > 0) + (m_li > 0) + (m_liu > 0)
DF_work$BldgMass_Fishman <- ifelse(n_pos > 0, (m_esch + m_li + m_liu) / n_pos, 0)

DF_work$total_Fishman <- DF_work$BldgMass_Fishman + DF_work$mobility_mass_tons
cat("Approach 3 computed. Non-zero cities:", sum(DF_work$total_Fishman > 0, na.rm = TRUE), "\n\n")

# =============================================================================
# 8. Compare building mass distributions
# =============================================================================
cat("=== Building Mass Comparison (building component only) ===\n")
cat(sprintf("  Global Avg median: %s tonnes\n",
            format(median(DF_work$BldgMass_GlobalAvg[DF_work$BldgMass_GlobalAvg > 0]), big.mark = ",")))
cat(sprintf("  Baseline median:   %s tonnes\n",
            format(median(DF_work$BuildingMass_AverageTotal[DF_work$BuildingMass_AverageTotal > 0], na.rm = TRUE), big.mark = ",")))
cat(sprintf("  Fishman median:    %s tonnes\n",
            format(median(DF_work$BldgMass_Fishman[DF_work$BldgMass_Fishman > 0]), big.mark = ",")))

# Correlation between approaches
cat("\nCorrelation (log10 building mass) between approaches:\n")
valid <- DF_work$BldgMass_GlobalAvg > 0 & DF_work$BuildingMass_AverageTotal > 0 &
         DF_work$BldgMass_Fishman > 0
cat(sprintf("  Global vs Baseline: r = %.4f\n",
            cor(log10(DF_work$BldgMass_GlobalAvg[valid]),
                log10(DF_work$BuildingMass_AverageTotal[valid]))))
cat(sprintf("  Global vs Fishman:  r = %.4f\n",
            cor(log10(DF_work$BldgMass_GlobalAvg[valid]),
                log10(DF_work$BldgMass_Fishman[valid]))))
cat(sprintf("  Baseline vs Fishman: r = %.4f\n",
            cor(log10(DF_work$BuildingMass_AverageTotal[valid]),
                log10(DF_work$BldgMass_Fishman[valid]))))

# =============================================================================
# 9. De-centered OLS for all 3 approaches
# =============================================================================

run_decentered_ols <- function(df, mass_col, pop_col = "population_2015",
                               country_col = "CTR_MN_NM", min_cities = 5) {
  # Filter to positive mass and countries with enough cities
  df_clean <- df %>%
    filter(.data[[mass_col]] > 0, .data[[pop_col]] > 0)

  city_counts <- table(df_clean[[country_col]])
  valid_countries <- names(city_counts[city_counts >= min_cities])
  df_clean <- df_clean %>% filter(.data[[country_col]] %in% valid_countries)

  n_cities <- nrow(df_clean)
  n_countries <- length(valid_countries)

  # De-center by country
  df_clean <- df_clean %>%
    mutate(
      log_pop = log10(.data[[pop_col]]),
      log_mass = log10(.data[[mass_col]])
    ) %>%
    group_by(.data[[country_col]]) %>%
    mutate(
      log_pop_centered = log_pop - mean(log_pop),
      log_mass_centered = log_mass - mean(log_mass)
    ) %>%
    ungroup()

  # OLS on pooled de-centered data
  mod <- lm(log_mass_centered ~ log_pop_centered, data = df_clean)
  s <- summary(mod)

  beta <- coef(mod)[2]
  se <- s$coefficients[2, 2]
  r2 <- s$r.squared
  ci_lower <- beta - 1.96 * se
  ci_upper <- beta + 1.96 * se
  intercept <- coef(mod)[1]

  list(
    beta = beta,
    se = se,
    r2 = r2,
    ci_lower = ci_lower,
    ci_upper = ci_upper,
    intercept = intercept,
    n_cities = n_cities,
    n_countries = n_countries,
    data = df_clean
  )
}

cat("\n=== De-centered OLS Results ===\n\n")

res_global  <- run_decentered_ols(DF_work, "total_GlobalAvg")
res_baseline <- run_decentered_ols(DF_work, "total_Baseline")
res_fishman <- run_decentered_ols(DF_work, "total_Fishman")

# Build comparison table
comparison <- tibble(
  Approach = c("Global Average (Haberl ROW)",
               "Baseline (Haberl 5-Region)",
               "Fishman 2024 RASMI (32-Region)"),
  Beta = c(res_global$beta, res_baseline$beta, res_fishman$beta),
  SE = c(res_global$se, res_baseline$se, res_fishman$se),
  CI_lower = c(res_global$ci_lower, res_baseline$ci_lower, res_fishman$ci_lower),
  CI_upper = c(res_global$ci_upper, res_baseline$ci_upper, res_fishman$ci_upper),
  R2 = c(res_global$r2, res_baseline$r2, res_fishman$r2),
  N_cities = c(res_global$n_cities, res_baseline$n_cities, res_fishman$n_cities),
  N_countries = c(res_global$n_countries, res_baseline$n_countries, res_fishman$n_countries)
)

cat("Comparison Table:\n")
print(comparison, width = 120)

# =============================================================================
# 10. Comparison Figure
# =============================================================================
cat("\n=== Generating Comparison Figure ===\n")

# Common theme for scatter plots
theme_scatter <- function() {
  theme_bw() +
    theme(
      axis.ticks = element_line(colour = "black", linewidth = 0.1),
      axis.line = element_line(colour = "black", linewidth = 0.5),
      text = element_text(size = 9),
      axis.text = element_text(size = 8),
      panel.grid.major.y = element_line(linewidth = 0.2, color = "grey90", linetype = 2),
      panel.grid.major.x = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      plot.title = element_text(size = 10, face = "bold")
    )
}

# Determine common axis limits across all 3 approaches
all_x <- c(res_global$data$log_pop_centered, res_baseline$data$log_pop_centered,
           res_fishman$data$log_pop_centered)
all_y <- c(res_global$data$log_mass_centered, res_baseline$data$log_mass_centered,
           res_fishman$data$log_mass_centered)
x_lim <- range(all_x, na.rm = TRUE)
y_lim <- range(all_y, na.rm = TRUE)

make_scatter <- function(dc_data, result, title_text) {
  ggplot(dc_data, aes(x = log_pop_centered, y = log_mass_centered)) +
    geom_point(
      shape = 21, color = "black", fill = "#8da0cb",
      stroke = 0.3, alpha = 0.5, size = 1.2
    ) +
    stat_smooth(method = "lm", se = FALSE, color = "#fc8d62", linewidth = 0.5) +
    geom_abline(intercept = 0, slope = 1, linetype = 2, linewidth = 0.5, color = "black") +
    coord_cartesian(xlim = x_lim, ylim = y_lim) +
    scale_x_continuous(
      name = "Log population (deviation from country mean)",
      expand = c(0.02, 0.02)
    ) +
    scale_y_continuous(
      name = "Log mass (deviation from country mean)",
      expand = c(0.02, 0.02)
    ) +
    theme_scatter() +
    ggtitle(title_text) +
    annotate("text",
             x = x_lim[1] + 0.3 * diff(x_lim),
             y = y_lim[2] - 0.05 * diff(y_lim),
             size = 3, hjust = 0,
             label = sprintf("beta==%.3f~~(n==%d)", round(result$beta, 3), result$n_cities),
             parse = TRUE) +
    annotate("text",
             x = x_lim[1] + 0.3 * diff(x_lim),
             y = y_lim[2] - 0.12 * diff(y_lim),
             size = 3, hjust = 0,
             label = sprintf("R^2==%.3f", round(result$r2, 3)),
             parse = TRUE) +
    annotate("text",
             x = x_lim[1] + 0.3 * diff(x_lim),
             y = y_lim[2] - 0.19 * diff(y_lim),
             size = 2.5, hjust = 0, color = "grey40",
             label = sprintf("95%% CI: [%.3f, %.3f]", result$ci_lower, result$ci_upper))
}

p1 <- make_scatter(res_global$data, res_global,
                   "A. Global Average MI (Haberl ROW)")
p2 <- make_scatter(res_baseline$data, res_baseline,
                   "B. Baseline (Haberl 5-Region)")
p3 <- make_scatter(res_fishman$data, res_fishman,
                   "C. Fishman 2024 RASMI (32-Region)")

# Forest plot of beta estimates with CIs
forest_data <- comparison %>%
  mutate(
    Approach_short = c("Global Average", "Baseline\n(5-Region)", "Fishman RASMI\n(32-Region)"),
    Approach_short = factor(Approach_short, levels = rev(Approach_short))
  )

p_forest <- ggplot(forest_data, aes(x = Beta, y = Approach_short)) +
  geom_vline(xintercept = 1, linetype = 2, color = "black", linewidth = 0.5) +
  geom_errorbarh(aes(xmin = CI_lower, xmax = CI_upper),
                 height = 0.2, linewidth = 0.5, color = "grey30") +
  geom_point(size = 3, color = "#fc8d62") +
  scale_x_continuous(
    name = expression("Scaling exponent ("*beta*")"),
    expand = c(0.05, 0)
  ) +
  ylab(NULL) +
  theme_bw() +
  theme(
    text = element_text(size = 9),
    axis.text = element_text(size = 8),
    panel.grid.major.x = element_line(linewidth = 0.2, color = "grey90"),
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    plot.title = element_text(size = 10, face = "bold")
  ) +
  ggtitle("D. Comparison of Scaling Exponents (95% CI)")

# Assemble
top_row <- (p1 | p2 | p3) + plot_layout(ncol = 3)
final_figure <- (top_row / p_forest) +
  plot_layout(heights = c(3, 1.2))

print(final_figure)

# Save
current_date <- Sys.Date()
fig_dir <- file.path(REV_DIR, "figures")
if (!dir.exists(fig_dir)) dir.create(fig_dir, recursive = TRUE)

ggsave(
  filename = file.path(fig_dir, paste0("Fig2_MI_Sensitivity_Decentered_", current_date, ".pdf")),
  plot = final_figure,
  device = "pdf",
  width = 36,
  height = 16,
  units = "cm",
  dpi = 300
)
cat("Figure saved to:", file.path(fig_dir, paste0("Fig2_MI_Sensitivity_Decentered_", current_date, ".pdf")), "\n")

# =============================================================================
# 11. Save Summary
# =============================================================================
output_dir <- file.path(REV_DIR, "data")
if (!dir.exists(output_dir)) dir.create(output_dir, recursive = TRUE)

write_csv(comparison,
          file.path(output_dir, paste0("MI_sensitivity_decentered_comparison_", current_date, ".csv")))
cat("Summary saved to:", file.path(output_dir, paste0("MI_sensitivity_decentered_comparison_", current_date, ".csv")), "\n")

# =============================================================================
# 12. Summary Statistics
# =============================================================================
cat("\n", paste(rep("=", 80), collapse = ""), "\n")
cat("SUMMARY FOR REVISION LETTER\n")
cat(paste(rep("=", 80), collapse = ""), "\n\n")

cat("Material Intensity Sensitivity Analysis (De-centered OLS)\n")
cat("=========================================================\n\n")

for (i in seq_len(nrow(comparison))) {
  cat(sprintf("Approach %d: %s\n", i, comparison$Approach[i]))
  cat(sprintf("  Beta: %.4f [%.4f, %.4f]\n", comparison$Beta[i],
              comparison$CI_lower[i], comparison$CI_upper[i]))
  cat(sprintf("  SE: %.4f, R2: %.4f\n", comparison$SE[i], comparison$R2[i]))
  cat(sprintf("  N cities: %d, N countries: %d\n\n",
              comparison$N_cities[i], comparison$N_countries[i]))
}

# Range and variability
beta_range <- range(comparison$Beta)
beta_mean <- mean(comparison$Beta)
beta_sd <- sd(comparison$Beta)
beta_cv <- beta_sd / beta_mean * 100

cat(sprintf("Beta range: [%.4f, %.4f]\n", beta_range[1], beta_range[2]))
cat(sprintf("Beta mean: %.4f, SD: %.4f, CV: %.2f%%\n", beta_mean, beta_sd, beta_cv))
cat(sprintf("Max difference from baseline: %.4f\n",
            max(abs(comparison$Beta - comparison$Beta[2]))))

cat("\nNote: Fishman 2024 MI values are per floor-area (kg/m2).\n")
cat("Converted to per-volume (kg/m3) by dividing by 3 (3m floor height).\n")
cat(paste(rep("=", 80), collapse = ""), "\n")
