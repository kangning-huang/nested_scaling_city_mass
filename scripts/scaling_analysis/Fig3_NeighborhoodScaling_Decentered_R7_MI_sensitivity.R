# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - R7 Per-Resolution - MI Sensitivity
# =============================================================================
# Tests 3 MI approaches at R7 using per-resolution filtering (not R6 filter).
#
# MI Approaches:
#   1. Global Average (Haberl "Rest of the world")
#   2. Baseline (Haberl 5-Region, current approach)
#   3. Fishman 2024 RASMI (32-Region from SSP)
#
# Per-resolution filter at R7:
#   - pop >= 1, mass > 0
#   - City total pop > 50,000
#   - >= 10 neighborhoods per city
#   - >= 5 qualifying cities per country
#
# Output: figures/Table_Decentered_R7_PerRes_MI_Sensitivity.csv
#
# IMPORTANT: Fishman MI values are per floor-area (kg/m2).
# Converted to per-volume (kg/m3) by dividing by 3 (3m floor height).
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(readr, readxl, dplyr, tibble, tidyr, ggplot2, scales, patchwork)

# =============================================================================
# SETUP
# =============================================================================

rev_dir <- normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
data_dir <- file.path(rev_dir, "data")
figure_dir <- file.path(rev_dir, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

merged_file <- file.path(data_dir, "Fig3_Merged_Neighborhood_H3_Resolution7_2026-02-01.csv")
mass_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")

cat("=== Fig3 Decentered - R7 Per-Resolution Filter - MI Sensitivity ===\n\n")

# =============================================================================
# MI APPROACH DEFINITIONS (identical to multiscale MI script)
# =============================================================================

MI_GLOBAL <- c(RS = 299.45, RM = 394.77, NR = 367.81, LW = 154.20, HR = 329.98)

MI_5REGION <- list(
  RS = c("North America" = 284.87, "Japan" = 124.93, "China" = 526.00,
         "OECD" = 349.64, "ROW" = 299.45),
  RM = c("North America" = 314.72, "Japan" = 601.60, "China" = 662.06,
         "OECD" = 398.60, "ROW" = 394.77),
  NR = c("North America" = 280.82, "Japan" = 272.90, "China" = 654.06,
         "OECD" = 375.55, "ROW" = 367.81),
  LW = c("North America" = 151.30, "ROW" = 154.20),
  HR = c("North America" = 312.35, "ROW" = 329.98)
)

OECD_ISOS <- c("AUS", "AUT", "BEL", "CHL", "CZE", "DNK", "EST", "FIN", "FRA",
               "DEU", "GRC", "HUN", "ISL", "IRL", "ISR", "ITA", "KOR", "LUX",
               "NLD", "NZL", "NOR", "POL", "PRT", "SVK", "SVN", "ESP", "SWE",
               "CHE", "TUR", "GBR")

assign_haberl_region <- function(iso) {
  case_when(
    iso %in% c("USA", "CAN") ~ "North America",
    iso == "JPN" ~ "Japan",
    iso == "CHN" ~ "China",
    iso %in% OECD_ISOS ~ "OECD",
    TRUE ~ "ROW"
  )
}

get_mi_5region <- function(cls, region) {
  if (cls %in% c("LW", "HR")) {
    region <- ifelse(region == "North America", "North America", "ROW")
  }
  val <- MI_5REGION[[cls]][region]
  ifelse(is.na(val), MI_GLOBAL[cls], val)
}

# --- Fishman 2024 RASMI (32-Region) ---
r5_32_lookup <- read.csv(file.path(data_dir, "R32_ISO_to_region_lookup.csv"),
                         stringsAsFactors = FALSE)

eo_lookup <- read.csv(file.path(data_dir, "MI_values_Fishman2024", "EO_to_RASMI_lookup.csv"),
                      stringsAsFactors = FALSE)
eo_lookup_long <- eo_lookup %>%
  pivot_longer(cols = starts_with("EO_"), names_to = "EO_class", values_to = "RASMI_code") %>%
  mutate(EO_class = sub("^EO_", "", EO_class))

rasmi_map <- tribble(
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
eo_lookup_long <- eo_lookup_long %>% left_join(rasmi_map, by = "RASMI_code")

mi_excel <- file.path(data_dir, "MI_values_Fishman2024", "MI_data_20230905.xlsx")
materials <- c("concrete", "brick", "wood", "steel", "glass", "plastics", "aluminum", "copper")

cat("Reading Fishman MI data...\n")
mi_by_material <- list()
for (mat in materials) {
  df_mat <- read_excel(mi_excel, sheet = mat) %>%
    filter(!is.na(.data[["function"]]), !is.na(structure), !is.na(R5_32)) %>%
    rename(func = `function`, mi_value = !!mat)
  df_median <- df_mat %>%
    group_by(func, structure, R5_32) %>%
    arrange(increment_iterations) %>%
    filter(increment_iterations == min(increment_iterations)) %>%
    summarise(median_mi = median(mi_value, na.rm = TRUE), .groups = "drop") %>%
    rename(!!paste0("mi_", mat) := median_mi)
  mi_by_material[[mat]] <- df_median
}

mi_total <- mi_by_material[[1]] %>% select(func, structure, R5_32, starts_with("mi_"))
for (i in 2:length(materials)) {
  mi_total <- mi_total %>%
    full_join(mi_by_material[[i]] %>% select(func, structure, R5_32, starts_with("mi_")),
              by = c("func", "structure", "R5_32"))
}
mi_cols <- paste0("mi_", materials)
mi_total$total_MI_kgm2 <- rowSums(mi_total[, mi_cols], na.rm = TRUE)
mi_total$total_MI_kgm3 <- mi_total$total_MI_kgm2 / 3.0

fishman_lookup <- eo_lookup_long %>%
  left_join(mi_total %>% select(func, structure, R5_32, total_MI_kgm3),
            by = c("func" = "func", "struct" = "structure", "R5_32" = "R5_32"))

proxy_regions <- c(
  "ASIA_OAS-CPA" = "ASIA_OAS-M", "ASIA_PAK" = "ASIA_IND",
  "LAM_LAM-L" = "LAM_LAM-M", "LAM_MEX" = "LAM_LAM-M",
  "MAF_NAF" = "MAF_MEA-M", "MAF_SAF" = "MAF_SSA-L",
  "MAF_SSA-M" = "MAF_SSA-L", "OECD_EEU" = "OECD_EU12-H",
  "OECD_EU12-M" = "OECD_EU12-H", "REF_CAS" = "OECD_TUR",
  "REF_EEU-FSU" = "OECD_EU12-H", "REF_RUS" = "OECD_EU12-H"
)

for (i in seq_len(nrow(fishman_lookup))) {
  if (is.na(fishman_lookup$total_MI_kgm3[i])) {
    region <- fishman_lookup$R5_32[i]
    func_val <- fishman_lookup$func[i]
    struct_val <- fishman_lookup$struct[i]
    if (region %in% names(proxy_regions)) {
      proxy <- proxy_regions[region]
      proxy_mi <- mi_total$total_MI_kgm3[
        mi_total$func == func_val & mi_total$structure == struct_val & mi_total$R5_32 == proxy]
      if (length(proxy_mi) > 0 && !is.na(proxy_mi[1])) {
        fishman_lookup$total_MI_kgm3[i] <- proxy_mi[1]; next
      }
    }
    global_median <- median(
      mi_total$total_MI_kgm3[mi_total$func == func_val & mi_total$structure == struct_val],
      na.rm = TRUE)
    if (!is.na(global_median)) {
      fishman_lookup$total_MI_kgm3[i] <- global_median
    } else {
      fishman_lookup$total_MI_kgm3[i] <- MI_GLOBAL[fishman_lookup$EO_class[i]]
    }
  }
}

fishman_lkp_key <- paste(fishman_lookup$R5_32, fishman_lookup$EO_class, sep = "||")
fishman_lkp_val <- fishman_lookup$total_MI_kgm3
names(fishman_lkp_val) <- fishman_lkp_key

cat("Fishman lookup ready:", sum(!is.na(fishman_lkp_val)), "/", length(fishman_lkp_val), "entries\n\n")

# =============================================================================
# MASS CALCULATION FUNCTIONS
# =============================================================================

calc_bldg_mass_global <- function(df, source) {
  classes <- c("LW", "RS", "RM", "HR", "NR")
  mass <- rep(0, nrow(df))
  for (cls in classes) {
    vol_col <- paste0("vol_", source, "_", cls)
    if (vol_col %in% colnames(df)) {
      vol <- df[[vol_col]]; vol[is.na(vol)] <- 0
      mass <- mass + vol * MI_GLOBAL[cls] / 1000
    }
  }
  return(mass)
}

calc_bldg_mass_5region <- function(df, source) {
  classes <- c("LW", "RS", "RM", "HR", "NR")
  mass <- rep(0, nrow(df))
  for (cls in classes) {
    vol_col <- paste0("vol_", source, "_", cls)
    if (vol_col %in% colnames(df)) {
      vol <- df[[vol_col]]; vol[is.na(vol)] <- 0
      mi_vals <- mapply(get_mi_5region, cls, df$haberl_region)
      mass <- mass + vol * mi_vals / 1000
    }
  }
  return(mass)
}

calc_bldg_mass_fishman <- function(df, source) {
  classes <- c("LW", "RS", "RM", "HR", "NR")
  mass <- rep(0, nrow(df))
  for (cls in classes) {
    vol_col <- paste0("vol_", source, "_", cls)
    if (!(vol_col %in% colnames(df))) next
    vol <- df[[vol_col]]; vol[is.na(vol)] <- 0
    keys <- paste(df$R5_32, cls, sep = "||")
    mi_vals <- fishman_lkp_val[keys]
    mi_vals[is.na(mi_vals)] <- MI_GLOBAL[cls]
    mass <- mass + vol * as.numeric(mi_vals) / 1000
  }
  return(mass)
}

avg_nonzero <- function(a, b, c) {
  n_pos <- (a > 0) + (b > 0) + (c > 0)
  ifelse(n_pos > 0, (a + b + c) / n_pos, 0)
}

# =============================================================================
# STEP 1: Load R7 data and compute mass under 3 MI approaches
# =============================================================================

cat("=== Loading Resolution 7 data ===\n")

merged <- readr::read_csv(merged_file, show_col_types = FALSE)
cat("  Merged rows:", nrow(merged), "\n")

mass_df <- readr::read_csv(mass_file, show_col_types = FALSE) %>%
  select(h3index, mobility_mass_tons, ID_HDC_G0, CTR_MN_ISO, CTR_MN_NM,
         UC_NM_MN, population_2015)

df <- merged %>%
  inner_join(mass_df %>% select(h3index, mobility_mass_tons), by = "h3index")

cat("  After join:", nrow(df), "neighborhoods\n")

# Assign regions
df$haberl_region <- assign_haberl_region(df$CTR_MN_ISO)
df <- df %>%
  left_join(r5_32_lookup %>% select(ISO3, R5_32), by = c("CTR_MN_ISO" = "ISO3"))

# Compute building mass for each MI approach
for (mi_name in c("Global", "Baseline", "Fishman")) {
  calc_fn <- switch(mi_name,
    "Global"   = calc_bldg_mass_global,
    "Baseline" = calc_bldg_mass_5region,
    "Fishman"  = calc_bldg_mass_fishman
  )
  m_esch <- calc_fn(df, "Esch2022")
  m_li   <- calc_fn(df, "Li2022")
  m_liu  <- calc_fn(df, "Liu2024")
  bldg <- avg_nonzero(m_esch, m_li, m_liu)
  df[[paste0("total_", mi_name)]] <- bldg + df$mobility_mass_tons
}

cat("  Mass computed for 3 MI approaches\n\n")

# =============================================================================
# STEP 2: Apply per-resolution filter at R7
# =============================================================================

cat("=== Applying per-resolution filter at R7 ===\n")

d7 <- df %>%
  filter(population_2015 >= 1, total_Baseline > 0)

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
# STEP 3: Run de-centered OLS for 3 MI approaches
# =============================================================================

run_decentered <- function(df, mi_name, mass_col, filter_city_ids, filter_country_isos) {
  d <- df %>%
    filter(population_2015 >= 1, .data[[mass_col]] > 0,
           ID_HDC_G0 %in% filter_city_ids,
           CTR_MN_ISO %in% filter_country_isos) %>%
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

  cat(sprintf("  %-10s | delta=%.4f [%.4f, %.4f] R2=%.4f N=%d cities=%d countries=%d\n",
              mi_name, delta, delta - 1.96 * se, delta + 1.96 * se,
              s$r.squared, nrow(dc), n_distinct(dc$ID_HDC_G0), n_distinct(dc$CTR_MN_NM)))

  tibble(
    Resolution = 7,
    `Hex Area (km2)` = 5.161,
    MI_approach = mi_name,
    Delta = round(delta, 4),
    SE = round(se, 4),
    CI_low = round(delta - 1.96 * se, 4),
    CI_high = round(delta + 1.96 * se, 4),
    R2 = round(s$r.squared, 4),
    N_neighborhoods = nrow(dc),
    N_cities = n_distinct(dc$ID_HDC_G0),
    N_countries = n_distinct(dc$CTR_MN_NM)
  )
}

cat("=== Running De-centered OLS (3 MI approaches, R7 per-resolution filter) ===\n\n")

results <- list()
for (mi in c("Global", "Baseline", "Fishman")) {
  results[[mi]] <- run_decentered(df, mi, paste0("total_", mi), r7_city_ids, r7_country_isos)
}

summary_df <- bind_rows(results)

cat("\n=== SUMMARY TABLE ===\n")
print(as.data.frame(summary_df), row.names = FALSE)

outfile <- file.path(figure_dir, "Table_Decentered_R7_PerRes_MI_Sensitivity.csv")
write_csv(summary_df, outfile)
cat("\nSaved:", outfile, "\n")

# =============================================================================
# SUMMARY STATS
# =============================================================================

cat("\n========================================\n")
cat("R7 PER-RESOLUTION FILTER MI SENSITIVITY\n")
cat("========================================\n")
cat(sprintf("  Cities: %d, Countries: %d\n", length(r7_city_ids), length(r7_country_isos)))
cat(sprintf("  Delta range: [%.4f, %.4f] (span: %.4f)\n",
            min(summary_df$Delta), max(summary_df$Delta),
            max(summary_df$Delta) - min(summary_df$Delta)))
cat(sprintf("  CV: %.2f%%\n", sd(summary_df$Delta) / mean(summary_df$Delta) * 100))
cat("========================================\n")
