# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - Multiscale - MI Sensitivity
# =============================================================================
# 3 x 3 sensitivity: 3 H3 resolutions (5, 6, 7) x 3 MI approaches
#
# MI Approaches:
#   1. Global Average (Haberl "Rest of the world")
#   2. Baseline (Haberl 5-Region, current approach)
#   3. Fishman 2024 RASMI (32-Region from SSP)
#
# Methodology:
#   - Read merged files with raw building volumes by class
#   - Recalculate building mass with each MI approach
#   - Add constant mobility_mass_tons (road + pavement, unchanged)
#   - Apply R6 filter set for consistent sample across resolutions
#   - De-center by city mean, pooled OLS -> delta
#
# Output: 3x3 scatter grid, summary CSV, forest plot
#
# IMPORTANT: Fishman MI values are per floor-area (kg/m2).
# Converted to per-volume (kg/m3) by dividing by 3 (3m floor height).
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(readr, readxl, dplyr, tibble, tidyr, ggplot2, scales,
               cowplot, patchwork)

# =============================================================================
# SETUP
# =============================================================================

project_root <- rprojroot::find_root(rprojroot::has_file("CLAUDE.md"))
data_dir <- file.path(project_root, "_Revision1", "data")
figure_dir <- file.path(project_root, "_Revision1", "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

# Merged files (with raw building volumes by class)
merged_files <- list(
  "5" = file.path(data_dir, "Fig3_Merged_Neighborhood_H3_Resolution5_2026-02-02.csv"),
  "6" = file.path(data_dir, "Fig3_Merged_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  "7" = file.path(data_dir, "Fig3_Merged_Neighborhood_H3_Resolution7_2026-02-01.csv")
)

# Mass files (for mobility_mass_tons which stays constant)
mass_files <- list(
  "5" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution5_2026-02-02.csv"),
  "6" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  "7" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
)

hex_areas <- c("5" = 252.9, "6" = 36.13, "7" = 5.161)

cat("=== Fig3 Decentered - Multiscale - MI Sensitivity (3x3) ===\n\n")

# =============================================================================
# MI APPROACH DEFINITIONS
# =============================================================================

# --- Approach 1: Global Average (Haberl "Rest of the World") ---
MI_GLOBAL <- c(RS = 299.45, RM = 394.77, NR = 367.81, LW = 154.20, HR = 329.98)

# --- Approach 2: Baseline (Haberl 5-Region) ---
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

# Haberl region assignment
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

# --- Approach 3: Fishman 2024 RASMI (32-Region) ---
# Read R5_32 lookup
r5_32_lookup <- read.csv(file.path(data_dir, "R32_ISO_to_region_lookup.csv"),
                         stringsAsFactors = FALSE)

# Read EO-to-RASMI lookup
eo_lookup <- read.csv(file.path(data_dir, "MI_values_Fishman2024", "EO_to_RASMI_lookup.csv"),
                      stringsAsFactors = FALSE)
eo_lookup_long <- eo_lookup %>%
  pivot_longer(cols = starts_with("EO_"), names_to = "EO_class", values_to = "RASMI_code") %>%
  mutate(EO_class = sub("^EO_", "", EO_class))

# Parse RASMI codes
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

# Read Fishman MI from 8 material sheets
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

# Merge materials
mi_total <- mi_by_material[[1]] %>% select(func, structure, R5_32, starts_with("mi_"))
for (i in 2:length(materials)) {
  mi_total <- mi_total %>%
    full_join(mi_by_material[[i]] %>% select(func, structure, R5_32, starts_with("mi_")),
              by = c("func", "structure", "R5_32"))
}
mi_cols <- paste0("mi_", materials)
mi_total$total_MI_kgm2 <- rowSums(mi_total[, mi_cols], na.rm = TRUE)
mi_total$total_MI_kgm3 <- mi_total$total_MI_kgm2 / 3.0  # floor-area -> volume

# Build Fishman lookup with proxy fallbacks
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

# Pre-build fast lookup key
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
# STEP 1: Load & join data for each resolution, compute mass under 3 MI
# =============================================================================

load_and_compute <- function(res_label) {
  cat(sprintf("=== Loading Resolution %s ===\n", res_label))

  # Read merged (volumes)
  merged <- readr::read_csv(merged_files[[res_label]], show_col_types = FALSE)
  cat("  Merged rows:", nrow(merged), "\n")

  # Read mass (for mobility_mass_tons)
  mass_df <- readr::read_csv(mass_files[[res_label]], show_col_types = FALSE) %>%
    select(h3index, mobility_mass_tons, ID_HDC_G0, CTR_MN_ISO, CTR_MN_NM,
           UC_NM_MN, population_2015)

  # Join: keep only rows present in mass file (those with computed mass)
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
  return(df)
}

all_data <- list()
for (res in c("5", "6", "7")) {
  all_data[[res]] <- load_and_compute(res)
}

# =============================================================================
# STEP 2: Determine R6 filter set (same logic as multiscale script)
# =============================================================================

cat("=== Determining R6 filter set ===\n")

d6 <- all_data[["6"]] %>%
  filter(population_2015 >= 1, total_Baseline > 0)

# Cities with total pop > 50,000
city_pop_r6 <- d6 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(total_pop = sum(population_2015), .groups = "drop") %>%
  filter(total_pop > 50000)
d6 <- d6 %>% filter(ID_HDC_G0 %in% city_pop_r6$ID_HDC_G0)

# Cities with >= 10 neighborhoods
city_nbhd_r6 <- d6 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)
d6 <- d6 %>% filter(ID_HDC_G0 %in% city_nbhd_r6$ID_HDC_G0)

# Countries with >= 5 qualifying cities
country_city_r6 <- d6 %>%
  group_by(CTR_MN_ISO) %>%
  summarize(num_cities = n_distinct(ID_HDC_G0)) %>%
  filter(num_cities >= 5)
d6 <- d6 %>% filter(CTR_MN_ISO %in% country_city_r6$CTR_MN_ISO)

r6_city_ids <- unique(d6$ID_HDC_G0)
r6_country_isos <- unique(d6$CTR_MN_ISO)
cat("R6 filter:", length(r6_city_ids), "cities,", length(r6_country_isos), "countries\n\n")

# =============================================================================
# STEP 3: Run de-centered OLS for all 9 combinations
# =============================================================================

run_decentered <- function(df, res_label, mi_name, mass_col,
                           filter_city_ids, filter_country_isos) {
  # Filter
  d <- df %>%
    filter(population_2015 >= 1, .data[[mass_col]] > 0,
           ID_HDC_G0 %in% filter_city_ids,
           CTR_MN_ISO %in% filter_country_isos) %>%
    mutate(
      log_pop = log10(population_2015),
      log_mass = log10(.data[[mass_col]]),
      Country_City = paste(CTR_MN_NM, ID_HDC_G0, sep = "_")
    )

  # De-center by city mean
  dc <- d %>%
    group_by(Country_City) %>%
    filter(n() >= 2) %>%
    mutate(
      log_pop_centered = log_pop - mean(log_pop),
      log_mass_centered = log_mass - mean(log_mass)
    ) %>%
    ungroup()

  # OLS
  mod <- lm(log_mass_centered ~ log_pop_centered, data = dc)
  s <- summary(mod)
  delta <- coef(mod)[2]
  se <- s$coefficients[2, 2]
  r2 <- s$r.squared

  cat(sprintf("  Res %s | %-10s | delta=%.4f [%.4f, %.4f] R2=%.4f N=%d\n",
              res_label, mi_name, delta, delta - 1.96 * se, delta + 1.96 * se,
              r2, nrow(dc)))

  list(
    Resolution = res_label, MI = mi_name,
    delta = delta, se = se, r2 = r2,
    ci_low = delta - 1.96 * se, ci_high = delta + 1.96 * se,
    n_neighborhoods = nrow(dc),
    n_cities = n_distinct(dc$ID_HDC_G0),
    n_countries = n_distinct(dc$CTR_MN_NM),
    data = dc
  )
}

mi_approaches <- c("Global", "Baseline", "Fishman")
mi_labels <- c("Global" = "Global Avg", "Baseline" = "Haberl 5-Region", "Fishman" = "Fishman RASMI")

cat("=== Running 3x3 De-centered OLS ===\n\n")
results <- list()
for (res in c("5", "6", "7")) {
  for (mi in mi_approaches) {
    key <- paste(res, mi, sep = "_")
    results[[key]] <- run_decentered(
      all_data[[res]], res, mi, paste0("total_", mi),
      r6_city_ids, r6_country_isos
    )
  }
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_df <- bind_rows(lapply(results, function(r) {
  tibble(
    Resolution = r$Resolution,
    `Hex Area (km2)` = hex_areas[r$Resolution],
    MI_approach = r$MI,
    Delta = round(r$delta, 4),
    SE = round(r$se, 4),
    CI_low = round(r$ci_low, 4),
    CI_high = round(r$ci_high, 4),
    R2 = round(r$r2, 4),
    N_neighborhoods = r$n_neighborhoods,
    N_cities = r$n_cities,
    N_countries = r$n_countries
  )
}))

cat("\n=== SUMMARY TABLE (3x3) ===\n")
print(as.data.frame(summary_df), row.names = FALSE)

write_csv(summary_df, file.path(figure_dir, "Table_Decentered_Multiscale_MI_Sensitivity.csv"))
cat("\nSaved:", file.path(figure_dir, "Table_Decentered_Multiscale_MI_Sensitivity.csv"), "\n")

# =============================================================================
# FIGURE: 3x3 Scatter Grid
# =============================================================================

cat("\n=== Generating 3x3 Scatter Grid ===\n")

# Determine common axis limits across all 9 panels
all_x <- unlist(lapply(results, function(r) r$data$log_pop_centered))
all_y <- unlist(lapply(results, function(r) r$data$log_mass_centered))
x_lim <- c(max(min(all_x, na.rm = TRUE), -6), min(max(all_x, na.rm = TRUE), 4))
y_lim <- c(max(min(all_y, na.rm = TRUE), -6), min(max(all_y, na.rm = TRUE), 4))

make_panel <- function(r, show_ylab, show_xlab) {
  p <- ggplot(r$data, aes(x = log_pop_centered, y = log_mass_centered)) +
    geom_point(alpha = 0.03, color = "#8da0cb", size = 0.5) +
    geom_abline(slope = r$delta, intercept = 0, color = "#fc8d62", linewidth = 0.6) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.4) +
    coord_cartesian(xlim = x_lim, ylim = y_lim) +
    annotate("text", x = x_lim[1] + 0.15 * diff(x_lim),
             y = y_lim[2] - 0.06 * diff(y_lim),
             label = sprintf("delta == %.3f", round(r$delta, 3)),
             color = "#fc8d62", parse = TRUE, hjust = 0, size = 2.5) +
    annotate("text", x = x_lim[1] + 0.15 * diff(x_lim),
             y = y_lim[2] - 0.13 * diff(y_lim),
             label = sprintf("95%% CI: [%.3f, %.3f]",
                             round(r$ci_low, 3), round(r$ci_high, 3)),
             color = "#fc8d62", hjust = 0, size = 2) +
    annotate("text", x = x_lim[1] + 0.15 * diff(x_lim),
             y = y_lim[2] - 0.20 * diff(y_lim),
             label = sprintf("R^2 == %.3f", round(r$r2, 3)),
             parse = TRUE, hjust = 0, size = 2.2, color = "grey40") +
    annotate("text", x = x_lim[2] - 0.02 * diff(x_lim),
             y = y_lim[1] + 0.04 * diff(y_lim),
             label = sprintf("N=%s", format(r$n_neighborhoods, big.mark = ",")),
             hjust = 1, size = 2, color = "grey50") +
    theme_bw(base_size = 8) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(size = 7, face = "bold", hjust = 0.5),
      plot.margin = margin(2, 2, 2, 2)
    )

  if (show_xlab) {
    p <- p + xlab("Log pop (dev. from city mean)")
  } else {
    p <- p + xlab(NULL)
  }
  if (show_ylab) {
    p <- p + ylab("Log mass (dev. from city mean)")
  } else {
    p <- p + ylab(NULL)
  }

  return(p)
}

# Build 3x3 grid: rows = resolution, cols = MI approach
panels <- list()
for (i_res in seq_along(c("5", "6", "7"))) {
  res <- c("5", "6", "7")[i_res]
  for (i_mi in seq_along(mi_approaches)) {
    mi <- mi_approaches[i_mi]
    key <- paste(res, mi, sep = "_")
    r <- results[[key]]

    show_ylab <- (i_mi == 1)
    show_xlab <- (i_res == 3)

    p <- make_panel(r, show_ylab, show_xlab)

    # Add title: top row gets MI label, left column gets resolution label
    if (i_res == 1) {
      p <- p + ggtitle(mi_labels[mi])
    }
    if (i_mi == 1) {
      p <- p + ylab(paste0("Res ", res, " (", hex_areas[res], " km\u00b2)\n",
                           "Log mass (dev. from city mean)"))
    }

    panels[[key]] <- p
  }
}

# Assemble with patchwork
grid_plot <- (panels[["5_Global"]] | panels[["5_Baseline"]] | panels[["5_Fishman"]]) /
             (panels[["6_Global"]] | panels[["6_Baseline"]] | panels[["6_Fishman"]]) /
             (panels[["7_Global"]] | panels[["7_Baseline"]] | panels[["7_Fishman"]])

# =============================================================================
# FIGURE: Forest Plot
# =============================================================================

forest_data <- summary_df %>%
  mutate(
    label = paste0("Res ", Resolution, " | ", mi_labels[MI_approach]),
    label = factor(label, levels = rev(unique(label)))
  )

p_forest <- ggplot(forest_data, aes(x = Delta, y = label,
                                     color = MI_approach, shape = Resolution)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.3, linewidth = 0.6) +
  geom_point(size = 2.5) +
  scale_color_manual(
    values = c("Global" = "#66c2a5", "Baseline" = "#fc8d62", "Fishman" = "#8da0cb"),
    labels = mi_labels
  ) +
  scale_shape_manual(values = c("5" = 15, "6" = 16, "7" = 17)) +
  labs(
    x = expression("Scaling exponent " * delta),
    y = NULL,
    color = "MI Approach",
    shape = "Resolution"
  ) +
  theme_bw(base_size = 9) +
  theme(
    panel.grid.minor = element_blank(),
    panel.grid.major.y = element_blank(),
    legend.position = "bottom"
  )

# Combine: wrap grid_plot so patchwork treats 3x3 grid as single element
final_figure <- wrap_elements(grid_plot) / p_forest + plot_layout(heights = c(3, 1.3))

ggsave(file.path(figure_dir,
                 paste0("Fig3_Decentered_Multiscale_MI_Sensitivity_", Sys.Date(), ".pdf")),
       final_figure, width = 22, height = 24, units = "cm", dpi = 300)

cat("Figure saved.\n")

# =============================================================================
# HEATMAP-STYLE SUMMARY
# =============================================================================

cat("\n========================================\n")
cat("MI SENSITIVITY x SCALE SUMMARY\n")
cat("========================================\n\n")

# Print as matrix
cat("Delta values (rows=resolution, cols=MI):\n")
delta_matrix <- summary_df %>%
  select(Resolution, MI_approach, Delta) %>%
  pivot_wider(names_from = MI_approach, values_from = Delta)
print(as.data.frame(delta_matrix), row.names = FALSE)

# Variability stats
cat("\nVariability within each resolution (across MI approaches):\n")
by_res <- summary_df %>%
  group_by(Resolution) %>%
  summarise(
    mean_delta = mean(Delta),
    sd_delta = sd(Delta),
    range_delta = max(Delta) - min(Delta),
    cv_pct = sd(Delta) / mean(Delta) * 100,
    .groups = "drop"
  )
print(as.data.frame(by_res), row.names = FALSE)

cat("\nVariability within each MI approach (across resolutions):\n")
by_mi <- summary_df %>%
  group_by(MI_approach) %>%
  summarise(
    mean_delta = mean(Delta),
    sd_delta = sd(Delta),
    range_delta = max(Delta) - min(Delta),
    cv_pct = sd(Delta) / mean(Delta) * 100,
    .groups = "drop"
  )
print(as.data.frame(by_mi), row.names = FALSE)

cat("\nOverall:\n")
cat(sprintf("  Grand mean delta: %.4f\n", mean(summary_df$Delta)))
cat(sprintf("  Overall SD: %.4f\n", sd(summary_df$Delta)))
cat(sprintf("  Overall range: [%.4f, %.4f]\n", min(summary_df$Delta), max(summary_df$Delta)))
cat(sprintf("  CV (MI within resolution): %.2f%% - %.2f%%\n",
            min(by_res$cv_pct), max(by_res$cv_pct)))
cat(sprintf("  CV (resolution within MI): %.2f%% - %.2f%%\n",
            min(by_mi$cv_pct), max(by_mi$cv_pct)))
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
