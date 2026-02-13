# =============================================================================
# Fig3_NeighborhoodScaling_Decentered_Source_Sensitivity.R
#
# Tests robustness of neighborhood-level scaling exponent (delta) to building
# volume data source choice: Esch2022, Li2022, Liu2024.
#
# Methodology (matching Fig3_NeighborhoodScaling_Decentered_Multiscale_R6Filter.R):
#   1. Apply R6 filter set for consistent sample across resolutions
#   2. For each data source: total_mass = BuildingMass_{source} + mobility_mass
#   3. De-center log(pop) and log(mass) by city mean
#   4. OLS on pooled de-centered data -> delta
#
# Analyses per resolution (5, 6, 7):
#   A. Baseline (equal-weighted average of 3 sources)
#   B. Individual sources (Esch2022, Li2022, Liu2024)
#   C. Random source selection (100 iterations):
#      - Per hexagon, randomly pick 1 source from those with data (1-3)
#      - Re-run de-centered OLS
#      - Build distribution of delta
#
# Output: PDF figure + CSV summary
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(readr, dplyr, tibble, tidyr, ggplot2, scales, patchwork)

set.seed(42)

# =============================================================================
# SETUP
# =============================================================================

project_root <- rprojroot::find_root(rprojroot::has_file("CLAUDE.md"))
data_dir <- file.path(project_root, "data")
figure_dir <- file.path(project_root, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

N_ITERATIONS <- 100

mass_files <- list(
  "5" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution5_2026-02-02.csv"),
  "6" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  "7" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
)

hex_areas <- c("5" = 252.9, "6" = 36.13, "7" = 5.161)
sources <- c("Esch2022", "Li2022", "Liu2024")
source_labels <- c("Esch2022" = "Esch2022 (WSF3D)",
                    "Li2022" = "Li2022",
                    "Liu2024" = "Liu2024 (GUS3D)")

cat("=== Fig3 De-centered - Source Sensitivity (3 resolutions) ===\n\n")

# =============================================================================
# STEP 1: Determine R6 filter set
# =============================================================================

cat("=== Determining R6 filter set ===\n")

data_r6 <- readr::read_csv(mass_files[["6"]], show_col_types = FALSE) %>%
  filter(population_2015 >= 1, total_built_mass_tons > 0)

# Cities with total pop > 50,000
city_pop_r6 <- data_r6 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(total_pop = sum(population_2015), .groups = "drop") %>%
  filter(total_pop > 50000)
data_r6 <- data_r6 %>% filter(ID_HDC_G0 %in% city_pop_r6$ID_HDC_G0)

# Cities with >= 10 neighborhoods
city_nbhd_r6 <- data_r6 %>%
  group_by(CTR_MN_ISO, ID_HDC_G0) %>%
  summarize(n_nbhd = n(), .groups = "drop") %>%
  filter(n_nbhd >= 10)
data_r6 <- data_r6 %>% filter(ID_HDC_G0 %in% city_nbhd_r6$ID_HDC_G0)

# Countries with >= 5 qualifying cities
country_city_r6 <- data_r6 %>%
  group_by(CTR_MN_ISO) %>%
  summarize(num_cities = n_distinct(ID_HDC_G0)) %>%
  filter(num_cities >= 5)
data_r6 <- data_r6 %>% filter(CTR_MN_ISO %in% country_city_r6$CTR_MN_ISO)

r6_city_ids <- unique(data_r6$ID_HDC_G0)
r6_country_isos <- unique(data_r6$CTR_MN_ISO)
cat("R6 filter:", length(r6_city_ids), "cities,",
    length(r6_country_isos), "countries\n\n")

# =============================================================================
# DE-CENTERED OLS FUNCTION (neighborhood-level, de-centers by city)
# =============================================================================

run_decentered_nbhd <- function(df, mass_col) {
  d <- df %>%
    filter(population_2015 >= 1, .data[[mass_col]] > 0) %>%
    filter(ID_HDC_G0 %in% r6_city_ids,
           CTR_MN_ISO %in% r6_country_isos) %>%
    mutate(
      log_pop = log10(population_2015),
      log_mass = log10(.data[[mass_col]]),
      Country_City = paste(CTR_MN_NM, ID_HDC_G0, sep = "_")
    )

  # De-center by city mean (>= 2 neighborhoods per city)
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
# STEP 2: Load data, compute per-source mass, run analyses
# =============================================================================

all_results <- list()  # res -> list(baseline, sources, random_df)

for (res in c("5", "6", "7")) {
  cat(sprintf("======== Resolution %s ========\n", res))

  df <- readr::read_csv(mass_files[[res]], show_col_types = FALSE)
  cat("  Loaded:", nrow(df), "neighborhoods\n")

  # Compute per-source total mass
  for (src in sources) {
    bldg_col <- paste0("BuildingMass_Total_", src)
    total_col <- paste0("total_", src)
    df[[total_col]] <- df[[bldg_col]] + df$mobility_mass_tons
  }

  # A. Baseline
  cat("  --- Baseline ---\n")
  res_baseline <- run_decentered_nbhd(df, "total_built_mass_tons")
  cat(sprintf("    delta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d\n",
              res_baseline$delta, res_baseline$ci_low, res_baseline$ci_high,
              res_baseline$r2, res_baseline$n_neighborhoods))

  # B. Individual sources
  cat("  --- Individual sources ---\n")
  res_sources <- list()
  for (src in sources) {
    total_col <- paste0("total_", src)
    res_sources[[src]] <- run_decentered_nbhd(df, total_col)
    cat(sprintf("    %-12s delta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d\n",
                src, res_sources[[src]]$delta,
                res_sources[[src]]$ci_low, res_sources[[src]]$ci_high,
                res_sources[[src]]$r2, res_sources[[src]]$n_neighborhoods))
  }

  # C. Random source selection
  cat(sprintf("  --- Random (%d iterations) ---\n", N_ITERATIONS))

  # Pre-filter working set (same as baseline filter)
  work_df <- df %>%
    filter(population_2015 >= 1, total_built_mass_tons > 0,
           ID_HDC_G0 %in% r6_city_ids, CTR_MN_ISO %in% r6_country_isos) %>%
    mutate(Country_City = paste(CTR_MN_NM, ID_HDC_G0, sep = "_"))

  # Pre-compute building mass matrix
  bldg_matrix <- as.matrix(work_df[, paste0("BuildingMass_Total_", sources)])
  mobility <- work_df$mobility_mass_tons

  # Pre-compute which sources are available (> 0) per neighborhood
  available_mask <- bldg_matrix > 0  # logical matrix: N x 3

  iter_results <- vector("list", N_ITERATIONS)
  for (iter in seq_len(N_ITERATIONS)) {
    # For each neighborhood, randomly pick one source from those with data
    source_idx <- integer(nrow(work_df))
    for (i in seq_len(nrow(work_df))) {
      avail <- which(available_mask[i, ])
      if (length(avail) == 0L) {
        source_idx[i] <- NA_integer_
      } else if (length(avail) == 1L) {
        source_idx[i] <- avail
      } else {
        source_idx[i] <- sample(avail, 1L)
      }
    }
    random_bldg <- bldg_matrix[cbind(seq_len(nrow(work_df)), source_idx)]
    random_bldg[is.na(source_idx)] <- 0
    random_total <- random_bldg + mobility

    tmp <- work_df
    tmp$random_total <- random_total

    d <- tmp %>%
      filter(random_total > 0) %>%
      mutate(
        log_pop = log10(population_2015),
        log_mass = log10(random_total)
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

    iter_results[[iter]] <- tibble(
      iteration = iter,
      delta = coef(mod)[2],
      se = s$coefficients[2, 2],
      r2 = s$r.squared,
      n_neighborhoods = nrow(dc)
    )
  }

  iter_df <- bind_rows(iter_results)

  cat(sprintf("    Delta: mean = %.4f, sd = %.4f, 95%% CI = [%.4f, %.4f]\n",
              mean(iter_df$delta), sd(iter_df$delta),
              quantile(iter_df$delta, 0.025), quantile(iter_df$delta, 0.975)))

  all_results[[res]] <- list(
    baseline = res_baseline,
    sources = res_sources,
    iter_df = iter_df
  )
}

# =============================================================================
# SUMMARY TABLE
# =============================================================================

cat("\n=== SUMMARY TABLE ===\n")

summary_rows <- list()
for (res in c("5", "6", "7")) {
  ar <- all_results[[res]]

  summary_rows[[length(summary_rows) + 1]] <- tibble(
    Resolution = res, Hex_km2 = hex_areas[res],
    Approach = "Baseline (Average)", Delta = ar$baseline$delta,
    SE = ar$baseline$se, CI_low = ar$baseline$ci_low,
    CI_high = ar$baseline$ci_high, R2 = ar$baseline$r2,
    N_neighborhoods = ar$baseline$n_neighborhoods,
    N_cities = ar$baseline$n_cities, N_countries = ar$baseline$n_countries)

  for (src in sources) {
    r <- ar$sources[[src]]
    summary_rows[[length(summary_rows) + 1]] <- tibble(
      Resolution = res, Hex_km2 = hex_areas[res],
      Approach = src, Delta = r$delta, SE = r$se,
      CI_low = r$ci_low, CI_high = r$ci_high, R2 = r$r2,
      N_neighborhoods = r$n_neighborhoods,
      N_cities = r$n_cities, N_countries = r$n_countries)
  }

  summary_rows[[length(summary_rows) + 1]] <- tibble(
    Resolution = res, Hex_km2 = hex_areas[res],
    Approach = "Random (mean)", Delta = mean(ar$iter_df$delta),
    SE = sd(ar$iter_df$delta),
    CI_low = quantile(ar$iter_df$delta, 0.025),
    CI_high = quantile(ar$iter_df$delta, 0.975),
    R2 = mean(ar$iter_df$r2),
    N_neighborhoods = round(mean(ar$iter_df$n_neighborhoods)),
    N_cities = ar$baseline$n_cities,
    N_countries = ar$baseline$n_countries)
}

summary_df <- bind_rows(summary_rows)
print(as.data.frame(summary_df), row.names = FALSE)

write_csv(summary_df, file.path(figure_dir,
          paste0("Table_Decentered_Source_Sensitivity_Neighborhood_Long_", Sys.Date(), ".csv")))

# --- 3x5 wide-format table (rows=resolution, cols=source) ---
wide_delta <- summary_df %>%
  select(Resolution, Hex_km2, Approach, Delta) %>%
  pivot_wider(names_from = Approach, values_from = Delta) %>%
  rename(`Baseline (Avg)` = `Baseline (Average)`,
         `Random Mean` = `Random (mean)`)

cat("\n=== 3x5 DELTA TABLE (rows=resolution, cols=source) ===\n")
print(as.data.frame(wide_delta), row.names = FALSE)

write_csv(wide_delta, file.path(figure_dir,
          paste0("Table_Decentered_Source_Sensitivity_Neighborhood_Wide_", Sys.Date(), ".csv")))

# =============================================================================
# FIGURES
# =============================================================================

cat("\n=== Generating figures ===\n")

source_colors <- c("Esch2022" = "#e41a1c", "Li2022" = "#377eb8",
                    "Liu2024" = "#4daf4a")

# --- Row 1: Histograms (one per resolution) ---
hist_panels <- list()
for (res in c("5", "6", "7")) {
  ar <- all_results[[res]]
  idf <- ar$iter_df

  p <- ggplot(idf, aes(x = delta)) +
    geom_histogram(bins = 25, fill = "steelblue", color = "black", alpha = 0.7) +
    geom_vline(xintercept = ar$baseline$delta, color = "#fc8d62",
               linetype = "solid", linewidth = 0.8) +
    geom_vline(xintercept = ar$sources[["Esch2022"]]$delta, color = "#e41a1c",
               linetype = "dashed", linewidth = 0.5) +
    geom_vline(xintercept = ar$sources[["Li2022"]]$delta, color = "#377eb8",
               linetype = "dashed", linewidth = 0.5) +
    geom_vline(xintercept = ar$sources[["Liu2024"]]$delta, color = "#4daf4a",
               linetype = "dashed", linewidth = 0.5) +
    geom_vline(xintercept = quantile(idf$delta, 0.025), color = "grey40",
               linetype = "dotted", linewidth = 0.5) +
    geom_vline(xintercept = quantile(idf$delta, 0.975), color = "grey40",
               linetype = "dotted", linewidth = 0.5) +
    labs(
      title = paste0("Res ", res, " (", hex_areas[res], " km\u00b2)"),
      x = expression(delta),
      y = "Freq."
    ) +
    theme_bw(base_size = 8) +
    theme(panel.grid.minor = element_blank(),
          plot.title = element_text(size = 8, face = "bold", hjust = 0.5))

  # Add source labels only on first panel
  if (res == "5") {
    p <- p +
      annotate("text", x = ar$baseline$delta, y = Inf, label = "Avg",
               color = "#fc8d62", vjust = 1.5, hjust = -0.1, size = 2.2) +
      annotate("text", x = ar$sources[["Esch2022"]]$delta, y = Inf,
               label = "Esch", color = "#e41a1c", vjust = 3, hjust = -0.1, size = 2) +
      annotate("text", x = ar$sources[["Li2022"]]$delta, y = Inf,
               label = "Li", color = "#377eb8", vjust = 4.5, hjust = -0.1, size = 2) +
      annotate("text", x = ar$sources[["Liu2024"]]$delta, y = Inf,
               label = "Liu", color = "#4daf4a", vjust = 6, hjust = -0.1, size = 2)
  }

  hist_panels[[res]] <- p
}

# --- Row 2: Forest plot ---
forest_rows <- list()
for (res in c("5", "6", "7")) {
  ar <- all_results[[res]]

  forest_rows[[length(forest_rows) + 1]] <- tibble(
    Label = paste0("Res ", res, " | Baseline"),
    Resolution = res, Delta = ar$baseline$delta,
    CI_low = ar$baseline$ci_low, CI_high = ar$baseline$ci_high,
    Type = "Baseline")

  for (src in sources) {
    r <- ar$sources[[src]]
    forest_rows[[length(forest_rows) + 1]] <- tibble(
      Label = paste0("Res ", res, " | ", src),
      Resolution = res, Delta = r$delta,
      CI_low = r$ci_low, CI_high = r$ci_high,
      Type = "Individual")
  }

  forest_rows[[length(forest_rows) + 1]] <- tibble(
    Label = paste0("Res ", res, " | Random"),
    Resolution = res, Delta = mean(ar$iter_df$delta),
    CI_low = quantile(ar$iter_df$delta, 0.025),
    CI_high = quantile(ar$iter_df$delta, 0.975),
    Type = "Random")
}

forest_df <- bind_rows(forest_rows)
forest_df$Label <- factor(forest_df$Label, levels = rev(forest_df$Label))

p_forest <- ggplot(forest_df, aes(x = Delta, y = Label, color = Type,
                                   shape = Resolution)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.3,
                 linewidth = 0.6) +
  geom_point(size = 2.5) +
  geom_text(aes(label = sprintf("%.4f", Delta)), vjust = -1, size = 2.3,
            show.legend = FALSE) +
  scale_color_manual(values = c("Baseline" = "#fc8d62",
                                "Individual" = "#8da0cb",
                                "Random" = "#66c2a5")) +
  scale_shape_manual(values = c("5" = 15, "6" = 16, "7" = 17)) +
  labs(
    title = "Neighborhood Scaling Exponent by Data Source and Resolution",
    x = expression("Scaling exponent " * delta),
    y = NULL,
    color = NULL, shape = "Resolution"
  ) +
  theme_bw(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        legend.position = "bottom")

# Assemble
p_histrow <- hist_panels[["5"]] | hist_panels[["6"]] | hist_panels[["7"]]
final_figure <- p_histrow / p_forest + plot_layout(heights = c(1, 1.5))

ggsave(file.path(figure_dir,
       paste0("Fig3_Decentered_Source_Sensitivity_", Sys.Date(), ".pdf")),
       final_figure, width = 24, height = 20, units = "cm", dpi = 300)

cat("Figure saved.\n")

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================

cat("\n========================================\n")
cat("ROBUSTNESS ASSESSMENT\n")
cat("========================================\n\n")

for (res in c("5", "6", "7")) {
  ar <- all_results[[res]]
  idf <- ar$iter_df

  baseline_in_ci <- (ar$baseline$delta >= quantile(idf$delta, 0.025)) &
                    (ar$baseline$delta <= quantile(idf$delta, 0.975))

  source_deltas <- sapply(ar$sources, function(r) unname(r$delta))

  cat(sprintf("Resolution %s (%s km2):\n", res, hex_areas[res]))
  cat(sprintf("  Baseline delta: %.4f\n", ar$baseline$delta))
  cat(sprintf("  Individual sources: Esch=%.4f, Li=%.4f, Liu=%.4f\n",
              source_deltas[["Esch2022"]], source_deltas[["Li2022"]],
              source_deltas[["Liu2024"]]))
  cat(sprintf("  Source range: %.4f (%.2f%% of mean)\n",
              diff(range(source_deltas)),
              diff(range(source_deltas)) / mean(source_deltas) * 100))
  cat(sprintf("  Random mean: %.4f, sd: %.4f\n",
              mean(idf$delta), sd(idf$delta)))
  cat(sprintf("  Random 95%% CI: [%.4f, %.4f]\n",
              quantile(idf$delta, 0.025), quantile(idf$delta, 0.975)))
  cat(sprintf("  Baseline in CI: %s\n", ifelse(baseline_in_ci, "YES", "NO")))
  cat("\n")
}

cat("Conclusion: ")
# Overall check
all_robust <- TRUE
for (res in c("5", "6", "7")) {
  ar <- all_results[[res]]
  source_deltas <- sapply(ar$sources, function(r) r$delta)
  if (diff(range(source_deltas)) >= 0.05) all_robust <- FALSE
}
if (all_robust) {
  cat("ROBUST across all resolutions\n")
} else {
  cat("Some resolution(s) show sensitivity\n")
}
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
