# =============================================================================
# Fig2_UniversalScaling_Decentered_Source_Sensitivity.R
#
# Tests robustness of city-level scaling exponent to building volume data
# source choice: Esch2022 (WSF3D), Li2022, Liu2024 (GUS3D).
#
# Methodology (matching Fig2_UniversalScaling_Decentered.R):
#   1. For each data source: total_mass = BuildingMass_{source} + mobility_mass
#   2. De-center log(pop) and log(mass) by country mean
#   3. OLS on pooled de-centered data -> beta
#
# Analyses:
#   A. Baseline (equal-weighted average of 3 sources)
#   B. Individual sources (Esch2022, Li2022, Liu2024)
#   C. Random source selection (100 iterations):
#      - Per city, randomly pick 1 source from those with data (1-3)
#      - Re-run de-centered OLS
#      - Build distribution of beta
#
# Output: PDF figure + CSV summary
# =============================================================================

rm(list = ls())
if (!interactive()) pdf(NULL)

library(pacman)
pacman::p_load(readr, dplyr, tibble, ggplot2, scales, patchwork)

set.seed(42)

# =============================================================================
# SETUP
# =============================================================================

project_root <- rprojroot::find_root(rprojroot::has_file("CLAUDE.md"))
data_dir <- file.path(project_root, "data")
figure_dir <- file.path(project_root, "figures")
dir.create(figure_dir, showWarnings = FALSE, recursive = TRUE)

N_ITERATIONS <- 100

cat("=== Fig2 De-centered - Data Source Sensitivity ===\n\n")

# =============================================================================
# LOAD DATA
# =============================================================================

DF <- read.csv(file.path(data_dir, "MasterMass_ByClass20250616.csv"),
               stringsAsFactors = FALSE)

cat("Loaded:", nrow(DF), "cities\n")

# Per-source total mass = BuildingMass_{source} + mobility_mass_tons
sources <- c("Esch2022", "Li2022", "Liu2024")
source_labels <- c("Esch2022" = "Esch2022 (WSF3D)",
                    "Li2022" = "Li2022",
                    "Liu2024" = "Liu2024 (GUS3D)")

for (src in sources) {
  bldg_col <- paste0("BuildingMass_Total_", src)
  total_col <- paste0("total_", src)
  DF[[total_col]] <- DF[[bldg_col]] + DF$mobility_mass_tons
}

# Baseline is already total_built_mass_tons (average of 3 sources + mobility)

# =============================================================================
# DE-CENTERED OLS FUNCTION
# =============================================================================

run_decentered_ols <- function(df, mass_col, min_cities_per_country = 5) {
  # Filter to positive mass and valid countries
  d <- df %>%
    filter(.data[[mass_col]] > 0) %>%
    mutate(
      log_pop = log10(population_2015),
      log_mass = log10(.data[[mass_col]])
    )

  city_counts <- table(d$CTR_MN_NM)
  valid_countries <- names(city_counts[city_counts >= min_cities_per_country])
  d <- d %>% filter(CTR_MN_NM %in% valid_countries)

  # De-center by country mean
  d <- d %>%
    group_by(CTR_MN_NM) %>%
    mutate(
      log_pop_centered = log_pop - mean(log_pop),
      log_mass_centered = log_mass - mean(log_mass)
    ) %>%
    ungroup()

  # OLS

  mod <- lm(log_mass_centered ~ log_pop_centered, data = d)
  s <- summary(mod)
  beta <- coef(mod)[2]
  se <- s$coefficients[2, 2]
  r2 <- s$r.squared

  list(
    beta = beta, se = se, r2 = r2,
    ci_low = beta - 1.96 * se, ci_high = beta + 1.96 * se,
    n_cities = nrow(d),
    n_countries = length(unique(d$CTR_MN_NM)),
    data = d
  )
}

# =============================================================================
# A. BASELINE (average of 3 sources)
# =============================================================================

cat("--- Baseline (average) ---\n")
res_baseline <- run_decentered_ols(DF, "total_built_mass_tons")
cat(sprintf("  beta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d, Countries = %d\n",
            res_baseline$beta, res_baseline$ci_low, res_baseline$ci_high,
            res_baseline$r2, res_baseline$n_cities, res_baseline$n_countries))

# =============================================================================
# B. INDIVIDUAL SOURCES
# =============================================================================

cat("\n--- Individual sources ---\n")
res_sources <- list()
for (src in sources) {
  total_col <- paste0("total_", src)
  res_sources[[src]] <- run_decentered_ols(DF, total_col)
  cat(sprintf("  %-12s beta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d\n",
              src, res_sources[[src]]$beta,
              res_sources[[src]]$ci_low, res_sources[[src]]$ci_high,
              res_sources[[src]]$r2, res_sources[[src]]$n_cities))
}

# =============================================================================
# C. RANDOM SOURCE SELECTION (100 iterations)
# =============================================================================

cat(sprintf("\n--- Random source selection (%d iterations) ---\n", N_ITERATIONS))

# Pre-filter to cities that are valid across ALL sources
# (use the same country filter as baseline)
valid_df <- DF %>%
  filter(total_built_mass_tons > 0)
city_counts <- table(valid_df$CTR_MN_NM)
valid_countries <- names(city_counts[city_counts >= 5])
valid_df <- valid_df %>% filter(CTR_MN_NM %in% valid_countries)

cat(sprintf("  Working set: %d cities, %d countries\n",
            nrow(valid_df), length(unique(valid_df$CTR_MN_NM))))

# Pre-compute per-source building mass columns as a matrix for fast indexing
bldg_matrix <- as.matrix(valid_df[, paste0("BuildingMass_Total_", sources)])
mobility <- valid_df$mobility_mass_tons

iteration_results <- vector("list", N_ITERATIONS)

# Pre-compute which sources are available (> 0) per city
available_mask <- bldg_matrix > 0  # logical matrix: N x 3

for (iter in seq_len(N_ITERATIONS)) {
  # For each city, randomly pick one source from those with data
  source_idx <- integer(nrow(valid_df))
  for (i in seq_len(nrow(valid_df))) {
    avail <- which(available_mask[i, ])
    if (length(avail) == 0L) {
      source_idx[i] <- NA_integer_
    } else if (length(avail) == 1L) {
      source_idx[i] <- avail
    } else {
      source_idx[i] <- sample(avail, 1L)
    }
  }

  # Build mass vector: pick building mass from assigned source + mobility
  random_bldg <- bldg_matrix[cbind(seq_len(nrow(valid_df)), source_idx)]
  random_bldg[is.na(source_idx)] <- 0
  random_total <- random_bldg + mobility

  # Replace mass in dataframe and run analysis
  tmp_df <- valid_df
  tmp_df$random_total <- random_total

  d <- tmp_df %>%
    filter(random_total > 0) %>%
    mutate(
      log_pop = log10(population_2015),
      log_mass = log10(random_total)
    )

  # Re-filter countries (should be same set, but ensure)
  cc <- table(d$CTR_MN_NM)
  vc <- names(cc[cc >= 5])
  d <- d %>% filter(CTR_MN_NM %in% vc)

  d <- d %>%
    group_by(CTR_MN_NM) %>%
    mutate(
      log_pop_centered = log_pop - mean(log_pop),
      log_mass_centered = log_mass - mean(log_mass)
    ) %>%
    ungroup()

  mod <- lm(log_mass_centered ~ log_pop_centered, data = d)
  s <- summary(mod)

  iteration_results[[iter]] <- tibble(
    iteration = iter,
    beta = coef(mod)[2],
    se = s$coefficients[2, 2],
    r2 = s$r.squared,
    n_cities = nrow(d)
  )
}

iter_df <- bind_rows(iteration_results)

cat(sprintf("  Completed %d iterations\n", nrow(iter_df)))
cat(sprintf("  Beta: mean = %.4f, sd = %.4f\n", mean(iter_df$beta), sd(iter_df$beta)))
cat(sprintf("  Beta 95%% CI (percentile): [%.4f, %.4f]\n",
            quantile(iter_df$beta, 0.025), quantile(iter_df$beta, 0.975)))
cat(sprintf("  R2: mean = %.4f, sd = %.4f\n", mean(iter_df$r2), sd(iter_df$r2)))

# =============================================================================
# SUMMARY TABLE
# =============================================================================

summary_table <- bind_rows(
  tibble(Approach = "Baseline (Average)", Beta = res_baseline$beta,
         SE = res_baseline$se, CI_low = res_baseline$ci_low,
         CI_high = res_baseline$ci_high, R2 = res_baseline$r2,
         N_cities = res_baseline$n_cities, N_countries = res_baseline$n_countries),
  bind_rows(lapply(sources, function(src) {
    r <- res_sources[[src]]
    tibble(Approach = src, Beta = r$beta, SE = r$se,
           CI_low = r$ci_low, CI_high = r$ci_high, R2 = r$r2,
           N_cities = r$n_cities, N_countries = r$n_countries)
  })),
  tibble(Approach = "Random (mean)", Beta = mean(iter_df$beta),
         SE = sd(iter_df$beta), CI_low = quantile(iter_df$beta, 0.025),
         CI_high = quantile(iter_df$beta, 0.975), R2 = mean(iter_df$r2),
         N_cities = round(mean(iter_df$n_cities)),
         N_countries = res_baseline$n_countries)
)

cat("\n=== SUMMARY TABLE ===\n")
print(as.data.frame(summary_table), row.names = FALSE)

write_csv(summary_table, file.path(figure_dir,
          paste0("Table_Decentered_Source_Sensitivity_City_", Sys.Date(), ".csv")))

# =============================================================================
# FIGURES
# =============================================================================

cat("\n=== Generating figures ===\n")

# --- Panel A: Histogram of random betas ---
p_hist <- ggplot(iter_df, aes(x = beta)) +
  geom_histogram(bins = 25, fill = "steelblue", color = "black", alpha = 0.7) +
  geom_vline(xintercept = res_baseline$beta, color = "#fc8d62",
             linetype = "solid", linewidth = 0.8) +
  geom_vline(xintercept = res_sources[["Esch2022"]]$beta, color = "#e41a1c",
             linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = res_sources[["Li2022"]]$beta, color = "#377eb8",
             linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = res_sources[["Liu2024"]]$beta, color = "#4daf4a",
             linetype = "dashed", linewidth = 0.5) +
  geom_vline(xintercept = quantile(iter_df$beta, 0.025), color = "grey40",
             linetype = "dotted", linewidth = 0.5) +
  geom_vline(xintercept = quantile(iter_df$beta, 0.975), color = "grey40",
             linetype = "dotted", linewidth = 0.5) +
  annotate("text", x = res_baseline$beta, y = Inf, label = "Baseline",
           color = "#fc8d62", vjust = 2, hjust = -0.1, size = 2.8) +
  annotate("text", x = res_sources[["Esch2022"]]$beta, y = Inf,
           label = "Esch2022", color = "#e41a1c", vjust = 3.5,
           hjust = -0.1, size = 2.5) +
  annotate("text", x = res_sources[["Li2022"]]$beta, y = Inf,
           label = "Li2022", color = "#377eb8", vjust = 5,
           hjust = -0.1, size = 2.5) +
  annotate("text", x = res_sources[["Liu2024"]]$beta, y = Inf,
           label = "Liu2024", color = "#4daf4a", vjust = 6.5,
           hjust = -0.1, size = 2.5) +
  labs(
    title = sprintf("Random Source Selection (%d iterations)", N_ITERATIONS),
    x = expression("Scaling exponent " * beta),
    y = "Frequency"
  ) +
  theme_bw(base_size = 9) +
  theme(panel.grid.minor = element_blank())

# --- Panel B: Forest plot (sources + baseline + random CI) ---
forest_df <- bind_rows(
  tibble(Label = "Baseline (Average)", Beta = res_baseline$beta,
         CI_low = res_baseline$ci_low, CI_high = res_baseline$ci_high,
         Type = "Baseline"),
  bind_rows(lapply(sources, function(src) {
    r <- res_sources[[src]]
    tibble(Label = source_labels[src], Beta = r$beta,
           CI_low = r$ci_low, CI_high = r$ci_high, Type = "Individual")
  })),
  tibble(Label = "Random Selection (95% CI)",
         Beta = mean(iter_df$beta),
         CI_low = quantile(iter_df$beta, 0.025),
         CI_high = quantile(iter_df$beta, 0.975),
         Type = "Random")
)

forest_df$Label <- factor(forest_df$Label, levels = rev(forest_df$Label))

p_forest <- ggplot(forest_df, aes(x = Beta, y = Label, color = Type)) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "grey50") +
  geom_errorbarh(aes(xmin = CI_low, xmax = CI_high), height = 0.3,
                 linewidth = 0.7) +
  geom_point(size = 3) +
  geom_text(aes(label = sprintf("%.4f", Beta)), vjust = -1, size = 2.8,
            show.legend = FALSE) +
  scale_color_manual(values = c("Baseline" = "#fc8d62",
                                "Individual" = "#8da0cb",
                                "Random" = "#66c2a5")) +
  labs(
    title = "City-Level Scaling Exponent by Data Source",
    x = expression("Scaling exponent " * beta),
    y = NULL,
    color = NULL
  ) +
  theme_bw(base_size = 9) +
  theme(panel.grid.minor = element_blank(),
        panel.grid.major.y = element_blank(),
        legend.position = "bottom")

# --- Panel C: De-centered scatter for each source ---
scatter_panels <- list()
all_results <- c(list(Baseline = res_baseline), res_sources)
scatter_names <- c("Baseline", sources)
scatter_colors <- c("Baseline" = "#fc8d62", "Esch2022" = "#e41a1c",
                     "Li2022" = "#377eb8", "Liu2024" = "#4daf4a")

# Common axis limits
all_x <- unlist(lapply(all_results, function(r) r$data$log_pop_centered))
all_y <- unlist(lapply(all_results, function(r) r$data$log_mass_centered))
x_lim <- range(all_x, na.rm = TRUE) * c(1.05, 1.05)
y_lim <- range(all_y, na.rm = TRUE) * c(1.05, 1.05)

for (nm in scatter_names) {
  r <- all_results[[nm]]
  scatter_panels[[nm]] <- ggplot(r$data, aes(x = log_pop_centered,
                                              y = log_mass_centered)) +
    geom_point(shape = 21, fill = "#8da0cb", color = "black",
               stroke = 0.2, alpha = 0.3, size = 1) +
    geom_abline(slope = r$beta, intercept = 0,
                color = scatter_colors[nm], linewidth = 0.6) +
    geom_abline(slope = 1, intercept = 0, linetype = "dashed", linewidth = 0.4) +
    coord_cartesian(xlim = x_lim, ylim = y_lim) +
    annotate("text", x = x_lim[1] + 0.1 * diff(x_lim),
             y = y_lim[2] - 0.05 * diff(y_lim),
             label = sprintf("beta == %.4f", r$beta),
             color = scatter_colors[nm], parse = TRUE,
             hjust = 0, size = 2.8) +
    annotate("text", x = x_lim[1] + 0.1 * diff(x_lim),
             y = y_lim[2] - 0.12 * diff(y_lim),
             label = sprintf("R^2 == %.3f", r$r2),
             parse = TRUE, hjust = 0, size = 2.5, color = "grey40") +
    annotate("text", x = x_lim[2] - 0.02 * diff(x_lim),
             y = y_lim[1] + 0.03 * diff(y_lim),
             label = sprintf("N = %s", format(r$n_cities, big.mark = ",")),
             hjust = 1, size = 2.2, color = "grey50") +
    ggtitle(ifelse(nm == "Baseline", "Baseline (Average)",
                   source_labels[nm])) +
    theme_bw(base_size = 8) +
    theme(
      panel.grid = element_blank(),
      plot.title = element_text(size = 8, face = "bold", hjust = 0.5),
      plot.margin = margin(3, 3, 3, 3)
    ) +
    xlab(if (nm %in% c("Liu2024", "Baseline"))
           "Log pop (dev. from country mean)" else NULL) +
    ylab(if (nm %in% c("Baseline", "Li2022"))
           "Log mass (dev. from country mean)" else NULL)
}

# Assemble scatter row
p_scatter <- scatter_panels[["Baseline"]] | scatter_panels[["Esch2022"]] |
             scatter_panels[["Li2022"]] | scatter_panels[["Liu2024"]]

# Assemble full figure
final_figure <- p_scatter /
  (p_hist | p_forest) +
  plot_layout(heights = c(1, 1))

ggsave(file.path(figure_dir,
       paste0("Fig2_Decentered_Source_Sensitivity_", Sys.Date(), ".pdf")),
       final_figure, width = 28, height = 18, units = "cm", dpi = 300)

cat("Figure saved.\n")

# =============================================================================
# ROBUSTNESS ASSESSMENT
# =============================================================================

cat("\n========================================\n")
cat("ROBUSTNESS ASSESSMENT\n")
cat("========================================\n")

baseline_in_ci <- (res_baseline$beta >= quantile(iter_df$beta, 0.025)) &
                  (res_baseline$beta <= quantile(iter_df$beta, 0.975))

cat(sprintf("Baseline beta: %.4f\n", res_baseline$beta))
cat(sprintf("Random mean beta: %.4f\n", mean(iter_df$beta)))
cat(sprintf("Random 95%% CI: [%.4f, %.4f]\n",
            quantile(iter_df$beta, 0.025), quantile(iter_df$beta, 0.975)))
cat(sprintf("Baseline within random 95%% CI: %s\n",
            ifelse(baseline_in_ci, "YES", "NO")))
cat(sprintf("Abs difference (baseline vs random mean): %.5f\n",
            abs(res_baseline$beta - mean(iter_df$beta))))
cat(sprintf("Relative difference: %.2f%%\n",
            abs(res_baseline$beta - mean(iter_df$beta)) / res_baseline$beta * 100))

# Range across individual sources
source_betas <- sapply(res_sources, function(r) r$beta)
cat(sprintf("\nIndividual source betas: [%.4f, %.4f]\n",
            min(source_betas), max(source_betas)))
cat(sprintf("Range: %.4f (%.2f%% of mean)\n",
            diff(range(source_betas)),
            diff(range(source_betas)) / mean(source_betas) * 100))

cat("\nConclusion: ")
if (baseline_in_ci && diff(range(source_betas)) < 0.05) {
  cat("ROBUST - data source choice does not affect scaling conclusions\n")
} else {
  cat("SENSITIVE - data source choice may affect scaling exponent\n")
}
cat("========================================\n")
cat("Outputs in:", figure_dir, "\n")
