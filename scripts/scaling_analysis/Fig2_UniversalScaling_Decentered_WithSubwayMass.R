# =============================================================================
# Fig2_UniversalScaling_Decentered_WithSubwayMass.R
# =============================================================================
# City-level de-centered scaling (Bettencourt & Lobo 2016), comparing:
#   (1) total_built_mass_tons (baseline)
#   (2) total_built_mass_tons + subway_mass (hex-aggregated to city)
#
# This script is analysis-first and uses only base R + dplyr (no pacman/tidyverse).
#
# Run from: _Revision1/scripts/
# Outputs:
#   ../results/Fig2_Decentered_Beta_WithVsWithoutSubway.csv
# =============================================================================

rm(list = ls())

suppressPackageStartupMessages({
  library(dplyr)
})

data_dir <- file.path("..", "data")
results_dir <- file.path("..", "results")

# -----------------------------------------------------------------------------
# Load city master data
# -----------------------------------------------------------------------------
DF <- read.csv(file.path(data_dir, "MasterMass_ByClass20250616.csv"),
               stringsAsFactors = FALSE)

# Countries with >=5 cities
city_counts <- table(DF$CTR_MN_NM)
valid_countries <- names(city_counts[city_counts >= 5])

# -----------------------------------------------------------------------------
# De-centered beta WITHOUT subway
# -----------------------------------------------------------------------------
DF0 <- DF %>%
  filter(CTR_MN_NM %in% valid_countries, total_built_mass_tons > 0) %>%
  mutate(
    log_pop = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  )

DF0_dc <- DF0 %>%
  group_by(CTR_MN_NM) %>%
  mutate(
    log_pop_centered = log_pop - mean(log_pop),
    log_mass_centered = log_mass - mean(log_mass)
  ) %>%
  ungroup()

mod0 <- lm(log_mass_centered ~ log_pop_centered, data = DF0_dc)
beta0 <- coef(mod0)[2]
se0 <- summary(mod0)$coefficients[2, 2]
ci0 <- c(beta0 - 1.96 * se0, beta0 + 1.96 * se0)

# -----------------------------------------------------------------------------
# Add subway mass (aggregate from hex-level res=6)
# -----------------------------------------------------------------------------
subway_hex <- read.csv(file.path(results_dir, "subway_mass_by_hexagon_resolution6.csv"),
                       stringsAsFactors = FALSE)

subway_city <- subway_hex %>%
  group_by(city_id) %>%
  summarize(subway_mass_tonnes_city = sum(subway_mass_tonnes, na.rm = TRUE), .groups = "drop")

DF1 <- DF %>%
  left_join(subway_city, by = c("ID_HDC_G0" = "city_id")) %>%
  mutate(
    subway_mass_tonnes_city = ifelse(is.na(subway_mass_tonnes_city), 0, subway_mass_tonnes_city),
    total_built_mass_withSubway_tons = total_built_mass_tons + subway_mass_tonnes_city
  )

DF1f <- DF1 %>%
  filter(CTR_MN_NM %in% valid_countries, total_built_mass_withSubway_tons > 0) %>%
  mutate(
    log_pop = log10(population_2015),
    log_mass = log10(total_built_mass_withSubway_tons)
  )

DF1_dc <- DF1f %>%
  group_by(CTR_MN_NM) %>%
  mutate(
    log_pop_centered = log_pop - mean(log_pop),
    log_mass_centered = log_mass - mean(log_mass)
  ) %>%
  ungroup()

mod1 <- lm(log_mass_centered ~ log_pop_centered, data = DF1_dc)
beta1 <- coef(mod1)[2]
se1 <- summary(mod1)$coefficients[2, 2]
ci1 <- c(beta1 - 1.96 * se1, beta1 + 1.96 * se1)

# -----------------------------------------------------------------------------
# Export + console summary
# -----------------------------------------------------------------------------

out <- data.frame(
  model = c("without_subway", "with_subway"),
  beta = c(beta0, beta1),
  se = c(se0, se1),
  ci_low = c(ci0[1], ci1[1]),
  ci_high = c(ci0[2], ci1[2]),
  n_cities = c(nrow(DF0_dc), nrow(DF1_dc)),
  n_countries = c(length(unique(DF0_dc$CTR_MN_NM)), length(unique(DF1_dc$CTR_MN_NM)))
)

out$delta_beta_vs_without <- out$beta - beta0

write.csv(out, file.path(results_dir, "Fig2_Decentered_Beta_WithVsWithoutSubway.csv"), row.names = FALSE)

cat("\n================ FIG2 (CITY) DECENTERED RESULTS ================\n")
cat(sprintf("WITHOUT subway: beta=%.4f  95%%CI=[%.4f, %.4f]  (N=%d cities)\n", beta0, ci0[1], ci0[2], nrow(DF0_dc)))
cat(sprintf("WITH    subway: beta=%.4f  95%%CI=[%.4f, %.4f]  (N=%d cities)\n", beta1, ci1[1], ci1[2], nrow(DF1_dc)))
cat(sprintf("Difference (with - without): %.6f\n", beta1 - beta0))
cat("Output:", file.path(results_dir, "Fig2_Decentered_Beta_WithVsWithoutSubway.csv"), "\n")
cat("===============================================================\n")
