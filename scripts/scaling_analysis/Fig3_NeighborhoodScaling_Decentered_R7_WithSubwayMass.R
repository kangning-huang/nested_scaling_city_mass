# =============================================================================
# Fig3 Neighborhood Scaling - Decentered - R7 Per-Resolution - With Subway Mass
# =============================================================================
# Compares R7 neighborhood-level scaling exponent with and without subway mass.
#
# Per-resolution filter at R7 (determined from baseline WITHOUT subway):
#   - pop >= 1, mass > 0
#   - City total pop > 50,000
#   - >= 10 neighborhoods per city
#   - >= 5 qualifying cities per country
#
# Output: results/Fig3_Decentered_Delta_R7_PerRes_WithVsWithoutSubway.csv
# =============================================================================

rm(list = ls())

rev_dir <- normalizePath(file.path(getwd(), ".."), mustWork = TRUE)
data_dir <- file.path(rev_dir, "data")
results_dir <- file.path(rev_dir, "results")
dir.create(results_dir, showWarnings = FALSE, recursive = TRUE)

mass_file <- file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
subway_file <- file.path(results_dir, "subway_mass_by_hexagon_resolution7.csv")

cat("=== Fig3 De-centered - R7 Per-Resolution Filter - Subway Impact ===\n\n")

# =============================================================================
# LOAD DATA
# =============================================================================

load_mass <- function() {
  df <- read.csv(mass_file, stringsAsFactors = FALSE)
  keep <- c("h3index", "UC_NM_MN", "ID_HDC_G0", "CTR_MN_ISO", "CTR_MN_NM",
            "population_2015", "total_built_mass_tons")
  df <- df[, keep]
  names(df) <- c("h3index", "city_name", "city_id", "country_iso", "country",
                  "population", "mass")
  df$h3index <- as.character(df$h3index)
  df$city_id <- as.integer(df$city_id)
  df$population <- as.numeric(df$population)
  df$mass <- as.numeric(df$mass)
  df <- df[df$population >= 1 & df$mass > 0, ]
  df
}

load_subway <- function() {
  df <- read.csv(subway_file, stringsAsFactors = FALSE)
  df <- df[, c("city_id", "h3index", "subway_mass_tonnes")]
  names(df) <- c("city_id", "h3index", "subway_mass")
  df$city_id <- as.integer(df$city_id)
  df$h3index <- as.character(df$h3index)
  df$subway_mass <- as.numeric(df$subway_mass)
  df
}

fit_delta <- function(df) {
  df$Country_City <- paste(df$country, df$city_id, sep = "_")
  df$log_pop <- log10(df$population)
  df$log_mass <- log10(df$mass)

  # Keep only cities with >= 2 neighborhoods
  n_by_city <- ave(df$log_pop, df$Country_City, FUN = length)
  df <- df[n_by_city >= 2, ]

  df$log_pop_centered <- df$log_pop - ave(df$log_pop, df$Country_City, FUN = mean)
  df$log_mass_centered <- df$log_mass - ave(df$log_mass, df$Country_City, FUN = mean)

  mod <- lm(log_mass_centered ~ log_pop_centered, data = df)
  se <- summary(mod)$coefficients[2, 2]

  list(
    delta = coef(mod)[2],
    se = se,
    r2 = summary(mod)$r.squared,
    n_neighborhoods = nrow(df),
    n_cities = length(unique(df$city_id)),
    n_countries = length(unique(df$country))
  )
}

# =============================================================================
# STEP 1: Load R7 data and apply per-resolution filter (on baseline)
# =============================================================================

cat("=== Loading R7 data ===\n")
df <- load_mass()
cat("R7 after pop >= 1, mass > 0:", nrow(df), "\n")

# Cities with total pop > 50,000
agg_pop <- aggregate(population ~ country_iso + city_id, data = df, FUN = sum)
keep_city <- agg_pop$city_id[agg_pop$population > 50000]
df <- df[df$city_id %in% keep_city, ]

# Cities with >= 10 neighborhoods
agg_n <- aggregate(h3index ~ country_iso + city_id, data = df, FUN = length)
keep_city2 <- agg_n$city_id[agg_n$h3index >= 10]
df <- df[df$city_id %in% keep_city2, ]

# Countries with >= 5 qualifying cities
agg_c <- aggregate(city_id ~ country_iso,
  data = unique(df[, c("country_iso", "city_id")]), FUN = length)
keep_country <- agg_c$country_iso[agg_c$city_id >= 5]
df <- df[df$country_iso %in% keep_country, ]

r7_city_ids <- unique(df$city_id)
r7_country_isos <- unique(df$country_iso)

cat("R7 per-resolution filter:", length(r7_city_ids), "cities in",
    length(r7_country_isos), "countries\n\n")

# =============================================================================
# STEP 2: Fit delta without and with subway
# =============================================================================

# Without subway (baseline)
cat("--- Without subway ---\n")
base <- fit_delta(df)
cat(sprintf("  delta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d\n",
            base$delta, base$delta - 1.96 * base$se, base$delta + 1.96 * base$se,
            base$r2, base$n_neighborhoods))

# With subway
cat("--- With subway ---\n")
sub <- load_subway()
df_sub <- df  # copy
key_sub <- paste(sub$city_id, sub$h3index, sep = "__")
key_df <- paste(df_sub$city_id, df_sub$h3index, sep = "__")
m <- match(key_df, key_sub)
add <- sub$subway_mass[m]
add[is.na(add)] <- 0
df_sub$mass <- df_sub$mass + add

subw <- fit_delta(df_sub)
cat(sprintf("  delta = %.4f [%.4f, %.4f], R2 = %.4f, N = %d\n",
            subw$delta, subw$delta - 1.96 * subw$se, subw$delta + 1.96 * subw$se,
            subw$r2, subw$n_neighborhoods))

# =============================================================================
# OUTPUT TABLE
# =============================================================================

out <- data.frame(
  Resolution = c(7, 7),
  model = c("without_subway", "with_subway"),
  delta = c(base$delta, subw$delta),
  se = c(base$se, subw$se),
  ci_low = c(base$delta - 1.96 * base$se, subw$delta - 1.96 * subw$se),
  ci_high = c(base$delta + 1.96 * base$se, subw$delta + 1.96 * subw$se),
  r2 = c(base$r2, subw$r2),
  n_neighborhoods = c(base$n_neighborhoods, subw$n_neighborhoods),
  n_cities = c(base$n_cities, subw$n_cities),
  n_countries = c(base$n_countries, subw$n_countries),
  delta_change = c(0, subw$delta - base$delta),
  stringsAsFactors = FALSE
)

outfile <- file.path(results_dir, "Fig3_Decentered_Delta_R7_PerRes_WithVsWithoutSubway.csv")
write.csv(out, outfile, row.names = FALSE)

cat("\n========================================\n")
cat("R7 PER-RESOLUTION FILTER SUBWAY IMPACT\n")
cat("========================================\n")
cat(sprintf("  Cities: %d, Countries: %d\n", length(r7_city_ids), length(r7_country_isos)))
print(out)
cat(sprintf("\n  Delta change: %+.4f (%.2f%%)\n",
            subw$delta - base$delta,
            (subw$delta - base$delta) / base$delta * 100))
cat("Saved:", outfile, "\n")
cat("========================================\n")
