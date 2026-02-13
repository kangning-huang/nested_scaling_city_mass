# =============================================================================
# Fig3_NeighborhoodScaling_Decentered_Multiscale_R6Filter_WithSubwayMass.R
# =============================================================================
# Neighborhood (H3 hex) de-centered scaling across resolutions (5/6/7).
#
# Baseline: total_built_mass_tons
# With subway: total_built_mass_tons + subway_mass_tonnes (hex-level)
#
# IMPORTANT: We derive the R6 filter set WITHOUT subway (baseline), then apply
# the same filter set to all resolutions for both baseline and subway runs.
#
# Minimal deps: base R only (faster / fewer packages).
# Run from: _Revision1/scripts/
# Output:
#   ../results/Fig3_Decentered_Delta_R6Filter_WithVsWithoutSubway.csv
# =============================================================================

rm(list = ls())

data_dir <- file.path("..", "data")
results_dir <- file.path("..", "results")

mass_files <- list(
  "5" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution5_2026-02-02.csv"),
  "6" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution6_2025-06-24.csv"),
  "7" = file.path(data_dir, "Fig3_Mass_Neighborhood_H3_Resolution7_2026-02-02.csv")
)

subway_files <- list(
  "5" = file.path(results_dir, "subway_mass_by_hexagon_resolution5.csv"),
  "6" = file.path(results_dir, "subway_mass_by_hexagon_resolution6.csv"),
  "7" = file.path(results_dir, "subway_mass_by_hexagon_resolution7.csv")
)

load_mass <- function(res) {
  df <- read.csv(mass_files[[res]], stringsAsFactors = FALSE)
  keep <- c("h3index", "UC_NM_MN", "ID_HDC_G0", "CTR_MN_ISO", "CTR_MN_NM", "population_2015", "total_built_mass_tons")
  df <- df[, keep]
  names(df) <- c("h3index", "city_name", "city_id", "country_iso", "country", "population", "mass")
  df$h3index <- as.character(df$h3index)
  df$city_id <- as.integer(df$city_id)
  df$population <- as.numeric(df$population)
  df$mass <- as.numeric(df$mass)
  df <- df[df$population >= 1 & df$mass > 0, ]
  df
}

load_subway <- function(res) {
  df <- read.csv(subway_files[[res]], stringsAsFactors = FALSE)
  df <- df[, c("city_id", "h3index", "subway_mass_tonnes")]
  names(df) <- c("city_id", "h3index", "subway_mass")
  df$city_id <- as.integer(df$city_id)
  df$h3index <- as.character(df$h3index)
  df$subway_mass <- as.numeric(df$subway_mass)
  df
}

fit_delta <- function(df) {
  # De-center within city
  df$Country_City <- paste(df$country, df$city_id, sep = "_")
  df$log_pop <- log10(df$population)
  df$log_mass <- log10(df$mass)

  # Keep only cities with >=2 neighborhoods
  n_by_city <- ave(df$log_pop, df$Country_City, FUN = length)
  df <- df[n_by_city >= 2, ]

  df$log_pop_centered <- df$log_pop - ave(df$log_pop, df$Country_City, FUN = mean)
  df$log_mass_centered <- df$log_mass - ave(df$log_mass, df$Country_City, FUN = mean)

  mod <- lm(log_mass_centered ~ log_pop_centered, data = df)
  se <- summary(mod)$coefficients[2, 2]

  list(
    delta = coef(mod)[2],
    se = se,
    n_neighborhoods = nrow(df),
    n_cities = length(unique(df$city_id)),
    n_countries = length(unique(df$country))
  )
}

# -----------------------------------------------------------------------------
# STEP 1: Determine baseline R6 filter set (WITHOUT subway)
# -----------------------------------------------------------------------------

cat("\n=== Determining R6 filter set (baseline: without subway) ===\n")

df6 <- load_mass("6")

# Cities with total pop > 50k
agg_pop <- aggregate(population ~ country_iso + city_id, data = df6, FUN = sum)
keep_city <- agg_pop$city_id[agg_pop$population > 50000]

df6 <- df6[df6$city_id %in% keep_city, ]

# Cities with >=10 neighborhoods
agg_n <- aggregate(h3index ~ country_iso + city_id, data = df6, FUN = length)
keep_city2 <- agg_n$city_id[agg_n$h3index >= 10]

df6 <- df6[df6$city_id %in% keep_city2, ]

# Countries with >=5 qualifying cities
agg_c <- aggregate(city_id ~ country_iso, data = unique(df6[, c("country_iso", "city_id")]), FUN = length)
keep_country <- agg_c$country_iso[agg_c$city_id >= 5]

df6 <- df6[df6$country_iso %in% keep_country, ]

r6_city_ids <- unique(df6$city_id)
r6_country_isos <- unique(df6$country_iso)

cat("R6 filter set (baseline):", length(r6_city_ids), "cities in", length(r6_country_isos), "countries\n")

# -----------------------------------------------------------------------------
# STEP 2: Fit deltas (baseline vs with subway) for each resolution
# -----------------------------------------------------------------------------

run_one <- function(res, with_subway = FALSE) {
  df <- load_mass(res)
  df <- df[df$city_id %in% r6_city_ids & df$country_iso %in% r6_country_isos, ]

  if (with_subway) {
    sub <- load_subway(res)
    key_sub <- paste(sub$city_id, sub$h3index, sep = "__")
    key_df <- paste(df$city_id, df$h3index, sep = "__")
    m <- match(key_df, key_sub)
    add <- sub$subway_mass[m]
    add[is.na(add)] <- 0
    df$mass <- df$mass + add
  }

  fit_delta(df)
}

rows <- list()
for (res in c("5", "6", "7")) {
  base <- run_one(res, FALSE)
  subw <- run_one(res, TRUE)

  rows[[paste0(res, "_base")]] <- data.frame(
    Resolution = res,
    model = "without_subway",
    delta = base$delta,
    se = base$se,
    ci_low = base$delta - 1.96 * base$se,
    ci_high = base$delta + 1.96 * base$se,
    n_neighborhoods = base$n_neighborhoods,
    n_cities = base$n_cities,
    n_countries = base$n_countries,
    stringsAsFactors = FALSE
  )

  rows[[paste0(res, "_subw")]] <- data.frame(
    Resolution = res,
    model = "with_subway",
    delta = subw$delta,
    se = subw$se,
    ci_low = subw$delta - 1.96 * subw$se,
    ci_high = subw$delta + 1.96 * subw$se,
    n_neighborhoods = subw$n_neighborhoods,
    n_cities = subw$n_cities,
    n_countries = subw$n_countries,
    stringsAsFactors = FALSE
  )
}

out <- do.call(rbind, rows)

# change vs baseline within each resolution
out$delta_change_vs_base <- NA
for (res in c("5", "6", "7")) {
  b <- out$delta[out$Resolution == res & out$model == "without_subway"]
  out$delta_change_vs_base[out$Resolution == res] <- out$delta[out$Resolution == res] - b
}

out <- out[order(as.integer(out$Resolution), out$model), ]

write.csv(out, file.path(results_dir, "Fig3_Decentered_Delta_R6Filter_WithVsWithoutSubway.csv"), row.names = FALSE)

cat("\n================ FIG3 (NEIGHBORHOOD) RESULTS ================\n")
print(out)
cat("Output:", file.path(results_dir, "Fig3_Decentered_Delta_R6Filter_WithVsWithoutSubway.csv"), "\n")
cat("===========================================================\n")
