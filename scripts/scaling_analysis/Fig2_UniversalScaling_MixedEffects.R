# =============================================================================
# Fig2_UniversalScaling_MixedEffects.R
#
# Original submission approach (converted from Rmd to R)
# Uses mixed-effects normalization for Panel B
#
# Key method: Fig 2B uses mixed-effects model to extract random intercepts,
#   then normalizes mass by subtracting country-specific random intercepts
#   before running OLS on the pooled data.
#
# Sister script to Fig2_UniversalScaling_Decentered.R for comparison
# =============================================================================

rm(list = ls())

# Suppress default Rplots.pdf output when running non-interactively
if (!interactive()) pdf(NULL)

# =============================================================================
# 1. Packages
# =============================================================================
library(pacman)
p_load(tidyverse, gridGraphics, patchwork, zoo, scales, broom, lme4,
       cowplot, RColorBrewer, ggrepel)

# =============================================================================
# 2. Input Data
# =============================================================================
DF_MasterMass <- read.csv("../../data/processed/MasterMass_ByClass20250616.csv",
                          stringsAsFactors = FALSE)

# Filter to valid countries (>=5 cities) and positive mass
city_counts <- table(DF_MasterMass$CTR_MN_NM)
valid_countries <- names(city_counts[city_counts >= 5])  # 75 countries
data_panel_a <- subset(DF_MasterMass,
                       CTR_MN_NM %in% valid_countries & total_built_mass_tons > 0)

cat("Number of cities:", nrow(data_panel_a), "\n")
cat("Number of countries:", length(unique(data_panel_a$CTR_MN_NM)), "\n")

# =============================================================================
# 3. Country-level statistics (UNCHANGED from original)
# =============================================================================
country_stats <- data_panel_a %>%
  group_by(CTR_MN_NM) %>%
  filter(n() >= 5) %>%
  group_modify(~ {
    mod <- lm(log10(total_built_mass_tons) ~ log10(population_2015), data = .x)
    s   <- summary(mod)
    tibble(
      n_cities      = nrow(.x),
      Intercept     = coef(mod)[1],
      Intercept_SE  = s$coefficients[1, 2],
      Intercept_t   = 10^coef(mod)[1],
      Slope         = coef(mod)[2],
      Slope_SE      = s$coefficients[2, 2],
      R_squared     = s$r.squared,
      p_value_slope = s$coefficients[2, 4]
    )
  }) %>%
  ungroup() %>%
  dplyr::rename(Country = CTR_MN_NM) %>%
  arrange(desc(n_cities))

# Write country stats
write_csv(country_stats, "../data/country_summary_table_mixedeffects.csv")

# =============================================================================
# 4. Panel A: Example Countries (UNCHANGED from original)
# =============================================================================
common_ylims <- c(1e5, 1e10)
top_countries <- c("China", "United States", "India", "Indonesia")

plot_df <- data_panel_a %>%
  mutate(
    CountryGroup = ifelse(CTR_MN_NM %in% top_countries, CTR_MN_NM, "Other")
  )

palette5 <- brewer.pal(5, "Set1")
color_vec <- c(
  setNames(palette5, top_countries),
  Other = "grey80"
)

# Label positions for top countries
label_df <- plot_df %>%
  filter(CTR_MN_NM %in% top_countries) %>%
  group_by(CTR_MN_NM) %>%
  slice_max(population_2015, n = 1) %>%
  ungroup() %>%
  mutate(
    x = population_2015 * 0.95,
    y = total_built_mass_tons * 1.05
  ) %>%
  select(Country = CTR_MN_NM, x, y)

panel_exampleCountry <- ggplot(plot_df, aes(x = population_2015, y = total_built_mass_tons)) +
  # Background points (all grey)
  geom_point(
    shape  = 21,
    fill   = "grey",
    color  = "black",
    stroke = 0.2,
    alpha  = 0.2,
    size   = 1.5
  ) +
  # Grey regression lines for "Other" countries
  stat_smooth(
    data   = filter(plot_df, CountryGroup == "Other"),
    method = "lm",
    se     = FALSE,
    aes(group = CTR_MN_NM),
    color  = "grey80",
    size   = 0.3
  ) +
  # Colored regression lines for top countries
  stat_smooth(
    data   = filter(plot_df, CountryGroup %in% top_countries),
    method = "lm",
    se     = FALSE,
    aes(group = CTR_MN_NM, color = CountryGroup),
    size   = 0.7
  ) +
  scale_color_manual(
    values = color_vec,
    breaks = top_countries
  ) +
  # Direct labels
  geom_text_repel(
    data          = label_df,
    aes(x = x, y = y, label = Country, color = Country),
    direction     = "y",
    hjust         = 0,
    segment.color = NA,
    size          = 3,
    show.legend   = FALSE
  ) +
  # Log-log scales
  scale_x_log10(
    name   = "Urban population",
    expand = c(0.02, 0.02),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  scale_y_log10(
    name   = "Total built mass (tonnes)",
    limits = common_ylims,
    expand = c(0.02, 0.02),
    labels = scales::trans_format("log10", scales::math_format(10^.x))
  ) +
  # Theme
  theme_bw() +
  theme(
    legend.position    = "none",
    axis.ticks         = element_line(colour = "black", size = 0.1),
    axis.line          = element_line(colour = "black", size = 0.5),
    text               = element_text(size = 9),
    axis.text          = element_text(size = 8),
    panel.grid.major.y = element_line(size = 0.2, color = "grey90", linetype = 2),
    panel.grid.major.x = element_blank(),
    panel.grid.minor   = element_blank(),
    panel.background   = element_blank()
  ) +
  labs(
    x = "Urban population",
    y = "Total built mass (tonnes)"
  )

# =============================================================================
# 5. Histogram Insets (UPDATED: mean instead of median per reviewer)
# =============================================================================
# Slope histogram - use MEAN (orange line)
mean_slope <- mean(country_stats$Slope, na.rm = TRUE)
slope_limits <- range(country_stats$Slope, na.rm = TRUE)

panel_slope <- ggplot(country_stats, aes(x = Slope)) +
  geom_histogram(bins = 21, fill = "gray", color = "black", alpha = 0.7) +
  geom_vline(xintercept = mean_slope, linetype = "solid", color = "#fc8d62", size = 0.8) +
  geom_vline(xintercept = 1, linetype = "dashed", color = "black", size = 0.8) +
  scale_x_continuous(limits = slope_limits) +
  theme_bw(base_size = 9) +
  theme(
    axis.ticks  = element_blank(),
    axis.text.y = element_blank(),
    panel.grid  = element_blank(),
    plot.margin = margin(0, 0, 0, 0)
  ) +
  xlab(expression("Country-specific "*beta)) +
  ylab(NULL)

# Intercept histogram - use MEAN in log space, then back-transform
mean_log_intercept <- mean(country_stats$Intercept, na.rm = TRUE)  # mean of log values
mean_intercept_t <- 10^mean_log_intercept  # back-transform to linear scale
intercept_limits <- range(country_stats$Intercept_t, na.rm = TRUE)

panel_intercept <- ggplot(country_stats, aes(x = Intercept_t)) +
  geom_histogram(bins = 21, fill = "gray", color = "black", alpha = 0.7) +
  geom_vline(xintercept = mean_intercept_t, linetype = "solid", color = "#fc8d62", size = 0.8) +
  scale_x_log10(
    limits = intercept_limits,
    expand = c(0.02, 0.02),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  theme_bw(base_size = 9) +
  theme(
    axis.ticks  = element_blank(),
    axis.text.y = element_blank(),
    panel.grid  = element_blank(),
    plot.margin = margin(0, 0, 0, 0)
  ) +
  xlab(expression("Country-specific M"[0]*" (t / cap"^beta*")")) +
  ylab(NULL)

cat("Mean slope:", mean_slope, "\n")
cat("Mean intercept (t):", mean_intercept_t, "\n")

# =============================================================================
# 6. Panel B: Mixed-Effects Normalized Scatter (ORIGINAL APPROACH)
# =============================================================================

# Step 1: Fit mixed-effects model
mixed_model <- lmer(log10(total_built_mass_tons) ~ log10(population_2015) + (1 | CTR_MN_NM),
                    data = data_panel_a)

# Extract fixed and random effects
fixed_effects <- fixef(mixed_model)
random_effects <- ranef(mixed_model)$CTR_MN_NM

cat("\n=== Mixed-Effects Model ===\n")
cat("Fixed intercept:", round(fixed_effects[1], 4), "\n")
cat("Fixed slope:", round(fixed_effects[2], 4), "\n")

# Step 2: Normalize mass by removing country-specific random intercepts
normalized_data <- data_panel_a
normalized_data <- merge(normalized_data, random_effects,
                         by.x = "CTR_MN_NM", by.y = "row.names", all.x = TRUE)
colnames(normalized_data)[ncol(normalized_data)] <- "random_intercept"

# Normalize: subtract random intercept from log(mass), then back-transform
normalized_data$MassValue_normalized <- with(normalized_data,
                                              10^(log10(total_built_mass_tons) - random_intercept))

# Reference line for linear scaling (200 tonnes per capita)
mass_per_cap <- 200
normalized_data <- normalized_data %>%
  mutate(mass_linear_est = mass_per_cap * population_2015)

# Step 3: OLS on normalized data
mod_normalized <- lm(log10(MassValue_normalized) ~ log10(population_2015),
                     data = normalized_data)
summary_mod <- summary(mod_normalized)

# Extract statistics
beta_mixed <- coef(mod_normalized)[2]
se_mixed <- summary_mod$coefficients[2, 2]
r2_mixed <- summary_mod$r.squared
intercept_mixed <- coef(mod_normalized)[1]
n_cities <- nrow(normalized_data)

# 95% CI
ci_lower <- beta_mixed - 1.96 * se_mixed
ci_upper <- beta_mixed + 1.96 * se_mixed

cat("\n=== Mixed-Effects Normalized OLS Results ===\n")
cat("Beta:", round(beta_mixed, 4), "\n")
cat("95% CI: [", round(ci_lower, 4), ",", round(ci_upper, 4), "]\n")
cat("SE:", round(se_mixed, 4), "\n")
cat("R-squared:", round(r2_mixed, 3), "\n")
cat("Intercept:", round(intercept_mixed, 4), "\n")
cat("N cities:", n_cities, "\n")

# Create normalized scatter plot
plot_normalized <- ggplot(normalized_data,
                          aes(x = population_2015, y = MassValue_normalized)) +
  geom_point(
    shape = 21,
    color = "black",
    fill = "#8da0cb",
    stroke = 0.3,
    alpha = 0.5,
    size = 1.5
  ) +
  # Highlight New York
  geom_point(
    data = subset(normalized_data, UC_NM_MN == "New York"),
    aes(x = population_2015, y = MassValue_normalized),
    shape = 21,
    color = "black",
    fill = "gray",
    size = 3,
    stroke = 0.5
  ) +
  # Regression line
  stat_smooth(
    method = "lm",
    se = FALSE,
    color = "#fc8d62",
    size = 0.5,
    alpha = 0.8
  ) +
  # Reference line: linear scaling (mass = 200 * pop)
  geom_line(aes(x = population_2015, y = mass_linear_est),
            linetype = 2, linewidth = 0.5, color = "black") +
  # Log-log scales
  scale_x_log10(
    name = "Urban population",
    expand = c(0.02, 0.02),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  scale_y_log10(
    position = "right",
    name = "Normalized total built mass (tonnes)",
    limits = common_ylims,
    expand = c(0.02, 0.02),
    labels = trans_format("log10", math_format(10^.x))
  ) +
  theme(legend.position = "none") +
  theme_bw() +
  theme(
    axis.ticks = element_line(colour = "black", size = 0.1),
    axis.line = element_line(colour = "black", size = 0.5),
    text = element_text(size = 9),
    axis.text = element_text(size = 8),
    panel.grid.major.y = element_line(size = 0.2, color = "grey90", linetype = 2),
    panel.grid.major.x = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank()
  ) +
  # Annotations
  annotate("text", x = 1e5, y = 10^9.3, size = 3,
           label = sprintf("beta==%.2f~(n==%d)", round(beta_mixed, 2), n_cities),
           parse = TRUE) +
  annotate("text", x = 1e5, y = 10^9, size = 3,
           label = sprintf("R^2==%.2f", round(r2_mixed, 2)),
           parse = TRUE)

# =============================================================================
# 7. Per-capita Inset (ORIGINAL for normalized data)
# =============================================================================
theme_inset <- function() {
  theme_bw(base_size = 7) +
    theme(
      axis.text = element_blank(),
      axis.ticks = element_blank(),
      panel.grid = element_blank(),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.border = element_rect(color = "black", fill = NA),
      legend.position = "none"
    )
}

# Per-capita mass
data_panel_b <- normalized_data %>%
  mutate(PerCapitaMass = MassValue_normalized / population_2015)

plot_b <- ggplot(data_panel_b, aes(x = population_2015, y = PerCapitaMass)) +
  geom_point(pch = 21, alpha = 0.05) +
  geom_point(
    data = subset(data_panel_b, UC_NM_MN == "New York"),
    aes(x = population_2015, y = PerCapitaMass),
    shape = 21,
    color = "black",
    fill = "gray",
    size = 3,
    stroke = 0.5
  ) +
  scale_x_log10() +
  scale_y_log10() +
  stat_smooth(method = "lm", se = FALSE, color = "#fc8d62", size = 0.5) +
  # Reference line at 200 tonnes/capita
  geom_hline(yintercept = 200, linetype = 2, color = "black", size = 0.5) +
  xlab("Urban population") +
  ylab("Per-cap built mass (t/cap)") +
  theme_inset()

# =============================================================================
# 8. Assemble Final Figure
# =============================================================================

# Panel A with insets
panel_A <- panel_exampleCountry +
  inset_element(panel_slope,
                left   = 0.25, bottom = 0.02,
                right  = 0.6,  top    = 0.32) +
  inset_element(panel_intercept,
                left   = 0.64, bottom = 0.02,
                right  = 0.99, top    = 0.32)

# Panel B with per-capita inset
panel_B <- plot_normalized +
  inset_element(plot_b,
                left   = 0.45, bottom = 0.02,
                right  = 0.98, top    = 0.45)

# Combine side by side
final_figure <-
  (panel_A | panel_B) +
  plot_layout(ncol = 2, widths = c(7.5, 7.5))

print(final_figure)

# =============================================================================
# 9. Save Figure
# =============================================================================
current_date <- Sys.Date()

ggsave(
  filename = paste0("../figures/Fig2_MixedEffects_", current_date, ".pdf"),
  plot = final_figure,
  device = "pdf",
  width = 30,
  height = 12,
  units = "cm",
  dpi = 300
)

cat("\nFigure saved to: figures/Fig2_MixedEffects_", current_date, ".pdf\n")

# =============================================================================
# 10. Summary Statistics for Comparison
# =============================================================================
cat("\n========================================\n")
cat("SUMMARY: MIXED-EFFECTS APPROACH\n")
cat("========================================\n")
cat("Mixed-effects normalization approach:\n")
cat("  1. Fit lmer(log10(mass) ~ log10(pop) + (1|Country))\n")
cat("  2. Extract random intercepts per country\n")
cat("  3. Normalize: log10(mass) - random_intercept\n")
cat("  4. OLS on pooled normalized data\n")
cat("\nResults:\n")
cat("  Beta (slope):", round(beta_mixed, 4), "\n")
cat("  95% CI: [", round(ci_lower, 4), ",", round(ci_upper, 4), "]\n")
cat("  SE:", round(se_mixed, 4), "\n")
cat("  R-squared:", round(r2_mixed, 3), "\n")
cat("  Intercept:", round(intercept_mixed, 4), "\n")
cat("  N cities:", n_cities, "\n")
cat("  N countries:", length(unique(normalized_data$CTR_MN_NM)), "\n")
cat("========================================\n")
