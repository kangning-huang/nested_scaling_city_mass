# =============================================================================
# Fig2_UniversalScaling_Decentered.R
#
# Revision for Nature Cities R1 - Reviewer #1 Response
# Implements de-centering approach from Bettencourt & Lobo (2016)
#
# Key change: Fig 2B uses de-centering instead of mixed-effects normalization
#   - De-center both log(pop) and log(mass) by country means
#   - OLS on pooled de-centered data
#   - Axes show deviation from country means
#
# All other panels (2A, insets) remain identical to original
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
write_csv(country_stats, "../data/country_summary_table_decentered.csv")

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
# 6. Panel B: De-centered Pooled Scatter (NEW - Bettencourt & Lobo 2016)
# =============================================================================

# De-center by country: subtract country mean from both log(pop) and log(mass)
decentered_data <- data_panel_a %>%
  mutate(
    log_pop = log10(population_2015),
    log_mass = log10(total_built_mass_tons)
  ) %>%
  group_by(CTR_MN_NM) %>%
  mutate(
    mean_log_pop = mean(log_pop),
    mean_log_mass = mean(log_mass),
    # De-center both X and Y
    log_pop_centered = log_pop - mean_log_pop,
    log_mass_centered = log_mass - mean_log_mass
  ) %>%
  ungroup()

# OLS on pooled de-centered data
mod_decentered <- lm(log_mass_centered ~ log_pop_centered, data = decentered_data)
summary_mod <- summary(mod_decentered)

# Extract statistics
beta_decentered <- coef(mod_decentered)[2]
se_decentered <- summary_mod$coefficients[2, 2]
r2_decentered <- summary_mod$r.squared
intercept_decentered <- coef(mod_decentered)[1]  # Should be ~0
n_cities <- nrow(decentered_data)

# 95% CI
ci_lower <- beta_decentered - 1.96 * se_decentered
ci_upper <- beta_decentered + 1.96 * se_decentered

cat("\n=== De-centered OLS Results ===\n")
cat("Beta:", round(beta_decentered, 4), "\n")
cat("95% CI: [", round(ci_lower, 4), ",", round(ci_upper, 4), "]\n")
cat("SE:", round(se_decentered, 4), "\n")
cat("R-squared:", round(r2_decentered, 3), "\n")
cat("Intercept (should be ~0):", round(intercept_decentered, 6), "\n")
cat("N cities:", n_cities, "\n")

# Mark New York in de-centered data
decentered_data <- decentered_data %>%
  mutate(is_newyork = UC_NM_MN == "New York")

# Create de-centered scatter plot
plot_decentered <- ggplot(decentered_data,
                          aes(x = log_pop_centered, y = log_mass_centered)) +
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
    data = subset(decentered_data, is_newyork),
    aes(x = log_pop_centered, y = log_mass_centered),
    shape = 21,
    color = "black",
    fill = "gray",
    size = 3,
    stroke = 0.5
  ) +
  # Regression line (de-centered OLS)
  stat_smooth(
    method = "lm",
    se = FALSE,
    color = "#fc8d62",
    size = 0.5,
    alpha = 0.8
  ) +
  # Reference line: slope = 1 (linear scaling)
  geom_abline(intercept = 0, slope = 1, linetype = 2, linewidth = 0.5, color = "black") +
  # Axis labels for de-centered space
  scale_x_continuous(
    name = "Log population (deviation from country mean)",
    expand = c(0.02, 0.02)
  ) +
  scale_y_continuous(
    position = "right",
    name = "Log mass (deviation from country mean)",
    expand = c(0.02, 0.02)
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
  # Annotations with de-centered results
  annotate("text", x = min(decentered_data$log_pop_centered) + 0.3,
           y = max(decentered_data$log_mass_centered) - 0.2,
           size = 3, hjust = 0,
           label = sprintf("beta==%.2f~(n==%d)", round(beta_decentered, 2), n_cities),
           parse = TRUE) +
  annotate("text", x = min(decentered_data$log_pop_centered) + 0.3,
           y = max(decentered_data$log_mass_centered) - 0.5,
           size = 3, hjust = 0,
           label = sprintf("R^2==%.2f", round(r2_decentered, 2)),
           parse = TRUE)

# =============================================================================
# 7. Per-capita Inset (UPDATED for de-centered data)
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

# Per-capita in de-centered space: log(M/P) - country_mean vs log(P) - country_mean
# Note: log(M/P) = log(M) - log(P)
# In de-centered terms: (log_mass - mean_log_mass) - (log_pop - mean_log_pop)
data_panel_b <- decentered_data %>%
  mutate(
    PerCapita_centered = log_mass_centered - log_pop_centered
  )

plot_b <- ggplot(data_panel_b, aes(x = log_pop_centered, y = PerCapita_centered)) +
  geom_point(pch = 21, alpha = 0.05) +
  geom_point(
    data = subset(data_panel_b, is_newyork),
    aes(x = log_pop_centered, y = PerCapita_centered),
    shape = 21,
    color = "black",
    fill = "gray",
    size = 3,
    stroke = 0.5
  ) +
  stat_smooth(method = "lm", se = FALSE, color = "#fc8d62", size = 0.5) +
  # Reference line at y = 0 (constant per-capita)
  geom_hline(yintercept = 0, linetype = 2, color = "black", size = 0.5) +
  xlab("Log pop (centered)") +
  ylab("Log per-cap mass (centered)") +
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
panel_B <- plot_decentered +
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
  filename = paste0("../figures/Fig2_Decentered_", current_date, ".pdf"),
  plot = final_figure,
  device = "pdf",
  width = 30,
  height = 12,
  units = "cm",
  dpi = 300
)

cat("\nFigure saved to: figures/Fig2_Decentered_", current_date, ".pdf\n")

# =============================================================================
# 10. Summary Statistics for Revision Letter
# =============================================================================
cat("\n========================================\n")
cat("SUMMARY FOR REVISION LETTER\n")
cat("========================================\n")
cat("De-centering approach (Bettencourt & Lobo 2016):\n")
cat("  - Each city's log(pop) and log(mass) centered by country mean\n")
cat("  - OLS on pooled de-centered data\n")
cat("\nResults:\n")
cat("  Beta (slope):", round(beta_decentered, 4), "\n")
cat("  95% CI: [", round(ci_lower, 4), ",", round(ci_upper, 4), "]\n")
cat("  R-squared:", round(r2_decentered, 3), "\n")
cat("  N cities:", n_cities, "\n")
cat("  N countries:", length(unique(decentered_data$CTR_MN_NM)), "\n")
cat("\nNote: Intercept ~0 by construction (", round(intercept_decentered, 6), ")\n")
cat("========================================\n")
