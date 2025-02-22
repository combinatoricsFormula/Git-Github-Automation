#!/bin/bash

# Prompt the user for the file path
echo "Please enter the path to your CSV file:"
read file_path

# Create a temporary R script
r_script=$(mktemp)

# Write the R script to the temporary file
cat <<EOL > $r_script
# Load necessary libraries
library(ggplot2)
library(readr)
library(nortest)
library(e1071)
library(dplyr)
library(purrr)
library(bayesplot)
library(rjags)

# Import data
data <- read_csv("$file_path")

# Assuming the data is in the first column
data_values <- data[[1]]

# Data Cleaning
clean_data <- na.omit(data_values)

# Descriptive Statistics
descriptive_stats <- data.frame(
  Mean = mean(clean_data),
  Median = median(clean_data),
  Std_Deviation = sd(clean_data),
  Variance = var(clean_data),
  Skewness = skewness(clean_data),
  Kurtosis = kurtosis(clean_data),
  Summary = summary(clean_data)
)
print("Descriptive Statistics:")
print(descriptive_stats)

# Standardization (Z-scores)
z_scores <- scale(clean_data)
print("Z-scores:")
print(z_scores)

# Distribution Fitting and Tests
ggplot(data.frame(clean_data), aes(x = clean_data)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  ggtitle("Histogram and Density Plot")

qqnorm(clean_data)
qqline(clean_data, col = "red")

# Normality Tests
shapiro_test <- shapiro.test(clean_data)
ad_test <- ad.test(clean_data)

# Poisson Distribution Fit
pois_lambda <- mean(clean_data)
pois_fit <- dpois(clean_data, pois_lambda)

# Bayesian Inference
model_string <- "model {
  for (i in 1:length(y)) {
    y[i] ~ dnorm(mu, tau)
  }
  mu ~ dnorm(0, 0.0001)
  tau <- pow(sigma, -2)
  sigma ~ dunif(0, 100)
}"
data_jags <- list(y = clean_data)
params <- c("mu", "sigma")
inits <- function() {list(mu = 0, sigma = 1)}

jags_model <- jags.model(textConnection(model_string), data = data_jags, inits = inits, n.chains = 3)
update(jags_model, 1000)  # Burn-in
mcmc_samples <- coda.samples(jags_model, variable.names = params, n.iter = 5000)
print(mcmc_samples)

# Nelson's Rules for Control Charts
control_limits <- function(x) {
  mean_x <- mean(x)
  sd_x <- sd(x)
  upper_control_limit <- mean_x + 3 * sd_x
  lower_control_limit <- mean_x - 3 * sd_x
  return(c(upper_control_limit, lower_control_limit))
}

cl <- control_limits(clean_data)
ucl <- cl[1]
lcl <- cl[2]
mean_val <- mean(clean_data)

ggplot(data.frame(index = 1:length(clean_data), value = clean_data), aes(x = index, y = value)) +
  geom_line() +
  geom_hline(yintercept = mean_val, col = "blue") +
  geom_hline(yintercept = ucl, col = "red", linetype = "dashed") +
  geom_hline(yintercept = lcl, col = "red", linetype = "dashed") +
  ggtitle("Control Chart with Nelson's Rules")

# Save all plots and results
ggsave("hist_density_plot.png")
png("qq_plot.png")
qqnorm(clean_data)
qqline(clean_data, col = "red")
dev.off()

# Save results to a text file
write.table(descriptive_stats, "results.txt", append = TRUE, col.names = NA, sep = "\t")
write(paste("Shapiro-Wilk Test:", shapiro_test), "results.txt", append = TRUE)
write(paste("Anderson-Darling Test:", ad_test), "results.txt", append = TRUE)
write(paste("Poisson Lambda:", pois_lambda), "results.txt", append = TRUE)
write("Bayesian Inference Results:", "results.txt", append = TRUE)
write(summary(mcmc_samples), "results.txt", append = TRUE)

print("Analysis complete. Results saved to 'results.txt'.")
EOL

# Execute the R script
Rscript $r_script

# Clean up the temporary R script
rm $r_script
