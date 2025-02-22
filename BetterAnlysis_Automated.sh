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
library(caret)
library(randomForest)
library(ggpubr)
library(psych)
library(MASS)
library(xgboost)
library(ROCR)

# Import data
data <- read_csv("$file_path")

# Assuming the data is in the first column
data_values <- data[[1]]

# Data Cleaning
clean_data <- na.omit(data_values)

# Step 1: Descriptive Statistics
descriptive_stats <- describe(clean_data)
print("Descriptive Statistics:")
print(descriptive_stats)

# Step 2: Basic Plotting
# Histogram and Density Plot
ggplot(data.frame(clean_data), aes(x = clean_data)) +
  geom_histogram(aes(y = ..density..), bins = 30, fill = "blue", alpha = 0.7) +
  geom_density(color = "red", size = 1) +
  ggtitle("Histogram and Density Plot")

# QQ-Plot
qqnorm(clean_data)
qqline(clean_data, col = "red")

# Step 3: Probability and Probability Distributions
# Normal Distribution Tests
shapiro_test <- shapiro.test(clean_data)
ad_test <- ad.test(clean_data)

# Poisson Distribution Fit
pois_lambda <- mean(clean_data)
pois_fit <- dpois(clean_data, pois_lambda)

# Step 4: Bayesian Inference
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

# Step 5: Confidence Intervals
conf_interval <- t.test(clean_data)$conf.int
print("Confidence Interval:")
print(conf_interval)

# Step 6: Hypothesis Testing
t_test <- t.test(clean_data, mu = mean(clean_data))
print(t_test)

# Step 7: Chi-Square Test
chisq_test <- chisq.test(table(clean_data))
print(chisq_test)

# Step 8: Group Mean Differences
# Assuming data has a grouping variable in the second column
group_means <- t.test(data[[1]] ~ data[[2]], data = data)
print(group_means)

# Step 9: Correlation and Regression
correlation <- cor(clean_data, data[[2]])
print("Correlation:")
print(correlation)

regression <- lm(data[[1]] ~ data[[2]], data = data)
print(summary(regression))

# Step 10: Machine Learning Model Comparison
# Split the data into training and testing sets
set.seed(123)
trainIndex <- createDataPartition(data[[2]], p = .8, list = FALSE, times = 1)
trainData <- data[trainIndex, ]
testData <- data[-trainIndex, ]

# Random Forest Model
rf_model <- randomForest(data[[2]] ~ ., data = trainData)
rf_predictions <- predict(rf_model, newdata = testData)
rf_conf_matrix <- confusionMatrix(rf_predictions, testData[[2]])
print("Random Forest Confusion Matrix:")
print(rf_conf_matrix)

# Logistic Regression Model
logistic_model <- glm(data[[2]] ~ ., family = binomial(), data = trainData)
logistic_predictions <- predict(logistic_model, newdata = testData, type = "response")
logistic_predictions <- ifelse(logistic_predictions > 0.5, 1, 0)
logistic_conf_matrix <- confusionMatrix(logistic_predictions, testData[[2]])
print("Logistic Regression Confusion Matrix:")
print(logistic_conf_matrix)

# XGBoost Model
xgb_train <- xgb.DMatrix(data = as.matrix(trainData[-1]), label = trainData[[2]])
xgb_test <- xgb.DMatrix(data = as.matrix(testData[-1]), label = testData[[2]])
xgb_model <- xgboost(data = xgb_train, max.depth = 3, nrounds = 100, objective = "binary:logistic")
xgb_predictions <- predict(xgb_model, xgb_test)
xgb_predictions <- ifelse(xgb_predictions > 0.5, 1, 0)
xgb_conf_matrix <- confusionMatrix(xgb_predictions, testData[[2]])
print("XGBoost Confusion Matrix:")
print(xgb_conf_matrix)

# Model Performance Comparison
models <- list(rf = rf_model, logistic = logistic_model, xgb = xgb_model)
predictions <- list(rf = rf_predictions, logistic = logistic_predictions, xgb = xgb_predictions)

performance <- function(predictions, actual) {
  pred <- prediction(predictions, actual)
  perf <- performance(pred, "tpr", "fpr")
  auc <- performance(pred, measure = "auc")
  return(list(perf = perf, auc = auc))
}

rf_performance <- performance(rf_predictions, testData[[2]])
logistic_performance <- performance(logistic_predictions, testData[[2]])
xgb_performance <- performance(xgb_predictions, testData[[2]])

print("Random Forest AUC:")
print(as.numeric(rf_performance$auc@y.values))
print("Logistic Regression AUC:")
print(as.numeric(logistic_performance$auc@y.values))
print("XGBoost AUC:")
print(as.numeric(xgb_performance$auc@y.values))

# Plot ROC Curves
par(mfrow = c(1, 3))
plot(rf_performance$perf, col = "blue", main = "Random Forest ROC")
plot(logistic_performance$perf, col = "green", main = "Logistic Regression ROC")
plot(xgb_performance$perf, col = "red", main = "XGBoost ROC")

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
write(paste("Confidence Interval:", conf_interval), "results.txt", append = TRUE)
write(t_test, "results.txt", append = TRUE)
write(chisq_test, "results.txt", append = TRUE)
write(group_means, "results.txt", append = TRUE)
write(paste("Correlation:", correlation), "results.txt", append = TRUE)
write(summary(regression), "results.txt", append = TRUE)
write(rf_conf_matrix, "results.txt", append = TRUE)
write(logistic_conf_matrix, "results.txt", append = TRUE)
write(xgb_conf_matrix, "results.txt", append = TRUE)
write(paste("Random Forest AUC:", as.numeric(rf_performance$auc@y.values)), "results.txt", append = TRUE)
write(paste("Logistic Regression AUC:", as.numeric(logistic_performance$auc@y.values)), "results.txt", append = TRUE)
write(paste("XGBoost AUC:", as.numeric(xgb_performance$auc@y.values)), "results.txt", append = TRUE)

print("Analysis complete. Results saved to 'results.txt'.")
EOL

# Execute the R script
Rscript $r_script

# Clean up the temporary R script
rm $r_script
