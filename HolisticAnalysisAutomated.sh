#!/bin/bash

# Check if input file is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <input_csv_file>"
  exit 1
fi

# Install required packages
pip install pandas numpy matplotlib seaborn scipy scikit-learn

# Python script content
PYTHON_SCRIPT=$(cat <<EOF
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns
import sys

def descriptive_statistics(data):
    stats_dict = {
        'mean': np.mean(data),
        'median': np.median(data),
        'mode': stats.mode(data)[0][0],
        'variance': np.var(data),
        'std_deviation': np.std(data),
        'range': np.ptp(data),
        'iqr': stats.iqr(data),
        'skewness': stats.skew(data),
        'kurtosis': stats.kurtosis(data),
        'cv': np.std(data) / np.mean(data),  # Coefficient of Variation
        'geometric_mean': stats.gmean(data),  # Geometric Mean
        'harmonic_mean': stats.hmean(data)   # Harmonic Mean
    }
    return stats_dict

def confidence_interval_mean(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data)
    std_err = stats.sem(data)
    h = std_err * stats.t.ppf((1 + confidence) / 2, n - 1)
    return mean - h, mean + h

def confidence_interval_proportion(p_hat, n, confidence=0.95):
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    margin_of_error = z * np.sqrt((p_hat * (1 - p_hat)) / n)
    return p_hat - margin_of_error, p_hat + margin_of_error

def hypothesis_testing(data, popmean):
    # One-sample t-test
    t_stat, p_value = stats.ttest_1samp(data, popmean)
    return t_stat, p_value

def z_test(data, popmean, popstd):
    # Z-test for population mean when population std is known
    z_stat = (np.mean(data) - popmean) / (popstd / np.sqrt(len(data)))
    p_value = 2 * (1 - stats.norm.cdf(np.abs(z_stat)))
    return z_stat, p_value

def anova(data_list):
    # One-way ANOVA for comparing multiple groups
    f_stat, p_value = stats.f_oneway(*data_list)
    return f_stat, p_value

def chi_square_test(observed, expected):
    # Chi-square test for independence
    chi2_stat, p_value = stats.chisquare(observed, expected)
    return chi2_stat, p_value

def f_test(data1, data2):
    # F-test for comparing variances between two populations
    f_stat = np.var(data1) / np.var(data2)
    dfn = len(data1) - 1
    dfd = len(data2) - 1
    p_value = 1 - stats.f.cdf(f_stat, dfn, dfd)
    return f_stat, p_value

def regression_analysis(x, y):
    # Linear Regression
    model = LinearRegression()
    model.fit(x.reshape(-1, 1), y)
    slope = model.coef_[0]
    intercept = model.intercept_
    r_squared = model.score(x.reshape(-1, 1), y)
    return slope, intercept, r_squared

def multiple_regression_analysis(x, y):
    # Multiple Regression (more than one independent variable)
    model = LinearRegression()
    model.fit(x, y)
    return model.intercept_, model.coef_, model.score(x, y)

def correlation_analysis(x, y):
    pearson_corr, pearson_p = stats.pearsonr(x, y)
    spearman_corr, spearman_p = stats.spearmanr(x, y)
    return pearson_corr, pearson_p, spearman_corr, spearman_p

def mann_whitney_u_test(data1, data2):
    # Mann-Whitney U test for comparing differences between two independent groups
    u_stat, p_value = stats.mannwhitneyu(data1, data2)
    return u_stat, p_value

def wilcoxon_signed_rank_test(data1, data2):
    # Wilcoxon Signed-Rank Test for comparing two related groups
    stat, p_value = stats.wilcoxon(data1, data2)
    return stat, p_value

def kruskal_wallis_test(data_list):
    # Kruskal-Wallis test for comparing multiple groups
    h_stat, p_value = stats.kruskal(*data_list)
    return h_stat, p_value

def bayesian_inference(prior, likelihood, evidence):
    # Bayes' Theorem for posterior probability
    posterior = (likelihood * prior) / evidence
    return posterior

def descriptive_plot(data):
    # Plotting histogram and boxplot for data visualization
    sns.histplot(data, kde=True)
    plt.show()
    sns.boxplot(x=data)
    plt.show()

def main(input_file):
    df = pd.read_csv(input_file)
    
    for column in df.columns:
        data = df[column].dropna().values
        print(f'Descriptive Statistics for {column}:')
        desc_stats = descriptive_statistics(data)
        for key, value in desc_stats.items():
            print(f'{key}: {value}')
        
        # Plot the data
        print(f'Plotting for {column}...')
        descriptive_plot(data)
        
        # Confidence Interval for Mean
        ci_mean = confidence_interval_mean(data)
        print(f'Confidence Interval for Mean: {ci_mean}')
        
        # Confidence Interval for Proportion
        p_hat = np.mean(data)  # Assuming data is binary (0 or 1)
        n = len(data)
        ci_prop = confidence_interval_proportion(p_hat, n)
        print(f'Confidence Interval for Proportion: {ci_prop}')
        
        # Hypothesis Testing
        popmean = np.mean(data)  # Use sample mean as population mean for demonstration
        t_stat, p_value = hypothesis_testing(data, popmean)
        print(f'T-Test: t_stat={t_stat}, p_value={p_value}')
        
        # Z-Test example
        popstd = np.std(data)  # Use sample std for demonstration
        z_stat, z_p_value = z_test(data, popmean, popstd)
        print(f'Z-Test: z_stat={z_stat}, p_value={z_p_value}')
        
        # ANOVA if there are multiple groups
        if len(df.columns) > 1:
            data_list = [df[col].dropna().values for col in df.columns if col != column]
            f_stat, anova_p_value = anova(data_list)
            print(f'ANOVA: F-stat={f_stat}, p_value={anova_p_value}')
        
        # Chi-Square Test example (for categorical data)
        observed = np.array([10, 20, 30])  # Example observed data
        expected = np.array([15, 15, 30])  # Example expected data
        chi2_stat, chi2_p_value = chi_square_test(observed, expected)
        print(f'Chi-Square Test: chi2_stat={chi2_stat}, p_value={chi2_p_value}')
        
        # F-Test example for two groups
        f_stat, f_p_value = f_test(df['column1'].dropna().values, df['column2'].dropna().values)
        print(f'F-Test: f_stat={f_stat}, p_value={f_p_value}')
        
        # Linear Regression and Correlation Analysis
        for other_column in df.columns:
            if column != other_column:
                x = df[column].dropna().values
                y = df[other_column].dropna().values
                slope, intercept, r_squared = regression_analysis(x, y)
                print(f'Linear Regression between {column} and {other_column}: slope={slope}, intercept={intercept}, r_squared={r_squared}')
                pearson_corr, pearson_p, spearman_corr, spearman_p = correlation_analysis(x, y)
                print(f'Correlation between {column} and {other_column}: Pearson={pearson_corr}, p={pearson_p}, Spearman={spearman_corr}, p={spearman_p}')
                
        # Non-parametric Tests
        u_stat, u_p_value = mann_whitney_u_test(df['group1'].dropna().values, df['group2'].dropna().values)
        print(f'Mann-Whitney U Test: u_stat={u_stat}, p_value={u_p_value}')
        
        wilcoxon_stat, wilcoxon_p_value = wilcoxon_signed_rank_test(df['pre'].dropna().values, df['post'].dropna().values)
        print(f'Wilcoxon Signed-Rank Test: stat={wilcoxon_stat}, p_value={wilcoxon_p_value}')
        
        h_stat, kruskal_p_value = kruskal_wallis_test([df['group1'].dropna().values, df['group2'].dropna().values])
        print(f'Kruskal-Wallis Test: h_stat={h_stat}, p_value={kruskal_p_value}')
        
        # Bayesian Inference Example (Assuming some values for prior, likelihood, and evidence)
        prior = 0.3
        likelihood = 0.7
        evidence = 0.5
        posterior = bayesian_inference(prior, likelihood, evidence)
        print(f'Bayesian Inference: posterior={posterior}')
        
        print('')

if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
EOF
)

# Run the Python script
python -c "$PYTHON_SCRIPT" "$1"
