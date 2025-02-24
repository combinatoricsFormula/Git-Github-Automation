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
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
import sys

def check_normality(data):
    results = {}
    
    # Descriptive statistics
    results['mean'] = np.mean(data)
    results['median'] = np.median(data)
    results['mode'] = stats.mode(data)[0][0]
    results['skewness'] = stats.skew(data)
    results['kurtosis'] = stats.kurtosis(data)
    
    # Shapiro-Wilk Test
    shapiro_test = stats.shapiro(data)
    results['shapiro_statistic'] = shapiro_test.statistic
    results['shapiro_p_value'] = shapiro_test.pvalue
    
    # Kolmogorov-Smirnov Test
    ks_test = stats.kstest(data, 'norm', args=(np.mean(data), np.std(data)))
    results['ks_statistic'] = ks_test.statistic
    results['ks_p_value'] = ks_test.pvalue
    
    # Anderson-Darling Test
    ad_test = stats.anderson(data, dist='norm')
    results['ad_statistic'] = ad_test.statistic
    results['ad_critical_values'] = ad_test.critical_values
    
    return results

def standardize_data(data):
    scaler = StandardScaler()
    z_scores = scaler.fit_transform(data.reshape(-1, 1)).flatten()
    return z_scores

def plot_data(data):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data, kde=True)
    plt.title('Histogram')
    
    plt.subplot(1, 3, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title('Q-Q Plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data)
    plt.title('Box Plot')
    
    plt.show()

def main(input_file):
    df = pd.read_csv(input_file)
    
    for column in df.columns:
        data = df[column].dropna().values
        print(f'Checking normality for {column}:')
        
        # Check normality
        results = check_normality(data)
        for key, value in results.items():
            print(f'{key}: {value}')
        
        # Standardize data
        z_scores = standardize_data(data)
        print(f'Z-scores: {z_scores[:5]}')  # Display first 5 z-scores
        
        # Plot data
        plot_data(data)
        print('')

if __name__ == "__main__":
    input_file = sys.argv[1]
    main(input_file)
EOF
)

# Create a temporary Python script file
TEMP_PYTHON_SCRIPT=$(mktemp)
echo "$PYTHON_SCRIPT" > "$TEMP_PYTHON_SCRIPT"

# Run the Python script with the input file
python "$TEMP_PYTHON_SCRIPT" "$1"

# Clean up the temporary Python script file
rm "$TEMP_PYTHON_SCRIPT"
