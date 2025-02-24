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
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PowerTransformer, QuantileTransformer
import sys

def plot_data(data, title):
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    sns.histplot(data, kde=True)
    plt.title(f'{title} - Histogram')
    
    plt.subplot(1, 3, 2)
    stats.probplot(data, dist="norm", plot=plt)
    plt.title(f'{title} - Q-Q Plot')
    
    plt.subplot(1, 3, 3)
    sns.boxplot(data)
    plt.title(f'{title} - Box Plot')
    
    plt.show()

def transform_data(data):
    # Original Data
    plot_data(data, 'Original Data')
    
    # Log Transformation
    log_data = np.log(data)
    plot_data(log_data, 'Log Transformation')
    
    # Square Root Transformation
    sqrt_data = np.sqrt(data)
    plot_data(sqrt_data, 'Square Root Transformation')
    
    # Cube Root Transformation
    cube_root_data = np.cbrt(data)
    plot_data(cube_root_data, 'Cube Root Transformation')
    
    # Box-Cox Transformation
    boxcox_data, _ = stats.boxcox(data)
    plot_data(boxcox_data, 'Box-Cox Transformation')
    
    # Yeo-Johnson Transformation
    pt = PowerTransformer(method='yeo-johnson')
    yeo_johnson_data = pt.fit_transform(data.reshape(-1, 1)).flatten()
    plot_data(yeo_johnson_data, 'Yeo-Johnson Transformation')
    
    # Reciprocal Transformation
    reciprocal_data = 1 / data
    plot_data(reciprocal_data, 'Reciprocal Transformation')
    
    # Exponential Transformation
    exp_data = np.exp(data)
    plot_data(exp_data, 'Exponential Transformation')
    
    # Standardization (Z-Score)
    z_score_data = stats.zscore(data)
    plot_data(z_score_data, 'Z-Score Standardization')
    
    # Min-Max Scaling
    min_max_scaler = MinMaxScaler()
    min_max_data = min_max_scaler.fit_transform(data.reshape(-1, 1)).flatten()
    plot_data(min_max_data, 'Min-Max Scaling')
    
    # Robust Scaling
    robust_scaler = RobustScaler()
    robust_data = robust_scaler.fit_transform(data.reshape(-1, 1)).flatten()
    plot_data(robust_data, 'Robust Scaling')
    
    # Quantile Transformation
    quantile_transformer = QuantileTransformer(output_distribution='normal')
    quantile_data = quantile_transformer.fit_transform(data.reshape(-1, 1)).flatten()
    plot_data(quantile_data, 'Quantile Transformation')

    # Winsorization
    lower_percentile = 0.05
    upper_percentile = 0.95
    winsorized_data = stats.mstats.winsorize(data, limits=[lower_percentile, upper_percentile])
    plot_data(winsorized_data, 'Winsorization')

    # Clipping
    lower_clip = np.percentile(data, 5)
    upper_clip = np.percentile(data, 95)
    clipped_data = np.clip(data, lower_clip, upper_clip)
    plot_data(clipped_data, 'Clipping')

def main(input_file):
    df = pd.read_csv(input_file)
    
    for column in df.columns:
        data = df[column].dropna().values
        transform_data(data)

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
