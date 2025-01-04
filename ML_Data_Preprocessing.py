"""
****************************************************************************************
Author: Amear Hussein Mathews
Date: 1/1/2025
Project: Machine Learning
Topic: Data Preprocessing
Lesson: Convert to Dummies
*****************************************************************************************
-----------------------------------------------------------------------------------------
*****************************************************************************************
Introduction - One-hot encoding:
*****************************************************************************************
Python & other object-oriented programming languages have revolutionized programming.
Here are four ways that we can accomplish preprocessing tasks using Python libraries.

Data Inspection - Useful Functions: 
--> data.shape
--> data.info()
--> data.describe()
--> data.head()
--> dat.dtypes
--> data.isnull().sum()
--> data.to_datetime(data['column'], errors='coerce')

************************ Using Pandas get_dummies ***************************************
"""
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
import numpy as np

# Load the dataset
music_df = pd.read_csv('/workspaces/Git-Github-Automation/Machine Learning/music_clean.csv')

# Convert categorical variable into dummy/indicator variables
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)

# Concatenate the original DataFrame with the dummy variables
music_dummies = pd.concat([music_df, music_dummies], axis=1)

# Drop the original 'genre' column
music_dummies = music_dummies.drop("genre", axis=1)

# Display the columns of the resulting DataFrame
print(music_dummies.columns)

# Define features and target variable
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Initialize the Linear Regression model
linreg = LinearRegression()

# Perform cross-validation and calculate the negative mean squared error
linreg_cv = cross_val_score(linreg, X_train, y_train, cv=kf, scoring="neg_mean_squared_error")

# Print the root mean squared error for each fold
print(np.sqrt(-linreg_cv))

############################ Using TensorFlow (TensorFlow) ###############################

import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split, KFold

# Load the dataset
music_df = pd.read_csv('/workspaces/Git-Github-Automation/Machine Learning/music_clean.csv')

# Convert categorical variable into dummy/indicator variables
music_dummies = pd.get_dummies(music_df["genre"], drop_first=True)

# Concatenate the original DataFrame with the dummy variables
music_dummies = pd.concat([music_df, music_dummies], axis=1)

# Drop the original 'genre' column
music_dummies = music_dummies.drop("genre", axis=1)

# Define features and target variable
X = music_dummies.drop("popularity", axis=1).values
y = music_dummies["popularity"].values

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize KFold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],))
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Perform cross-validation
mse_scores = []
for train_index, val_index in kf.split(X_train):
    X_train_fold, X_val_fold = X_train[train_index], X_train[val_index]
    y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
    
    # Train the model
    model.fit(X_train_fold, y_train_fold, epochs=50, verbose=0)
    
    # Evaluate the model
    mse = model.evaluate(X_val_fold, y_val_fold, verbose=0)
    mse_scores.append(mse)

# Calculate the root mean squared error for each fold
rmse_scores = np.sqrt(mse_scores)

print(rmse_scores)

############################ Using CategoryEncoders (category_encoders) ##################
import pandas as pd
import category_encoders as ce

# Load the dataset
music_df = pd.read_csv('/workspaces/Git-Github-Automation/Machine Learning/music_clean.csv')

# Initialize the OneHotEncoder
encoder = ce.OneHotEncoder(cols=['genre'], use_cat_names=True)

# Fit and transform the data
music_dummies = encoder.fit_transform(music_df)

# Display the first few rows of the resulting DataFrame
print(music_dummies.head())

############################ Using DictVectorizer (scikit-learn) #########################
import pandas as pd
from sklearn.feature_extraction import DictVectorizer

# Load the dataset
music_df = pd.read_csv('/workspaces/Git-Github-Automation/Machine Learning/music_clean.csv')

# Convert the DataFrame to a list of dictionaries
data_dict = music_df.to_dict(orient='records')

# Initialize the DictVectorizer
vec = DictVectorizer(sparse=False)

# Fit and transform the data
encoded_data = vec.fit_transform(data_dict)

# Convert the result to a DataFrame for better readability
encoded_df = pd.DataFrame(encoded_data, columns=vec.get_feature_names_out())

print(encoded_df.head())

############################ Using sklearn OneHotEncoder ################################
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load the dataset
music_df = pd.read_csv('/workspaces/Git-Github-Automation/Machine Learning/music_clean.csv')

# Initialize the OneHotEncoder
encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform the data
encoded_data = encoder.fit_transform(music_df[['genre']])

# Convert the result to a DataFrame for better readability
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['genre']))

print(encoded_df.head())
