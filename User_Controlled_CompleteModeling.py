import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import filedialog
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load data
def load_data():
    choice = input("Do you want to load a dataset from your local machine? (yes/no): ").strip().lower()
    if choice == 'yes':
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename()
        if file_path:
            df = pd.read_csv(file_path)
            return df
        else:
            print("No file selected. Exiting...")
            exit()
    else:
        print("Generating a sample dataset...")
        dataset_choice = input("Choose a dataset (iris/wine/breast_cancer) or press enter for a random dataset: ").strip().lower()
        if dataset_choice == 'iris':
            data = load_iris()
        elif dataset_choice == 'wine':
            data = load_wine()
        elif dataset_choice == 'breast_cancer':
            data = load_breast_cancer()
        elif dataset_choice == '':
            print("No dataset chosen, generating random data...")
            # Create random data with 10 features and 100 samples
            n_features = 10
            n_samples = 100
            np.random.seed(42)
            data = np.random.rand(n_samples, n_features)
            df = pd.DataFrame(data, columns=[f"feature{i}" for i in range(1, n_features + 1)])
            # Add a random target variable
            df['target'] = np.random.randint(0, 2, size=n_samples)  # Binary target (0 or 1)
            return df
        else:
            print("Invalid choice. Defaulting to iris dataset.")
            data = load_iris()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        return df

# Function to handle missing values, duplicates, and outliers
def preprocess_data(df):
    # Handle missing values
    df.fillna(df.mean(), inplace=True)
    # Remove duplicate rows
    df.drop_duplicates(inplace=True)
    # Handle outliers (example: capping at 99th percentile)
    for col in df.columns:
        if df[col].dtype != 'object':
            upper_limit = df[col].quantile(0.99)
            df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])
    return df

# Function to encode categorical variables
def encode_categorical(df):
    categorical_cols = df.select_dtypes(include=['category', 'object']).columns
    if len(categorical_cols) > 0:
        encoder = OneHotEncoder(drop='first')
        encoded_features = encoder.fit_transform(df[categorical_cols]).toarray()
        encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
        df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    return df

# Function to perform feature engineering (adding interaction terms)
def feature_engineering(df):
    # Example: Create interaction terms if columns exist
    if 'feature1' in df.columns and 'feature2' in df.columns:
        df['feature1_feature2_interaction'] = df['feature1'] * df['feature2']
    return df

# Function to perform feature selection
def feature_selection(X, y):
    model = LinearRegression()
    rfe = RFE(model, n_features_to_select=5)
    X_rfe = rfe.fit_transform(X, y)
    lasso = Lasso(alpha=0.01)
    lasso.fit(X, y)
    selected_features = X.columns[(lasso.coef_ != 0)]
    return selected_features

# Function to handle multicollinearity
def handle_multicollinearity(X):
    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    
    # Use the 'feature' column to filter columns with VIF < 5
    selected_features = vif_data.loc[vif_data["VIF"] < 5, "feature"]
    X_final = X[selected_features]
    
    return X_final

# Function to evaluate models
def evaluate_models(X_train, X_test, y_train, y_test, models):
    results = []
    plt.figure(figsize=(15, 10))
    for i, (name, model) in enumerate(models.items()):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_prob)
        
        results.append({
            'Model': name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1 Score': f1,
            'ROC-AUC': roc_auc
        })
        
        # Print model evaluation
        print(f"Model: {name}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC-AUC:", roc_auc)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        
        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.subplot(2, 3, i+1)
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.title(f'ROC Curve: {name}')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()

# Main function to run the pipeline
def main():
    # Load and preprocess data
    df = load_data()
    print("\nInitial Data Preview (First 5 Rows):")
    print(df.head())  # Show the first 5 rows of the dataset for the user to inspect
    
    df = preprocess_data(df)
    df = encode_categorical(df)
    
    # Split data into features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Handle multicollinearity
    X_train = handle_multicollinearity(X_train)
    X_test = X_test[X_train.columns]  # Ensure that test set has the same features as train set
    
    # Define models to evaluate
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(),
        'Gradient Boosting': GradientBoostingClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier(),
        'Support Vector Classifier': SVC(),
        'Decision Tree': DecisionTreeClassifier()
    }
    
    # Evaluate models
    evaluate_models(X_train, X_test, y_train, y_test, models)

if __name__ == '__main__':
    main()
