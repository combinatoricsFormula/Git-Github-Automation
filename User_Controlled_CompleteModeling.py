import pandas as pd
import numpy as np
from sklearn.feature_selection import RFE
from sklearn.linear_model import Lasso, LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from statsmodels.stats.outliers_influence import variance_inflation_factor
import matplotlib.pyplot as plt
import seaborn as sns
import tkinter as tk
from tkinter import filedialog

# Function to load data
def load_data():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename()
    if file_path:
        df = pd.read_csv(file_path)
        return df
    else:
        print("No file selected. Exiting...")
        exit()

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
    encoder = OneHotEncoder(drop='first')
    encoded_features = encoder.fit_transform(df[categorical_cols]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols))
    df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
    return df

# Function to perform feature engineering
def feature_engineering(df):
    # Example: Create interaction terms
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
    X_final = X.loc[:, vif_data["VIF"] < 5]
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

        print(f"Model: {name}")
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("ROC-AUC:", roc_auc)
        print("Classification Report:\n", classification_report(y_test, y_pred))
        print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
        print("\n")

        # Plot ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        plt.subplot(2, 3, i+1)
        plt.plot(fpr, tpr, label=f'{name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend(loc="lower right")

    plt.tight_layout()
    plt.show()

    return pd.DataFrame(results)

# Main function
def main():
    df = load_data()
    df = preprocess_data(df)
    df = encode_categorical(df)
    df = feature_engineering(df)

    X = df.drop('target', axis=1)
    y = df['target']

    selected_features = feature_selection(X, y)
    X = X[selected_features]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_final = handle_multicollinearity(pd.DataFrame(X_scaled, columns=selected_features))

    X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

    models = {
        'Logistic Regression': LogisticRegression(),
        'Decision Tree': DecisionTreeClassifier(),
        'Random Forest': RandomForestClassifier(),
        'Support Vector Machine': SVC(probability=True),
        'Gradient Boosting': GradientBoostingClassifier(),
        'K-Nearest Neighbors': KNeighborsClassifier()
    }

    run_all_models = input("Do you want to run all models? (yes/no): ").strip().lower()
    if run_all_models == 'yes':
        results_df = evaluate_models(X_train, X_test, y_train, y_test, models)
    else:
        model_name = input("Enter the model name you want to run (Logistic Regression, Decision Tree, Random Forest, Support Vector Machine, Gradient Boosting, K-Nearest Neighbors): ").strip()
        if model_name in models:
            results_df = evaluate_models(X_train, X_test, y_train, y_test, {model_name: models[model_name]})
        else:
            print("Invalid model name. Exiting...")
            exit()

    best_model = results_df.loc[results_df['F1 Score'].idxmax()]
    print("Best Model Based on F1 Score:")
    print(best_model)

    # Hyperparameter tuning for Random Forest using GridSearchCV
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_rf_model = grid_search.best_estimator_

    # Evaluate the best Random Forest model
    y_pred_best_rf = best_rf_model.predict(X_test)
    y_prob_best_rf = best_rf_model.predict_proba(X_test)[:, 1]

    print("Best Random Forest Model")
    print("Accuracy:", accuracy_score(y_test, y_pred_best_rf))
    print("Precision:", precision_score(y_test, y_pred_best_rf))
    print("Recall:", recall_score(y_test, y_pred_best_rf))
    print("F1 Score:", f1_score(y_test, y_pred_best_rf))
    print("ROC-AUC:", roc_auc_score(y_test, y_prob_best_rf))
    print("Classification Report:\n", classification_report(y_test, y_pred_best_rf))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_best_rf))

    # Visualize feature importances for the best Random Forest model
    importances = best_rf_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = selected_features

    plt.figure(figsize=(10, 6))
    plt.title('Feature Importances (Best Random Forest)')
    plt.bar(range(X_final.shape[1]), importances[indices], align='center')
    plt.xticks(range(X_final.shape[1]), features[indices], rotation=45)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.show()

if __name__ == "__main__":
    main()
