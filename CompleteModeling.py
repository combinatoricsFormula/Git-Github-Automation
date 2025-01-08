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

# Sample dataset
data = {
    'feature1': np.random.rand(100),
    'feature2': np.random.rand(100),
    'feature3': np.random.rand(100),
    'feature4': np.random.rand(100),
    'feature5': np.random.rand(100),
    'feature6': np.random.rand(100),
    'feature7': np.random.rand(100),
    'feature8': np.random.rand(100),
    'feature9': np.random.rand(100),
    'feature10': np.random.rand(100),
    'target': np.random.randint(0, 2, 100)
}

df = pd.DataFrame(data)

# Introduce some missing values and duplicates for demonstration
df.loc[5:10, 'feature1'] = np.nan
df.loc[15:20, 'feature2'] = np.nan
df = pd.concat([df, df.iloc[0:5]], ignore_index=True)

# Handle missing values
df.fillna(df.mean(), inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Handle outliers (example: capping at 99th percentile)
for col in df.columns:
    if df[col].dtype != 'object':
        upper_limit = df[col].quantile(0.99)
        df[col] = np.where(df[col] > upper_limit, upper_limit, df[col])

# Encode categorical variables (if any)
# Assuming 'feature10' is categorical for demonstration
df['feature10'] = df['feature10'].astype('category')
encoder = OneHotEncoder(drop='first')
encoded_features = encoder.fit_transform(df[['feature10']]).toarray()
encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['feature10']))
df = pd.concat([df.drop('feature10', axis=1), encoded_df], axis=1)

# Feature Engineering: Create new features (example: interaction terms)
df['feature1_feature2_interaction'] = df['feature1'] * df['feature2']

X = df.drop('target', axis=1)
y = df['target']

# Feature Selection using RFE
model = LinearRegression()
rfe = RFE(model, n_features_to_select=5)
X_rfe = rfe.fit_transform(X, y)

# Feature Selection using Lasso
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)
selected_features = X.columns[(lasso.coef_ != 0)]

# Feature Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X[selected_features])

# Handling Multicollinearity
vif_data = pd.DataFrame()
vif_data["feature"] = selected_features
vif_data["VIF"] = [variance_inflation_factor(X_scaled, i) for i in range(X_scaled.shape[1])]
X_final = X_scaled[:, vif_data["VIF"] < 5]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_final, y, test_size=0.2, random_state=42)

# Train multiple models
models = {
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'Support Vector Machine': SVC(probability=True),
    'Gradient Boosting': GradientBoostingClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier()
}

# Evaluate models
results = []
for name, model in models.items():
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

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Identify the best model based on F1 Score
best_model = results_df.loc[results_df['F1 Score'].idxmax()]

print("Best Model Based on F1 Score:")
print(best_model)

# Hyperparameter tuning for Random Forest using GridSearchCV
param_grid = {
    'n_estim
