# Databricks notebook source
# DBTITLE 1,Introduction
# MAGIC %md
# MAGIC # South Summit 2026 - End-to-End ML Pipeline
# MAGIC ## Titanic Survival Prediction
# MAGIC
# MAGIC This notebook demonstrates a complete ML workflow on Databricks:
# MAGIC 1. **Exploratory Data Analysis** — understand the dataset
# MAGIC 2. **Data Preprocessing** — handle missing values, encode features
# MAGIC 3. **Model Training** — compare Logistic Regression, Random Forest, and LightGBM
# MAGIC 4. **Model Registration** — register the best model to Unity Catalog
# MAGIC 5. **Model Deployment** — serve the model via a REST endpoint

# COMMAND ----------

# DBTITLE 1,Install Libraries
# MAGIC %pip install lightgbm category_encoders --quiet
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Load Data
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Load Titanic dataset from Unity Catalog Volume
df = pd.read_csv("/Volumes/classic_stable_1ogj68/genaids/genaids/titanic.csv")
print(f"Dataset shape: {df.shape}")
print(f"\nColumn types:\n{df.dtypes}")
print(f"\nMissing values:\n{df.isnull().sum()}")
display(df.head(10))

# COMMAND ----------

# DBTITLE 1,EDA Section
# MAGIC %md
# MAGIC ## 1. Exploratory Data Analysis

# COMMAND ----------

# DBTITLE 1,EDA - Dataset Overview
# Dataset statistics
print("=" * 60)
print("DATASET OVERVIEW")
print("=" * 60)
print(f"\nShape: {df.shape[0]} rows x {df.shape[1]} columns")
print(f"\nDescriptive Statistics:")
display(df.describe())

# Feature type table
feature_types = pd.DataFrame({
    "Feature": ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp", "Parch", "Ticket", "Fare", "Cabin", "Embarked"],
    "Type": ["Numeric (ID)", "Binary Target", "Categorical Ordinal", "Text (drop)", "Categorical Nominal", "Numeric Float", "Numeric Integer", "Numeric Integer", "Text (drop)", "Numeric Float", "Categorical (high null)", "Categorical Nominal"]
})
print("\nFeature Types:")
display(feature_types)

# COMMAND ----------

# DBTITLE 1,EDA - Survival Distribution
# Target variable distribution
fig, axes = plt.subplots(1, 3, figsize=(16, 4))

# Survival counts
sns.countplot(data=df, x='Survived', ax=axes[0], palette='Set2')
axes[0].set_title('Survival Distribution')
axes[0].set_xticklabels(['Died (0)', 'Survived (1)'])

survival_pct = df['Survived'].value_counts(normalize=True)
print(f"Survival rate: {survival_pct[1]:.1%} survived, {survival_pct[0]:.1%} died")

# Survival by Sex
sns.countplot(data=df, x='Sex', hue='Survived', ax=axes[1], palette='Set2')
axes[1].set_title('Survival by Sex')
axes[1].legend(['Died', 'Survived'])

# Survival by Pclass
sns.countplot(data=df, x='Pclass', hue='Survived', ax=axes[2], palette='Set2')
axes[2].set_title('Survival by Passenger Class')
axes[2].legend(['Died', 'Survived'])

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA - Correlogram and Age Distribution
# Correlogram of numeric features
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

numeric_cols = ['Survived', 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
corr_matrix = df[numeric_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', center=0, ax=axes[0], fmt='.2f')
axes[0].set_title('Feature Correlation Matrix')

# Age distribution by survival
sns.histplot(data=df, x='Age', hue='Survived', kde=True, ax=axes[1], palette='Set2', bins=30)
axes[1].set_title('Age Distribution by Survival')
axes[1].legend(['Died', 'Survived'])

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,EDA - Fare and Embarked
# Fare distribution and Embarked analysis
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

# Fare by survival (log scale for better viz)
sns.boxplot(data=df, x='Survived', y='Fare', ax=axes[0], palette='Set2')
axes[0].set_title('Fare by Survival Status')
axes[0].set_xticklabels(['Died', 'Survived'])
axes[0].set_ylim(0, 150)

# Survival rate by Embarked
embarked_survival = df.groupby('Embarked')['Survived'].mean().reset_index()
sns.barplot(data=embarked_survival, x='Embarked', y='Survived', ax=axes[1], palette='Set2')
axes[1].set_title('Survival Rate by Embarkation Port')
axes[1].set_ylabel('Survival Rate')

plt.tight_layout()
plt.show()

# COMMAND ----------

# DBTITLE 1,Preprocessing Section
# MAGIC %md
# MAGIC ## 2. Data Preprocessing

# COMMAND ----------

# DBTITLE 1,Preprocessing Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import category_encoders as ce

# Select features
features_to_use = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
target = 'Survived'

# Prepare data
X = df[features_to_use].copy()
y = df[target].copy()

# Fill missing Embarked with mode
X['Embarked'].fillna(X['Embarked'].mode()[0], inplace=True)

# Encode categorical features
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})
X = pd.get_dummies(X, columns=['Embarked'], drop_first=True)

# Split data (stratified to preserve class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"\nClass distribution in training set:")
print(y_train.value_counts(normalize=True))

# Impute Age and scale numeric features
numeric_features = ['Age', 'Fare', 'SibSp', 'Parch']
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numeric_features)
    ],
    remainder='passthrough'
)

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

print(f"\nProcessed feature shape: {X_train_processed.shape}")

# COMMAND ----------

# DBTITLE 1,Model Training Section
# MAGIC %md
# MAGIC ## 3. Model Training & Comparison
# MAGIC Training three algorithms:
# MAGIC - **Logistic Regression** — interpretable baseline
# MAGIC - **Random Forest** — ensemble of decision trees
# MAGIC - **LightGBM** — gradient boosted trees (state-of-the-art)

# COMMAND ----------

# DBTITLE 1,Train Models with MLflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix, RocCurveDisplay
)

# Set MLflow experiment
mlflow.set_experiment("/Users/caio.moreno@databricks.com/south-summit-2026/titanic-survival")

# Define models to train
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, class_weight='balanced'),
    "LightGBM": LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=42, class_weight='balanced', verbose=-1)
}

results = {}

for name, model in models.items():
    with mlflow.start_run(run_name=name) as run:
        # Train
        model.fit(X_train_processed, y_train)
        
        # Predict
        y_pred = model.predict(X_test_processed)
        y_proba = model.predict_proba(X_test_processed)[:, 1]
        
        # Metrics
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        
        # Log parameters and metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("roc_auc", auc)
        
        # Log model with signature
        signature = infer_signature(X_train_processed, y_pred)
        mlflow.sklearn.log_model(model, "model", signature=signature, input_example=X_train_processed[:5])
        
        results[name] = {"accuracy": acc, "f1_score": f1, "roc_auc": auc, "run_id": run.info.run_id}
        print(f"\n{name}: Accuracy={acc:.4f} | F1={f1:.4f} | ROC-AUC={auc:.4f}")

print("\n" + "=" * 60)
print("All models trained and logged to MLflow!")

# COMMAND ----------

# DBTITLE 1,Compare Models
# Model comparison table
results_df = pd.DataFrame(results).T
results_df = results_df.drop(columns=['run_id']).astype(float)
results_df = results_df.sort_values('roc_auc', ascending=False)
print("Model Comparison (sorted by ROC-AUC):")
display(results_df)

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Metrics comparison bar chart
results_df[['accuracy', 'f1_score', 'roc_auc']].plot(kind='bar', ax=axes[0], colormap='Set2')
axes[0].set_title('Model Comparison')
axes[0].set_ylabel('Score')
axes[0].set_ylim(0.6, 1.0)
axes[0].legend(loc='lower right')
axes[0].tick_params(axis='x', rotation=15)

# ROC Curves
for name, model in models.items():
    y_proba = model.predict_proba(X_test_processed)[:, 1]
    RocCurveDisplay.from_predictions(y_test, y_proba, name=name, ax=axes[1])
axes[1].set_title('ROC Curves')
axes[1].plot([0, 1], [0, 1], 'k--', label='Random')
axes[1].legend()

plt.tight_layout()
plt.show()

# Identify best model
best_model_name = results_df.index[0]
print(f"\n\U0001f3c6 Best model: {best_model_name} (ROC-AUC: {results_df.loc[best_model_name, 'roc_auc']:.4f})")

# COMMAND ----------

# DBTITLE 1,Confusion Matrix - Best Model
# Confusion matrix for best model
best_model = models[best_model_name]
y_pred_best = best_model.predict(X_test_processed)

fig, ax = plt.subplots(figsize=(6, 5))
cm = confusion_matrix(y_test, y_pred_best)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Died', 'Survived'], yticklabels=['Died', 'Survived'])
ax.set_title(f'Confusion Matrix - {best_model_name}')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted')
plt.show()

print(f"\nClassification Report - {best_model_name}:")
print(classification_report(y_test, y_pred_best, target_names=['Died', 'Survived']))

# COMMAND ----------

# DBTITLE 1,Registration Section
# MAGIC %md
# MAGIC ## 4. Register Best Model to Unity Catalog

# COMMAND ----------

# DBTITLE 1,Register Model to UC
import mlflow
from mlflow import MlflowClient

# Set Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Register the best model
catalog = "classic_stable_1ogj68"
schema = "genaids"
model_name = f"{catalog}.{schema}.titanic_survival_model"

# Get the best run ID
best_run_id = results[best_model_name]["run_id"]
model_uri = f"runs:/{best_run_id}/model"

# Register model
registered_model = mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"\n\u2705 Model registered: {model_name}")
print(f"   Version: {registered_model.version}")

# Set alias 'champion'
client = MlflowClient()
client.set_registered_model_alias(model_name, "champion", registered_model.version)
print(f"   Alias 'champion' set to version {registered_model.version}")

# COMMAND ----------

# DBTITLE 1,Deployment Section
# MAGIC %md
# MAGIC ## 5. Deploy Model to Serving Endpoint

# COMMAND ----------

# DBTITLE 1,Create Serving Endpoint
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
import time

w = WorkspaceClient()

endpoint_name = "titanic-survival-endpoint"

# Create or update the serving endpoint
try:
    endpoint = w.serving_endpoints.create(
        name=endpoint_name,
        config=EndpointCoreConfigInput(
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=registered_model.version,
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ]
        ),
    )
    print(f"\u2705 Endpoint '{endpoint_name}' created!")
    print(f"   Model: {model_name} v{registered_model.version}")
    print(f"   Status: Provisioning (may take 5-10 minutes)")
except Exception as e:
    if "already exists" in str(e).lower():
        # Update existing endpoint
        w.serving_endpoints.update_config(
            name=endpoint_name,
            served_entities=[
                ServedEntityInput(
                    entity_name=model_name,
                    entity_version=registered_model.version,
                    scale_to_zero_enabled=True,
                    workload_size="Small",
                )
            ],
        )
        print(f"\u2705 Endpoint '{endpoint_name}' updated with new model version!")
    else:
        raise e

print(f"\n\U0001f517 Endpoint URL: /serving-endpoints/{endpoint_name}/invocations")

# COMMAND ----------

# DBTITLE 1,Test Endpoint
# Test the endpoint with sample data (wait for it to be ready)
import json

# Sample passenger data for inference
sample_data = {
    "dataframe_records": [
        {"Pclass": 1, "Sex": 1, "Age": 29.0, "SibSp": 0, "Parch": 0, "Fare": 100.0, "Embarked_Q": 0, "Embarked_S": 0},
        {"Pclass": 3, "Sex": 0, "Age": 35.0, "SibSp": 1, "Parch": 2, "Fare": 15.0, "Embarked_Q": 0, "Embarked_S": 1},
    ]
}

print("Sample request payload:")
print(json.dumps(sample_data, indent=2))
print(f"\n\U0001f4a1 Once the endpoint is READY, test it with:")
print(f"   w.serving_endpoints.query(name='{endpoint_name}', dataframe_records=sample_data['dataframe_records'])")
print(f"\n   Or via REST API:")
print(f"   POST /serving-endpoints/{endpoint_name}/invocations")