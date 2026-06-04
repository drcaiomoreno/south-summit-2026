# South Summit 2026 (Madrid)

### Basic Prompt:

**My role:**
I am a Senior GenAI Data Scientist.

**Dataset path:**
`/Volumes/classic_stable_1ogj68/genaids/genaids/titanic.csv`

**My goal:**
I want to train a Machine Learning model, evaluate multiple algorithms, compare their performance, and perform an Exploratory Data Analysis (EDA) on the dataset. I also want to register the best model and deploy it to production through a serving endpoint. Use Databricks to build the complete end-to-end Machine Learning pipeline, including data ingestion, feature engineering, model training, evaluation, model registration, and deployment.


### Advanced Prompt:

## Role

You are a Senior Generative AI and Machine Learning Data Scientist with expertise in Databricks, MLflow, Unity Catalog, Model Serving, and MLOps best practices.

## Dataset

Dataset location:

```python
/Volumes/classic_stable_1ogj68/genaids/genaids/titanic.csv
```

## Objective

Build an end-to-end Machine Learning solution on Databricks using the Titanic dataset. The solution should follow industry best practices and be production-ready.

## Requirements

### 1. Exploratory Data Analysis (EDA)

Perform a comprehensive Exploratory Data Analysis including:

* Dataset overview
* Schema inspection
* Missing value analysis
* Descriptive statistics
* Target variable distribution
* Correlation analysis
* Feature distributions
* Outlier detection
* Visualizations and business insights
* Data quality assessment

### 2. Data Preparation

Prepare the data for Machine Learning by:

* Handling missing values
* Encoding categorical variables
* Feature engineering
* Feature selection
* Train/validation/test split
* Data normalization or scaling when appropriate

### 3. Model Training

Train and evaluate multiple Machine Learning algorithms, including:

* Logistic Regression
* Random Forest
* Gradient Boosting
* XGBoost
* LightGBM (if available)
* Decision Tree

For each model:

* Perform hyperparameter tuning
* Use cross-validation
* Track all experiments using MLflow

### 4. Model Evaluation

Compare all models using:

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Confusion Matrix

Create a comparison table and identify the best-performing model.

### 5. MLflow Experiment Tracking

Use MLflow to:

* Track parameters
* Track metrics
* Log artifacts
* Register the best model in Unity Catalog
* Maintain model lineage

### 6. Model Registration

Register the champion model in Unity Catalog Model Registry following Databricks best practices.

Example:

```python
catalog.schema.titanic_survival_model
```

### 7. Production Deployment

Deploy the selected model to a Databricks Model Serving Endpoint.

Requirements:

* Create the serving endpoint programmatically
* Enable autoscaling
* Configure inference logging
* Use the registered model version
* Demonstrate online inference

### 8. Inference Examples

Provide example code to:

* Query the serving endpoint
* Perform batch inference
* Perform real-time inference
* Score a single passenger

### 9. Monitoring and Governance

Implement:

* Model monitoring
* Data drift detection
* Performance monitoring
* Alerting recommendations
* Unity Catalog governance controls

### 10. Deliverables

Generate:

1. Complete Databricks Notebook
2. EDA Report
3. Model Comparison Report
4. MLflow Experiment Tracking
5. Registered Model
6. Model Serving Endpoint
7. Inference Examples
8. Architecture Diagram of the End-to-End ML Pipeline

### Technical Constraints

* Use Databricks Runtime ML.
* Use Unity Catalog for governance.
* Use MLflow for experiment tracking and model registry.
* Follow Databricks MLOps best practices.
* Write clean, modular, production-ready code.
* Include detailed comments and explanations for every step.
* Use Spark whenever appropriate for scalability.

### Expected Outcome

A fully operational Databricks Machine Learning pipeline that:

* Ingests the Titanic dataset from Unity Catalog Volumes.
* Performs EDA and feature engineering.
* Trains and compares multiple models.
* Selects the best model automatically.
* Registers the model in Unity Catalog.
* Deploys the model to a Databricks Model Serving Endpoint.
* Demonstrates real-time and batch inference.
* Provides monitoring and governance recommendations suitable for production environments.
