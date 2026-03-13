# Semiconductor Failure Prediction Using Machine Learning

## Project Overview
This project focuses on predicting semiconductor manufacturing failures using machine learning techniques applied to the SECOM dataset. Semiconductor fabrication processes generate large volumes of sensor data, and identifying faulty products early can significantly reduce manufacturing costs and improve production yield.

The dataset contains hundreds of sensor measurements collected from the semiconductor manufacturing process. The objective is to classify whether a product will **Pass (0)** or **Fail (1)** during the quality control stage.

This project implements a complete machine learning pipeline including **data preprocessing, handling missing values, class imbalance correction, model training, evaluation, and visualization**.

---

## Objectives
- Analyze high-dimensional semiconductor sensor data.
- Handle missing values and noisy features.
- Address **class imbalance** in failure prediction.
- Train and compare multiple machine learning models.
- Identify important features influencing semiconductor failures.
- Evaluate models using appropriate classification metrics.

---

## Dataset
**Dataset:** SECOM Manufacturing Dataset  
**Source:** UCI Machine Learning Repository

### Dataset Characteristics
- ~590 sensor features
- Binary target variable:
  - `0` → Pass
  - `1` → Fail
- High dimensionality
- Large number of missing values
- Strong class imbalance (few failure samples)

---

## Technologies & Tools Used
- **Programming Language:** Python  
- **Libraries:**
  - Pandas
  - NumPy
  - Scikit-learn
  - Seaborn
  - Matplotlib
  - Imbalanced-learn (SMOTE)

---

## Machine Learning Pipeline

### 1. Exploratory Data Analysis (EDA)
- Examined dataset structure and feature distribution.
- Visualized missing value ratios across features.
- Analyzed class distribution of pass vs fail samples.

### 2. Data Preprocessing
- Removed features with more than **50% missing values**.
- Applied **Mean Imputation** to handle remaining missing values.
- Performed **feature scaling using StandardScaler**.

### 3. Handling Class Imbalance
The dataset contains significantly fewer failure samples.

To address this:
- Implemented **SMOTE (Synthetic Minority Oversampling Technique)**.
- Generated synthetic samples for the minority class.

This improves the model's ability to detect rare failure cases.

---

## Machine Learning Models Used
The following classification models were implemented and compared:

- Logistic Regression
- Decision Tree
- Random Forest
- Support Vector Machine (SVM)
- K-Nearest Neighbors (KNN)

---

## Model Evaluation Metrics
Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix
- Precision-Recall Curve

These metrics help assess performance, especially for **imbalanced datasets**.

---

## Visualizations
The project generates several visual insights:

- Missing value distribution
- Class distribution plots
- Model ROC curves
- Precision-Recall curves
- Confusion matrices
- Feature importance plots
- Model accuracy comparison graphs

---

## Feature Importance Analysis
Using the **Random Forest model**, feature importance scores were extracted to identify the most influential process parameters affecting semiconductor failures.

This helps understand which sensor measurements contribute most to defect prediction.

---

## Key Insights
- Semiconductor manufacturing datasets are highly **high-dimensional and noisy**.
- **Class imbalance significantly affects model performance**.
- SMOTE improves detection of minority class (failures).
- **Random Forest performed best** due to its robustness with high-dimensional data.
- Feature importance analysis helps identify critical process variables.
