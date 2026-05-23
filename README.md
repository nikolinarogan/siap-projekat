# Phone Addiction Level Prediction

Machine learning project focused on predicting smartphone addiction levels among teenagers using behavioral, psychological, and lifestyle data.

---

## Overview

This project explores the problem of smartphone addiction among young adults through machine learning techniques. The goal was to predict addiction levels based on digital habits, psychological indicators, and lifestyle factors, while also identifying the most influential features contributing to addiction.

The project combines:
- synthetic Kaggle data,
- real survey data collected from 100 participants,
- exploratory data analysis,
- imbalance handling techniques,
- model optimization and explainability.


---

## Project Preview

### Exploratory Data Analysis

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/2288eb1f-06fa-4c7e-97bd-0cbc0b49da33" />


<img width="500" height="380" alt="image" src="https://github.com/user-attachments/assets/f15d7553-c13e-42cc-9330-b737be2f4983" />


<img width="629" height="470" alt="image" src="https://github.com/user-attachments/assets/47c9b491-f47c-4fe3-8b4c-9470537e52e1" />


<img width="2125" height="765" alt="image" src="https://github.com/user-attachments/assets/5725786e-17c2-445f-b02a-072de0caaa64" />


---

### Model Performance & Results


<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/53f56f49-4d43-424d-af28-2d772688cfbc" />

<img width="800" height="500" alt="image" src="https://github.com/user-attachments/assets/9901639e-f3f7-4d80-b868-70f29e912469" />

---

### SHAP Explainability

<img width="650" height="800" alt="image" src="https://github.com/user-attachments/assets/ace8753d-b93f-4cf3-be08-1edd009f24b8" />


---


##  Dataset

The base dataset was obtained from Kaggle:

- Teen Phone Addiction Dataset
- ~3000 synthetic samples
- 25 features related to:
  - daily phone usage,
  - social media activity,
  - gaming habits,
  - sleep patterns,
  - anxiety and depression levels,
  - lifestyle indicators.

To improve realism and credibility of the research, the dataset was extended with:
-  100 real survey responses collected manually.

### Target Variable
`Addiction_Level` — continuous value in the range 1–10.

---

##  Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Imbalanced-learn (SMOTE)
- SHAP
- Matplotlib

---

##  Project Pipeline

### 1. Data Preprocessing
- handling missing values,
- removing duplicates and outliers,
- one-hot encoding categorical features,
- feature scaling for neural networks.

### 2. Exploratory Data Analysis
Performed:
- correlation analysis,
- heatmaps,
- histograms,
- boxplots,
- distribution analysis.

Main findings:
- Daily phone usage was the strongest predictor.
- Social media usage strongly correlated with addiction level.
- Higher addiction levels were associated with reduced sleep quality.

### 3. Handling Imbalanced Data
Several approaches were tested:
- SMOTE oversampling,
- weighted regression,
- logarithmic target transformations.

SMOTE combined with MLP produced the best results.

### 4. Model Training
The following models were trained and compared:
- Linear Regression
- KNN
- XGBoost
- MLP Neural Network

Validation strategy:
- 70/30 train-test split
- 5-fold cross-validation
- manual grid search hyperparameter tuning

---

##  Results

| Model | MAE | R² |
|---|---|---|
| XGBoost | 0.31 | ~0.91 |
| MLP Neural Network | **0.118** | **~0.95** |

### Best Performing Model
MLP with:
- hidden layers `(128, 64, 32)`
- `tanh` activation
- SMOTE balancing
- feature scaling

---

## Explainability with SHAP

SHAP analysis was used to interpret feature importance.

Most important features:
- Daily Usage Hours
- Number of Apps Used
- Phone Checking Frequency
- Social Media Usage
- Sleep Hours

Demographic features had significantly lower impact.

---

##  Key Conclusions

- Behavioral phone usage patterns are stronger predictors than demographic factors.
- Combining real and synthetic data improved dataset realism.
- Neural networks significantly outperformed traditional regression approaches.
- Data balancing techniques greatly improved prediction quality for underrepresented addiction levels.

---
