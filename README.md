# Churn Modeling 📉👥  

This project focuses on predicting customer churn by leveraging machine learning techniques and addressing imbalanced data challenges.  

## 📝 Problem Statement  
Customer churn is a critical problem for businesses, where retaining customers is often more cost-effective than acquiring new ones. The goal is to predict churn accurately, enabling better decision-making to improve customer retention.  

## 🔍 Steps Followed  

### 1️⃣ Understanding the Data  
- Explored columns and data types.  
- Described numerical and categorical features.  

### 2️⃣ Feature Extraction + Exploratory Data Analysis (EDA)  
- Engineered new features like **Product Engagement** to capture customer behavior:  

```python
def extract_product_engagment(row):
    is_active, n_products = row['IsActiveMember'], row['NumOfProducts']
    
    if is_active == 0:
        return 'very_low_engagment'
    elif is_active == 1 and n_products == 1:
        return 'small_engagment'
    elif is_active == 1 and n_products == 2:
        return 'avg_engagment'
    elif is_active == 1 and n_products == 3:
        return 'above_avg_engagment'
    elif is_active == 1 and n_products == 4:
        return 'high_engagment'
```
- **Univariate Analysis:**  
  - Analyzed distributions using histograms and distplots.  
  - Examined categorical feature frequencies with pie charts and count plots.  

- **Bivariate Analysis:**  
  - **Numerical vs Numerical:** Used scatter and line plots to study relationships.  
  - **Numerical vs Categorical:** Applied box plots, violin plots, and strip plots to observe data distributions.  
  - **Categorical vs Categorical:** Visualized comparisons using bar plots and count plots.  

- **Multivariate Analysis:**  
  - Conducted pair plot analysis for feature relationships.  
  - Generated correlation heatmaps for insights.  

### 3️⃣ Pre-Processing  
- **Duplicate Handling:** Identified and removed duplicate records.  
- **Train-Test Split:** Divided data into training and testing sets.  
- **Missing Values:** Detected and imputed missing values.  
- **Outliers:** Addressed outliers using robust statistical techniques.  
- **Encoding:**  
  - Used **OrdinalEncoder** and **LabelEncoder** for ordinal data.  
  - Applied **OneHotEncoder** for nominal data with fewer categories and **BinaryEncoder** for those with more categories.  
- **Scaling:** Standardized features using **StandardScaler**, **MinMaxScaler**, and **RobustScaler**.  
- **Imbalanced Data:** Balanced data using SMOTE for oversampling and undersampling techniques.  

### 4️⃣ Modeling  
- **Baseline Models:** Trained initial models to assess performance.  
- **Data Balancing Techniques:**  
  - Compared models trained with `class_weight='balanced'`.  
  - Evaluated models with undersampling and oversampling techniques.  
- **Model Comparison:** Analyzed SVM, Random Forest, AdaBoost, and GradientBoosting models.  
- **Ensemble Learning:**  
  - Experimented with **Voting (hard/soft)** and **Stacking** methods.  
  - Best Model: **Stacked Ensemble with Soft Voting**.  
- **Threshold Optimization:**  
  - Used PR Curve to identify a threshold that prioritized **80% recall**.  
- **Hyperparameter Tuning:**  
  - Applied **GridSearchCV** and **RandomizedSearchCV** for optimal settings.  

### ✅ Results  
- **Validation Accuracy:** **85.15%**  
- **Test Accuracy:** **74.2%**  
- **Recall:** **77.64%**  
- **Precision:** **42.64%**  

### 5️⃣ Model Deployment  
- Saved the best-performing stacked ensemble model for future inference.  
