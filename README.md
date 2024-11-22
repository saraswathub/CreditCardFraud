# Credit Card Fraud Detection


Credit card firms must detect fraudulent credit card transactions to prevent consumers from being charged for products they did not buy. Data Science can address such a challenge, and its significance, coupled with Machine Learning, cannot be emphasized.

The dataset utilized covers credit card transactions done by European cardholders in September 2013. This dataset contains 492 frauds out of 284,807 transactions over two days. The dataset is unbalanced, with the positive class (frauds) accounting for 0.172 percent of all transactions.

This project demonstrates an end-to-end process for detecting fraudulent credit card transactions using machine learning techniques, with a focus on handling imbalanced datasets. The dataset used is publicly available and contains anonymized credit card transaction records.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)  
2. [Installation](#installation)  
3. [Data Exploration](#data-exploration)  
   - Fraud Case Distribution  
   - Proportion Analysis  
   - Distribution of Time and Amount Features  
   - Missing Values Check  
4. [Feature Scaling and Engineering](#feature-scaling-and-engineering)  
   - Scaling of Time and Amount Features  
   - Outlier Removal Using IQR  
   - Feature Reordering  
5. [Modeling and Evaluation](#modeling-and-evaluation)  
   - Train-Test Split  
   - Metrics Used  
   - Stratified k-Fold Cross-Validation  
6. [Handling Imbalanced Data](#handling-imbalanced-data)  
   - Undersampling with NearMiss  
   - Oversampling with SMOTE  
7. [Visualizations](#visualizations)  
   - Distribution of Features  
   - Boxplots  
   - Feature Distributions by Class  
   - ROC Curves  
8. [How to Run](#how-to-run)  
9. [Conclusion](#conclusion)

---

## Dataset Overview

The dataset contains 31 columns:
- **Time**: Time elapsed since the first transaction.  
- **Amount**: Transaction amount.  
- **V1-V28**: Principal components derived from a PCA transformation to anonymize data.  
- **Class**: Target variable (0 = Genuine, 1 = Fraudulent).

### Fraud Case Distribution
The dataset is highly imbalanced, with fraudulent transactions comprising a small fraction of the total.

---

## Installation

Ensure you have Python installed (>= 3.8). Install the required dependencies:

```bash
pip install sklearn==0.24.2 imbalanced-learn numpy pandas matplotlib seaborn
```


## Data Exploration

### Fraud Case Distribution:
- Fraudulent and valid transactions were counted and compared.  
- Proportion of fraudulent transactions was calculated.

### Distribution of Features:
- KDE plots were created to visualize the distribution of `Time` and `Amount`.

### Missing Values Check:
- Confirmed there were no missing values in the dataset.

---

## Feature Scaling and Engineering

### Scaling:
- Used `RobustScaler` to scale `Time` and `Amount` to reduce the impact of outliers.

### Outlier Removal:
- Outliers were removed using the IQR method.

### Feature Reordering:
- `Time` and `Amount` were placed at the beginning of the dataset for easier access.

---

## Modeling and Evaluation

### Train-Test Split:
- Dataset was split into 80% training and 20% testing subsets.

### Metrics Used:
- Accuracy, Precision, Recall, F1-Score, and AUC-ROC were used for evaluation.

### Cross-Validation:
- Stratified k-fold cross-validation was applied for robust model evaluation.

---

## Handling Imbalanced Data

### Undersampling:
- `NearMiss` undersampling technique was used to reduce the size of the majority class.

### Oversampling:
- `SMOTE` oversampling technique was employed to synthetically increase the size of the minority class.

---

## Visualizations

### Key Visualizations:
1. KDE plots for `Time` and `Amount`.  
2. Boxplots for features.  
3. Distributions of all features for Fraudulent vs Genuine transactions.  
4. ROC curves for various models.

---

## How to Run

1. Clone the repository:  
   ```bash
   git clone https://github.com/your-repository/credit-card-fraud-detection.git
   cd credit-card-fraud-detection
Run the script:

    python cc_code.py

## Conclusion

This project demonstrated effective techniques for handling imbalanced data and building a robust fraud detection system. It highlights the importance of data preprocessing, feature engineering, and evaluation strategies for real-world machine learning applications.
