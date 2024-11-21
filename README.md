# CreditCardFraud


Credit card firms must detect fraudulent credit card transactions to prevent consumers from being charged for products they did not buy. Data Science can address such a challenge, and its significance, coupled with Machine Learning, cannot be emphasized.

The dataset utilized covers credit card transactions done by European cardholders in September 2013. This dataset contains 492 frauds out of 284,807 transactions over two days. The dataset is unbalanced, with the positive class (frauds) accounting for 0.172 percent of all transactions.


This project demonstrates an end-to-end process for detecting fraudulent credit card transactions using machine learning techniques, with a focus on handling imbalanced datasets. The dataset used is publicly available and contains anonymized credit card transaction records.

---

## Table of Contents

1. [Dataset Overview](#dataset-overview)  
2. [Installation](#installation)  
3. [Data Exploration](#data-exploration)  
4. [Feature Scaling and Engineering](#feature-scaling-and-engineering)  
5. [Modeling and Evaluation](#modeling-and-evaluation)  
6. [Handling Imbalanced Data](#handling-imbalanced-data)  
7. [Visualizations](#visualizations)  
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
