# DA5401: A4 – GMM-Based Synthetic Sampling for Imbalanced Data

**Author:** Yash Purswani\
**Roll Number:** ME22B214

## Overview
This notebook implements Gaussian Mixture Model (GMM)-based synthetic sampling to address class imbalance in the **Credit Card Fraud Detection dataset** (Kaggle: [mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)).


## Contents

- **Part A – Baseline Model and Data Analysis**
  - Load and explore the dataset
  - Train a baseline Logistic Regression model on the imbalanced data
  - Discuss why accuracy is not informative in imbalanced problems

- **Part B – GMM for Synthetic Sampling**
  - Fit a Gaussian Mixture Model (GMM) on the minority (fraud) class
  - Select the optimal number of components using BIC
  - Generate synthetic fraud samples from the fitted GMM
  - Compare theoretical difference between SMOTE and GMM

- **Part C – Clustering-Based Undersampling (CBU) + GMM**
  - Apply MiniBatchKMeans to undersample the majority class efficiently
  - Combine undersampled majority with real and synthetic minority samples
  - Create a balanced training dataset

- **Part D – Model Training and Evaluation**
  - Train Logistic Regression on the balanced datasets (GMM-only and CBU+GMM)
  - Evaluate on the original imbalanced test set
  - Compare models using precision, recall, and F1-score for the fraud class
  - Visualize results with radar plots and confusion matrices

## Key Insights
- The **baseline model** shows high accuracy but poor recall on fraud cases due to imbalance.
- **GMM-based oversampling** improves recall and F1-score while keeping precision reasonable.
- **CBU+GMM** can further increase recall but often reduces precision, leading to more false alarms.
- Overall, GMM oversampling alone strikes the best balance between detecting fraud and limiting false positives.

## Requirements
- Python 3.x
- Libraries: `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`

## File Structure

- `ME22B214_A4.ipynb` → Main notebook with code, plots, and explanations.

- `creditcard.csv` → Dataset (downloaded from Kaggle).

## How to Run  
1. Install required libraries:  
   ```bash
   pip install pandas numpy scikit-learn imbalanced-learn seaborn matplotlib kagglehub
2. Place `creditcard.csv` in the working directory OR let the notebook fetch it automatically from Kaggle.
3. Open the notebook:
    ```bash
    jupyter notebook ME22B214_A4.ipynb
4. Run all cells to reproduce results.