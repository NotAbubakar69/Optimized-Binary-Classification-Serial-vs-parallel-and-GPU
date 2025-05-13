# Optimized-Binary-Classification-Serial-vs-parallel-and-GPU

## Overview

This project implements and benchmarks a binary classification pipeline using tree-based (XGBoost) and deep learning (PyTorch) models under various compute configurations: serial CPU, parallel CPU, and GPU. The focus was to optimize training time without sacrificing classification accuracy. The project achieved up to *71.8% reduction in processing time*, maintaining accuracy around \~50–52%.

## WorkFlow
  <img src="https://github.com/user-attachments/assets/b281b535-c3ff-4152-a699-899b345a0b30" width="200">

## Key Highlights

* *Data Preprocessing*: Missing value imputation, one-hot encoding, quantile transformation, and SMOTE for class imbalance.
* *Model Comparison*:

  * XGBoost (Serial, Parallel, GPU)
  * PyTorch Neural Network (CPU, GPU)
* *Performance Metrics*:

  * Accuracy: XGBoost \~52%, PyTorch \~49–51%
  * F1 Score: PyTorch GPU highest at 0.4592
  * Speed: XGBoost Parallel CPU fastest at 0.38s, PyTorch GPU took 37.34s

## Dataset

* File: pdc_dataset_with_target.csv
* Features: 4 numerical, 3 categorical
* Size: \~40,100 samples with moderate class imbalance

## Environment

* *Hardware*: Intel Core i7 (8 threads), Tesla T4 GPU (via Google Colab)
* *Software*: Python 3.12, scikit-learn, imbalanced-learn, XGBoost 1.7, PyTorch 2.0

## Data Preprocessing Steps

1. *Missing Values*: Imputed using mean (symmetric) or median (skewed)
2. *Encoding*: One-hot encoding for categorical features
3. *Normalization*: Quantile transformation on numerical features
4. *Imbalance Handling*: SMOTE on training set
5. *Final Shape*: (Pre-SMOTE) 30,750 samples → (Post-SMOTE) 36,990 samples
<img src="https://github.com/user-attachments/assets/508de6a5-fc15-426e-9ef1-c91bc0480c90" width="700">
<img src="https://github.com/user-attachments/assets/4eef7070-44dd-4396-a565-d4c3549df7ef" width="700">
<img src="https://github.com/user-attachments/assets/6c56455d-ddb5-46b3-baf2-58075ef59ade" width="700">
<img src="https://github.com/user-attachments/assets/67bda1fb-4533-49e3-aba2-e266c5f3630f" width="700">
<img src="https://github.com/user-attachments/assets/fc58739a-b5ca-4dd2-abca-338dd92bd9b2" width="700">
<img src="https://github.com/user-attachments/assets/d158d354-1c46-4d23-acf2-2a9bdd2eca12" width="700">

## Model Implementations

### XGBoost

* *Serial*: n_jobs=1
* *Parallel*: n_jobs=-1
* *GPU*: tree_method='gpu_hist', predictor='gpu_predictor'

### PyTorch Neural Network

* Layers: \[Input → 32 → 64 → 16 → 1], ReLU + Sigmoid
* Optimizer: Adam
* Loss: Binary Cross Entropy
* AMP: Enabled for GPU
* Batch Size: 64 (CPU), 1024 (GPU)
<img src="https://github.com/user-attachments/assets/b58b4b8f-bc0c-4adc-a52e-c5d2aedb653e" width="700">

## Results Summary
<img src="https://github.com/user-attachments/assets/e273b39d-8e17-4c3f-9e54-a91429dc646a" width="700">

## Observations

* *XGBoost Parallel CPU* was fastest with no drop in performance.
* *PyTorch GPU* had best F1 Score but slower than all XGBoost variants.
* Dataset quality limited max accuracy (\~52%) due to:

  * Missing values
  * Outliers and skew
  * Weak feature-label relationships

## Challenges and Learnings

* *GPU Overhead*: GPU XGBoost showed minimal benefit due to small dataset size.
* *PyTorch Optimization*: Neural nets performed poorly on tabular data and were sensitive to hyperparameters.
* *Resource Utilization*: Tree-based models were significantly more efficient.

## Future Work

* Improve data quality and feature engineering
* Experiment with deeper or hybrid models (e.g., TabNet, LightGBM)
* Scale to larger datasets for better GPU utilization
* Conduct systematic hyperparameter tuning
* Explore alternate imbalance handling methods

## References

* Chen, T. & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. ACM SIGKDD.
* Paszke, A. et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. NeurIPS.

## Team Members

* Muhammad Qasim
* Ayaan Khan
* Muhammad Abubakar Nadeem
