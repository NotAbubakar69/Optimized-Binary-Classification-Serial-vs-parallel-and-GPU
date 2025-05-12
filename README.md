# Optimized-Binary-Classification-Serial-vs-parallel-and-GPU

## Overview

This project implements and benchmarks a binary classification pipeline using tree-based (XGBoost) and deep learning (PyTorch) models under various compute configurations: serial CPU, parallel CPU, and GPU. The focus was to optimize training time without sacrificing classification accuracy. The project achieved up to *71.8% reduction in processing time*, maintaining accuracy around \~50–52%.

## Team Members

* Muhammad Qasim (22I-1994)
* Ayaan Khan (22I-2066)
* Muhammad AbuBakAr Nadeem (22I-2003)

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

## Results Summary

| Model            | Accuracy | F1 Score   | Time (s) | Speedup vs Serial |
| ---------------- | -------- | ---------- | -------- | ----------------- |
| XGBoost Serial   | 0.5245   | 0.4327     | 0.61     | —                 |
| XGBoost Parallel | 0.5245   | 0.4327     | 0.38     | 37.7%             |
| XGBoost GPU      | 0.5197   | 0.4321     | 0.57     | 6.55%             |
| PyTorch CPU      | 0.5084   | 0.4513     | 119.37   | —                 |
| PyTorch GPU      | 0.4898   | *0.4592* | 37.34    | 71.8%             |

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
