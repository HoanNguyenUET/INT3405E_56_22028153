# Predicting Problematic Internet Use in Children

This repository contains two versions of a model designed to predict Problematic Internet Use (PIU) in children, developed for the [Child Mind Institute - Problematic Internet Use](https://www.kaggle.com/competitions/child-mind-institute-problematic-internet-use/overview) competition on Kaggle.

The model utilizes a **Dual-Head Neural Network** architecture to process both **tabular** and **time-series** data.

## Repository Structure

This repository contains two main branches, representing two versions of the model:

*   **`ensemble-operation-base.ipynb` (Base Version):** This notebook implements the initial version of the model. It uses **Conv1D** layers for processing the time-series data within the Time-Series Head.
*   **`ensemble-operation-Optimize.ipynb` (Optimize Version):** This notebook implements an optimized version of the model. It replaces the Conv1D layers with **LSTM** layers in the Time-Series Head for better temporal dependency capture. It also incorporates **Dropout** and **L2 Regularization** to prevent overfitting.

## Model Overview

Both versions of the model share the following core structure:

1. **Data Preprocessing:**
    *   **Tabular Data:** Missing values are handled using mean imputation for numerical features and mode imputation for categorical features. The data is then standardized using `StandardScaler`.
    *   **Time-Series Data:** Missing values are filled with 0, and the sequences are truncated or padded to a fixed length of 500 timesteps.

2. **Dual-Head Neural Network:**
    *   **Tabular Head:** Processes tabular data using Dense layers with ReLU activation and Batch Normalization.
    *   **Time-Series Head:**
        *   **Base Version:** Uses Conv1D, MaxPooling1D, and GlobalAveragePooling1D layers.
        *   **Optimize Version:** Uses LSTM layers with Dropout and L2 Regularization added to the Dense layers of Tabular Head.
    *   **Concatenation:** The outputs of the two heads are concatenated.
    *   **Output:** A Dense layer with Softmax activation produces a 4-class prediction (PIU levels).

## Key Differences Between Versions

| Feature          | Base Version (`ensemble-operation-base.ipynb`) | Optimize Version (`ensemble-operation-Optimize.ipynb`) |
| ---------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Time-Series Head | Conv1D, MaxPooling1D, GlobalAveragePooling1D | LSTM                                                    |
| Regularization   | None in Time-Series Head, No L2 Regularization in Dense layers of Tabular Head.                         | Dropout (both heads), L2 Regularization (Dense Layers of Tabular Head)                    |
| Complexity       | Lower                                          | Higher                                                 |
| Training Speed   | Faster                                         | Slower                                                  |

## Getting Started

To explore the code, simply open the respective Jupyter Notebook files (`ensemble-operation-base.ipynb` and `ensemble-operation-Optimize.ipynb`). The notebooks contain detailed code and explanations for each step of the process.

## Future Improvements

Potential future improvements for the model include:

*   Addressing class imbalance using techniques like oversampling (SMOTE), undersampling, or class weights.
*   Hyperparameter tuning using methods like Grid Search or Random Search.
*   Feature engineering to create more informative features from the existing data.
*   Data augmentation, particularly for the time-series data.
*   Model ensembling to combine predictions from multiple models.

## Author

HoanNguyenUET
