# Stress Detection using XGBoost

A machine learning project for **multi-class stress level classification** using physiological and sleep-related features. The project uses an **XGBoost Classifier** with **GridSearchCV** for hyperparameter tuning and evaluates the trained model using accuracy, classification report, confusion matrix, cross-validation, and feature importance.

## Project Overview

Stress can be associated with measurable physiological and behavioral signals such as heart rate, respiration rate, blood oxygen, sleep duration, body temperature, snoring, and eye movement.

This project builds a supervised machine learning pipeline that takes these features as input and predicts a person's **stress level on a scale from 0 to 4**.

### Objective

- Build a machine learning model for stress-level classification.
- Use physiological and sleep-related measurements as predictive features.
- Optimize the XGBoost model using GridSearchCV.
- Evaluate model performance using multiple classification metrics.
- Save the trained model for future integration into an application or web system.

## Features Used

The dataset used by the notebook contains the following input features:

| Feature | Description |
|---|---|
| `snoring range` | Snoring-related measurement |
| `respiration rate` | Respiration rate |
| `body temperature` | Body temperature measurement |
| `limb movement` | Limb movement measurement |
| `blood oxygen` | Blood oxygen measurement |
| `eye movement` | Eye movement measurement |
| `hours of sleep` | Sleep duration |
| `heart rate` | Heart rate measurement |

### Target Variable

`Stress Levels`

The target contains **five classes: 0, 1, 2, 3, and 4**.

## Machine Learning Workflow

```text
Dataset (data_stress.csv)
        |
        v
Data Loading & Inspection
        |
        v
Feature / Target Separation
        |
        v
Train-Test Split (80/20)
        |
        v
XGBoost Classifier
        |
        v
GridSearchCV Hyperparameter Tuning
        |
        v
Best Model Selection
        |
        v
Stress Level Prediction
        |
        +--------------------+
        |                    |
        v                    v
   Model Evaluation     Feature Importance
        |
        v
Saved Model (.pkl)
```

## Model

The project uses **XGBoost (Extreme Gradient Boosting)** for multi-class classification.

### Why XGBoost?

XGBoost was selected because it:

- Performs well on structured/tabular datasets.
- Can model nonlinear relationships between physiological features.
- Provides feature-importance information.
- Supports regularization and hyperparameter tuning.
- Is suitable for deployment after training.

## Hyperparameter Optimization

`GridSearchCV` with **3-fold cross-validation** was used to search across:

- Learning rate: `0.01, 0.05, 0.1`
- Maximum depth: `3, 5, 7`
- Number of estimators: `100, 200, 300`
- Subsample: `0.7, 0.8, 1`
- Column sampling by tree: `0.7, 0.8, 1`

The notebook evaluated **243 parameter combinations**, resulting in **729 cross-validation fits**.

### Best Parameters

```text
learning_rate   = 0.1
max_depth       = 3
n_estimators    = 200
subsample       = 0.7
colsample_bytree = 0.7
```

## Results

The trained model achieved the following results on the held-out test set:

| Metric | Result |
|---|---:|
| Test Accuracy | **97.62%** |
| Cross-Validation Accuracy | **99.21%** |
| Macro F1-score | **0.98** |
| Weighted F1-score | **0.98** |

### Classification Performance

| Stress Class | Precision | Recall | F1-score |
|---:|---:|---:|---:|
| 0 | 1.00 | 0.96 | 0.98 |
| 1 | 0.96 | 0.96 | 0.96 |
| 2 | 0.96 | 0.96 | 0.96 |
| 3 | 0.96 | 1.00 | 0.98 |
| 4 | 1.00 | 1.00 | 1.00 |

The confusion matrix showed only **3 misclassified samples out of 126 test samples**.

## Model Evaluation

The notebook generates:

- Accuracy score
- Classification report
- Confusion matrix
- Cross-validation accuracy
- XGBoost feature importance plot
- Actual vs. predicted stress-level scatter plot
- Actual vs. predicted stress-level line plot

## Dataset

The notebook expects a CSV file named:

```text
data_stress.csv
```

The CSV should contain the eight input features listed above and the target column:

```text
Stress Levels
```

The dataset is loaded using Pandas.

> **Dataset note:** The notebook does not document the original source, collection methodology, or licensing information for `data_stress.csv`. Add the original dataset citation/license here if you publish the dataset or know its source.

## Installation

Clone the repository:

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

Install the required Python packages:

```bash
pip install pandas xgboost scikit-learn matplotlib seaborn joblib jupyter
```

## Running the Project

### Option 1: Google Colab

1. Open `Stress_Detection_model.ipynb` in Google Colab.
2. Upload `data_stress.csv` when prompted.
3. Run the notebook cells from top to bottom.
4. The optimized model will be trained and evaluated.
5. The trained model will be saved as:

```text
stress_detection_xgb_model.pkl
```

### Option 2: Jupyter Notebook

Place the following files in the same project directory:

```text
Stress_Detection_model.ipynb
data_stress.csv
```

Then start Jupyter:

```bash
jupyter notebook
```

Open the notebook and execute the cells sequentially.

## Saved Model

The final trained XGBoost model is exported using Joblib:

```python
joblib.dump(best_model, "stress_detection_xgb_model.pkl")
```

The saved `.pkl` model can later be loaded for prediction:

```python
import joblib

model = joblib.load("stress_detection_xgb_model.pkl")
prediction = model.predict(new_data)
print(prediction)
```

## Example Prediction Input

A new prediction should contain the same feature columns used during training:

```python
new_data = [[
    70.0,   # snoring range
    20.0,   # respiration rate
    96.0,   # body temperature
    10.0,   # limb movement
    95.0,   # blood oxygen
    85.0,   # eye movement
    7.0,    # hours of sleep
    60.0    # heart rate
]]

prediction = model.predict(new_data)
print("Predicted Stress Level:", prediction[0])
```

## Repository Structure

```text
.
├── Stress_Detection_model.ipynb
├── data_stress.csv
├── stress_detection_xgb_model.pkl
├── README.md
└── requirements.txt
```

`data_stress.csv` and the trained `.pkl` model may be excluded from GitHub if they contain restricted, private, or licensed data.

## Important Implementation Note

The current notebook initializes XGBoost with:

```python
num_class=3
```

However, the dataset contains **five target classes (0–4)**. Before using the project as a production or deployment system, this should be corrected to:

```python
num_class=5
```

or the parameter can be omitted and allowed to be inferred appropriately by the XGBoost implementation.

The notebook also produces a warning that `use_label_encoder` is not used by the installed XGBoost version. For a cleaner modern implementation, that deprecated/unused parameter can be removed.

## Limitations

- The project is a machine learning prototype and is **not a clinical diagnostic system**.
- Model performance depends on the quality and representativeness of the dataset.
- The dataset documentation in the notebook does not establish clinical validity.
- The model predicts the labels present in the training data; the README does not assign medical meanings to classes 0–4 because the notebook does not define them.
- Additional validation on independent datasets would be required before real-world deployment.
- Physiological measurements can be affected by sensor quality, motion, environmental conditions, and individual variability.

## Future Improvements

- Add data preprocessing and validation pipelines.
- Add independent external-test-set evaluation.
- Perform feature engineering and feature selection.
- Compare XGBoost with Random Forest, LightGBM, CatBoost, and neural-network approaches.
- Add explainable AI using SHAP.
- Build a Streamlit or Flask/FastAPI prediction interface.
- Add real-time physiological sensor integration.
- Add model monitoring and retraining support.
- Containerize the application using Docker.
- Create an API for integration with a healthcare or wearable-device application.

## Research Context

The broader research material associated with this project discusses wearable physiological sensing and intelligent healthcare systems. It emphasizes that physiological sensing is useful only when the measured signal is stable, actionable, and sufficiently connected to the therapeutic or preventive decision. It also recommends separating sensing, estimation, control, and safety functions in intelligent wearable systems.

This project focuses specifically on the **machine-learning stress classification component**, rather than autonomous medical treatment or drug delivery.

## Disclaimer

This project is intended for **educational, research, and prototype purposes only**. It should not be used to diagnose, treat, or make clinical decisions about stress or any medical condition.

