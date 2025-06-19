# shap_formel_xai Package Documentation

## Overview

The `shap_formel_xai` package is designed to provide comprehensive explainability for machine learning models, combining SHAP (SHapley Additive exPlanations) values with formal, rule-based explanations. This package is particularly suited for interpretable AI applications, such as medical diagnostics, where both feature importance and human-readable rules are crucial.

---

## Main Components

### 1. Visualization Utilities (`shap_formel_xai.utils.Visualization`)

- **show_SHAP_plot**:  
  Displays SHAP value plots (e.g., waterfall plots) to visualize the contribution of each feature to a specific prediction.

- **show_formel_explanations**:  
  Prints formal explanations derived from rule-based methods in a clear, interpretable format.

---

### 2. Explanation Utilities (`shap_formel_xai.utils.Explanation`)

- **formel_explanation**:  
  Generates formal, rule-based explanations of the prediction based on the dataset, model, and instance.

- **get_optimal_formel**:  
  Filters and selects the most relevant formal explanations based on top SHAP feature importance.

- **shap_explanation**:  
  Computes SHAP values for the model and instance, extracting the most important features that drive the prediction.

---

### 3. Interpretation Utilities (`shap_formel_xai.utils.Interpretation`)

- **interprete_results**:  
  Provides a contextual interpretation of the results, incorporating categorical variables and domain-specific metrics (such as medical intervals).

---

### 4. Validation Utilities (`shap_formel_xai.utils.Validation`)

- **validate_input**:  
  Checks the validity and consistency of inputs (model, data, instance, columns) before running explanations to avoid errors.

---

## Core Function: `SHAP_Formel`

This function integrates all components to explain the prediction of a machine learning model for a given instance using SHAP values alongside formal rule-based explanations.

### Parameters

| Parameter           | Type          | Description                                                                                   |
|---------------------|---------------|-----------------------------------------------------------------------------------------------|
| `model`             | Trained model | The trained machine learning model (e.g., `RandomForestClassifier`) to explain.                |
| `instance`          | `pd.DataFrame`| A single-row dataframe representing the instance to explain.                                  |
| `data`              | `pd.DataFrame`| The training or reference dataset without missing values.                                    |
| `target_column`     | `str`         | Name of the target (label) column in the dataset.                                            |
| `categorical_columns`| `list[str]`   | Optional list of categorical column names in the data.                                       |
| `medical_metrics`   | `list[dict]`  | Optional list of dictionaries describing medical metrics and their intervals for interpretation. |

### Function Workflow

1. **Input Validation:**  
   Uses `validate_input` to verify the model and data inputs.

2. **SHAP Values Calculation:**  
   Computes SHAP values with `shap_explanation` and identifies the top 10 important features.

3. **Formal Explanation Generation:**  
   Extracts formal explanations through `formel_explanation`.

4. **Optimal Explanation Selection:**  
   Filters explanations to include only the most relevant features with `get_optimal_formel`.

5. **Visualization:**  
   Displays the SHAP waterfall plot (`show_SHAP_plot`) and prints formal explanations (`show_formel_explanations`).

6. **Interpretation:**  
   Provides detailed interpretation using `interprete_results`, which can include categorical data and medical metrics.

---

## Example Usage

```python
from shap_formel_xai import SHAP_Formel
import shap
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

# Assume model is a trained RandomForestClassifier
# instance is a pd.DataFrame with one row
# data is the training dataset

SHAP_Formel(
    model=model,
    instance=instance,
    data=data,
    target_column="Outcome",
    categorical_columns=["Gender", "Smoker"],
    medical_metrics=[
        {
            "column_name": "Blood Pressure",
            "intervals": [
                {"min": 0, "max": 80, "interpretation": "Low"},
                {"min": 81, "max": 120, "interpretation": "Normal"},
                {"min": 121, "max": 200, "interpretation": "High"}
            ]
        }
    ]
)
```
## Note:

You can find a real example demonstrating the usage of this package in the folder named **`exemple de application`** within this project repository.This example illustrates how to apply the `SHAP_Formel` function with actual data and models.

