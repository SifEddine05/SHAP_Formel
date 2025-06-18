from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import pandas as pd


def verify_model(model):
    """
    Verify that the model is a fitted RandomForestClassifier.
    
    Args:
        model: The machine learning model to verify.
        
    Returns:
        (bool, str): Tuple where bool indicates validity, str contains error message if invalid.
    """
    if isinstance(model, RandomForestClassifier):
        try:
            # Check if the model has been fitted/trained
            check_is_fitted(model)
        except NotFittedError:
            return False, "The model has not been trained yet."
    else:
        return False, "Only RandomForestClassifier is supported in this version."
    return True, ""


def verify_data(data):
    """
    Verify that the input data is a pandas DataFrame without missing values.
    
    Args:
        data: Input dataset to verify.
        
    Returns:
        (bool, str): Tuple with validity and error message.
    """
    if not isinstance(data, pd.DataFrame):
        return False, "Input data must be a pandas DataFrame."
    if data.isnull().values.any():
        return False, "Input data contains missing values."
    return True, ""


def verify_instance(instance, data, target_column):
    """
    Verify that the instance to explain is a single-row DataFrame
    with the same columns as data except for the target column.
    
    Args:
        instance: DataFrame with a single row.
        data: Reference DataFrame for columns.
        target_column: Name of the target column.
        
    Returns:
        (bool, str): Validity and error message.
    """
    if not isinstance(instance, pd.DataFrame) or instance.shape[0] != 1:
        return False, "Instance must be a pandas DataFrame with exactly one row."
    
    expected_columns = [col for col in data.columns if col != target_column]
    if list(instance.columns) != expected_columns:
        return False, f"Instance must contain the same columns as data, excluding the target column '{target_column}'."
    
    if target_column not in data.columns:
        return False, f"Target column '{target_column}' is not present in the dataset."
    
    return True, ""


def verify_medical_metrics(medical_metrics):
    """
    Verify the format of medical metrics list with intervals and interpretations.
    
    Args:
        medical_metrics: List of dicts describing medical metrics.
        
    Returns:
        (bool, str): Validity and error message.
    """
    if not isinstance(medical_metrics, list):
        return False, "medical_metrics must be a list."
    
    for metric in medical_metrics:
        if not isinstance(metric, dict):
            return False, "Each item in medical_metrics must be a dict."
        if "column_name" not in metric or "intervalles" not in metric:
            return False, "Each metric must have 'column_name' and 'intervalles' keys."
        if not isinstance(metric["column_name"], str):
            return False, "'column_name' must be a string."
        intervals = metric["intervalles"]
        if not isinstance(intervals, list):
            return False, "'intervalles' must be a list."
        for interval in intervals:
            if not isinstance(interval, dict):
                return False, "Each interval in 'intervalles' must be a dict."
            if not all(k in interval for k in ("min", "max", "interpretation")):
                return False, "Each interval must have 'min', 'max', and 'interpretation' keys."
            if not (isinstance(interval["min"], (int, float)) and isinstance(interval["max"], (int, float))):
                return False, "'min' and 'max' must be numeric."
            if not isinstance(interval["interpretation"], str):
                return False, "'interpretation' must be a string."
    return True, ""


def verify_categorical_columns(categorical_columns, data):
    """
    Verify that categorical_columns is a list of strings and columns exist in data.
    
    Args:
        categorical_columns: List of categorical column names.
        data: Reference DataFrame.
        
    Returns:
        (bool, str): Validity and error message.
    """
    if not isinstance(categorical_columns, list):
        return False, "categorical_columns must be a list."
    
    for col in categorical_columns:
        if not isinstance(col, str):
            return False, f"'{col}' in categorical_columns is not a string."
    
    missing_cols = [col for col in categorical_columns if col not in data.columns]
    if missing_cols:
        return False, f"The following columns are missing in the dataset: {missing_cols}"
    
    return True, ""


def validate_input(model, instance, data, target_column, categorical_columns=None, medical_metrics=None):
    """
    Validate all inputs needed for the explanation function.
    
    Args:
        model: Trained model to verify.
        instance: Single instance to explain.
        data: Reference dataset.
        target_column: Target column name.
        categorical_columns: Optional list of categorical column names.
        medical_metrics: Optional list of medical metrics definitions.
        
    Returns:
        (bool, str): Overall validity and error message.
    """
    is_valid, msg = verify_data(data)
    if not is_valid:
        return is_valid, msg
    
    is_valid, msg = verify_model(model)
    if not is_valid:
        return is_valid, msg
    
    is_valid, msg = verify_instance(instance, data, target_column)
    if not is_valid:
        return is_valid, msg
    
    if categorical_columns is not None:
        is_valid, msg = verify_categorical_columns(categorical_columns, data)
        if not is_valid:
            return is_valid, msg
    
    if medical_metrics is not None:
        is_valid, msg = verify_medical_metrics(medical_metrics)
        if not is_valid:
            return is_valid, msg
    
    return True, ""
