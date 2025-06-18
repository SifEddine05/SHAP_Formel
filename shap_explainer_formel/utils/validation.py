
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted
import pandas as pd


def verify_model(model) : 
    if isinstance(model, RandomForestClassifier):
        # Check if it has been trained (fitted)
        try:
            check_is_fitted(model)
        except NotFittedError:
            return False, "The model has not been trained yet."
    else:
        return False, "Only RandomForestClassifier is supported in this version."
    return True, ""


def verify_data(data) :
    if not isinstance(data, pd.DataFrame):
        return False, "Input data must be a pandas DataFrame."
    if data.isnull().values.any():
        return False, "Input data contains missing values."   

    return True, ""

def verify_instance(instance, data,cibleName) : 
    if not isinstance(instance, pd.DataFrame) or instance.shape[0] != 1:
        return False, "Instance must be a pandas DataFrame with exactly one row."

    # 6. Vérifier que les colonnes de instance sont exactement celles de data sauf la colonne cible
    expected_columns = [col for col in data.columns if col != cibleName]
    if list(instance.columns) != expected_columns:
        return False, f"Instance must contain the same columns as data, excluding the target column '{cibleName}'."
    
    if cibleName not in data.columns.tolist():
        return False, f"Target column '{cibleName}' is not present in the dataset."
    
    return True , ""

def verify_medical(medicalMetrics) : 
    if not isinstance(medicalMetrics, list):
        return False, "medicalMetrics must be a list."

    for metric in medicalMetrics:
        if not isinstance(metric, dict):
            return False, "Each item in medicalMetrics must be a dict."
        
        if "column_name" not in metric or "intervalles" not in metric:
            return False, "Each metric must have 'column_name' and 'intervalles' keys."
        
        if not isinstance(metric["column_name"], str):
            return False, "'column_name' must be a string."
        
        intervalles = metric["intervalles"]
        if not isinstance(intervalles, list):
            return False, "'intervalles' must be a list."
        
        for intervalle in intervalles:
            if not isinstance(intervalle, dict):
                return False, "Each interval in 'intervalles' must be a dict."
            if not all(k in intervalle for k in ("min", "max", "interpretation")):
                return False, "Each interval must have 'min', 'max', and 'interpretation' keys."
            if not (isinstance(intervalle["min"], (int, float)) and isinstance(intervalle["max"], (int, float))):
                return False, "'min' and 'max' must be numbers."
            if not isinstance(intervalle["interpretation"], str):
                return False, "'interpretation' must be a string."
    
    return True, ""

def verify_categorical_columns(categoriel_columns, data):
    
    # Vérifier que c'est une liste
    if not isinstance(categoriel_columns, list):
        return False, "categoriel_columns doit être une liste."

    # Vérifier que chaque élément est une chaîne de caractères
    for col in categoriel_columns:
        if not isinstance(col, str):
            return False, f"'{col}' dans categoriel_columns n'est pas une chaîne de caractères."

    # Vérifier que toutes les colonnes existent dans data
    missing_cols = [col for col in categoriel_columns if col not in data.columns]
    if missing_cols:
        return False, f"Les colonnes suivantes sont manquantes dans le dataset : {missing_cols}"

    return True, ""


def validate_input(model, instance, data, cibleName,categoriel_columns,medicalMetrics): 
    isTrue , msg = verify_data(data=data)
    if(not isTrue) : return isTrue , msg 
    isTrue , msg = verify_model(model=model)
    if(not isTrue) : return isTrue , msg 

    isTrue , msg = verify_instance(instance=instance,data=data,cibleName=cibleName)
    if(not isTrue) : return isTrue , msg 


    if(categoriel_columns!=None) : 
        isTrue , msg = verify_categorical_columns(categoriel_columns, data)
        if(not isTrue) : return isTrue , msg 
    if(medicalMetrics!=None) :
        isTrue , msg = verify_medical(medicalMetrics=medicalMetrics)
        if(not isTrue) : return isTrue , msg 
    return True, ""



   
