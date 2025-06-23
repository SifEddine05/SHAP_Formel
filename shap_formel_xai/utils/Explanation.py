from pyxai import Learning, Explainer
from sklearn.base import is_classifier, is_regressor
from xgboost import  XGBClassifier

def shap_explanation(data, shap, cibleName, model, instance):
    """
    Generate SHAP explanations for a given instance using a tree-based model.

    Args:
        data (pd.DataFrame): Reference dataset including features and target.
        shap (module): SHAP module imported externally.
        cibleName (str): Name of the target column in the dataset.
        model (object): Trained tree-based model compatible with SHAP TreeExplainer.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.

    Returns:
        tuple:
            - shap_values_class (np.ndarray): SHAP values for the predicted class.
            - top_10_shap_feature_names (list): List of top 10 feature names based on SHAP values.
            - explainer_shap (shap.TreeExplainer): Initialized SHAP TreeExplainer object.
    """
    X = data.drop(columns=[cibleName])
    explainer_shap = shap.TreeExplainer(model, model_output="raw", data=None)
    classe_predite = model.predict(instance)
    shap_values = explainer_shap.shap_values(instance)

    if is_classifier(model) or isinstance(model, XGBClassifier)  :
        
        shap_values_class = shap_values[:, :, int(classe_predite[0])]

    else : 
        shap_values_class = shap_values
    top_10_shap_features = abs(shap_values_class[0]).argsort()[-10:][::-1]
    top_10_shap_feature_names = X.columns[top_10_shap_features].tolist()

    return shap_values_class[0], top_10_shap_feature_names, explainer_shap


def formel_explanation(data, cibleName, model, instance):
    """
    Generate formal explanations for a given instance using PyXAI explainer.

    Args:
        data (pd.DataFrame): Reference dataset including features and target.
        cibleName (str): Name of the target column in the dataset.
        model (object): Trained model compatible with PyXAI Learning and Explainer.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.

    Returns:
        dict: Dictionary mapping feature names to their formal explanations (sufficient reasons).
    """
    try : 
        feature_names = data.columns.tolist()
        prediction = model.predict(instance)
        learner, model1 = Learning.import_models(model, feature_names)
        explainer = Explainer.initialize(model1, instance=instance.values[0])
        sufficient = explainer.sufficient_reason()
        sufficient_features = explainer.to_features(sufficient)
        sufficient_features_dict = {feature.split()[0]: feature for feature in sufficient_features}
        return sufficient_features_dict
    except : 
        print("No Formel Explanation model not supported")
        return []
    


def get_optimal_formel(sufficient_features_dict, top_10_features):
    """
    Filter formal explanations to retain only those for the top important features.

    Args:
        sufficient_features_dict (dict): Dictionary mapping feature names to formal explanations.
        top_10_features (list): List of top 10 feature names to filter by.

    Returns:
        dict: Filtered dictionary containing only features in top_10_features.
    """
    filtered_dict = {}
    for feature_name in top_10_features:
        if feature_name in sufficient_features_dict:
            filtered_dict[feature_name] = sufficient_features_dict[feature_name]
    
    return filtered_dict
