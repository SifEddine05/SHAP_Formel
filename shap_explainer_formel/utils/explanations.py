
from pyxai import Tools, Learning, Explainer


def shap_explanation(data,shap,cibleName,model,instance) : 
    X=data.drop(columns=[cibleName])

    explainer_shap = shap.TreeExplainer(model,model_output="raw",data=None)

    classe_predite = model.predict(instance)
    shap_values = explainer_shap.shap_values(instance)
    shap_values_class = shap_values[:, :, int(classe_predite[0])]

    top_10_shap_features = abs(shap_values_class[0]).argsort()[-10:][::-1]


    top_10_shap_feature_names = X.columns[top_10_shap_features].tolist()

    return shap_values_class[0],top_10_shap_feature_names,explainer_shap


def formel_explanation(data,cibleName,model,instance) : 
    feature_names = data.columns.tolist()
    
    prediction = model.predict(instance)
    learner, model1 = Learning.import_models(model, feature_names)
    explainer = Explainer.initialize(model1, instance=instance.values[0])
    sufficient = explainer.sufficient_reason()
    sufficient_features = explainer.to_features(sufficient)
    sufficient_features_dict = {feature.split()[0]: feature for feature in sufficient_features}  # dictionary of features and formel explanations
    return sufficient_features_dict


def get_optimal_formel(sufficient_features_dict, top_10_features):
    filtered_dict = {}
    for feature_name in top_10_features:
        if feature_name in sufficient_features_dict:
            filtered_dict[feature_name] = sufficient_features_dict[feature_name]
    return filtered_dict