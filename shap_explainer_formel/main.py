from shap_explainer_formel.utils.Affichage import show_SHAP_plot, show_formel_explanations
from shap_explainer_formel.utils.explanations import formel_explanation, get_optimal_formel, shap_explanation
from shap_explainer_formel.utils.interpretation import interprete_results
from shap_explainer_formel.utils.validation import validate_input
import shap



def SHAP_Formel(model, instance, data, cibleName,categoriel_columns=None,medicalMetrics=None):
    # model est le model choisit ici le model autoriser c'est random forest Classifier entrainer
    # instance est pd.DataFrame	d'un seul row 
    # data c'est une dataframe sans manquantes 
    # cibleName c'est une string (le nom de la colonne cible)
    # medicalMetrics est sous la forme [{"column_name": "Blood Pressure",
    #     "intervalles": [
    #     {"min": 0, "max": 80, "interpretation": "Low"},
    #     {"min": 81, "max": 120, "interpretation": "Normal"},
    #     {"min": 121, "max": 200, "interpretation": "High"}
    #   ]}]
    # categoriel_columns list des colonnes catégoriel s'il existent ["col1","col2"]

    # Vérifie la validité de l'input 
    isvalide,msg = validate_input(model, instance, data, cibleName,categoriel_columns,medicalMetrics)
    if (not isvalide) : 
        print(msg)
        return
    if(categoriel_columns is None) : categoriel_columns = []
    if(medicalMetrics is None) : medicalMetrics = []


    shap.initjs()
    shap_values,top_10_features,explainer_shap = shap_explanation(data,shap,cibleName,model,instance)

    sufficient_features = formel_explanation(data,cibleName,model,instance)
    optimal_formel = get_optimal_formel(sufficient_features, top_10_features)

    show_SHAP_plot(shap, shap_values, explainer_shap, instance, model)
    show_formel_explanations(optimal_formel)
    interprete_results(top_10_features,categoriel_columns,medicalMetrics,cibleName,data,instance)


    



    