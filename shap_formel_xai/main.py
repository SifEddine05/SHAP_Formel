from shap_formel_xai.utils.Visualization import show_SHAP_plot, show_formel_explanations
from shap_formel_xai.utils.Explanation import formel_explanation, get_optimal_formel, shap_explanation
from shap_formel_xai.utils.Interpretation import interprete_results
from shap_formel_xai.utils.Validation import validate_input
import shap

def SHAP_Formel(model, instance, data, target_column, categorical_columns=None, medical_metrics=None):
    """
    Explain a prediction of a given instance using SHAP values and formal rule-based explanations.

    Parameters:
    - model: trained model (e.g., RandomForestClassifier)
    - instance: pd.DataFrame with a single row representing the instance to explain
    - data: pd.DataFrame without missing values (training or reference data)
    - target_column: str, name of the target column in data
    - categorical_columns: list of categorical column names (optional)
    - medical_metrics: list of dicts describing medical metrics and intervals (optional)
      Format example:
      [
        {
          "column_name": "Blood Pressure",
          "intervals": [
            {"min": 0, "max": 80, "interpretation": "Low"},
            {"min": 81, "max": 120, "interpretation": "Normal"},
            {"min": 121, "max": 200, "interpretation": "High"}
          ]
        }
      ]

    Returns:
    None, prints explanations and plots
    """

    # Validate inputs before running explanation
    is_valid, message = validate_input(model, instance, data, target_column, categorical_columns, medical_metrics)
    if not is_valid:
        print(message)
        return

    # Default empty lists if not provided
    if categorical_columns is None:
        categorical_columns = []
    if medical_metrics is None:
        medical_metrics = []

    # Initialize JS visualization for SHAP plots
    shap.initjs()

    # Compute SHAP values and get top 10 important features for the prediction
    shap_values, top_10_features, explainer_shap = shap_explanation(data, shap, target_column, model, instance)

    # Generate formal explanations based on sufficient features
    sufficient_features = formel_explanation(data, target_column, model, instance)

    # Filter formal explanations to only top 10 SHAP features
    optimal_formel = get_optimal_formel(sufficient_features, top_10_features)

    # Display SHAP waterfall plot for the instance prediction
    show_SHAP_plot(shap, shap_values, explainer_shap, instance, model)

    # Print the formal explanations in a readable format
    show_formel_explanations(optimal_formel)

    # Interpret the results based on top features, categorical columns, and medical metrics
    interprete_results(optimal_formel, categorical_columns, medical_metrics, target_column, data, instance)
