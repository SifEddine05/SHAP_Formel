from sklearn.base import is_classifier
from xgboost import XGBClassifier


def show_SHAP_plot(shap, shap_values, explainer, instance, model):
    """
    Display a SHAP waterfall plot explaining the prediction for a given instance.

    Args:
        shap (module): SHAP module, used to generate plots.
        shap_values (np.ndarray): SHAP values for the instance.
        explainer (shap.Explainer): SHAP explainer object with expected values.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.
        model (object): Trained model used to predict the instance's class.

    Returns:
        None: This function prints output and displays a plot directly.
    """
    # Predict the class for the instance (get the first prediction)
    predicted_class = model.predict(instance.values)[0]

    print("\n" + "-"*70)
    print("🔍 Explanation of the prediction for the given instance")
    print("-"*70)
    print(f"➡️  Predicted class: {predicted_class}")
    print("\n📊 SHAP values contribution analysis:\n")

    if is_classifier(model) or isinstance(model, XGBClassifier):
        expected_value = explainer.expected_value[int(predicted_class)]
    else : 
        expected_value = explainer.expected_value
    # Display the SHAP waterfall plot
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values,
            base_values=expected_value,
            data=instance.values[0],
            feature_names=instance.columns.tolist()  # optional but good practice
        ),
        max_display=10  # Show top 10 important features
    )
    print("-"*70 + "\n")


def show_formel_explanations(formel_dic):
    """
    Print a summary of formal explanations for the features of an instance.

    Args:
        formel_dic (dict): Dictionary mapping feature names to formal explanations.

    Returns:
        None: This function prints output directly.
    """
    print("\n📝 Formal explanations summary:")
    print("-" * 70)
    if len(formel_dic.items()) == 0:
        print("No formel explanation available for this type of model")
    for feature_name, explanation in formel_dic.items():
        print(f"• {feature_name}: {explanation}")
    print("-" * 70 + "\n")
