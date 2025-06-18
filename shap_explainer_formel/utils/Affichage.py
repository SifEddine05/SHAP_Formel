

from matplotlib import pyplot as plt


def show_SHAP_plot(shap, shap_values, explainer, instance, model):
    # Predict the class for the instance (get the first prediction)
    predicted_class = model.predict(instance.values)[0]

    print("\n" + "-"*70)
    print("🔍 Explanation of the prediction for the given instance")
    print("-"*70)
    print(f"➡️  Predicted class: {predicted_class}")
    print("\n📊 SHAP values contribution analysis:\n")


    # Display the SHAP waterfall plot
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values,
            base_values=explainer.expected_value[int(predicted_class)],
            data=instance.values[0],
            feature_names=instance.columns.tolist()  # optional but good practice

        ),
        max_display=10  # Show top 10 important features
    )
   

    print("-"*70 + "\n")


def show_formel_explanations(formel_dic):
    print("\n📝 Formal explanations summary:")
    print("-" * 70)
    for feature_name, explanation in formel_dic.items():
        print(f"• {feature_name}: {explanation}")
    print("-" * 70 + "\n")






