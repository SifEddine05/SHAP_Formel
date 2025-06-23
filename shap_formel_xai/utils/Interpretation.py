def interprete_medical(medical, instance, top_10_features):
    """
    Interpret medical features of the instance based on defined intervals.

    Args:
        medical (list): List of medical metrics with 'column_name' and 'intervalles' definitions.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.
        top_10_features (list): List of top 10 important feature names.

    Returns:
        None: Prints interpretation for each medical feature found in top_10_features.
    """
    for feature_info in medical:
        col = feature_info.get("column_name")
        if col in top_10_features:
            val = instance.iloc[0][col]  # valeur de la colonne dans l'instance (ligne unique)
            intervalles = feature_info.get("intervalles", [])
            interpretation_found = False
            for intervalle in intervalles:
                min_val = intervalle.get("min")
                max_val = intervalle.get("max")
                if min_val is not None and max_val is not None:
                    if min_val <= val <= max_val:
                        interpretation_found = True
                        print(f"{col} = {val}: {intervalle['interpretation']}")

            if not interpretation_found:
                return print(f"{col} = {val}: Value is outside the defined intervals.")


def interprete_numeric(numeric, instance, top_10_features, data):
    """
    Interpret numeric features of the instance compared to dataset averages.

    Args:
        numeric (list): List of numeric feature names.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.
        top_10_features (list): List of top 10 important feature names.
        data (pd.DataFrame): Reference dataset used to calculate averages.

    Returns:
        None: Prints interpretation comparing instance value with average.
    """
    for feature_info in numeric:
        if feature_info in top_10_features:
            val = instance.iloc[0][feature_info]
            moy = data[feature_info].mean()
            diff = val - moy

            if val > moy:
                print(f"{feature_info} = {val:.2f}: ➕ Higher than average (+{abs(diff):.2f})")
            elif val < moy:
                print(f"{feature_info} = {val:.2f}: ➖ Lower than average (-{abs(diff):.2f})")
            else:
                print(f"{feature_info} = {val:.2f}: ➡️ Equal to the average")


def interprete_categoriel(categoriel, instance, top_10_features, data):
    """
    Interpret categorical features of the instance with distribution percentages.

    Args:
        categoriel (list): List of categorical feature names.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.
        top_10_features (list): List of top 10 important feature names.
        data (pd.DataFrame): Reference dataset to calculate value distributions.

    Returns:
        None: Prints interpretation with percentage of individuals in the same category.
    """
    for feature_info in categoriel:
        if feature_info in top_10_features:
            val = instance.iloc[0][feature_info]

            # Calculate percentage of individuals in the same category
            percent = data[feature_info].value_counts(normalize=True).get(val, 0) * 100

            print(f"{feature_info} = {val} 🧠 ({percent:.2f}% of individuals)")


def interprete_results(formel_dic, categoriel, medical, cibleName, data, instance):
    """
    Aggregate and print interpretations for medical, numeric, and categorical features.

    Args:
        top_10_features (list): List of top 10 important feature names.
        categoriel (list): List of categorical feature names.
        medical (list): List of medical metrics definitions.
        cibleName (str): Name of the target column in dataset.
        data (pd.DataFrame): Reference dataset.
        instance (pd.DataFrame): Single-row dataframe representing the instance to explain.

    Returns:
        None: Prints a structured interpretation report.
    """
    print("\n📝 Interpretation of Explanations :")
    print("-" * 70)
    X = data.drop(columns=[cibleName])
    top_features = list(formel_dic.keys())

    medical_cols = [m["column_name"] for m in medical] if medical else []
    numeric = [col for col in X.columns if col not in categoriel + medical_cols]
    interprete_medical(medical, instance, top_features)
    interprete_numeric(numeric, instance, top_features, data)
    interprete_categoriel(categoriel, instance, top_features, data)
    print("-" * 70 + "\n")
