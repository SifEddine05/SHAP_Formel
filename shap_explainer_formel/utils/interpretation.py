

def interprete_medical(medical, instance, top_10_features):
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
                        print (f"{col} = {val}: {intervalle['interpretation']}")

            if not interpretation_found:
                return print(f"{col} = {val}: Value is outside the defined intervals.")


def interprete_numeric(numeric, instance, top_10_features,data):
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

def interprete_categoriel(categoriel,instance,top_10_features,data) : 
    for feature_info in categoriel:
        if feature_info in top_10_features:
            val = instance.iloc[0][feature_info]

            # Calculate percentage of individuals in the same category
            percent = data[feature_info].value_counts(normalize=True).get(val, 0) * 100

            print(f"{feature_info} = {val} 🧠 ({percent:.2f}% of individuals)")





def interprete_results(top_10_features,categoriel,medical,cibleName,data,instance):
    print("\n📝 Interpretation of Explanations :")
    print("-" * 70)
    X=data.drop(columns=[cibleName])
    numeric = [col for col in X.columns if col not in categoriel and col not in medical]
    interprete_medical(medical, instance, top_10_features)
    interprete_numeric(numeric, instance, top_10_features,data)
    interprete_categoriel(categoriel,instance,top_10_features,data)
    print("-" * 70 + "\n")
