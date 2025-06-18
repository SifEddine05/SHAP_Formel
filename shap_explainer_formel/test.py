import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from shap_explainer_formel.main import SHAP_Formel


import os
import pandas as pd

# Get directory of the current script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construct full path to df.csv relative to the script directory
csv_path = os.path.join(script_dir, 'df.csv')

df = pd.read_csv(csv_path)
df_cleaned = df.copy()
df_cleaned = df_cleaned.drop(['Id','DATEV0','VERSION', 'DATEV1', 'DATEV2'], axis=1)
data= df_cleaned.copy()

from sklearn.preprocessing import LabelEncoder

# Initialiser le LabelEncoder
from sklearn.preprocessing import LabelEncoder

# Initialiser le LabelEncoder
le = LabelEncoder()

# Liste des colonnes à encoder
colonnes_a_encoder = ['SEXE', 'ETUDE', 'SAOS', 'SJSR', 'CSP', 'REG']

# Dictionnaire pour stocker les mappings
label_mappings = {}

# Appliquer le LabelEncoder à chaque colonne et stocker le mapping
for col in colonnes_a_encoder:
    data[col] = le.fit_transform(data[col])
    label_mappings[col] = dict(zip(le.classes_, le.transform(le.classes_)))

# Afficher le mapping complet pour toutes les colonnes
print("Mappings :", label_mappings)
from sklearn.preprocessing import StandardScaler, LabelEncoder
scaler = StandardScaler()
df_scaled = scaler.fit_transform(data)
from sklearn.cluster import KMeans

k_best = 2 # Remplace par la valeur optimale après analyse du coude
kmeans = KMeans(n_clusters=k_best, random_state=42, n_init=10)
data['Cluster'] = kmeans.fit_predict(df_scaled)  # Ajouter les clusters au dataset

# Importation des bibliothèques nécessaires
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

X = data.drop(columns=['Cluster'])  # Caractéristiques (features)
y = data['Cluster']  # Variable cible (target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
model = RandomForestClassifier(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
instance =  instance = X.iloc[[100]]


isi_interpretation = [{
    "column_name": "ISIV1",
    "intervalles": [
        {"min": 0,  "max": 7,  "interpretation": "Absence d'insomnie"},
        {"min": 8,  "max": 14, "interpretation": "Insomnie sub-clinique"},
        {"min": 15, "max": 21, "interpretation": "Insomnie clinique (modérée)"},
        {"min": 22, "max": 28, "interpretation": "Insomnie clinique (sévère)"}
    ]
}]


SHAP_Formel(model, instance, data, target_column="Cluster",medical_metrics=isi_interpretation,categorical_columns=['IMC'])
