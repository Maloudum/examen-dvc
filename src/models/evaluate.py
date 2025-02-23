import json
import pickle
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
# Chargement des données de test
X_test_scaled = pd.read_csv("data/processed_data/X_test_scaled.csv")
y_test = pd.read_csv("data/processed_data/y_test.csv")

# Chargement du modèle entraîné
with open("models/best_rf_model.pkl", "rb") as f:
    best_model = pickle.load(f)

# Prédictions sur l'ensemble de test
y_pred_ = best_model.predict(X_test_scaled)

# Calcul des métriques
mse = mean_squared_error(y_test, y_pred_)
r2 = r2_score(y_test, y_pred_)

y_pred = pd.DataFrame(y_pred_, columns=y_test.columns)
y_pred.to_csv("data/processed_data/predictions.csv")
# Enregistrement des métriques dans un fichier JSON
metrics = {"MSE": mse, "R2": r2}

with open("metrics/scores.json", "w") as f:
    json.dump(metrics, f, indent=4)

print("Évaluation terminée. Résultats enregistrés dans metrics/scores.json")
