import pickle
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Chargement des données
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# Chargement des meilleurs hyperparamètres
with open("models/best_rf_params.pkl", "rb") as f:
    best_params = pickle.load(f)

# Entraînement du modèle avec les meilleurs paramètres
best_model = RandomForestRegressor(**best_params)
best_model.fit(X_train_scaled, y_train.values.ravel())  # .ravel() pour éviter les erreurs de dimensions

# Sauvegarde du modèle entraîné
with open("models/best_rf_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Modèle entraîné et sauvegardé avec succès !")
