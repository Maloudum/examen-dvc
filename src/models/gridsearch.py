from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
import pickle
import pandas as pd

# Définition du modèle
rf = RandomForestRegressor()

# Définition des hyperparamètres à tester
param_grid = {
    'n_estimators': [100, 200],  # Nombre d'arbres
    'max_depth': [10, 20],  # Profondeur maximale
    'min_samples_split': [2, 5],  # Nombre min d'échantillons pour un split
    'min_samples_leaf': [1, 2, 4]  # Nombre min d'échantillons par feuille
}

# Chargement des données
X_train_scaled = pd.read_csv("data/processed_data/X_train_scaled.csv")
y_train = pd.read_csv("data/processed_data/y_train.csv")

# GridSearch avec validation croisée et scoring MSE et R2
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    verbose=2,
    scoring='neg_mean_squared_error'  
)

# Exécution de GridSearch
grid_search.fit(X_train_scaled, y_train.values.ravel())  # .ravel() pour éviter les warnings

# Sauvegarde des meilleurs paramètres
best_params = grid_search.best_params_

with open("models/best_rf_params.pkl", "wb") as f:
    pickle.dump(best_params, f)

print("Meilleurs paramètres trouvés :", best_params)
