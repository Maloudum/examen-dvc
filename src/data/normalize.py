import pandas as pd
from sklearn.preprocessing import StandardScaler

X_train = pd.read_csv("data/processed_data/X_train.csv")
X_test = pd.read_csv("data/processed_data/X_test.csv")

scaler = StandardScaler()

# Appliquer la normalisation sur les ensembles d'entraînement et de test
X_train_scaled_ = scaler.fit_transform(X_train)
X_test_scaled_ = scaler.transform(X_test)

X_train_scaled = pd.DataFrame(X_train_scaled_,columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled_,columns=X_test.columns)

# Sauvegarder les ensembles normalisés dans data/processed
X_train_scaled.to_csv('data/processed_data/X_train_scaled.csv', index=False)
X_test_scaled.to_csv('data/processed_data/X_test_scaled.csv', index=False)
