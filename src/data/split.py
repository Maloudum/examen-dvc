from sklearn.model_selection import train_test_split
import pandas as pd

data = pd.read_csv("data/raw_data/raw.csv")
X = data.drop(["date","silica_concentrate"], axis = 'columns')
y = data['silica_concentrate']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42)

print(len(X))
print(len(X_test))

X_train.to_csv('data/processed_data/X_train.csv', index=False)
X_test.to_csv('data/processed_data/X_test.csv', index=False)
y_train.to_csv('data/processed_data/y_train.csv', index=False)
y_test.to_csv('data/processed_data/y_test.csv', index=False)

