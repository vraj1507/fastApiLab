import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from pathlib import Path
from data import get_data

X, y = get_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "wine_model.pkl"
with open(MODEL_PATH, "wb") as f:
    pickle.dump(model, f)

print("Model saved! Test accuracy:", round(model.score(X_test, y_test), 4))
