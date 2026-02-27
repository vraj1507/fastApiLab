import pickle
import numpy as np
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parents[1] / "model" / "wine_model.pkl"

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

def predict_data(features):
    prediction = model.predict(np.array(features))
    return int(prediction[0])
