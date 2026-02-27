from fastapi import FastAPI, status, HTTPException
from pydantic import BaseModel
from predict import predict_data

app = FastAPI()

class WineData(BaseModel):
    alcohol: float
    malic_acid: float
    ash: float
    alcalinity_of_ash: float
    magnesium: float
    total_phenols: float
    flavanoids: float
    nonflavanoid_phenols: float
    proanthocyanins: float
    color_intensity: float
    hue: float
    od280_od315_of_diluted_wines: float
    proline: float

class WineResponse(BaseModel):
    response: int
    label: str

WINE_CLASSES = {0: "Class 0 (Cultivar 1)", 1: "Class 1 (Cultivar 2)", 2: "Class 2 (Cultivar 3)"}

@app.get("/", status_code=status.HTTP_200_OK)
async def health_ping():
    return {"status": "healthy", "model": "Wine Classifier"}

@app.post("/predict", response_model=WineResponse)
async def predict_wine(wine_features: WineData):
    try:
        features = [[
            wine_features.alcohol, wine_features.malic_acid, wine_features.ash,
            wine_features.alcalinity_of_ash, wine_features.magnesium,
            wine_features.total_phenols, wine_features.flavanoids,
            wine_features.nonflavanoid_phenols, wine_features.proanthocyanins,
            wine_features.color_intensity, wine_features.hue,
            wine_features.od280_od315_of_diluted_wines, wine_features.proline
        ]]
        prediction = predict_data(features)
        return WineResponse(response=prediction, label=WINE_CLASSES[prediction])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
