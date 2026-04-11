from fastapi import FastAPI
from services.forecast_service import run_forecast

app = FastAPI()

@app.get("/")
def home():
    return {"message": "TempCast running"}

@app.post("/predict")
def predict(text: str):
    result = run_forecast(text)
    return {"prediction": result}