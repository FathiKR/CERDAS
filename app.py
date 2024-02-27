import model_build.iris_model.classifier as clf
from fastapi import FastAPI
from joblib import load
from model_build.api_creation import Iris

app = FastAPI(title="Iris ML API", description="API for iris dataset ml model", version="1.0")

@app.on_event('startup')
async def load_model():
    clf.model = load('model_build/iris_model/iris_model_base.joblib')

@app.post('/predict', tags=["predictions"])
async def get_prediction(iris: Iris):
    
    data = dict(iris)['data_input']
    
    prediction = clf.model.predict(data).tolist()
    # log_proba = clf.model.predict_log_proba(data).tolist()
    
    return {"prediction": prediction}