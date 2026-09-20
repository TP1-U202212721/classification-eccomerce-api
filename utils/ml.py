import joblib
import pandas as pd

model = None
MODEL_PATH = "models/estimation_risk_model.pkl"

def get_model():
    global model
    if model is None:
        model = joblib.load(MODEL_PATH)
    return model


def predict_risk(data):
    model = get_model()
    df = pd.DataFrame([data])
    prediction = model.predict(df)
    return prediction[0]