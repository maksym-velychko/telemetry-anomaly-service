import joblib
import pandas as pd
from fastapi import APIRouter

from api.schemas import InputData

router_v1 = APIRouter(prefix='/api/v1', tags=['TelemetryV1'])

model_path = 'models/isolation_forest.pkl'
model = joblib.load(model_path)

@router_v1.post('predict/')
async def model_prediction(data: InputData):
    input_data = pd.DataFrame([{
        'CPU_Usage': data.cpu_usage,
        'Memory_Usage': data.memory_usage,
    }])

    prediction = model.predict(input_data)[0]
    is_anomaly = bool(prediction == -1)

    return {
        'is_anomaly': is_anomaly,
        'status': 'CRITICAL' if is_anomaly else 'OK',
    }