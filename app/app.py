from fastapi import FastAPI
from detection import detect_anomalies, test_dataloader

model = ""

app = FastAPI(
    title="Anomaly Detection",
    # description="Finding Anomaly", 
    version="0.1")


@app.get("/")
async def health_check():
    return {'health':'OK'}


@app.get("/detect/")
async def detect():
    anomalies, errors = detect_anomalies(model, test_dataloader, threshold=0.01)
    
    return f"Number of anomalies detected: {sum(anomalies)}"