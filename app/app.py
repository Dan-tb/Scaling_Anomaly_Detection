from fastapi import FastAPI
from detection import detect_anomalies
from pydantic import BaseModel
from typing import List
from preprocess import ImageDataset, transform
import torch
from model import ConvVAE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ConvVAE().to(device)

model_name = "anomaly_model.pth"
model_path = "C:/Users/USER/Documents/Projects/AI projects/Anomaly_app/Scaling_Anomaly_Detection/data/" + model_name
state_dict = torch.load(model_path, weights_only=True, map_location="cpu")
model = model.load_state_dict(state_dict)


image_dir = "C:/Users/USER/Downloads/Garba/Uninfected/"
dataset = ImageDataset(image_dir=image_dir, transform=transform)
test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

app = FastAPI(
    title="Anomaly Detection",
    # description="Finding Anomaly", 
    version="0.1")


class Anomaly_Item(BaseModel):
    name_id: str
    anomaly_status: bool


@app.get("/")
async def health_check():
    return {'health':'OK'}


@app.get("/detect/", response_model=List[Anomaly_Item])
async def detect():
    anomalies, errors = detect_anomalies(model, test_dataloader, threshold=0.01)
    
    return [
        {"image": anomaly["image"], "is_anomaly": anomaly["is_anomaly"]}
        for anomaly in anomalies
    ]