from fastapi import FastAPI, UploadFile, File, HTTPException
from detection import detect_anomalies
from pydantic import BaseModel
from typing import List
import torch
from model import ConvVAE
import zipfile
import rarfile
import io
from PIL import Image
from preprocess import ImageDataset, transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize the model and load the state dictionary
model = ConvVAE().to(device)
model_name = "anomaly_model.pth"
model_path = "C:/Users/USER/Documents/Projects/AI projects/Anomaly_app/Scaling_Anomaly_Detection/data/" + model_name
state_dict = torch.load(model_path, map_location=device)  
model.load_state_dict(state_dict)  
model.eval()

app = FastAPI(
    title="Anomaly Detection",
    version="0.1"
)

class Anomaly_Item(BaseModel):
    name_id: str
    anomaly_status: bool

@app.get("/")
async def health_check():
    return {'health': 'OK'}

@app.post("/detect/", response_model=List[Anomaly_Item])
async def detect(file: UploadFile = File(...)):
    file_ext = file.filename.split('.')[-1].lower()
    if file_ext not in ["zip", "rar"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a zip or rar file.")

    images = []
    image_names = []

    # Process ZIP file
    if file_ext == "zip":
        with zipfile.ZipFile(io.BytesIO(await file.read())) as zip_file:
            for filename in zip_file.namelist():
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    with zip_file.open(filename) as image_file:
                        image = Image.open(image_file).convert("RGB")
                        images.append(image)
                        image_names.append(filename.split('/')[-1])

    # Process RAR file
    elif file_ext == "rar":
        with rarfile.RarFile(io.BytesIO(await file.read())) as rar_file:
            for filename in rar_file.namelist():
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    with rar_file.open(filename) as image_file:
                        image = Image.open(image_file).convert("RGB")
                        images.append(image)
                        image_names.append(filename.split('/')[-1])

    # Create the dataset and dataloader with in-memory images
    dataset = ImageDataset(images=images, image_names=image_names, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False)

    # Run anomaly detection
    anomalies, errors = detect_anomalies(model, test_dataloader, threshold=0.01)

    return [
        {"name_id": anomaly["image"], "anomaly_status": anomaly["is_anomaly"]}
        for anomaly in anomalies
    ]