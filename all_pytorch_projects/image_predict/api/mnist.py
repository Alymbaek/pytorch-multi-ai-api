import torch
import torch.nn as nn
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
import io
import torchvision.transforms as transforms
from PIL import Image
from pathlib import Path

from all_pytorch_projects.database.models import Mnist
from all_pytorch_projects.database.db import SessionLocal
from sqlalchemy.orm import Session

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

mnist_router = APIRouter(prefix='/mnist', tags=['MNIST NUMBERS'])

class ImageSVM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 14 * 14, 64),
            nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / 'pytorch_model' / 'model.pth'
model = ImageSVM()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
])


@mnist_router.post('/predict_mnist/')
async def predict_numbers(file: UploadFile=File(...), db:Session=Depends(get_db)):
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data))
        image_to_tensor = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_to_tensor)
            pred = torch.argmax(y_pred, dim=1).item()

            data = Mnist(label=file.filename, predict=pred)
            db.add(data)
            db.commit()
            db.refresh(data)

        return {'predict': pred}
    except Exception as e:
        return {'Error': str(e)}










