import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
import io
from pathlib import Path

from all_pytorch_projects.database.models import FashionMnist
from all_pytorch_projects.database.db import SessionLocal
from sqlalchemy.orm import Session

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

fashion_router = APIRouter(prefix='/fashion', tags=['Fashion MNIST'])

class FashionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        x = self.conv_block(x)
        x = self.fc(x)
        return x

BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / 'pytorch_model' / 'fashion_cnn.pth'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = FashionCNN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((28, 28)),
    transforms.ToTensor()
])

@fashion_router.post('/predict_fashion/')
async def predict_fashion(file: UploadFile=File(...), db:Session=Depends(get_db)):
    try:
        img_data = await file.read()
        img_by = Image.open(io.BytesIO(img_data))
        image_to_tensor = transform(img_by).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_to_tensor)
            result = torch.argmax(y_pred, dim=1).item()

            data = FashionMnist(label=file.filename, predict=result)
            db.add(data)
            db.commit()
            db.refresh(data)

        return {'predict': result}
    except Exception as e:
        return {'Error': str(e)}
















