import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
import io
from pathlib import Path
import json


from all_pytorch_projects.database.models import Cifar
from all_pytorch_projects.database.db import SessionLocal
from sqlalchemy.orm import Session

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

class VGGCheckImage(nn.Module):
  def __init__(self):
    super().__init__()
    self.conv = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),

        nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(),
        nn.MaxPool2d(2),
    )

    self.fc = nn.Sequential(
        nn.Flatten(),
        nn.Linear(256 * 4 *4, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, 100)
    )

  def forward(self, x):
    x = self.conv(x)
    x = self.fc(x)
    return x

cifar_router = APIRouter(prefix='/cifar', tags=['Cifar 100'])


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BASE_DIR = Path(__file__).resolve().parent.parent.parent
labels = BASE_DIR / 'pytorch_model' / 'cifar100_classes.json'
model_path = BASE_DIR / 'pytorch_model' / 'model_cnn.pth'
model = VGGCheckImage()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


with open(labels, 'r', encoding='utf-8') as f:
    CLASS_NAMES = json.load(f)


transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor()
])

@cifar_router.post('/predict_cifar/')
async def predict_image(file: UploadFile=File(...), db:Session=Depends(get_db)):
    try:
        image_data = await file.read()
        image_by = Image.open(io.BytesIO(image_data)).convert('RGB')
        image_to_tensor = transform(image_by).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(image_to_tensor)
            result = torch.argmax(y_pred, dim=1).item()

            data = Cifar(label=file.filename, predict=result)
            db.add(data)
            db.commit()
            db.refresh(data)

            return {'predict': CLASS_NAMES[result]}
    except Exception as e:
        return {'Error': str(e)}












