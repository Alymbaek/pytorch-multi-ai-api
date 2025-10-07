import torch
import torch.nn as nn
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import torch.nn.functional as F
import soundfile as sf
from torchaudio import transforms
from pathlib import Path
import json
import io

from all_pytorch_projects.database.models import Gtzan
from all_pytorch_projects.database.db import SessionLocal
from sqlalchemy.orm import Session

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

gtzan_router = APIRouter(prefix='/gtzan', tags=['Gtzan Audio'])


class Music(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8))
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = self.fc(x)
        return x


BASE_DIR = Path(__file__).resolve().parent.parent.parent
model_path = BASE_DIR / 'pytorch_model' / 'model_music.pth'

with open(BASE_DIR / 'pytorch_model' / "music_classes.json", "r", encoding="utf-8") as f:
    labels = json.load(f)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

index = {idx: label for idx, label in enumerate(labels)}
model = Music()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

transform = transforms.MelSpectrogram(
    sample_rate=16000,
    n_mels=64
)
max_len = 100

def change_audio(waveform, sample_rate):
    # Конвертируем в tensor если это еще не tensor
    if not isinstance(waveform, torch.Tensor):
        waveform = torch.tensor(waveform, dtype=torch.float32)

    # Добавляем batch dimension для обработки
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)

    # Ресемплинг если нужно
    if sample_rate != 16000:
        resampler = transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)

    # Убираем batch dimension для спектрограммы
    waveform = waveform.squeeze(0)

    spec = transform(waveform)

    if spec.shape[1] > max_len:
        spec = spec[:, :max_len]
    elif spec.shape[1] < max_len:
        count_zero = max_len - spec.shape[1]
        spec = F.pad(spec, (0, count_zero))

    return spec



@gtzan_router.post('/predict_gtzan/')
async def predict_gtzan(file: UploadFile = File(..., ), db:Session=Depends(get_db)):
    try:
        audio = await file.read()
        if not audio:
            raise HTTPException(status_code=400, detail='Empty')
        wf, sr = sf.read(io.BytesIO(audio), dtype='float32')

        spec = (change_audio(wf, sr))
        x = spec.unsqueeze(0).unsqueeze(0).to(device)

        with torch.no_grad():
            y_pred = model(x)
            pred = torch.argmax(y_pred, dim=1).item()
            label = index[pred]
            
            data = Gtzan(label=file.filename, predict=label)
            db.add(data)
            db.commit()
            db.refresh(data)

            return {'class':label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))




