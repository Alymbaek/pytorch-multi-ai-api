import torch
from fastapi import APIRouter, UploadFile, File, HTTPException, Depends
import torch.nn.functional as F
from torchaudio import transforms as T
import soundfile as sf
import torch.nn as nn
from torchaudio import transforms
import torchaudio
from pathlib import Path
import json
import io

from all_pytorch_projects.database.models import Urban
from all_pytorch_projects.database.db import SessionLocal
from sqlalchemy.orm import Session

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


urban_router = APIRouter(prefix='/urban', tags=['Urban Audio'])


class URBAN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((8, 8)),
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
model_path = BASE_DIR / 'pytorch_model' / 'model_urban.pth'

labels = {'fold1': 0,
 'fold10': 1,
 'fold2': 2,
 'fold3': 3,
 'fold4': 4,
 'fold5': 5,
 'fold6': 6,
 'fold7': 7,
 'fold8': 8,
 'fold9': 9}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

index = {idx: label for idx, label in enumerate(labels)}
model = URBAN()
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()


transform = transforms.MelSpectrogram(
    sample_rate=22050,
    n_mels=64
)
max_len = 500
amp2db = T.AmplitudeToDB()

def change_audio(waveform, sample_rate):
    # waveform: numpy array from soundfile (shape: (n,) или (n, channels))
    # 1) to tensor float32
    waveform = torch.tensor(waveform, dtype=torch.float32)

    # 2) make mono: if shape (n_samples, n_channels) -> average channels -> (n_samples,)
    if waveform.dim() == 2:
        # soundfile returns (frames, channels) -> transpose to (channels, frames) preferred by torchaudio
        waveform = waveform.mean(dim=1)  # среднее по каналам -> (n_samples,)

    # 3) ensure shape [1, samples] as expected by Resample and MelSpectrogram
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # -> [1, samples]

    # 4) resample to 22050 if needed
    if sample_rate != 22050:
        waveform = torchaudio.functional.resample(waveform, orig_freq=sample_rate, new_freq=22050)

    # now waveform: [1, samples]
    # 5) create spectrogram: MelSpectrogram accepts (channel, time) -> returns (channel, n_mels, T)
    spec = transform(waveform)               # -> [1, n_mels, T]
    spec = amp2db(spec)                      # лог-амплитуда

    # drop channel dim -> [n_mels, T]
    if spec.dim() == 3:
        spec = spec.squeeze(0)

    # debug: (можешь удалить потом)
    # print("spec.shape after transform:", spec.shape)

    # 6) pad / truncate along time dimension (dim=1)
    T_len = spec.shape[1]
    if T_len > max_len:
        spec = spec[:, :max_len]
    elif T_len < max_len:
        pad_amt = max_len - T_len
        # F.pad expects (left, right) for last dim — это корректно для [n_mels, T]
        spec = F.pad(spec, (0, pad_amt))

    return spec  # shape [n_mels, max_len]


@urban_router.post('/predict_urban/')
async def predict_urban(file: UploadFile = File(..., ), db: Session=Depends(get_db)):
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

            data = Urban(label=file.filename, predict=label)
            db.add(data)
            db.commit()
            db.refresh(data)

            return {'class':label}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))





