from pathlib import Path
from all_pytorch_projects.database.schema import AgNews
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from fastapi import APIRouter, Depends

from all_pytorch_projects.database.models import Text_ag_news
from all_pytorch_projects.database.db import SessionLocal
from sqlalchemy.orm import Session

async def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


ag_news = APIRouter(prefix='/ag_news', tags=['Ag News'])

BASE_DIR = Path(__file__).resolve().parent.parent.parent
vocab_path = BASE_DIR / 'pytorch_model' / 'vocab_agnews.pth'
model_path = BASE_DIR / 'pytorch_model' / 'model_news.pth'
print(BASE_DIR)
print(vocab_path)
class CheckText(nn.Module):
  def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, output_dim=4):
    super().__init__()
    self.emb = nn.Embedding(vocab_size, embed_dim)
    self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
    self.fc = nn.Linear(hidden_dim, output_dim)

  def forward(self, x):
    x = self.emb(x)
    _, (hidden, _) = self.lstm(x)
    output = self.fc(hidden[-1])
    return output

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab = torch.load(vocab_path)
model = CheckText(len(vocab))
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()

labels = {
  0:'World',
  1:'Sports',
  2:'Business',
  3:'Sci/Tech'
}
tokenizer = get_tokenizer('basic_english')

def change_text_to_int(text):
    idx = list(tokenizer(text))
    idx_to_int = vocab(idx)
    return idx_to_int

@ag_news.post('/text_predict')
async def text_pred(text: AgNews, db:Session=Depends(get_db)):
    text_to_int = torch.tensor(change_text_to_int(text.text)).unsqueeze(0).to(device)
    pred = model(text_to_int)
    result = torch.argmax(pred, dim=1).item()

    data = Text_ag_news(label=text.text, predict=result)
    db.add(data)
    db.commit()
    db.refresh(data)

    return {'predict': labels[result]}

