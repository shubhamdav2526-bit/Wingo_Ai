# app.py
import time
import math
import requests
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
from flask import Flask, jsonify, render_template_string

# =========================
# Model + Config
# =========================
GAME_TYPE = "WinGo_30S"
SEQ_LEN   = 30
DEVICE    = "cpu"

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

class TinyWinGoTransformer(nn.Module):
    def __init__(self, seq_len: int, d_model: int = 64, nhead: int = 4, nlayers: int = 2, dim_ff: int = 128, dropout: float = 0.1):
        super().__init__()
        self.embed = nn.Embedding(2, d_model)
        self.pos = PositionalEncoding(d_model, max_len=seq_len+1)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_ff, dropout=dropout, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 2)
        )
    def forward(self, x):
        emb = self.embed(x)
        emb = self.pos(emb)
        z = self.encoder(emb)
        z = self.norm(z[:, -1, :])
        return self.head(z)

# Load trained model
model = TinyWinGoTransformer(seq_len=SEQ_LEN)
try:
    checkpoint = torch.load("winfo_transformer.pt", map_location=DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    print("✅ Model loaded")
except Exception as e:
    print("⚠️ Model not loaded:", e)

model.to(DEVICE)
model.eval()

# =========================
# Helpers
# =========================
def fetch_history_once():
    ts = int(time.time() * 1000)
    url = f"https://draw.ar-lottery01.com/WinGo/{GAME_TYPE}/GetHistoryIssuePage.json?ts={ts}"
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
        return res.json().get("data", {}).get("list", [])
    except Exception as e:
        print("⚠️ API error:", e)
        return []

def to_small_big(game_list):
    out = []
    for entry in game_list:
        num = int(entry["number"])
        label = 0 if num < 5 else 1   # 0=small, 1=big
        out.append((entry["issueNumber"], label))
    return out

def load_history():
    data = fetch_history_once()
    hist = to_small_big(data)
    hist.sort(key=lambda x: x[0])  # chronological
    return hist

def predict_live():
    hist = load_history()
    if len(hist) < SEQ_LEN + 1:
        return {"error": "Not enough history"}
    labels = [x[1] for x in hist]
    ctx = labels[-SEQ_LEN:]
    ctx_tensor = torch.tensor([ctx], dtype=torch.long, device=DEVICE)
    with torch.no_grad():
        logits = model(ctx_tensor)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        pred = int(probs.argmax())
    return {
        "prediction": "BIG" if pred == 1 else "SMALL",
        "confidence": float(probs[pred]),
        "probs": {"small": float(probs[0]), "big": float(probs[1])},
        "time": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    }

# =========================
# Flask API + Dashboard
# =========================
app = Flask(__name__)

@app.route("/predict")
def api_predict():
    return jsonify(predict_live())

@app.route("/")
def dashboard():
    data = predict_live()
    if "error" in data:
        return f"<h2>Error: {data['error']}</h2>"

    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>WinGo Predictor</title>
        <meta http-equiv="refresh" content="10"> <!-- auto refresh every 10 sec -->
        <style>
            body { font-family: Arial, sans-serif; text-align: center; background: #f4f4f9; }
            .card { margin: 50px auto; padding: 20px; width: 300px; background: white;
                    border-radius: 15px; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
            .big { color: #e74c3c; font-weight: bold; }
            .small { color: #3498db; font-weight: bold; }
            .time { margin-top: 10px; font-size: 0.9em; color: gray; }
        </style>
    </head>
    <body>
        <div class="card">
            <h2>🎯 Next Prediction</h2>
            <h1 class="{{ data['prediction'].lower() }}">{{ data['prediction'] }}</h1>
            <p>Confidence: {{ (data['confidence']*100)|round(2) }}%</p>
            <p>Small: {{ (data['probs']['small']*100)|round(1) }}% | Big: {{ (data['probs']['big']*100)|round(1) }}%</p>
            <div class="time">Updated: {{ data['time'] }}</div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, data=data)

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)