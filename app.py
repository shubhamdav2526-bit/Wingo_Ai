import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, render_template
import random, csv, os
from datetime import datetime
import matplotlib.pyplot as plt

app = Flask(__name__)

# ==========================
#  Transformer Model
# ==========================
class WinfoTransformer(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes, num_heads, num_layers):
        super(WinfoTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        x = self.embedding(x).unsqueeze(1)
        x = self.transformer(x)
        x = x.mean(dim=1)
        return self.fc(x)

# ==========================
#  Globals
# ==========================
INPUT_SIZE = 1
HIDDEN_SIZE = 64
NUM_CLASSES = 2
NUM_HEADS = 2
NUM_LAYERS = 2
MODEL_PATH = "winfo_model.pth"
CSV_FILE = "predictions.csv"

model = WinfoTransformer(INPUT_SIZE, HIDDEN_SIZE, NUM_CLASSES, NUM_HEADS, NUM_LAYERS)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Load model if exists
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH))

# Ensure CSV exists
if not os.path.exists(CSV_FILE):
    with open(CSV_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time", "prediction", "confidence", "actual", "win"])

latest_prediction = {"prediction": None, "confidence": None, "time": None}

# ==========================
#  Training & Prediction
# ==========================
def simulate_data():
    return torch.tensor([[random.randint(0, 1)]], dtype=torch.float32)

def predict_and_log():
    global latest_prediction
    model.eval()

    with torch.no_grad():
        data = simulate_data()
        output = model(data)
        probs = torch.softmax(output, dim=1).numpy()[0]
        pred = int(probs.argmax())
        conf = float(probs.max())

    # Simulate actual result (like casino output)
    actual = random.randint(0, 1)
    win = int(pred == actual)

    # Save in CSV
    with open(CSV_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now().isoformat(), pred, conf, actual, win])

    # Save model progress
    torch.save(model.state_dict(), MODEL_PATH)

    latest_prediction = {"prediction": pred, "confidence": conf, "time": datetime.now().isoformat(), "actual": actual, "win": win}
    return latest_prediction

# ==========================
#  Routes
# ==========================
@app.route("/latest")
def get_latest():
    return jsonify(latest_prediction)

@app.route("/")
def index():
    # Load last 20 results
    history = []
    try:
        with open(CSV_FILE, "r") as f:
            reader = list(csv.DictReader(f))
            history = reader[-20:]
    except:
        history = []

    # Chart for wins/losses
    wins = [int(row["win"]) for row in history] if history else []
    plt.figure(figsize=(5,3))
    plt.plot(wins, marker="o", linestyle="-")
    plt.title("Last Wins (1=Win, 0=Loss)")
    plt.xlabel("Game")
    plt.ylabel("Result")
    chart_path = "static/chart.png"
    plt.savefig(chart_path)
    plt.close()

    return render_template("index.html", latest=latest_prediction, history=history, chart=chart_path)

# ==========================
#  Background Loop
# ==========================
import threading, time
def loop_predictions():
    while True:
        predict_and_log()
        time.sleep(5)  # every 5 sec

threading.Thread(target=loop_predictions, daemon=True).start()

# ==========================
#  Run App
# ==========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)