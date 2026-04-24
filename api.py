from fastapi import FastAPI
import torch
from src.model import Net
from src.utils import calculate_sparsity

app = FastAPI()

model = Net()
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()


@app.get("/")
def home():
    return {"message": "Self-Pruning Neural Network API"}


@app.get("/sparsity")
def get_sparsity():
    sparsity = calculate_sparsity(model)
    return {"sparsity_percentage": round(sparsity, 2)}


@app.get("/predict")
def predict():
    x = torch.randn(1, 3, 32, 32)
    output = model(x)
    pred = torch.argmax(output, dim=1).item()
    return {"predicted_class": pred}