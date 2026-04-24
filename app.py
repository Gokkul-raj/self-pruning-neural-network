import streamlit as st
import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from src.model import Net
from src.utils import calculate_sparsity

# Page config
st.set_page_config(page_title="Self-Pruning NN Dashboard", layout="wide")

# Title
st.title("🧠 Self-Pruning Neural Network Dashboard")

# Load model
@st.cache_resource
def load_model():
    model = Net()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# CIFAR-10 labels
classes = ['airplane', 'automobile', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck']

# Load dataset
@st.cache_data
def load_data():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform
    )
    return dataset

dataset = load_data()

# Layout
col1, col2 = st.columns(2)

# LEFT SIDE → Image + Prediction
with col1:
    st.subheader("🔍 Image Prediction")

    if st.button("Generate Random Image"):
        idx = np.random.randint(0, len(dataset))
        image, label = dataset[idx]

        st.image(np.transpose(image.numpy(), (1, 2, 0)),
                 caption=f"Actual: {classes[label]}")

        with torch.no_grad():
            output = model(image.unsqueeze(0))
            pred = torch.argmax(output, dim=1).item()

        st.success(f"Predicted: {classes[pred]}")

# RIGHT SIDE → Metrics
with col2:
    st.subheader("📊 Model Metrics")

    sparsity = calculate_sparsity(model)
    st.metric("Sparsity (%)", f"{sparsity:.2f}")

    st.info("Model achieved high sparsity using L1 regularization on gates.")

# FULL WIDTH → Graph
st.subheader("📈 Gate Distribution")

gates_all = []

for module in model.modules():
    if hasattr(module, 'gate_scores'):
        gates = torch.sigmoid(module.gate_scores).detach().numpy()
        gates_all.extend(gates.flatten())

fig, ax = plt.subplots()
ax.hist(gates_all, bins=50)
ax.set_title("Gate Value Distribution")
ax.set_xlabel("Gate Value")
ax.set_ylabel("Frequency")

st.pyplot(fig)