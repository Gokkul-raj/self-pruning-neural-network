import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from src.model import Net
from src.utils import sparsity_loss, calculate_sparsity

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

# Model
model = Net().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Hyperparameters
lambda_val = 1e-3
epochs = 10

# Training
for epoch in range(epochs):
    model.train()
    total_loss = 0

    for images, labels in trainloader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        cls_loss = criterion(outputs, labels)
        sp_loss = sparsity_loss(model)

        loss = cls_loss + lambda_val * sp_loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Evaluation
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)

        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
sparsity = calculate_sparsity(model)

print(f"\nFinal Accuracy: {accuracy:.2f}%")
print(f"Sparsity: {sparsity:.2f}%")

# Save model
torch.save(model.state_dict(), "model.pth")

# Save results
with open("results.txt", "a") as f:
    f.write(f"Lambda: {lambda_val}, Accuracy: {accuracy:.2f}, Sparsity: {sparsity:.2f}\n")

# Plot gate distribution
gates_all = []

for module in model.modules():
    if hasattr(module, 'gate_scores'):
        gates = torch.sigmoid(module.gate_scores).detach().cpu().numpy()
        gates_all.extend(gates.flatten())

plt.hist(gates_all, bins=50)
plt.title("Gate Distribution")
plt.xlabel("Gate Value")
plt.ylabel("Frequency")
plt.show()