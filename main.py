import os
import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)

import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# ============================================================
# Dataset
# ============================================================
class ORIGADataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.data = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = os.path.basename(self.data.iloc[idx]["ImageName"])
        label = int(self.data.iloc[idx]["glaucoma"])

        img_path = os.path.join(self.img_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)


# ============================================================
# Transforms
# ============================================================
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])


# ============================================================
# Dataset / Loader
# ============================================================
train_dataset = ORIGADataset(
    "./ORIGA Retinal Fundus Image Dataset/ORIGA_train.csv",
    "./ORIGA Retinal Fundus Image Dataset/ORIGA/train",
    transform=train_transform
)

test_dataset = ORIGADataset(
    "./ORIGA Retinal Fundus Image Dataset/ORIGA_test.csv",
    "./ORIGA Retinal Fundus Image Dataset/ORIGA/test",
    transform=test_transform
)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=16, shuffle=False)


# ============================================================
# Model
# ============================================================
device = "cuda" if torch.cuda.is_available() else "cpu"

model = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# ============================================================
# Training & Evaluation
# ============================================================
def train_one_epoch(model, loader):
    model.train()
    total_loss = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    preds, trues, probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            outputs = model(imgs)

            softmax = torch.softmax(outputs, dim=1)[:, 1]

            preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            trues.extend(labels.numpy())
            probs.extend(softmax.cpu().numpy())

    acc       = accuracy_score(trues, preds)
    f1        = f1_score(trues, preds)
    auc       = roc_auc_score(trues, probs)
    precision = precision_score(trues, preds)
    recall    = recall_score(trues, preds)
    cm        = confusion_matrix(trues, preds)

    return acc, f1, auc, precision, recall, cm, preds, trues


# ============================================================
# History Logging
# ============================================================
history = {"loss": [], "acc": [], "f1": [], "auc": [], "precision": [], "recall": []}

num_epochs = 20

for epoch in range(num_epochs):
    train_loss = train_one_epoch(model, train_loader)
    acc, f1, auc, precision, recall, cm, preds, trues = evaluate(model, test_loader)

    history["loss"].append(train_loss)
    history["acc"].append(acc)
    history["f1"].append(f1)
    history["auc"].append(auc)
    history["precision"].append(precision)
    history["recall"].append(recall)

    print(f"[Epoch {epoch+1}] Loss: {train_loss:.4f} | Acc: {acc:.4f} | "
          f"F1: {f1:.4f} | AUC: {auc:.4f} | Precision: {precision:.4f} | Recall: {recall:.4f}")


# ============================================================
# Confusion Matrix Plot
# ============================================================
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Normal", "Glaucoma"],
            yticklabels=["Normal", "Glaucoma"])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()


# ============================================================
# Metric Graphs
# ============================================================
plt.figure(figsize=(16, 10))

plt.subplot(3, 2, 1)
plt.plot(history["loss"], marker="o")
plt.title("Training Loss")
plt.grid(True)

plt.subplot(3, 2, 2)
plt.plot(history["acc"], marker="o")
plt.title("Accuracy")
plt.grid(True)

plt.subplot(3, 2, 3)
plt.plot(history["f1"], marker="o")
plt.title("F1 Score")
plt.grid(True)

plt.subplot(3, 2, 4)
plt.plot(history["auc"], marker="o")
plt.title("AUC")
plt.grid(True)

plt.subplot(3, 2, 5)
plt.plot(history["precision"], marker="o")
plt.title("Precision")
plt.grid(True)

plt.subplot(3, 2, 6)
plt.plot(history["recall"], marker="o")
plt.title("Recall")
plt.grid(True)

plt.tight_layout()
plt.show()


# ============================================================
# Grad-CAM
# ============================================================
def generate_gradcam(model, image_tensor):
    model.eval()
    image_tensor = image_tensor.unsqueeze(0).to(device)

    features = None
    gradients = None

    # Forward Hook
    def forward_hook(module, input, output):
        nonlocal features
        features = output

    # Backward Hook
    def backward_hook(module, grad_input, grad_output):
        nonlocal gradients
        gradients = grad_output[0]

    last_conv = model.features[-1]

    last_conv.register_forward_hook(forward_hook)
    last_conv.register_full_backward_hook(backward_hook)

    # Forward
    output = model(image_tensor)
    pred_class = output.argmax(dim=1)
    output[0, pred_class].backward()

    weights = gradients.mean(dim=(2, 3), keepdim=True)

    cam = (weights * features).sum(dim=1).squeeze()
    cam = cam.detach().cpu().numpy()

    cam = np.maximum(cam, 0)
    cam = cam / cam.max()

    return cam


# ============================================================
# Grad-CAM Visualization
# ============================================================
sample_img, _ = test_dataset[0]
cam = generate_gradcam(model, sample_img)

orig = sample_img.permute(1, 2, 0).numpy()

heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
heatmap = cv2.resize(heatmap, (orig.shape[1], orig.shape[0]))

overlay = 0.5 * heatmap + 0.5 * (orig * 255)

plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Original")
plt.imshow(orig)

plt.subplot(1, 2, 2)
plt.title("Grad-CAM")
plt.imshow(np.uint8(overlay))
plt.show()