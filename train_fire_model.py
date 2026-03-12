"""
FIRE CLASSIFIER TRAINING SCRIPT
================================
Fine-tunes MobileNetV3-Small (designed for edge/low-power devices)
into a binary fire/no-fire classifier.

Dataset: Uses images from the local 'dataset/' folder.
  - Put fire images in:     dataset/fire/
  - Put non-fire images in: dataset/no_fire/

If you want a bigger dataset, download the ForestFire dataset from Kaggle:
  https://www.kaggle.com/datasets/kutaykutlu/forest-fire
Then unzip into dataset/fire/ and dataset/no_fire/ accordingly.

The script will save the trained weights to: models/fire_classifier.pth
"""

import os
import torch
import torch.nn as nn
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import time

# ---- CONFIG ----
DATA_DIR    = "./dataset"
MODEL_DIR   = "./models"
MODEL_OUT   = os.path.join(MODEL_DIR, "fire_classifier.pth")
EPOCHS      = 10
BATCH_SIZE  = 16
LR          = 1e-4
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "fire"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "no_fire"), exist_ok=True)

print(f"🚀 [TRAINING] Using device: {DEVICE}")

# ---- TRANSFORMS ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- DATASET ----
dataset = datasets.ImageFolder(root=DATA_DIR, transform=transform)
print(f"📂 [TRAINING] Found {len(dataset)} images. Classes: {dataset.classes}")

if len(dataset) == 0:
    print("\n❌ No images found!")
    print("   Add fire images to:     wildfire_prototype/dataset/fire/")
    print("   Add non-fire images to: wildfire_prototype/dataset/no_fire/")
    print("   Download from: https://www.kaggle.com/datasets/kutaykutlu/forest-fire")
    exit(1)

# 80/20 train-val split
train_size = int(0.8 * len(dataset))
val_size   = len(dataset) - train_size
train_ds, val_ds = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ---- MODEL ----
weights = models.MobileNet_V3_Small_Weights.DEFAULT
model = models.mobilenet_v3_small(weights=weights)

# Replace the final classification layer with binary (fire / no_fire)
in_features = model.classifier[-1].in_features
model.classifier[-1] = nn.Linear(in_features, 2)
model = model.to(DEVICE)

# Freeze backbone, only train the new head first (faster + avoids overfitting on small dataset)
for param in model.features.parameters():
    param.requires_grad = False
for param in model.classifier.parameters():
    param.requires_grad = True

# ---- TRAIN ----
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

best_val_acc = 0.0
print(f"\n🔥 [TRAINING] Starting {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    t0 = time.time()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total   += labels.size(0)

    train_loss = running_loss / total
    train_acc  = correct / total

    # Validation
    model.eval()
    val_correct, val_total = 0, 0
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            val_correct += (preds == labels).sum().item()
            val_total   += labels.size(0)

    val_acc = val_correct / val_total if val_total > 0 else 0
    elapsed = time.time() - t0
    print(f"  Epoch [{epoch+1}/{EPOCHS}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2%} | Val Acc: {val_acc:.2%} | {elapsed:.1f}s")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'classes': dataset.classes,
            'val_acc': val_acc
        }, MODEL_OUT)
        print(f"  ✅ Saved best model → {MODEL_OUT} (Val Acc: {val_acc:.2%})")

print(f"\n🎯 [TRAINING COMPLETE] Best validation accuracy: {best_val_acc:.2%}")
print(f"   Model saved to: {MODEL_OUT}")
