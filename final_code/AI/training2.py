from pathlib import Path
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt


# =========================
# USTAWIENIA
# =========================
DATA_DIR = Path(__file__).parent      # folder, w którym są katalogi True/ i False/
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3

PATIENCE = 10                          # early stopping
MIN_DELTA = 1e-4                      # minimalna poprawa val_loss
ETA_MIN = 1e-6                        # minimalny learning rate dla cosine annealing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Urządzenie:", device)


# =========================
# TRANSFORMACJE
# =========================
transform_train = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


# =========================
# WRAPPER NA RÓŻNE TRANSFORMY
# =========================
class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]   # PIL.Image, label
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# =========================
# DANE
# =========================
# Struktura:
# DATA_DIR/
#   False/
#   True/
base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)

print("Liczba wszystkich obrazów:", len(base_dataset))
print("Mapowanie klas:", base_dataset.class_to_idx)

# Podział train/val = 80/20
num_train = int(0.8 * len(base_dataset))
num_val = len(base_dataset) - num_train

train_subset, val_subset = random_split(
    base_dataset,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

# Wagi klas na nierównowagę
targets = np.array(base_dataset.targets)
train_indices = train_subset.indices
train_targets = targets[train_indices]

counts = np.bincount(train_targets, minlength=2)
print("Liczność klas w train:", counts)

class_weights = counts.max() / counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Wagi klas:", class_weights.tolist())

# Datasety z różnymi transformacjami
train_dataset = TransformSubset(train_subset, transform_train)
val_dataset = TransformSubset(val_subset, transform_val)

# Loadery
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODEL
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 64 -> 32

            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 32 -> 16

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),      # 16 -> 8
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)

# CosineAnnealingLR
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=EPOCHS,
    eta_min=ETA_MIN
)


# =========================
# HISTORIA TRENINGU
# =========================
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []
learning_rates = []


# =========================
# FUNKCJE TRENINGU / WALIDACJI
# =========================
def train_one_epoch(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"[Train] Epoka {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


def evaluate(epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    print(f"[Val]   Epoka {epoch}: loss={epoch_loss:.4f}, acc={epoch_acc:.4f}")
    return epoch_loss, epoch_acc


# =========================
# TRENING + EARLY STOPPING
# =========================
best_val_loss = float("inf")
best_val_acc = 0.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    current_lr = optimizer.param_groups[0]["lr"]
    learning_rates.append(current_lr)
    print(f"\n=== Epoka {epoch}/{EPOCHS} | LR = {current_lr:.6f} ===")

    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = evaluate(epoch)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # najlepsze accuracy tylko informacyjnie
    if val_acc > best_val_acc:
        best_val_acc = val_acc

    # early stopping na val_loss
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model2.pth")
        print(f"✅ Zapisuję najlepszy model (val_loss={best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"Brak poprawy val_loss: {epochs_without_improvement}/{PATIENCE}")

        if epochs_without_improvement >= PATIENCE:
            print("🛑 Early stopping - zatrzymuję trening.")
            break

    # scheduler aktualizujemy po epoce
    scheduler.step()

print("\nTrening zakończony.")
print(f"Najlepsze val_loss: {best_val_loss:.4f}")
print(f"Najlepsze val_acc:  {best_val_acc:.4f}")


# =========================
# WCZYTANIE NAJLEPSZEGO MODELU
# =========================
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()
print("Wczytano najlepszy model z best_model.pth")


# =========================
# WYKRESY
# =========================
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(15, 4))

# Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoka")
plt.ylabel("Loss")
plt.title("Wykres uczenia - Loss")
plt.legend()
plt.grid(True)

# Accuracy
plt.subplot(1, 3, 2)
plt.plot(epochs_range, train_accuracies, label="Train Accuracy")
plt.plot(epochs_range, val_accuracies, label="Val Accuracy")
plt.xlabel("Epoka")
plt.ylabel("Accuracy")
plt.title("Wykres uczenia - Accuracy")
plt.legend()
plt.grid(True)

# Learning Rate
plt.subplot(1, 3, 3)
plt.plot(epochs_range, learning_rates, label="Learning Rate")
plt.xlabel("Epoka")
plt.ylabel("LR")
plt.title("CosineAnnealingLR")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("training_curves2.png", dpi=200)
plt.show()