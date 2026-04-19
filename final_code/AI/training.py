from pathlib import Path
import numpy as np

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


# =========================
# USTAWIENIA
# =========================
DATA_DIR = Path(__file__).parent      # w tym samym folderze co True/ i False/
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 200
LR = 1e-3

PATIENCE = 5
MIN_DELTA = 1e-5

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


class TransformSubset(Dataset):
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# =========================
# DANE
# =========================
base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
print("Liczba wszystkich obrazów:", len(base_dataset))
print("Mapowanie klas:", base_dataset.class_to_idx)

num_train = int(0.8 * len(base_dataset))
num_val = len(base_dataset) - num_train
train_subset, val_subset = random_split(
    base_dataset,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)

print(f"Train: {len(train_subset)}, Val: {len(val_subset)}")

targets = np.array(base_dataset.targets)
train_indices = train_subset.indices
train_targets = targets[train_indices]

counts = np.bincount(train_targets, minlength=2)
print("Liczność klas w train:", counts)

class_weights = counts.max() / counts
class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
print("Wagi klas:", class_weights.tolist())

train_dataset = TransformSubset(train_subset, transform_train)
val_dataset = TransformSubset(val_subset, transform_val)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODEL – prosty CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


model = SimpleCNN().to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=LR)


# =========================
# HISTORIA TRENINGU
# =========================
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []


# =========================
# PĘTLA TRENINGOWA
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
# EARLY STOPPING
# =========================
best_val_loss = float("inf")
best_acc = 0.0
epochs_without_improvement = 0

for epoch in range(1, EPOCHS + 1):
    train_loss, train_acc = train_one_epoch(epoch)
    val_loss, val_acc = evaluate(epoch)

    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)

    # zapis najlepszego accuracy tylko informacyjnie
    if val_acc > best_acc:
        best_acc = val_acc

    # early stopping działa na val_loss
    if val_loss < best_val_loss - MIN_DELTA:
        best_val_loss = val_loss
        epochs_without_improvement = 0
        torch.save(model.state_dict(), "best_model.pth")
        print(f"✅ Zapisuję najlepszy model (val_loss={best_val_loss:.4f})")
    else:
        epochs_without_improvement += 1
        print(f"Brak poprawy val_loss przez {epochs_without_improvement}/{PATIENCE} epok.")

        if epochs_without_improvement >= PATIENCE:
            print("🛑 Early stopping - zatrzymuję trening.")
            break

print("Trening zakończony.")
print("Najlepsze val accuracy:", best_acc)
print("Najlepsze val loss:", best_val_loss)


# =========================
# WYKRES UCZENIA
# =========================
epochs_range = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Strata treningowa")
plt.plot(epochs_range, val_losses, label="Strata walidacyjna")
plt.xlabel("Epoka")
plt.ylabel("Wartość funkcji straty")
plt.title("Przebieg uczenia – strata")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(epochs_range, train_accuracies, label="Dokładność treningowa")
plt.plot(epochs_range, val_accuracies, label="Dokładność walidacyjna")
plt.xlabel("Epoka")
plt.ylabel("Dokładność")
plt.title("Przebieg uczenia – dokładność")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("wykres_uczenia.png", dpi=200)
plt.show()