from pathlib import Path
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, confusion_matrix


# =========================
# USTAWIENIA
# =========================
DATA_DIR = Path(__file__).parent      # folder z True/ i False/
IMG_SIZE = 64
BATCH_SIZE = 32

MODEL_PATH = Path(__file__).resolve().parent / "best_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Urządzenie:", device)


# =========================
# TRANSFORMACJE (tylko walidacja)
# =========================
transform_val = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


class TransformSubset(Dataset):
    """Wrapper: pozwala mieć transformy mimo że base_dataset ma transform=None."""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        img, label = self.subset[idx]  # PIL.Image
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# =========================
# DANE (ten sam split 80/20)
# =========================
base_dataset = datasets.ImageFolder(root=DATA_DIR, transform=None)
print("Liczba wszystkich obrazów:", len(base_dataset))
print("Mapowanie klas:", base_dataset.class_to_idx)  # np. {'False': 0, 'True': 1}

num_train = int(0.8 * len(base_dataset))
num_val = len(base_dataset) - num_train

_, val_subset = random_split(
    base_dataset,
    [num_train, num_val],
    generator=torch.Generator().manual_seed(42)
)

print(f"Liczba obrazów w walidacji: {len(val_subset)}")

val_dataset = TransformSubset(val_subset, transform_val)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


# =========================
# MODEL – identyczna architektura
# =========================
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 -> 32

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32 -> 16

            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 16 -> 8
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


if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Nie znaleziono modelu: {MODEL_PATH.resolve()}")

model = SimpleCNN().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()
print(f"Wczytano model: {MODEL_PATH}")


# =========================
# PREDYKCJE na walidacji
# =========================
all_labels = []
all_probs = []  # P(True) – zakładamy klasa 1 = True

with torch.no_grad():
    for images, labels in val_loader:
        images = images.to(device)
        outputs = model(images)
        probs = torch.softmax(outputs, dim=1)[:, 1]

        all_labels.append(labels.numpy())
        all_probs.append(probs.cpu().numpy())

y_true = np.concatenate(all_labels)
y_score = np.concatenate(all_probs)

print("Zebrano predykcje dla:", y_true.shape[0], "obrazów")


# =========================
# ROC curve (podpisy po polsku)
# =========================
fpr, tpr, thresholds = roc_curve(y_true, y_score)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "--", label="Linia losowego zgadywania")

# progi, które chcesz zaznaczyć
thr_list = [0.6]

for thr in thr_list:
    # znajdź indeks najbliższego progu
    idx = np.argmin(np.abs(thresholds - thr))

    plt.scatter(fpr[idx], tpr[idx], color="red")

plt.xlabel("Odsetek fałszywych alarmów (FPR)")
plt.ylabel("Odsetek trafień (TPR)")
plt.title("Krzywa ROC – walidacja")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve.png", dpi=150)
plt.close()

print(f"Zapisano: roc_curve.png | AUC (walidacja): {roc_auc:.4f}")


# =========================
# Macierze pomyłek dla progów (podpisy po polsku)
# =========================
thr_list = [0.5, 0.6, 0.7, 0.8]
n = len(thr_list)

rows = 2
cols = int(np.ceil(n / rows))

fig, axes = plt.subplots(
    rows,
    cols,
    figsize=(4.8 * cols, 4.2 * rows)
)
axes = axes.flatten()

fig.subplots_adjust(
    left=0.08,
    right=0.88,
    bottom=0.08,
    top=0.90,
    wspace=0.45,
    hspace=0.55
)

im = None
for k, thr in enumerate(thr_list):
    ax = axes[k]

    y_pred = (y_score >= thr).astype(int)
    cm = confusion_matrix(y_true, y_pred)

    # zabezpieczenie przed dzieleniem przez 0 (gdy brak klasy w y_true)
    denom = cm.sum(axis=1, keepdims=True).astype(float)
    denom[denom == 0] = 1.0
    cm_norm = cm.astype(float) / denom

    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)

    ax.set_title(f"Próg prawdopodobieństwa = {thr:.2f}")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["False", "True"])
    ax.set_yticklabels(["False", "True"])
    ax.set_xlabel("Predykcja")
    ax.set_ylabel("Prawdziwa klasa")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm_norm[i, j]:.2f}", ha="center", va="center", color="black")

for k in range(n, len(axes)):
    axes[k].axis("off")

if im is not None:
    cbar = fig.colorbar(im, ax=axes[:n], fraction=0.03, pad=0.04)
    cbar.set_label("Udział (znormalizowane)")

plt.savefig("confusion_matrices_thresholds.png", dpi=150)
plt.close()

print("Zapisano: confusion_matrices_thresholds.png")