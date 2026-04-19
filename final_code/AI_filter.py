import base64
from io import BytesIO
from pathlib import Path
import os
import numpy as np

import pandas as pd
from PIL import Image
import torch
from torch import nn
from torchvision import transforms
from concurrent.futures import ProcessPoolExecutor, as_completed

from config_paths import AI_MODEL_PATH, RESULTS_DIR

# =========================
# ŚCIEŻKI / USTAWIENIA
# =========================
BASE_PATH = RESULTS_DIR
FILE = Path("data/detections_filtered.csv")
MODEL_PATH = AI_MODEL_PATH

IMG_SIZE = 64
THRESHOLD = 0.6  # próg P(True) – powyżej = zostaje, poniżej = wyrzucamy
MAX_WORKERS = max(1, int(0.9 * os.cpu_count()))
# większe batch'e na GPU, mniejsze na CPU
BATCH_SIZE_AI = 256 if torch.cuda.is_available() else 64

# =========================
# ARCHITEKTURA MUSI BYĆ IDENTYCZNA JAK W TRAININGU
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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: SimpleCNN | None = None


def get_model() -> SimpleCNN:
    """Ładowanie leniwe — działa też w procesach potomnych (Windows spawn)."""
    global _model
    if _model is None:
        if not MODEL_PATH.is_file():
            raise FileNotFoundError(
                f"Brak modelu sieci: {MODEL_PATH}\n"
                "Umieść plik best_model.pth w katalogu final_code/AI/ (patrz README)."
            )
        _model = SimpleCNN().to(device)
        _model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        _model.eval()
    return _model


# transformacja jak przy trenowaniu
img_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def b64_to_tensor(b64_str: str) -> torch.Tensor | None:
    """Base64 -> tensor [1,1,H,W]; zwraca None jak coś pójdzie nie tak."""
    try:
        raw = base64.b64decode(b64_str)
        img = Image.open(BytesIO(raw)).convert("RGB")
        t = img_transform(img)          # [1, H, W]
        return t.unsqueeze(0)           # [1, 1, H, W]
    except Exception:
        return None


def filter_detections_for_device(dev_dir: Path) -> tuple[int, int]:
    """
    Filtruje detections.csv dla jednego urządzenia.
    Zwraca (n_rows, n_removed).
    """
    model = get_model()
    det_path = dev_dir / FILE
    if not det_path.exists():
        print(f"[{dev_dir.name}] Brak pliku {FILE}, pomijam.")
        return 0, 0

    bad_lines_count = 0

    def _bad_line_handler(bad_line):
        """Każdy błędny wiersz zwiększa licznik i jest wyrzucany."""
        nonlocal bad_lines_count
        bad_lines_count += 1
        return None  # None = wiersz odrzucony

    df = pd.read_csv(
        det_path,
        on_bad_lines=_bad_line_handler,
        engine="python"
    )

    if bad_lines_count > 0:
        print(f"[{dev_dir.name}] Usunięto błędnych wierszy: {bad_lines_count}")

    if "frame_content" not in df.columns:
        print(f"[{dev_dir.name}] Brak kolumny 'frame_content', pomijam.")
        return len(df), 0

    original_len = len(df)
    if original_len == 0:
        print(f"[{dev_dir.name}] Pusty plik, pomijam.")
        return 0, 0

    # przygotuj dane do klasyfikacji
    keep_mask = np.ones(original_len, dtype=bool)  # True = zostaje

    batch_tensors = []
    batch_indices = []

    for idx, b64 in enumerate(df["frame_content"]):
        if pd.isna(b64) or not isinstance(b64, str) or b64.strip() == "":
            # brak obrazu – nie ruszamy tego wiersza (zostaje)
            continue

        t = b64_to_tensor(b64)
        if t is None:
            # nie udało się zdekodować – na wszelki wypadek zostawiamy
            continue

        batch_tensors.append(t)
        batch_indices.append(idx)

        # gdy uzbiera się batch – przepchaj przez model
        if len(batch_tensors) == BATCH_SIZE_AI:
            batch = torch.cat(batch_tensors, dim=0).to(device)  # [B,1,64,64]
            with torch.no_grad():
                logits = model(batch)
                probs = torch.softmax(logits, dim=1)[:, 1]  # P(True)

            probs = probs.cpu().numpy()
            for bi, p in zip(batch_indices, probs):
                if p < THRESHOLD:
                    keep_mask[bi] = False

            batch_tensors.clear()
            batch_indices.clear()

    # resztka batcha
    if batch_tensors:
        batch = torch.cat(batch_tensors, dim=0).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.softmax(logits, dim=1)[:, 1]
        probs = probs.cpu().numpy()
        for bi, p in zip(batch_indices, probs):
            if p < THRESHOLD:
                keep_mask[bi] = False

    # filtruj dataframe
    filtered_df = df[keep_mask].reset_index(drop=True)
    removed = original_len - len(filtered_df)

    # zapisz nowy plik
    out_path = dev_dir / FILE
    filtered_df.to_csv(out_path, index=False)
    print(f"[{dev_dir.name}] oryginalnie: {original_len}, usunięto: {removed}, zostało: {len(filtered_df)}")

    return original_len, removed


def main():
    try:
        get_model()
    except FileNotFoundError as e:
        print(e)
        raise SystemExit(1)

    total_rows = 0
    total_removed = 0

    base_path = BASE_PATH
    dev_dirs = [p for p in base_path.iterdir() if p.is_dir()]

    print(f"Znaleziono {len(dev_dirs)} urządzeń.")

    # Na GPU trzymamy jeden proces z dużymi batch'ami (wiele procesów zjada pamięć GPU).
    # Na CPU, jeśli jest wiele urządzeń, korzystamy z wielu procesów.
    use_parallel = device.type == "cpu" and len(dev_dirs) > 1

    if use_parallel:
        print(f"Uruchamiam równolegle na CPU ({MAX_WORKERS} procesów).")
        with ProcessPoolExecutor(max_workers=MAX_WORKERS) as ex:
            futures = {ex.submit(filter_detections_for_device, dev_dir): dev_dir for dev_dir in dev_dirs}

            for fut in as_completed(futures):
                dev = futures[fut]
                try:
                    n_all, n_removed = fut.result()
                except Exception as e:
                    print(f"[{dev.name}] Błąd podczas filtrowania: {e}")
                    continue

                total_rows += n_all
                total_removed += n_removed
    else:
        if device.type == "cuda":
            print("Uruchamiam sekwencyjnie z dużymi batch'ami na GPU.")
        else:
            print("Mało urządzeń – uruchamiam sekwencyjnie na CPU (bez narzutu multiprocessing).")

        for dev_dir in dev_dirs:
            try:
                n_all, n_removed = filter_detections_for_device(dev_dir)
            except Exception as e:
                print(f"[{dev_dir.name}] Błąd podczas filtrowania: {e}")
                continue

            total_rows += n_all
            total_removed += n_removed

    if total_rows > 0:
        percent_removed = 100.0 * total_removed / total_rows
    else:
        percent_removed = 0.0

    print("\n=== PODSUMOWANIE FILTROWANIA AI ===")
    print(f"Łącznie wierszy: {total_rows}")
    print(f"Usuniętych (False wg modelu): {total_removed} ({percent_removed:.2f}%)")

if __name__ == "__main__":
    main()
