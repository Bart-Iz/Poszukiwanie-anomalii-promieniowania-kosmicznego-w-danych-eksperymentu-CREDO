import base64
import sys
from io import BytesIO
from pathlib import Path

import pandas as pd
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_paths import RESULTS_DIR

# =========================
# USTAWIENIA
# =========================
CSV_PATH = RESULTS_DIR / "13157" / "data" / "detections.csv"
device = "1276"
TRUE_DIR = Path(r"True")
FALSE_DIR = Path(r"False")
PROGRESS_FILE = Path("progress.txt")

COLUMN_NAME = "frame_content"


def load_progress():
    if PROGRESS_FILE.exists():
        try:
            return int(PROGRESS_FILE.read_text())
        except:
            return -1
    return -1


def save_progress(i):
    PROGRESS_FILE.write_text(str(i))


def decode_base64_to_np(b64_str: str):
    raw = base64.b64decode(b64_str)
    img = Image.open(BytesIO(raw)).convert("RGB")
    return np.array(img)


def count_labeled():
    TRUE_DIR.mkdir(exist_ok=True)
    FALSE_DIR.mkdir(exist_ok=True)
    return len(list(TRUE_DIR.glob("*.png"))), len(list(FALSE_DIR.glob("*.png")))


def iter_rows(start):
    for idx, chunk in enumerate(pd.read_csv(CSV_PATH, chunksize=1)):
        if idx < start:
            continue
        yield idx, chunk.iloc[0]


def main():
    last = load_progress()
    start = last + 1

    print(f"Startuję od: {start}")

    # INICJALIZACJA STAŁEGO OKNA MATPLOTLIB
    plt.ion()
    fig, ax = plt.subplots(figsize=(5, 5))
    img_display = None

    true_count, false_count = count_labeled()

    for idx, row in iter_rows(start):
        print("=" * 40)
        print(f"Wiersz {idx} | True: {true_count}, False: {false_count}")

        b64 = row[COLUMN_NAME]

        if pd.isna(b64) or not isinstance(b64, str) or b64.strip() == "":
            print("Brak obrazu – pomijam.")
            save_progress(idx)
            continue

        try:
            img_np = decode_base64_to_np(b64)
        except Exception as e:
            print(f"Błąd dekodowania: {e}")
            save_progress(idx)
            continue

        # ✔️ FILTR: pomijamy obrazy mniejsze niż 30x30
        if img_np.shape[0] < 30 or img_np.shape[1] < 30:
            print(f"Obraz zbyt mały ({img_np.shape[1]}x{img_np.shape[0]}) – pomijam.")
            save_progress(idx)
            continue

        # Aktualizacja obrazu bez zamykania okna
        if img_display is None:
            img_display = ax.imshow(img_np)
            ax.set_title(f"Wiersz {idx}")
            ax.axis("off")
            plt.show()
        else:
            img_display.set_data(img_np)
            ax.set_title(f"Wiersz {idx}")
            fig.canvas.draw()
            fig.canvas.flush_events()

        plt.pause(0.001)

        # INPUT użytkownika
        user_input = input("t=True, f=False, q=wyjście: ").strip().lower()

        if user_input == "q":
            print("Kończę pracę.")
            break

        if user_input not in ("t", "f"):
            print("Niepoprawna etykieta – pomijam.")
            continue

        # Zapis
        out_dir = TRUE_DIR if user_input == "t" else FALSE_DIR
        out_path = out_dir / f"{device}_row_{idx}.png"

        Image.fromarray(img_np).save(out_path)

        if user_input == "t":
            true_count += 1
        else:
            false_count += 1

        save_progress(idx)

    plt.ioff()
    plt.close("all")


if __name__ == "__main__":
    main()
