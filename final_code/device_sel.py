import os
import time
from datetime import datetime
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt  # na razie nie używany, ale możesz zostawić

from config_paths import (
    LIST_DEVICES_TXT,
    RAW_DETECTIONS_CSV,
    RAW_PINGS_CSV,
    RESULTS_DIR,
)

# =========================
# USTAWIENIA GLOBALNE (ścieżki względem katalogu repozytorium)
# =========================
BASE_PATH = str(RESULTS_DIR)
DET_FILE = str(RAW_DETECTIONS_CSV)
PING_FILE = str(RAW_PINGS_CSV)
DEVICES_TXT = str(LIST_DEVICES_TXT)

CHUNK = 2_000_000   # wielkość chunka do czytania data.csv


# ==========================================
# Wybór urządzeń na podstawie warunków:
# - liczba detekcji z visible == True
# - minimalny czas działania (on_time)
# ==========================================
def device_selection(
    hours: int,
    det_file: str = DET_FILE,
    ping_file: str = PING_FILE,
    min_detections: int = 1000,
    chunk_size: int = CHUNK,
):
    """
    Wybiera urządzenia spełniające dwa warunki:
      - mają co najmniej min_detections detekcji z visible == True,
      - suma on_time >= hours (w godzinach).

    Implementacja jest strumieniowa (chunkami), więc działa również dla
    bardzo dużych plików, nie ładując wszystkiego do pamięci naraz.
    """
    # --- filtr na detekcje (visible == True), strumieniowo ---
    print("Liczenie detekcji per device_id (visible == True)...")
    det_counts = {}
    for chunk in pd.read_csv(
        det_file,
        usecols=["device_id", "visible"],
        chunksize=chunk_size,
    ):
        # filtrujemy tylko widoczne
        chunk = chunk[chunk["visible"] == True]
        if chunk.empty:
            continue

        vc = chunk["device_id"].value_counts()
        for dev_id, cnt in vc.items():
            det_counts[dev_id] = det_counts.get(dev_id, 0) + int(cnt)

    n_detections = pd.Series(det_counts, dtype="int64")
    valid_detections = n_detections[n_detections >= min_detections]

    print(f"Liczba device_id po warunku detekcji: {valid_detections.shape[0]}")

    # --- filtr na czas działania (on_time), strumieniowo ---
    print("Liczenie on_time per device_id...")
    ms_threshold = hours * 60 * 60 * 1000
    on_time_acc = {}

    for chunk in pd.read_csv(
        ping_file,
        usecols=["device_id", "on_time"],
        chunksize=chunk_size,
    ):
        grp = chunk.groupby("device_id")["on_time"].sum()
        for dev_id, s in grp.items():
            on_time_acc[dev_id] = on_time_acc.get(dev_id, 0) + int(s)

    on_time_sum = pd.Series(on_time_acc, dtype="int64")
    valid_time = on_time_sum[on_time_sum >= ms_threshold].index

    # --- przecięcie ---
    valid_devices = set(valid_detections.index) & set(valid_time)

    print(f"Liczba device_id po warunku detekcji + on_time: {len(valid_devices)}")

    # --- zapis ---
    with open(DEVICES_TXT, "w") as f:
        for device_id in sorted(valid_devices):
            f.write(f"{int(device_id)}\n")

# ==========================================
# Tworzenie folderów per urządzenie
# ==========================================
def make_folders(base_path: str, device: int):
    """
    Tworzy katalog:
      base_path/{device}/data
    jeśli nie istnieje.
    """
    device_path = os.path.join(base_path, str(device))
    os.makedirs(device_path, exist_ok=True)
    os.makedirs(os.path.join(device_path, "data"), exist_ok=True)


# ==========================================
# Jedno przejście po dużym pliku data.csv
# i zapis per urządzenie
# ==========================================
def process_data_files(
    DEVICES: List[int],
    det_file: str = DET_FILE,
    ping_file: str = PING_FILE,
    base_path: str = BASE_PATH,
    CHUNK: int = CHUNK,
):
    """
    Jedno przejście chunkami przez duży plik detekcji:
      - w każdym chunku filtrujemy tylko interesujące device_id,
      - rozbijamy chunk na grupy po device_id,
      - dopisujemy do odpowiednich plików:
           base_path/{dev_id}/data/detections.csv

    Potem:
      - wczytujemy pings.csv JEDEN raz,
      - filtrujemy po DEVICES,
      - rozbijamy po device_id i zapisujemy:
           base_path/{dev_id}/data/pings.csv
    """

    # dla szybkiego sprawdzania przynależności O(1)
    DEV_SET = set(DEVICES)

    # --- 1) Detections: jedno przejście chunkami ---
    print("Przetwarzanie detekcji (jedno przejście chunkami)...")
    for chunk in pd.read_csv(det_file, chunksize=CHUNK):
        # filtrujemy tylko urządzenia z listy
        if "device_id" not in chunk.columns:
            raise ValueError("Brak kolumny 'device_id' w pliku data.csv")

        chunk = chunk[chunk["device_id"].isin(DEV_SET)]
        if chunk.empty:
            continue

        # rozbij na urządzenia i dopisuj do ich plików
        for dev_id, g in chunk.groupby("device_id", sort=False):
            dev_dir = os.path.join(base_path, str(dev_id), "data")
            os.makedirs(dev_dir, exist_ok=True)

            out_path = os.path.join(dev_dir, "detections.csv")
            write_header = not os.path.exists(out_path)

            g.to_csv(out_path, mode="a", header=write_header, index=False)

    # --- 2) Pings: jedno wczytanie, potem groupby ---
    print("Przetwarzanie pingów (jedno wczytanie)...")
    pings_all = pd.read_csv(
        ping_file,
        dtype={"device_id": "int32", "timestamp": "int64", "on_time": "int64"},
    )
    pings_all = pings_all[pings_all["device_id"].isin(DEV_SET)]

    for dev_id, g in pings_all.groupby("device_id", sort=False):
        dev_dir = os.path.join(base_path, str(dev_id), "data")
        os.makedirs(dev_dir, exist_ok=True)
        out_path = os.path.join(dev_dir, "pings.csv")
        g.to_csv(out_path, index=False)

    print("Zakończono rozbijanie danych na urządzenia.")


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    start = time.time()
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    os.makedirs(BASE_PATH, exist_ok=True)
    # 1) Jeśli nie ma listy urządzeń -> wyznacz ją
    if not os.path.exists(DEVICES_TXT):
        device_selection(72)

    # 2) Wczytanie listy urządzeń
    with open(DEVICES_TXT, "r", encoding="utf-8") as f:
        devices = [int(line.strip()) for line in f if line.strip()]

    print(f"Urządzeń do przetworzenia: {len(devices)}")

    # 3) Tworzenie folderów per urządzenie (szybka pętla)
    for device in tqdm(devices, desc="tworzenie folderów"):
        make_folders(BASE_PATH, device)

    # 4) Jedno przejście przez pliki i zapis per urządzenie
    process_data_files(devices)

    end = time.time()
    duration = end - start
    print("Czas [s]:", duration)
