# ============================================
# Importy i ustawienia równoległości / narzędzia
# ============================================
from config_paths import REPO_ROOT, RESULTS_DIR

import os
import math
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import base64
from PIL import Image
import io
import time
from tqdm import tqdm
from datetime import datetime
import itertools
from typing import List, Tuple, Dict
from concurrent.futures import ProcessPoolExecutor, as_completed


# ============================================================
# Usuwanie katalogu: najpierw rmtree, jak się nie da -> rename
# ============================================================
def safe_remove_or_rename(folder_path: str):
    """
    Próbuje usunąć folder. Jeśli się nie uda (np. Windows blokuje),
    zmienia nazwę katalogu na remove_<stara_nazwa>_(opcjonalny licznik).
    """
    try:
        shutil.rmtree(folder_path)
        return "removed"
    except Exception:
        base = os.path.basename(folder_path)
        parent = os.path.dirname(folder_path)
        new_name = f"remove_{base}"
        new_path = os.path.join(parent, new_name)

        # jeśli nazwa już istnieje, dodaj licznik
        counter = 1
        while os.path.exists(new_path):
            new_name = f"remove_{base}_{counter}"
            new_path = os.path.join(parent, new_name)
            counter += 1

        try:
            os.rename(folder_path, new_path)
            return f"renamed_to_{new_name}"
        except Exception:
            return "failed"


# ==================================================
# Filtr grey_filter – wycina podejrzanie jasne ramki
# (na podstawie liczby jasnych pikseli w obrazie)
# ==================================================
def grey_filter(data: pd.DataFrame, folder_path: str,
                bright_pixels: int = 70, threshold: int = 70) -> pd.DataFrame:
    """
    Przechodzi po wierszach z frame_content, liczy liczbę jasnych pikseli
    i zostawia tylko te, które mają (0, bright_pixels). Reszta wypada.
    """
    good_rows = []
    data = data.reset_index(drop=True)
    idx_col = data["idx"].astype(int).to_numpy()
    frames = data["frame_content"].to_numpy()

    for i, idx in enumerate(idx_col):
        # wyciągnij base64 (po przecinku)
        b64 = frames[i].split(",")[-1]
        # konwersja na obraz w skali szarości
        img = Image.open(io.BytesIO(base64.b64decode(b64, validate=False))).convert("L")
        gray = np.array(img, dtype=np.uint8)
        # ile pikseli powyżej progu jasności
        sum_value = int((gray > threshold).sum())
        if 0 < sum_value < bright_pixels:
            good_rows.append(idx)

    return data[data["idx"].isin(good_rows)].reset_index(drop=True)


# ==============================================
# Filtr dead_pixel – usuwa długie serie (x,y)
# ==============================================
def dead_pixel(data: pd.DataFrame, min_run: int = 3):
    """
    Szuka serii jednakowych współrzędnych (x, y); zostawia tylko
    wiersze będące w seriach krótszych niż min_run.
    """
    change = data[["x", "y"]].ne(data[["x", "y"]].shift()).any(axis=1)
    grp = change.cumsum()
    run_len = grp.groupby(grp).transform("size")
    out = data[run_len < min_run]
    return out.reset_index(drop=True)


# =======================================================
# remove_duplicates – usuwa duplikaty timestampów (det/pings)
# =======================================================
def remove_duplicates(det: pd.DataFrame, pings: pd.DataFrame):
    """
    Usuwa duplikaty po timestamp w detekcjach i pingach.
    Dodatkowo sprząta zbędne kolumny (idx, altitude, visible).
    """
    det = det.copy()
    pings = pings.copy()

    # detekcje
    det["timestamp"] = det["timestamp"].astype("int64")
    det = det.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")
    if "idx" in det.columns:
        det = det.drop(columns=["idx"])
    det = det.drop(columns=["altitude", "visible"], errors="ignore")

    # pingi
    pings["timestamp"] = pings["timestamp"].astype("int64")
    pings["on_time"] = pings["on_time"].astype("int64")
    pings = pings.sort_values("timestamp").drop_duplicates(subset="timestamp", keep="first")

    return det.reset_index(drop=True), pings.reset_index(drop=True)


# =================================================
# Stała do klasyfikatora too_often (licznik artefaktów)
# =================================================
ARTIFACT_TOO_OFTEN = "artifact_too_often"


# ======================================================
# grupowanie timestampów po dzielniku (np. co 1 ms)
# ======================================================
def group_by_timestamp_division(detections: List[dict], divisor: int) -> Dict[int, List[dict]]:
    """
    Grupuje listę detekcji (dict) po timestamp//divisor.
    Zwraca: słownik key -> lista detekcji.
    """
    out: Dict[int, List[dict]] = {}
    for d in detections:
        ts = int(d.get("timestamp"))
        key = ts // divisor
        out.setdefault(key, []).append(d)
    return out


# ======================================
# get_and_set – inicjalizacja pola w dict
# ======================================
def get_and_set(d: dict, key: str, default):
    """
    Jeśli w dict nie ma klucza, ustawia wartość domyślną.
    Zwraca aktualną wartość d[key].
    """
    if key not in d:
        d[key] = default
    return d[key]


# =====================================================
# classify_by_lambda – rozdziela listę detekcji na
# (spełniające warunek, niespełniające)
# =====================================================
def classify_by_lambda(detections: List[dict], predicate) -> Tuple[List[dict], List[dict]]:
    """
    Rozdziela listę detekcji na dwie listy: classified i unclassified,
    w zależności od tego, czy predicate(d) zwróci True.
    """
    classified, unclassified = [], []
    for d in detections:
        (classified if predicate(d) else unclassified).append(d)
    return classified, unclassified


# ================================================================
# too_often – klasyfikator "zbyt często" na podstawie timestampów
# ================================================================
def too_often(detections: List[dict],
              often: int = 10,
              time_window: int = 60000) -> Tuple[List[dict], List[dict]]:
    """
    Klasyfikator 'too_often':
      - grupuje po identycznych timestampach (divisor=1),
      - dla każdej pary grup zbliżonych w czasie (< time_window) zwiększa licznik
        ARTIFACT_TOO_OFTEN dla detekcji z obu grup,
      - jako artefakty klasyfikuje te, które mają licznik >= often.
    Zwraca (classified, unclassified).
    """
    # sortowanie po timestamp (niekonieczne, ale porządkuje)
    detections.sort(key=lambda x: x.get("timestamp"))
    grouped = group_by_timestamp_division(detections, 1)

    if len(grouped) == 1:
        # tylko jedna grupa czasowa – licznik = 0
        for group in grouped.values():
            for d in group:
                get_and_set(d, ARTIFACT_TOO_OFTEN, 0)
    else:
        # wszystkie pary kluczy
        keys = sorted(grouped.keys())
        for key, key_prim in itertools.combinations(keys, 2):
            # inicjalizacja licznika (nie resetujemy, jeśli już jest)
            for d in itertools.chain(grouped.get(key, ()), grouped.get(key_prim, ())):
                get_and_set(d, ARTIFACT_TOO_OFTEN, 0)
            # sprawdzamy odległość w czasie pomiędzy kluczami
            if abs(key - key_prim) < time_window:
                for d in grouped[key]:
                    d[ARTIFACT_TOO_OFTEN] += 1
                for d in grouped[key_prim]:
                    d[ARTIFACT_TOO_OFTEN] += 1

    # klasyfikacja po progu 'often'
    return classify_by_lambda(detections, lambda x: x.get(ARTIFACT_TOO_OFTEN, 0) >= often)


# ==================================================
# Filtr size_filter – zostawia tylko kwadratowe framy
# ==================================================
def size_filter(data: pd.DataFrame,
                min_size: int = 30,
                frame_col: str = "frame_content") -> pd.DataFrame:
    """
    Zostawia tylko wiersze, dla których obraz:
      - jest kwadratowy (szerokość == wysokość),
      - ma rozmiar co najmniej min_size x min_size pikseli.
    Czyli odrzucamy:
      - wszystko co niekwadratowe (w != h),
      - wszystko co ma bok < min_size.
    """
    good_rows = []
    data = data.reset_index(drop=True)

    if "idx" not in data.columns:
        data["idx"] = range(len(data))

    idx_col = data["idx"].astype(int).to_numpy()
    frames = data[frame_col].astype(str).to_numpy()

    for i, idx in enumerate(idx_col):
        try:
            b64 = frames[i].split(",")[-1]  # z lub bez prefixu data:
            img = Image.open(io.BytesIO(base64.b64decode(b64, validate=False)))
            w, h = img.size
        except Exception:
            # jak coś się nie da odczytać – traktujemy jako zły obrazek
            continue

        # warunek: kwadrat i bok >= min_size
        if (w == h) and (w >= min_size):
            good_rows.append(idx)

    return data[data["idx"].isin(good_rows)].reset_index(drop=True)


# ==============================================
# process_dir – pipeline filtrów dla jednego
# katalogu urządzenia (równoległy worker)
# ==============================================
def process_dir(dev_dir: str, often=10, time_window=60000):
    """
    Przetwarza dane jednego urządzenia:
      1. wczytuje detections.csv i pings.csv,
      2. odcina dane < 2018,
      3. dead_pixel,
      4. size_filter,
      5. grey_filter,
      6. remove_duplicates (NA KOŃCU),
      7. zapisuje wyniki do tych samych plików.

    Zwraca:
      (dir, status,
       n_start,
       n_after_ts2018,
       n_after_dead,
       n_after_size,
       n_after_grey,
       n_after_duplicates)
    """
    try:
        data_dir = os.path.join(dev_dir, "data")
        det_path = os.path.join(data_dir, "detections.csv")
        ping_path = os.path.join(data_dir, "pings.csv")
        out_path = os.path.join(data_dir, "detections_filtered.csv")

        # brak wymaganych plików -> pomijamy urządzenie
        if not (os.path.exists(det_path) and os.path.exists(ping_path)):
            return dev_dir, "missing_files", 0, 0, 0, 0, 0, 0

        # Czytaj szybciej; jeśli nie masz pyarrow, usuń argument engine
        detections_df = pd.read_csv(det_path, engine="pyarrow")
        pings_df = pd.read_csv(ping_path, engine="pyarrow")

        if "timestamp" not in detections_df.columns:
            return dev_dir, "no_timestamp", 0, 0, 0, 0, 0, 0

        # --- startowa liczba detekcji ---
        n_start = len(detections_df)

        # ---- 1) usunięcie danych sprzed 2018 ----
        TS_2018 = int(pd.Timestamp("2018-01-01", tz="UTC").timestamp() * 1000)
        detections_df = detections_df[detections_df["timestamp"] >= TS_2018]
        pings_df = pings_df[pings_df["timestamp"] >= TS_2018]
        n_after_ts2018 = len(detections_df)

        if n_after_ts2018 == 0:
            safe_remove_or_rename(dev_dir)
            return (
                dev_dir,
                "empty_after_ts2018",
                n_start,
                n_after_ts2018,
                0,
                0,
                0,
                0,
            )

        # ---- 2) dead_pixel (NA POCZĄTKU) ----
        detections_df = dead_pixel(detections_df.copy())
        n_after_dead = len(detections_df)
        if n_after_dead == 0:
            safe_remove_or_rename(dev_dir)
            return (
                dev_dir,
                "empty_after_dead_pixel",
                n_start,
                n_after_ts2018,
                n_after_dead,
                0,
                0,
                0,
            )

        # ---- 3) size_filter ----
        detections_df["idx"] = range(len(detections_df))
        detections_df = size_filter(detections_df.copy(), min_size=30)
        n_after_size = len(detections_df)
        if n_after_size == 0:
            safe_remove_or_rename(dev_dir)
            return (
                dev_dir,
                "empty_after_size_filter",
                n_start,
                n_after_ts2018,
                n_after_dead,
                n_after_size,
                0,
                0,
            )

        # ---- 4) grey_filter ----
        detections_df["idx"] = range(len(detections_df))
        detections_df = grey_filter(detections_df.copy(), folder_path=data_dir)
        n_after_grey = len(detections_df)
        if n_after_grey == 0:
            safe_remove_or_rename(dev_dir)
            return (
                dev_dir,
                "empty_after_grey",
                n_start,
                n_after_ts2018,
                n_after_dead,
                n_after_size,
                n_after_grey,
                0,
            )

        # ---- 5) remove_duplicates (NA KOŃCU) ----
        detections_df, pings_df = remove_duplicates(detections_df, pings_df)
        n_after_duplicates = len(detections_df)
        if n_after_duplicates == 0:
            safe_remove_or_rename(dev_dir)
            return (
                dev_dir,
                "empty_after_duplicates",
                n_start,
                n_after_ts2018,
                n_after_dead,
                n_after_size,
                n_after_grey,
                n_after_duplicates,
            )

        # --- zapis ---
        os.makedirs(dev_dir, exist_ok=True)
        detections_df.to_csv(out_path, index=False)
        pings_df.to_csv(ping_path, index=False)

        return (
            dev_dir,
            "ok",
            n_start,
            n_after_ts2018,
            n_after_dead,
            n_after_size,
            n_after_grey,
            n_after_duplicates,
        )

    except Exception as e:
        # niech błąd jednego katalogu nie zatrzyma całości
        return dev_dir, f"error: {repr(e)}", 0, 0, 0, 0, 0, 0

# =========================
# MAIN – uruchomienie równoległe
# =========================
if __name__ == "__main__":
    # Ustaw wątki BLAS dla procesów potomnych (jeśli używasz numpy/MKL)
    os.environ.setdefault("MKL_NUM_THREADS", "8")   # liczba rdzeni fizycznych
    os.environ.setdefault("OMP_NUM_THREADS", "8")

    # znacznik startu
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    start = time.time()

    base_path = str(RESULTS_DIR)

    # zbierz katalogi urządzeń (podkatalogi w base_path)
    dirs = [
        os.path.join(base_path, d)
        for d in os.listdir(base_path)
        if os.path.isdir(os.path.join(base_path, d))
    ]

    results = []
    max_workers = int(0.9 * os.cpu_count())

    # równoległe przetwarzanie katalogów
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        fut_map = {ex.submit(process_dir, d, 10, 60000): d for d in dirs}
        for fut in tqdm(as_completed(fut_map), total=len(fut_map),
                        desc="filtrowanie danych (parallel)"):
            results.append(fut.result())

    df_sum = pd.DataFrame(
        results,
        columns=[
            "dir",
            "status",
            "n_start",
            "n_after_ts2018",
            "n_after_dead",
            "n_after_size",
            "n_after_grey",
            "n_after_duplicates",
        ],
    )

    print(df_sum["status"].value_counts())

    if (df_sum["status"].str.startswith("error")).any():
        print("\nErrors:")
        print(df_sum[df_sum["status"].str.startswith("error")])

    # ============================
    # GLOBALNE PODSUMOWANIE
    # ============================
    total_start = df_sum["n_start"].sum()

    if total_start > 0:
        total_ts = df_sum["n_after_ts2018"].sum()
        total_dead = df_sum["n_after_dead"].sum()
        total_size = df_sum["n_after_size"].sum()
        total_grey = df_sum["n_after_grey"].sum()
        total_dup = df_sum["n_after_duplicates"].sum()  # OSTATECZNA LICZBA

        def pct(x):
            return 100.0 * x / total_start

        # --- liczenie liczby odrzuconych na etapach (zgodnie z KOLEJNOŚCIĄ) ---
        rej_ts = total_start - total_ts          # odcięcie <2018
        rej_dead = total_ts - total_dead         # dead_pixel
        rej_size = total_dead - total_size       # size_filter
        rej_grey = total_size - total_grey       # grey_filter
        rej_dup = total_grey - total_dup         # remove_duplicates (na końcu)

        total_rejected = total_start - total_dup

        print("\n📊 GLOBALNE PODSUMOWANIE FILTRACJI")
        print("------------------------------------")
        print(f"Całkowita liczba detekcji: {total_start}")
        print(f"Ostatecznie pozostało:     {total_dup}")
        print(f"Łącznie odrzucono:         {total_rejected} ({pct(total_rejected):.2f}%)\n")

        print("Rozbicie na etapy:")
        print(f" - Odrzucone <2018:        {rej_ts} ({pct(rej_ts):.2f}%)")
        print(f" - Dead pixel:             {rej_dead} ({pct(rej_dead):.2f}%)")
        print(f" - Size filter:            {rej_size} ({pct(rej_size):.2f}%)")
        print(f" - Grey filter:            {rej_grey} ({pct(rej_grey):.2f}%)")
        print(f" - Duplicates (na końcu):  {rej_dup} ({pct(rej_dup):.2f}%)")

        # ================================
        # ZAPIS DO PLIKU CSV (ŁADNA FORMA)
        # ================================
        summary_df = pd.DataFrame([
            ["Start (wszystkie detekcje)", total_start, "—"],
            ["Odrzucone <2018", rej_ts, f"{pct(rej_ts):.2f}%"],
            ["Odrzucone: dead_pixel", rej_dead, f"{pct(rej_dead):.2f}%"],
            ["Odrzucone: size_filter", rej_size, f"{pct(rej_size):.2f}%"],
            ["Odrzucone: grey_filter", rej_grey, f"{pct(rej_grey):.2f}%"],
            ["Odrzucone: duplicates (na końcu)", rej_dup, f"{pct(rej_dup):.2f}%"],
            ["\nŁĄCZNIE ODRZUCONE", total_rejected, f"{pct(total_rejected):.2f}%"],
            ["Pozostałe po filtracji", total_dup, f"{100 - pct(total_rejected):.2f}%"],
        ], columns=["Etap", "Liczba", "Procent"])

        out_path = REPO_ROOT / "filters_summary.csv"
        summary_df.to_csv(out_path, index=False)
        print(f"\n📁 Wyniki zapisano do pliku: {out_path}")

    end = time.time()
    print("Czas [s]:", end - start)
