from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple, Dict, Any, Set
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import poisson


# =========================
# KONFIG (ścieżki względem katalogu repozytorium)
# =========================
_REPO = Path(__file__).resolve().parent
BASE_RESULTS = _REPO / "results"
DETECTIONS_FILE = Path("data/detections_filtered.csv")
PINGS_FILE = Path("data/pings.csv")

POISSON_DIR = _REPO / "wyniki_poisson"
TURN_POISSON_FILE = _REPO / "wyniki" / "turn_poisson.txt"

FREQ = "5min"
BIN_SECONDS = pd.Timedelta(FREQ).total_seconds()

MIN_ON_TIME_S = 150.0
ALPHA = 0.00135   # jednostronne ~3 sigma


# =========================
# Pobranie listy urządzeń
# =========================
def get_devices_from_wyniki_poisson(poisson_dir: Path) -> Set[str]:
    if not poisson_dir.exists():
        return set()
    return {p.name for p in poisson_dir.iterdir() if p.is_dir()}


def get_devices_from_turn_poisson(turn_poisson_file: Path) -> Set[str]:
    devices = set()
    if not turn_poisson_file.exists():
        return devices

    with open(turn_poisson_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) >= 1:
                devices.add(parts[0])

    return devices


def get_all_selected_devices() -> List[str]:
    dev_poisson = get_devices_from_wyniki_poisson(POISSON_DIR)
    dev_turn = get_devices_from_turn_poisson(TURN_POISSON_FILE)
    devices = sorted(dev_poisson | dev_turn)
    return devices


# =========================
# Ekspozycja z pingów
# =========================
def count_on_time(pings: pd.DataFrame, freq: str = "5min") -> pd.DataFrame:
    p = pings.rename(columns={"timestamp": "start", "delta_time": "dt", "on_time": "active_ms"}).copy()

    p["start"] = pd.to_datetime(p["start"], unit="ms", utc=True, errors="coerce")
    p["dt"] = pd.to_timedelta(p["dt"], unit="ms", errors="coerce")
    p["end"] = p["start"] + p["dt"]
    p["active_seconds"] = p["active_ms"] / 1000.0

    p = p.dropna(subset=["start", "end", "dt", "active_seconds"])
    p = p[(p["dt"] > pd.Timedelta(0)) & (p["end"] > p["start"])]

    if p.empty:
        return pd.DataFrame(columns=["window_start", "window_end", "on_time_seconds"])

    p = p.sort_values("start").reset_index(drop=True)
    bin_size = pd.Timedelta(freq)

    grid_start = p["start"].min().floor(freq)
    grid_end = p["end"].max().ceil(freq)

    grid = pd.DataFrame({"window_start": pd.date_range(grid_start, grid_end, freq=freq, tz="UTC")})
    grid["window_end"] = grid["window_start"] + bin_size

    p["bin0"] = p["start"].dt.floor(freq)
    p["bin1"] = (p["end"] - pd.Timedelta(microseconds=1)).dt.floor(freq)

    n_bins = ((p["bin1"] - p["bin0"]) // bin_size + 1).astype(int)
    keep = n_bins > 0
    p = p.loc[keep].copy()
    n_bins = n_bins.loc[keep]

    if p.empty:
        grid["on_time_seconds"] = 0.0
        return grid

    idx = np.repeat(p.index.to_numpy(), n_bins.to_numpy())
    offset = np.concatenate([np.arange(k, dtype=int) for k in n_bins.to_numpy()])

    tmp = p.loc[idx, ["start", "end", "bin0", "active_seconds"]].copy()
    tmp["window_start"] = tmp["bin0"] + offset * bin_size
    tmp["window_end"] = tmp["window_start"] + bin_size

    left = tmp[["start", "window_start"]].max(axis=1)
    right = tmp[["end", "window_end"]].min(axis=1)
    overlap_s = (right - left).dt.total_seconds().clip(lower=0.0)

    interval_s = (tmp["end"] - tmp["start"]).dt.total_seconds()
    interval_s = np.where(interval_s <= 0, np.nan, interval_s)

    tmp["on_time_seconds"] = overlap_s * (tmp["active_seconds"] / interval_s)

    agg = tmp.groupby("window_start")["on_time_seconds"].sum().reset_index()
    out = grid.merge(agg, on="window_start", how="left")
    out["on_time_seconds"] = out["on_time_seconds"].fillna(0.0)
    return out


# =========================
# x i q dla danego lambda
# =========================
def threshold_x_and_q(lam: float, alpha: float = ALPHA) -> Tuple[int, float]:
    """
    x = najmniejsze całkowite x takie, że P(X >= x | lam) <= alpha
    q = rzeczywiste P(X >= x | lam)
    """
    if not np.isfinite(lam) or lam < 0:
        return -1, np.nan

    x = int(poisson.isf(alpha, lam)) + 1
    q = float(poisson.sf(x - 1, lam))
    return x, q


def compute_x_q_arrays(lam_array: np.ndarray, alpha: float = ALPHA) -> Tuple[np.ndarray, np.ndarray]:
    x_arr = np.empty(len(lam_array), dtype=int)
    q_arr = np.empty(len(lam_array), dtype=float)

    for i, lam in enumerate(lam_array):
        x, q = threshold_x_and_q(float(lam), alpha)
        x_arr[i] = x
        q_arr[i] = q

    return x_arr, q_arr


# =========================
# Jedno urządzenie
# =========================
def compute_device_windows(device_id: str, alpha: float = ALPHA) -> pd.DataFrame:
    det_path = BASE_RESULTS / device_id / DETECTIONS_FILE
    ping_path = BASE_RESULTS / device_id / PINGS_FILE

    if not det_path.exists() or not ping_path.exists():
        return pd.DataFrame()

    try:
        det = pd.read_csv(det_path, usecols=["timestamp"])
        pings = pd.read_csv(ping_path, usecols=["timestamp", "on_time", "delta_time"])
    except Exception:
        return pd.DataFrame()

    if det.empty or pings.empty:
        return pd.DataFrame()

    det["time"] = pd.to_datetime(det["timestamp"], unit="ms", utc=True, errors="coerce")
    det = det.dropna(subset=["time"]).sort_values("time").reset_index(drop=True)
    if det.empty:
        return pd.DataFrame()

    windows = count_on_time(pings, freq=FREQ)
    if windows.empty:
        return pd.DataFrame()

    det["window_start"] = det["time"].dt.floor(FREQ)
    counts = det.groupby("window_start").size().rename("count").reset_index()

    windows = windows.merge(counts, on="window_start", how="left")
    windows["count"] = windows["count"].fillna(0).astype(int)

    w = windows[windows["on_time_seconds"] >= MIN_ON_TIME_S].copy()
    if w.empty:
        return pd.DataFrame()

    total_on_time = float(w["on_time_seconds"].sum())
    if total_on_time <= 0:
        return pd.DataFrame()

    rate_per_s = float(w["count"].sum()) / total_on_time

    w["lam"] = rate_per_s * w["on_time_seconds"].astype(float)
    w["p_obs"] = poisson.sf(w["count"].astype(int) - 1, w["lam"].astype(float))

    x_arr, q_arr = compute_x_q_arrays(w["lam"].to_numpy(dtype=float), alpha)
    w["x"] = x_arr
    w["q"] = q_arr

    w["is_overactive_obs"] = w["p_obs"] <= alpha
    w["rate_obs"] = w["count"] / w["on_time_seconds"]
    w["rate_per_s_device"] = rate_per_s
    w["rate_threshold_x"] = w["x"] / w["on_time_seconds"]

    w.insert(0, "device_id", device_id)

    return w[
        [
            "device_id",
            "window_start",
            "window_end",
            "on_time_seconds",
            "count",
            "rate_obs",
            "rate_per_s_device",
            "lam",
            "p_obs",
            "x",
            "q",
            "rate_threshold_x",
            "is_overactive_obs",
        ]
    ].copy()


# =========================
# Worker równoległy
# =========================
def worker_device(device_id: str, alpha: float) -> pd.DataFrame:
    return compute_device_windows(device_id, alpha=alpha)


def build_all_devices_df_parallel(device_ids: List[str], alpha: float = ALPHA) -> pd.DataFrame:
    frames = []

    if not device_ids:
        return pd.DataFrame()

    max_workers = max(1, int((os.cpu_count() or 4) * 0.9))

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(worker_device, device_id, alpha): device_id for device_id in device_ids}

        total = len(futures)
        done = 0

        for fut in as_completed(futures):
            done += 1
            if done % 25 == 0 or done == total:
                print(f"[{done}/{total}] przetworzono urządzenia")

            dev = futures[fut]
            try:
                df_dev = fut.result()
                if isinstance(df_dev, pd.DataFrame) and not df_dev.empty:
                    frames.append(df_dev)
            except Exception as e:
                print(f"Błąd dla urządzenia {dev}: {e}")

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["window_start", "device_id"]).reset_index(drop=True)
    return out


# =========================
# Tło statystyczne na okno
# =========================
def coincidence_background_per_window(df_all: pd.DataFrame) -> pd.DataFrame:
    rows = []

    if df_all.empty:
        return pd.DataFrame()

    for window_start, g in df_all.groupby("window_start", sort=True):
        g = g[g["on_time_seconds"] > 0].copy()
        n = len(g)

        # pomijamy okna z mniej niż 2 aktywnymi urządzeniami
        if n < 2:
            continue

        q = g["q"].to_numpy(dtype=float)

        # dokładne P(>=2) = 1 - P(0) - P(1)
        p0 = float(np.prod(1.0 - q))

        denom = 1.0 - q
        safe = denom > 0
        p1 = float(p0 * np.sum(q[safe] / denom[safe]))

        p_ge_2_exact = 1.0 - p0 - p1
        p_ge_2_exact = max(0.0, min(1.0, p_ge_2_exact))

        # przybliżenie: suma po parach q_i q_j
        sum_q = float(np.sum(q))
        sum_q2 = float(np.sum(q * q))
        p_ge_2_approx = 0.5 * (sum_q * sum_q - sum_q2)

        n_overactive_obs = int(g["is_overactive_obs"].sum())
        has_ge_2_obs = bool(n_overactive_obs >= 2)

        rows.append(
            {
                "window_start": window_start,
                "n_active_devices": n,
                "n_overactive_devices_obs": n_overactive_obs,
                "has_coincidence_obs": has_ge_2_obs,
                "p_background_ge_2_exact": p_ge_2_exact,
                "p_background_ge_2_approx": p_ge_2_approx,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("window_start").reset_index(drop=True)


# =========================
# Raport globalny
# =========================
def summarize_background(df_all: pd.DataFrame, df_bg_win: pd.DataFrame) -> Dict[str, Any]:
    if df_all.empty:
        return {
            "n_device_window_rows": 0,
            "n_unique_windows_all": 0,
            "n_windows_with_at_least_2_active_devices": 0,
            "expected_single_overactive_all_rows": 0.0,
            "expected_single_overactive_only_windows_ge_2_active": 0.0,
            "expected_windows_ge_2_exact": 0.0,
            "expected_windows_ge_2_approx": 0.0,
            "observed_windows_ge_2": 0,
        }

    n_unique_windows_all = int(df_all["window_start"].nunique())

    active_counts = (
        df_all[df_all["on_time_seconds"] > 0]
        .groupby("window_start")["device_id"]
        .nunique()
    )
    windows_ge_2 = set(active_counts[active_counts >= 2].index)

    n_windows_with_at_least_2_active_devices = len(windows_ge_2)

    expected_single_all_rows = float(df_all["q"].sum())

    expected_single_only_ge2 = float(
        df_all[df_all["window_start"].isin(windows_ge_2)]["q"].sum()
    )

    if df_bg_win.empty:
        return {
            "n_device_window_rows": int(len(df_all)),
            "n_unique_windows_all": n_unique_windows_all,
            "n_windows_with_at_least_2_active_devices": n_windows_with_at_least_2_active_devices,
            "expected_single_overactive_all_rows": expected_single_all_rows,
            "expected_single_overactive_only_windows_ge_2_active": expected_single_only_ge2,
            "expected_windows_ge_2_exact": 0.0,
            "expected_windows_ge_2_approx": 0.0,
            "observed_windows_ge_2": 0,
        }

    return {
        "n_device_window_rows": int(len(df_all)),
        "n_unique_windows_all": n_unique_windows_all,
        "n_windows_with_at_least_2_active_devices": n_windows_with_at_least_2_active_devices,
        "expected_single_overactive_all_rows": expected_single_all_rows,
        "expected_single_overactive_only_windows_ge_2_active": expected_single_only_ge2,
        "expected_windows_ge_2_exact": float(df_bg_win["p_background_ge_2_exact"].sum()),
        "expected_windows_ge_2_approx": float(df_bg_win["p_background_ge_2_approx"].sum()),
        "observed_windows_ge_2": int(df_bg_win["has_coincidence_obs"].sum()),
    }


# =========================
# Raport drukowany
# =========================
def print_report(device_ids: List[str], df_all: pd.DataFrame, df_bg_win: pd.DataFrame, summary: Dict[str, Any]) -> None:
    print("\n" + "=" * 75)
    print("GLOBALNY RAPORT ANALIZY TŁA STATYSTYCZNEGO KOINCYDENCJI")
    print("=" * 75)

    print(f"Źródła urządzeń:")
    print(f"  - folder {POISSON_DIR}")
    print(f"  - plik   {TURN_POISSON_FILE}")
    print(f"Liczba unikalnych urządzeń użytych do analizy: {len(device_ids)}")
    print(f"Częstość binowania: {FREQ}")
    print(f"Minimalny on_time w oknie: {MIN_ON_TIME_S:.1f} s")
    print(f"Próg istotności alpha: {ALPHA}")
    print("-" * 75)

    print("ROZMIAR ANALIZY:")
    print(f"Liczba wszystkich par (urządzenie, okno) po filtrach: {summary['n_device_window_rows']}")
    print(f"Liczba wszystkich unikalnych okien w danych: {summary['n_unique_windows_all']}")
    print(f"Liczba okien z co najmniej 2 aktywnymi urządzeniami: {summary['n_windows_with_at_least_2_active_devices']}")
    print("-" * 75)

    print("TŁO DLA POJEDYNCZYCH NADAKTYWNOŚCI:")
    print(
        "Oczekiwana liczba pojedynczych nadaktywnych urządzeń od samego tła "
        f"(po wszystkich parach urządzenie-okno): {summary['expected_single_overactive_all_rows']:.12g}"
    )
    print(
        "Oczekiwana liczba pojedynczych nadaktywnych urządzeń od samego tła "
        f"(tylko w oknach, gdzie aktywne były >=2 urządzenia): "
        f"{summary['expected_single_overactive_only_windows_ge_2_active']:.12g}"
    )
    print("-" * 75)

    print("TŁO DLA KOINCYDENCJI >=2 URZĄDZEŃ:")
    print(
        "Spodziewana liczba okien z koincydencją >=2 urządzeń od samego tła "
        f"(dokładnie): {summary['expected_windows_ge_2_exact']:.12g}"
    )
    print(
        "Spodziewana liczba okien z koincydencją >=2 urządzeń od samego tła "
        f"(przybliżenie suma q_i q_j): {summary['expected_windows_ge_2_approx']:.12g}"
    )
    print(
        "Rzeczywiście zaobserwowana liczba okien, w których >=2 urządzenia "
        f"były nadaktywne jednocześnie: {summary['observed_windows_ge_2']}"
    )

    if summary["expected_windows_ge_2_exact"] > 0:
        ratio = summary["observed_windows_ge_2"] / summary["expected_windows_ge_2_exact"]
        print(
            "Stosunek obserwacji do spodziewanego tła dla koincydencji >=2: "
            f"{ratio:.6g}"
        )

    print("=" * 75)


# =========================
# MAIN
# =========================
def main() -> None:
    device_ids = get_all_selected_devices()

    if not device_ids:
        print("Nie znaleziono żadnych urządzeń do analizy.")
        print(f"Sprawdź folder: {POISSON_DIR}")
        print(f"Sprawdź plik:   {TURN_POISSON_FILE}")
        return

    print("Start analizy globalnej...")
    print(f"Liczba urządzeń na wejściu: {len(device_ids)}")

    df_all = build_all_devices_df_parallel(device_ids, alpha=ALPHA)

    if df_all.empty:
        print("Brak danych po przetworzeniu urządzeń.")
        return

    df_bg_win = coincidence_background_per_window(df_all)
    summary = summarize_background(df_all, df_bg_win)

    print_report(
        device_ids=device_ids,
        df_all=df_all,
        df_bg_win=df_bg_win,
        summary=summary,
    )


if __name__ == "__main__":
    main()