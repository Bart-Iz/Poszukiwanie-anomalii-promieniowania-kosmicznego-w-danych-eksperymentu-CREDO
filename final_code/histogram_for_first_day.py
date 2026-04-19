import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config_paths import REPO_ROOT, RESULTS_DIR

# =========================
# USTAWIENIA
# =========================
BASE_PATH = RESULTS_DIR
DATA_SUBDIR = "data"
DETECTIONS_FILE = "detections_filtered.csv"
PINGS_FILE = "pings.csv"

TIME_WINDOW_MS = 24 * 60 * 60 * 1000  # 24h w ms

FACTOR_CUT = 5.0  # jeśli factor > 5 -> wycinamy pierwszą dobę z detections_filtered.csv


# ============================================
# Budowanie 24-godzinnych okien z on_time
# ============================================
def build_24h_windows_from_pings(pings: pd.DataFrame):
    if pings.empty:
        return []

    p = pings.copy()
    p["timestamp"] = p["timestamp"].astype("int64")
    p["on_time"] = p["on_time"].astype("int64")
    p = p.sort_values("timestamp").reset_index(drop=True)

    ts = p["timestamp"].to_numpy(dtype="int64")
    on = p["on_time"].to_numpy(dtype="int64")

    cum = on.cumsum()
    total_on = int(cum[-1])
    if total_on <= 0:
        return []

    n_full = total_on // TIME_WINDOW_MS
    windows = []

    if n_full > 0:
        targets = np.arange(1, n_full + 1, dtype="int64") * TIME_WINDOW_MS
        idx_end = np.searchsorted(cum, targets, side="left")
        end_ts = ts[idx_end]

        start_ts = int(ts[0])
        for i in range(int(n_full)):
            windows.append({
                "start_ts": int(start_ts),
                "end_ts": int(end_ts[i]),
                "uptime_ms": int(TIME_WINDOW_MS),
            })
            start_ts = int(end_ts[i])

        last_start_ts = int(start_ts)
    else:
        last_start_ts = int(ts[0])

    leftover_on = total_on - n_full * TIME_WINDOW_MS
    if leftover_on > 0:
        last_end_ts = int(ts[-1])
        windows.append({
            "start_ts": int(last_start_ts),
            "end_ts": int(last_end_ts),
            "uptime_ms": int(leftover_on),
        })

    return windows


# ============================================
# Liczenie detekcji w zadanych oknach
# ============================================
def count_detections_in_windows(det: pd.DataFrame, windows):
    if det.empty or not windows:
        return [0] * len(windows)

    d = det.copy()
    d["timestamp"] = d["timestamp"].astype("int64")
    d = d.sort_values("timestamp")
    ts_det = d["timestamp"].to_numpy(dtype="int64")

    counts = []
    for w in windows:
        s = w["start_ts"]
        e = w["end_ts"]
        mask = (ts_det >= s) & (ts_det < e)
        counts.append(int(mask.sum()))
    return counts


# ============================================
# Wycinanie pierwszej doby z detections (nadpisanie pliku)
# ============================================
def drop_first_day_from_detections(det_path: str, first_window: dict) -> tuple[bool, int, int]:
    """
    Usuwa z detections_filtered.csv wszystkie wiersze z timestamp w [start_ts, end_ts)
    i NADPISUJE plik.

    Zwraca:
      (changed, before_rows, after_rows)
    """
    s = int(first_window["start_ts"])
    e = int(first_window["end_ts"])

    df = pd.read_csv(det_path)
    if "timestamp" not in df.columns:
        raise ValueError(f"Brak kolumny 'timestamp' w pliku: {det_path}")

    before = len(df)

    ts = df["timestamp"].astype("int64")
    keep = ~((ts >= s) & (ts < e))
    df2 = df.loc[keep].copy()

    after = len(df2)
    changed = after != before

    if changed:
        tmp_path = det_path + ".tmp"
        df2.to_csv(tmp_path, index=False)
        os.replace(tmp_path, det_path)

    return changed, before, after


# ============================================
# Liczenie mnożnika + (opcjonalnie) wycinanie pierwszego dnia
# ============================================
def compute_first_day_factor_for_device(dev_dir: str):
    device_id = os.path.basename(dev_dir)
    data_dir = os.path.join(dev_dir, DATA_SUBDIR)
    det_path = os.path.join(data_dir, DETECTIONS_FILE)
    ping_path = os.path.join(data_dir, PINGS_FILE)

    if not (os.path.isfile(det_path) and os.path.isfile(ping_path)):
        return None

    det = pd.read_csv(det_path, usecols=["timestamp"])
    pings = pd.read_csv(ping_path, usecols=["timestamp", "on_time"])

    if det.empty or pings.empty:
        return None

    windows = build_24h_windows_from_pings(pings)
    if not windows:
        return None

    counts = count_detections_in_windows(det, windows)

    rows = []
    for w, c in zip(windows, counts):
        uptime_ms = int(w["uptime_ms"])
        uptime_h = uptime_ms / (60 * 60 * 1000)
        if uptime_ms <= 0:
            rate_24h = np.nan
        else:
            rate_24h = c * 24.0 / uptime_h

        rows.append({
            "device": device_id,
            "start_ts": int(w["start_ts"]),
            "end_ts": int(w["end_ts"]),
            "uptime_ms": uptime_ms,
            "uptime_h": uptime_h,
            "detections": int(c),
            "rate_24h": rate_24h,
        })

    day_df = pd.DataFrame(rows)

    if len(day_df) < 2:
        return None

    first_rate = float(day_df.loc[0, "rate_24h"])
    rest_rates = day_df["rate_24h"].iloc[1:].to_numpy(dtype=float)
    rest_rates = rest_rates[~np.isnan(rest_rates)]
    if rest_rates.size == 0:
        return None

    mean_rest = float(rest_rates.mean())
    if mean_rest == 0:
        return None

    factor = float(first_rate / mean_rest) if not np.isnan(first_rate) else np.nan

    # >>> NOWE: jeśli factor > 5, wycinamy pierwszą dobę z detections_filtered.csv <<<
    dropped = False
    before_rows = after_rows = None
    if not np.isnan(factor) and factor > FACTOR_CUT:
        try:
            dropped, before_rows, after_rows = drop_first_day_from_detections(det_path, windows[0])
        except Exception as exc:
            print(f"[{device_id}] ERROR przy wycinaniu pierwszej doby: {exc}")

    return {
        "device": device_id,
        "first_rate_24h": first_rate,
        "mean_rest_rate_24h": mean_rest,
        "factor_first_vs_rest": factor,
        "dropped_first_day": int(dropped),
        "rows_before": before_rows,
        "rows_after": after_rows,
        "per_day_df": day_df,
    }


# ============================================
# MAIN
# ============================================
def main():
    base = BASE_PATH
    dev_dirs = [str(p) for p in base.iterdir() if p.is_dir()]

    results = []
    per_device_days = []

    for dev_dir in dev_dirs:
        res = compute_first_day_factor_for_device(dev_dir)
        if res is None:
            continue

        results.append({
            "device": res["device"],
            "first_rate_24h": res["first_rate_24h"],
            "mean_rest_rate_24h": res["mean_rest_rate_24h"],
            "factor_first_vs_rest": res["factor_first_vs_rest"],
            "dropped_first_day": res["dropped_first_day"],
            "rows_before": res["rows_before"],
            "rows_after": res["rows_after"],
        })
        per_device_days.append(res["per_day_df"])

        if res["dropped_first_day"] == 1:
            print(f"[{res['device']}] factor>{FACTOR_CUT} -> wycięto pierwszą dobę z detections "
                  f"({res['rows_before']} -> {res['rows_after']})")

    if not results:
        print("Brak urządzeń, dla których da się policzyć statystykę.")
        return

    df_factors = pd.DataFrame(results)
    out_factors = REPO_ROOT / "first_day_factors_per_device.csv"
    df_factors.to_csv(out_factors, index=False)
    print(f"Zapisano → {out_factors}")

    df_days_all = pd.concat(per_device_days, ignore_index=True)
    out_days = REPO_ROOT / "per_device_24h_windows.csv"
    df_days_all.to_csv(out_days, index=False)
    print(f"Zapisano → {out_days}")

    factors = df_factors["factor_first_vs_rest"].to_numpy(dtype=float)
    factors = factors[~np.isnan(factors)]

    VIS_MAX = 31
    factors_vis = factors[factors <= VIS_MAX]

    plt.rcParams.update({
        "font.size": 22,
        "axes.titlesize": 28,
        "axes.labelsize": 22,
        "xtick.labelsize": 20,
        "ytick.labelsize": 20,
    })

    fig = plt.figure(figsize=(16, 8))
    ax = fig.add_subplot(111)
    ax.hist(factors_vis, bins=30)
    ax.set_xlabel(
        "ilość detekcji pierwszego dnia / średnia ilość detekcji na 24 h",
        fontsize=22,
    )
    ax.set_ylabel("liczba urządzeń", fontsize=22)
    ax.set_title(
        "Histogram wielokrotności średniej detekcji w ciągu pierwszych 24 godzin",
        fontsize=28,
    )
    ax.tick_params(axis="both", labelsize=20)
    ax.grid(True, which="major", alpha=0.35)
    fig.tight_layout()
    out_png = REPO_ROOT / "first_day.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)

    print(f"Zapisano → {out_png}")

    n_out = int(np.sum(factors > VIS_MAX))
    print(f"Outliery > {VIS_MAX:.2f}: {n_out} urządzeń")
    print(f"Liczba urządzeń uwzględnionych w histogramie: {len(factors)}")


if __name__ == "__main__":
    main()
