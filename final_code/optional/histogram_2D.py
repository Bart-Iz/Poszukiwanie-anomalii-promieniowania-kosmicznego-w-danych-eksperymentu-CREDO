from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config_paths import REPO_ROOT

ROOT = REPO_ROOT / "wyniki_non_poisson"
MIN_N = 10  # "powyżej 10" => > 10


def save_hist2d_xy(df_win: pd.DataFrame, out_png: Path, title: str) -> None:
    x = pd.to_numeric(df_win["x"], errors="coerce")
    y = pd.to_numeric(df_win["y"], errors="coerce")
    m = np.isfinite(x) & np.isfinite(y)

    x = x[m].astype(int)
    y = y[m].astype(int)

    if len(x) == 0:
        return

    # automatyczne biny: od min do max +1
    xmin, xmax = int(x.min()), int(x.max())
    ymin, ymax = int(y.min()), int(y.max())

    # zabezpieczenie, żeby nie wywalić pamięci jakby zakres był kosmiczny
    max_bins = 512
    nx = min(max_bins, (xmax - xmin + 1))
    ny = min(max_bins, (ymax - ymin + 1))

    fig = plt.figure(figsize=(6, 5))
    plt.hist2d(x, y, bins=[nx, ny], range=[[xmin, xmax + 1], [ymin, ymax + 1]])
    plt.xlabel("x (najjaśniejszy piksel)")
    plt.ylabel("y (najjaśniejszy piksel)")
    plt.title(title)
    plt.colorbar(label="Liczba detekcji")
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=250)
    plt.close(fig)


def main() -> None:
    suspicious_files = sorted(ROOT.glob("*/suspicious.csv"))
    print(f"Znaleziono suspicious.csv: {len(suspicious_files)}")

    for csv_path in suspicious_files:
        dev = csv_path.parent.name
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            print(f"[{dev}] błąd czytania: {e}")
            continue

        if df.empty:
            continue

        # window_start jako datetime (żeby ładnie sortować i formatować)
        df["window_start"] = pd.to_datetime(df["window_start"], utc=True, errors="coerce")
        df = df.dropna(subset=["window_start"])

        if df.empty:
            continue

        out_dir = csv_path.parent / "hist2d_xy_suspicious"
        out_dir.mkdir(parents=True, exist_ok=True)

        for ws, g in df.groupby("window_start"):
            if len(g) <= MIN_N:  # ma być powyżej 10
                continue

            ws_str = ws.strftime("%Y%m%d_%H%M%S")
            out_png = out_dir / f"hist2d_xy_{ws_str}_N{len(g)}.png"
            title = f"{dev} | okno {ws.strftime('%Y-%m-%d %H:%M:%S')} UTC | N={len(g)}"
            save_hist2d_xy(g, out_png, title)

    print("DONE.")


if __name__ == "__main__":
    main()