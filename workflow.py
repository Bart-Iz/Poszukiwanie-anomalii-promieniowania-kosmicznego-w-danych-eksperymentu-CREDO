import subprocess
import sys
from pathlib import Path

# Katalog główny repozytorium (tam, gdzie leży workflow.py) — wszystkie ścieżki są względem niego
REPO_ROOT = Path(__file__).resolve().parent

# =========================
# KONFIG
# =========================
# Skrypty w kolejności wykonania (względem REPO_ROOT)
SCRIPTS = [
    "final_code/device_sel.py",
    "final_code/filters.py",
    "final_code/AI_filter.py",
    "final_code/area_filter.py",
    "final_code/histogram_for_first_day.py",
    "final_code/full_poisson.py",
    "final_code/last_step.py",
]


# =========================
# FUNKCJA URUCHAMIAJĄCA
# =========================
def run_script(script_path: Path) -> None:
    print(f"\n🚀 Uruchamiam: {script_path}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=str(REPO_ROOT),
        capture_output=False,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ Błąd w {script_path}, zatrzymuję pipeline.")
        sys.exit(result.returncode)

    print(f"✅ Zakończono: {script_path}")


# =========================
# MAIN
# =========================
def main():
    for script in SCRIPTS:
        script_path = REPO_ROOT / script

        if not script_path.exists():
            print(f"❌ Plik nie istnieje: {script_path}")
            sys.exit(1)

        run_script(script_path)

    print("\n🎉 Pipeline zakończony sukcesem!")


if __name__ == "__main__":
    main()