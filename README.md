# Poszukiwanie anomalii promieniowania kosmicznego w danych eksperymentu CREDO

Repozytorium zawiera kod użyty w pracy magisterskiej (Uniwersytet Warszawski, Wydział Fizyki, kwiecień 2026): **Search for cosmic ray anomalies in the CREDO experiment data**.

**CREDO** (Cosmic Ray Extremely Distributed Observatory) to międzynarodowy projekt poszukujący wielkoskalowych korelacji w promieniowaniu kosmicznym z użyciem rozproszonej sieci detektorów (smartfony uczestników). Praca dotyczy **wtórnego promieniowania kosmicznego**: opracowania schematu poszukiwania anomalii po filtracji artefaktów, klasyfikacji obrazów siecią splotową oraz analizy statystycznej opartej o rozkład Poissona w oknach czasowych (m.in. test Fano Var/Śr, ogon rozkładu Poissona, korelacje między urządzeniami, „showery” detekcji).

## Wymagania

- Python 3.10+  
- Zależności: `pip install -r requirements.txt`  
- **PyTorch**: do kroku `AI_filter.py` potrzebny jest plik wag **`final_code/AI/best_model.pth`** (wytrenowany model CNN opisany w pracy). Bez tego pliku ten etap się nie uruchomi — patrz sekcja [Model sieci](#model-sieci).

## Dane wejściowe

1. **`pliki_csv/data.csv`** — zbiór detekcji (m.in. `device_id`, `visible`, …).  
2. **`pliki_csv/pings.csv`** — pingi (`device_id`, `timestamp`, `on_time`, …).

Katalog `pliki_csv/` jest w repozytorium z `.gitkeep`; same pliki CSV są w `.gitignore` (zbyt duże na Git). Umieść je lokalnie przed pierwszym uruchomieniem.

Skrypt `final_code/device_sel.py` (pierwszy krok pipeline) wybiera urządzenia spełniające progi (m.in. liczba detekcji z `visible == True`, czas pracy) i zapisuje listę do `final_code/list_of_devices.txt`. Jeśli ten plik już istnieje, wybór urządzeń jest pomijany.

## Uruchomienie (po `git clone`)

W katalogu głównym repozytorium:

```bash
pip install -r requirements.txt
python workflow.py
```

`workflow.py` ustawia katalog roboczy na **korzeń repozytorium** i kolejno uruchamia:

| Krok | Skrypt | Opis |
|------|--------|------|
| 1 | `final_code/device_sel.py` | Podział dużych CSV na katalogi `results/<device_id>/data/` |
| 2 | `final_code/filters.py` | Filtry obrazu (dead pixel, rozmiar, grey), dane od 2018, duplikaty → `detections_filtered.csv` |
| 3 | `final_code/AI_filter.py` | Klasyfikacja CNN na `frame_content` |
| 4 | `final_code/area_filter.py` | Redukcja hotspotów na matrycy CMOS |
| 5 | `final_code/histogram_for_first_day.py` | Analiza pierwszej doby vs reszta (opcjonalne wycinanie) |
| 6 | `final_code/full_poisson.py` | Okna 5 min, Poisson, histogramy, `wyniki/`, `wyniki_poisson/`, … |
| 7 | `final_code/last_step.py` | Anomalie, shower, korelacje → `anomalies/` |

Wyniki (foldery `results/`, `wyniki*`, `anomalies/`) powstają obok `workflow.py`.

## Ścieżki w kodzie

Wspólny moduł **`final_code/config_paths.py`** definiuje `REPO_ROOT` (katalog nadrzędny względem `final_code/`). Dzięki temu skrypty nie zależą od dysku ani od `C:\...`.

## Model sieci

Umieść wytrenowany plik **`best_model.pth`** w katalogu `final_code/AI/`. Architektura `SimpleCNN` w `AI_filter.py` musi być zgodna z treningiem (`final_code/AI/training.py` / `training2.py`).

## Skrypty pomocnicze (poza `workflow.py`)

W katalogu **`final_code/optional/`** znajdują się m.in.:

- `windows.py` — alternatywny eksport wybranych okien do `wybrane_okna/`  
- `graphics_overlay.py` — nakładka dwóch klatek z CSV  
- `detction_for_window_for_map.py`, `histogram_2D.py` — analizy pomocnicze  

Nie są wymagane do głównego pipeline’u.

## Streszczenie (na podstawie pracy)

Opracowano algorytm poszukiwania anomalii we wtórnym promieniowaniu kosmicznym i przetestowano go na danych grupy CREDO (citizen science, lata 2018–2020). W raporcie omówiono m.in. potencjalną kaskadę oraz korelacje czasowe podwyższonej liczby detekcji na wielu urządzeniach, z odniesieniem do szacowanego tła statystycznego.

---

**Autor:** Bartosz Izydorczyk · **Promotor:** prof. dr hab. Aleksander Filip Żarnecki  
