# Poszukiwanie anomalii promieniowania kosmicznego w danych eksperymentu CREDO

Kod z pracy magisterskiej (UW, Wydział Fizyki, 2026): analiza wtórnego promieniowania kosmicznego w danych CREDO — filtracja, CNN, testy Poissona, korelacje i „showery”.

## Pliki potrzebne do uruchomienia `workflow.py`

| Plik | Rola |
|------|------|
| **`data.csv`** | Detekcje: m.in. `device_id`, `visible`, `timestamp`, kolumny używane w filtrach (m.in. `frame_content`, `x`, `y` itd. zgodnie z pipeline). |
| **`pings.csv`** | Pingi: m.in. `device_id`, `timestamp`, `on_time` (oraz `delta_time` w późniejszych krokach). |
| **`best_model.pth`** | Wagi sieci CNN — **tylko** dla etapu filtrowania AI; muszą pasować do architektury w `AI_filter.py` (jak przy treningu w `training.py` / `training2.py`). |

Dodatkowo, **opcjonalnie**:

| Plik | Rola |
|------|------|
| **`list_of_devices.txt`** | Jedna liczba (`device_id`) na linię. Jeśli **nie** istnieje przy starcie, zostanie utworzony na podstawie `data.csv` i `pings.csv` według progów z pierwszego kroku pipeline. |

Żadnych innych plików wejściowych pipeline nie wymaga.

## Środowisko

- Python 3.10+
- `pip install -r requirements.txt`

## Uruchomienie

```bash
pip install -r requirements.txt
python workflow.py
```

Kolejność kroków w `workflow.py`: wybór/podział danych na urządzenia → filtry klasyczne → filtr AI → filtr obszaru → histogram pierwszej doby → analiza Poissona → krok końcowy (anomalie, shower).

---

**Autor:** Bartosz Izydorczyk · **Promotor:** prof. dr hab. Aleksander Filip Żarnecki
