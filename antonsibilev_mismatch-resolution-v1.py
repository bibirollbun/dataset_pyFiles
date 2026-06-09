import csv, multiprocessing as mp
from pathlib import Path
from functools import partial
import pydicom
from tqdm import tqdm



# ---------- configuration ----------
SOURCE_DIR = Path("/kaggle/input/rsna-intracranial-aneurysm-detection/series")   # 4405 dir
LOG_DIR    = Path("/kaggle/working/logs"); LOG_DIR.mkdir(parents=True, exist_ok=True)
CSV_OUT    = LOG_DIR / "dim_mismatch.csv"
N_WORKERS  = max(mp.cpu_count(), 2)
# -----------------------------------

def inspect_series(dir_path: Path):
    rows, cols = set(), set()
    dcm_files = sorted(dir_path.glob("*.dcm"))
    if not dcm_files:
        return (dir_path.name, "no_dcm_files", "", "", 0)

    try:
        for f in dcm_files:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            rows.add(int(ds.Rows))
            cols.add(int(ds.Columns))
            
            if len(rows) > 1 or len(cols) > 1:
                break
    except Exception as e:
        return (dir_path.name, f"read_error:{e.__class__.__name__}", "", "", 0)

    if len(rows) == 1 and len(cols) == 1:
        return None                      

    return (
        dir_path.name,
        ";".join(map(str, rows)),
        ";".join(map(str, cols)),
        len(dcm_files),
        "mismatch"
    )

def main():
    series_dirs = [p for p in SOURCE_DIR.iterdir() if p.is_dir()]
    fieldnames = ["SeriesDir", "RowsSet", "ColsSet", "n_files", "note"]

    with mp.Pool(N_WORKERS) as pool, CSV_OUT.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile); writer.writerow(fieldnames)
        for res in tqdm(pool.imap_unordered(inspect_series, series_dirs),
                        total=len(series_dirs), desc="Scanning"):
            if res is not None:
                writer.writerow(res)

    print(f"✔ Проверка завершена. Лог только проблемных серий → {CSV_OUT}")

if __name__ == "__main__":
    main()



import pandas as pd
df = pd.read_csv("/kaggle/working/logs/dim_mismatch.csv")


print("Всего серий с несоответствием:", df.shape[0])
print(df.head())        

df  


