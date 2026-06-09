
from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import os
import numpy as np
from numpy.lib.format import open_memmap
import pandas as pd
import cv2
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

# =====================
# CONFIG (ajuste aqui)
# =====================
TRAIN_CSV = "/kaggle/input/grand-xray-slam-division-a/train1.csv"
TEST_CSV  = "/kaggle/input/grand-xray-slam-division-a/sample_submission_1.csv"   # precisa ter 'Image_name'
TRAIN_DIR = "/kaggle/input/grand-xray-slam-division-a/train1"
TEST_DIR  = "/kaggle/input/grand-xray-slam-division-a/test1"

OUT_DIR    = "/kaggle/working/"
IMAGE_SIZE = 320
CHANNELS   = 1           # salvar compacto (grayscale). No treino repetimos p/ 3 canais.
DTYPE      = "uint8"     # "uint8" (recomendado) ou "float16"
OVERWRITE  = True
# Ajuste conforme o seu SSD/CPU. 8–16 costuma ser bom em NVMe. Em HDD, reduza.
N_WORKERS  = max(1, (os.cpu_count() or 2) - 1)

LABEL_COLUMNS = [
    'Atelectasis','Cardiomegaly','Consolidation','Edema','Enlarged Cardiomediastinum',
    'Fracture','Lung Lesion','Lung Opacity','No Finding','Pleural Effusion',
    'Pleural Other','Pneumonia','Pneumothorax','Support Devices'
]

# =====================
# Helpers
# =====================
def _ensure_outdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def _validate_files(file_paths: List[Path]):
    missing = [str(p) for p in file_paths if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"{len(missing)} imagens não encontradas. Ex.: {missing[:3]}")

def _compute_slices(n: int, n_workers: int, chunk_size: int | None = None) -> List[Tuple[int,int]]:
    """
    Divide [0, n) em fatias. Se chunk_size for fornecido, cria blocos com esse tamanho;
    caso contrário, divide em n_workers blocos aproximadamente iguais.
    Blocos menores -> barra de progresso mais fluida.
    """
    if chunk_size and chunk_size > 0:
        slices = []
        for s in range(0, n, chunk_size):
            e = min(n, s + chunk_size)
            if s < e:
                slices.append((s, e))
        return slices
    # fallback: dividir por workers
    n_workers = max(1, min(n_workers, n))
    base = n // n_workers
    rem  = n % n_workers
    slices = []
    start = 0
    for w in range(n_workers):
        size = base + (1 if w < rem else 0)
        end = start + size
        if start < end:
            slices.append((start, end))
        start = end
    return slices

def _worker_write_slice(
    out_imgs: str,
    files_chunk: List[str],
    idx_start: int,
    image_size: int,
):
    # Reabrir memmap neste processo
    arr = open_memmap(out_imgs, mode='r+')
    H = W = image_size

    # Array scratch para evitar realocações
    scratch = np.empty((H, W), dtype=np.uint8)

    for k, path_str in enumerate(files_chunk):
        i = idx_start + k
        # Leitura grayscale rápida
        img = cv2.imread(path_str, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Falha ao ler: {path_str}")

        # Resize (INTER_AREA é adequado para downscale)
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)

        # Garantir dtype
        if img.dtype != np.uint8:
            img = img.astype(np.uint8, copy=False)

        scratch[...] = img
        # Escrever (C,H,W) com C=1
        arr[i, 0, :, :] = scratch

    # Fechar view do memmap neste processo
    del arr
    return True

def _pack_split(csv_path: str, img_dir: str, is_train: bool):
    csv_path = Path(csv_path); img_dir = Path(img_dir)
    out_dir  = Path(OUT_DIR);  _ensure_outdir(out_dir)

    df = pd.read_csv(csv_path)
    if 'Image_name' not in df.columns:
        raise KeyError("CSV precisa conter a coluna 'Image_name'.")

    n = len(df)
    files: List[Path] = [img_dir / str(nm) for nm in df['Image_name'].tolist()]
    _validate_files(files)

    c, h, w = CHANNELS, IMAGE_SIZE, IMAGE_SIZE

    if is_train:
        out_imgs = out_dir / f"images_train_{IMAGE_SIZE}_c1_uint8.npy"
        out_lbls = out_dir / "labels_train.npy"
        out_idx  = out_dir / "index_train.csv"
    else:
        out_imgs = out_dir / f"images_test_{IMAGE_SIZE}_c1_uint8.npy"
        out_lbls = None
        out_idx  = out_dir / "index_test.csv"

    if out_imgs.exists() and not OVERWRITE:
        raise FileExistsError(f"{out_imgs} já existe. Defina OVERWRITE=True para sobrescrever.")

    print(f"=== PACK {'TRAIN' if is_train else 'TEST'} ===")
    print(f"N={n} | {h}x{w} | C={c} | dtype=uint8 -> {out_imgs}")

    # Criar arquivo .npy memmap e pré-alocar
    arr = open_memmap(str(out_imgs), mode='w+', dtype=np.uint8, shape=(n, c, h, w))
    del arr  # será reaberto pelos workers em 'r+'

    # Estratégia de progresso:
    # - blocos menores deixam a barra mais suave; 512/1024 são bons valores.
    # - ajuste chunk_size se quiser barras mais granulares.
    chunk_size = 1024
    slices = _compute_slices(n, N_WORKERS, chunk_size=chunk_size)
    paths_str = [str(p) for p in files]

    total_done = 0
    with ProcessPoolExecutor(max_workers=min(N_WORKERS, len(slices))) as ex, \
         tqdm(total=n, desc="Empacotando (paralelo)", unit="img", dynamic_ncols=True) as pbar:
        futures = []
        for (s, e) in slices:
            fut = ex.submit(
                _worker_write_slice,
                str(out_imgs),
                paths_str[s:e],
                s,
                IMAGE_SIZE,
            )
            futures.append((fut, e - s))

        # Conforme cada slice concluir, atualizamos o progresso
        for fut, size in futures:
            fut.result()  # Propaga erro, se houver
            total_done += size
            pbar.update(size)

    # CSV rápido
    if is_train:
        # Checar colunas
        missing = [c for c in LABEL_COLUMNS if c not in df.columns]
        if missing:
            raise KeyError(f"Colunas faltantes no CSV de treino: {missing}")

        out_df = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int32),
            "path": paths_str,  # opcional: salvar apenas Image_name
        })
        for col in LABEL_COLUMNS:
            out_df[col] = df[col].astype(np.float32).values
        out_df.to_csv(out_idx, index=False)

        # Labels separados (compatível com treino)
        labels = df[LABEL_COLUMNS].astype(np.float32).values
        np.save(out_lbls, labels)
        print(f"Labels salvos em: {out_lbls}")

    else:
        out_df = pd.DataFrame({
            "idx": np.arange(n, dtype=np.int32),
            "path": paths_str,
        })
        out_df.to_csv(out_idx, index=False)

    print(f"Imagens salvas em: {out_imgs}")
    print(f"Índice salvo em:  {out_idx}")

# =====================
# Main
# =====================
if __name__ == "__main__":
    _pack_split(TRAIN_CSV, TRAIN_DIR, is_train=True)
    _pack_split(TEST_CSV,  TEST_DIR,  is_train=False)


