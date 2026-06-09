# --- libraries ---

import os
from pathlib import Path
import ast
from typing import Dict, Any

import numpy as np
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
from tqdm import tqdm




# --- Utility: Extract pixel coords ---
def extract_coordinates(
    df: pd.DataFrame,
    coord_col: str = 'coordinates',
    x_col: str = 'x_pix',
    y_col: str = 'y_pix'
) -> pd.DataFrame:
    """
    Parse a coordinate field (dict or literal str) into integer pixel columns.
    """
    df = df.copy()
    def _to_dict(v):
        if isinstance(v, dict):
            return v
        return ast.literal_eval(v)

    coords = df[coord_col].apply(_to_dict).tolist()
    pix = pd.DataFrame(coords, index=df.index)[['x','y']].astype(int)
    df[x_col], df[y_col] = pix['x'], pix['y']
    return df



# --- Utility: DICOM geometry ---
def get_image_geometry(ds: pydicom.Dataset) -> Dict[str, Any]:
    """Return rows, columns, spacings, cosines, and origin."""
    rows = int(ds.get('Rows', 512)); cols = int(ds.get('Columns', 512))
    ps = ds.get('PixelSpacing', [1.0,1.0])
    rs, cs = float(ps[0]), float(ps[1])
    origin = np.array(ds.get('ImagePositionPatient', [0,0,0]), float)
    iop = list(map(float, ds.get('ImageOrientationPatient', [1,0,0,0,1,0])))
    row_cos, col_cos = np.array(iop[:3]), np.array(iop[3:])
    return dict(
        rows=rows, columns=cols,
        row_spacing_mm=rs, col_spacing_mm=cs,
        origin_xyz=origin,
        row_cosine=row_cos, col_cosine=col_cos
    )


# --- Compute physical & relative coords ---
def compute_physical_and_relative(
    df: pd.DataFrame,
    sop_col: str,
    x_pix: str,
    y_pix: str,
    dicom_index: Dict[str, Path]
) -> pd.DataFrame:
    """
    For each row, load/cached DICOM geometry and compute:
      - x_mm, y_mm (world patient coords)
      - x_rel, y_rel (normalized within image)
    Also adds 'rows' and 'columns'.
    """
    out = df.copy()
    cache: Dict[str, Dict[str, Any]] = {}
    # accumulator lists
    x_mm, y_mm, rows_list, cols_list, x_rel, y_rel = ([] for _ in range(6))

    for _, row in out.iterrows():
        uid = str(row[sop_col]).upper()
        px, py = int(row[x_pix]), int(row[y_pix])
        if uid not in cache:
            path = dicom_index.get(uid)
            ds = pydicom.dcmread(str(path), stop_before_pixels=True) if path and path.is_file() else pydicom.Dataset()
            cache[uid] = get_image_geometry(ds)
        geom = cache[uid]
        origin = geom['origin_xyz']
        rs, cs = geom['row_spacing_mm'], geom['col_spacing_mm']
        row_cos, col_cos = geom['row_cosine'], geom['col_cosine']
        rows, cols = geom['rows'], geom['columns']
        # world coords
        world = origin + px*cs*col_cos + py*rs*row_cos
        x_mm.append(world[0]); y_mm.append(world[1])
        # relative coords
        rows_list.append(rows); cols_list.append(cols)
        x_rel.append(px/cols); y_rel.append(py/rows)

    out['x_mm'], out['y_mm'] = x_mm, y_mm
    out['rows'], out['columns'] = rows_list, cols_list
    out['x_rel'], out['y_rel'] = x_rel, y_rel
    
    return out


# --- Plotting ---
def plot_global(
    df: pd.DataFrame,
    loc_col: str = 'location'
) -> None:
    """Global scatter of relative coords by location."""
    fig, ax = plt.subplots(figsize=(12,6))
    for loc, grp in df.groupby(loc_col):
        ax.scatter(grp['x_rel'], grp['y_rel'], alpha=0.6, label=loc)
    ax.set(xlabel='x_rel', ylabel='y_rel', title='Global Relative Scatter')
    ax.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.tight_layout(); plt.show()


def plot_per_location(
    df: pd.DataFrame,
    location: str
) -> None:
    """Per-location scatter and boxplots on relative coords."""
    sub = df[df['location']==location]
    fig, (ax1, ax2) = plt.subplots(1,2,figsize=(10,4))
    ax1.scatter(sub['x_rel'], sub['y_rel'], alpha=0.6)
    ax1.set(title=f"{location} Relative Scatter", xlabel='x_rel', ylabel='y_rel')
    ax2.boxplot([sub['x_rel'], sub['y_rel']], labels=['x_rel','y_rel'])
    ax2.set(title=f"{location} Relative Boxplots")
    plt.tight_layout(); plt.show()



# --- #Utility: Improved Indexing ---
def index_dicoms(root_dir: str) -> dict[str, str]:
    dicom_index = {
        f[:-4].upper(): Path(root, f)
        for root, _, files in os.walk("/kaggle/input/rsna-intracranial-aneurysm-detection/")
        for f in files
        if f.lower().endswith(".dcm")
    }
    return dicom_index


# --- Main ---
def main():
    print('reading csv file')
    df = pd.read_csv(Path('/kaggle/input/rsna-intracranial-aneurysm-detection/train_localizers.csv'))

    print('extracting given coordinates')
    df = extract_coordinates(df)

    print('building dicom index')
    #dicom_index = {p.stem.upper(): p for p in Path('/kaggle/input/rsna-intracranial-aneurysm-detection/').rglob('*.dcm')}
    #dicom_index = {p.stem.upper(): p for p in tqdm(Path("/kaggle/input/rsna-intracranial-aneurysm-detection/").rglob("*.dcm"), desc="Indexing DICOM", unit="file")} #2min to run
    dicom_index = index_dicoms("/kaggle/input/rsna-intracranial-aneurysm-detection/")
    print('dicom index built')

    print('computing physical and relative values')
    df = compute_physical_and_relative(
        df, sop_col='SOPInstanceUID', x_pix='x_pix', y_pix='y_pix', dicom_index=dicom_index
    )
    print('finished computing physical and relative values')
    
    plot_global(df)
    
    for loc in df['location'].unique():
        plot_per_location(df, loc)

if __name__=='__main__':
    main()

