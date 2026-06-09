from pathlib import Path
from Bio import SeqIO
import polars as pl


metadata_dir = Path("metadata")
if not metadata_dir.exists():
    metadata_dir.mkdir(parents=True)


import polars as pl


def preprocess_label(df, no_chain=False, invalid_value=-1e18):
    """
    - Convert wide to long format.
    - Filter out invalid values.
    """

    # convert to long format
    df = (
        df.unpivot(index=["ID", "resname", "resid"])
        .with_columns(
            [
                pl.col("variable").str.extract(r"^([xyz])", 1).alias("coord"),
                pl.col("variable")
                .str.extract(r"_(\d+)", 1)
                .cast(pl.Int32)
                .alias("design_id"),
            ]
        )
        .pivot(
            values="value",
            index=["ID", "resname", "resid", "design_id"],
            on="coord",
        )
        # remove invalid values
        .filter(pl.col("x").gt(invalid_value))
    )

    if no_chain:
        # convert to long format
        df = (
            # split ID columns
            df.with_columns(pl.col("ID").str.splitn("_", 2).alias("_a"))
            .unnest("_a")
            .rename({"field_0": "pdb_id", "field_1": "res_id"})
            .with_columns(pl.lit(None).alias("chain_id"))
        )
    else:
        df = (
            # split ID columns
            df.with_columns(pl.col("ID").str.splitn("_", 3).alias("_a"))
            .unnest("_a")
            .rename({"field_0": "pdb_id", "field_1": "chain_id", "field_2": "res_id"})
        )

    return (
        df.with_columns(
            pl.format("{}_{}", pl.col("pdb_id"), pl.col("res_id")).alias("target_id"),
        )
        .sort(["pdb_id", "design_id", "chain_id", "resid"])
        .select(
            "ID",
            "target_id",
            "pdb_id",
            "design_id",
            "chain_id",
            "resid",
            "resname",
            "x",
            "y",
            "z",
        )
    )


def preprocess_sequence(df):
    return (
        df.with_columns(pl.col("target_id").str.splitn("_", 2).alias("_a"))
        .unnest("_a")
        .rename({"field_0": "pdb_id", "field_1": "chain_id"})
        .with_columns(
            pl.col("temporal_cutoff")
            .str.strptime(pl.Date, format="%Y-%m-%d")
            .alias("temporal_cutoff")
        )
        .with_columns(
            pl.col("sequence").str.len_chars().alias("sequence_length"),
        )
        .sort(["pdb_id", "chain_id"])
        .select(
            "target_id",
            "pdb_id",
            "chain_id",
            "temporal_cutoff",
            "sequence",
            "all_sequences",
            "description",
            "sequence_length",
        )
    )


train_sequence = pl.read_csv(
    "/kaggle/input/stanford-rna-3d-folding/train_sequences.csv"
)
train_label = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/train_labels.csv")
val_label = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/validation_labels.csv")
val_sequence = pl.read_csv(
    "/kaggle/input/stanford-rna-3d-folding/validation_sequences.csv"
)
test_sequence = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/test_sequences.csv")
val_label = preprocess_label(val_label, no_chain=True)
train_label = preprocess_label(train_label)
val_sequence = preprocess_sequence(val_sequence)
train_sequence = preprocess_sequence(train_sequence)
test_sequence = preprocess_sequence(test_sequence)
submission = pl.read_csv("/kaggle/input/stanford-rna-3d-folding/sample_submission.csv")


train_label.write_parquet(metadata_dir / "train_label.parquet")
train_sequence.write_parquet(metadata_dir / "train_sequence.parquet")
val_label.write_parquet(metadata_dir / "val_label.parquet")
val_sequence.write_parquet(metadata_dir / "val_sequence.parquet")
test_sequence.write_parquet(metadata_dir / "test_sequence.parquet")


from tqdm import tqdm

msa_paths = sorted(Path("/kaggle/input/stanford-rna-3d-folding/MSA/").rglob("*.fasta"))
msa_names = [path.name for path in msa_paths]
len(msa_paths)
cols = [
    "target_id",
    "idx",
    "id",
    "name",
    "description",
    "seq",
]

records = []
for path in tqdm(msa_paths):
    for i, record in enumerate(SeqIO.parse(path, "fasta")):
        record = vars(record)
        record["target_id"] = path.name.split(".")[0]
        record["idx"] = i
        record["seq"] = str(record["_seq"])
        new_record = {col: record[col] for col in cols if col in record}
        records.append(new_record)

msa = pl.DataFrame(records)
display(msa)


msa.write_parquet(metadata_dir / "msa.parquet")


import matplotlib.pyplot as plt

_, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

train_sequence["sequence_length"].to_pandas().hist(
    bins=100, ax=ax1, label="sequence length", color="blue", alpha=0.5
)
val_sequence["sequence_length"].to_pandas().hist(
    bins=10, ax=ax2, label="sequence length", color="orange", alpha=0.5
)
tr_mean  = train_sequence["sequence_length"].mean()
va_mean = val_sequence["sequence_length"].mean()
ax1.axvline(tr_mean, color="red", linestyle="--", label=f"mean={tr_mean:.0f}")
ax2.axvline(va_mean, color="red", linestyle="--", label=f"mean={va_mean:.0f}")
ax1.set(
    xlabel="sequence length",
    ylabel="count",
    title=f"train ({train_sequence.shape[0]} samples)",
)
ax1.legend()
ax2.set(
    xlabel="sequence length",
    ylabel="count",
    title=f"val ({val_sequence.shape[0]} samples)",
)
ax2.legend()
plt.show()


pdb_dir = Path("pdb")
if not pdb_dir.exists():
    pdb_dir.mkdir(parents=True)


from tqdm import tqdm


def make_pdb(df, output_dir):
    num_groups = df.group_by("pdb_id", "chain_id", "design_id").len().shape[0]
    for (pdb_id, chain_id, design_id), df in tqdm(
        df.group_by("pdb_id", "chain_id", "design_id", maintain_order=True),
        total=num_groups,
    ):
        chain_id = chain_id if chain_id else "A"
        file_path = output_dir / f"{pdb_id}_{chain_id}_{design_id}.pdb"
        with open(file_path, "w") as f:
            for i, row in enumerate(df.iter_rows(named=True)):
                design_id = row["design_id"]
                atom_serial = i + 1
                atom_name = "C1"
                residue_name = row["resname"]
                residue_num = int(row["resid"])
                chain_id = row["chain_id"] if row["chain_id"] else "A"
                x, y, z = row["x"], row["y"], row["z"]

                line = (
                    f"ATOM  {atom_serial:5d} {atom_name:<4s} {residue_name:>3s} {chain_id:1s}"
                    f"{residue_num:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
                )
                f.write(line)
            f.write("END\n")


make_pdb(val_label, pdb_dir)


make_pdb(train_label, pdb_dir)


import PIL

PIL.Image.open("/kaggle/input/s3f-resources/RNA_chemical_structure.gif")


import py3Dmol


def load_pdb(pdb_file, hbondCutoff=4.0, width=800, height=600):
    """PDBファイルを読み込み、py3Dmolのビューを生成する"""
    with open(pdb_file, "r") as f:
        pdb_str = f.read()
    view = py3Dmol.view(js="https://3dmol.org/build/3Dmol.js", width=width, height=height)
    view.addModel(pdb_str, "pdb", {"hbondCutoff": hbondCutoff})
    return view


def apply_cartoon_style(view, filter_option, base_colors, radius):
    """cartoonスタイルの表示設定"""
    if filter_option == "C1":
        view.setStyle({"cartoon": {"color": "blue"}})
        for base, color in base_colors.items():
            view.setStyle({"resn": base}, {"sphere": {"color": color, "radius": radius}})
    else:
        view.setStyle({"cartoon": {"color": "spectrum"}})


def apply_chain_style(view, filter_option, chain, radius):
    """チェーン指定の場合の表示設定"""
    if filter_option == "C1":
        view.setStyle({"chain": chain}, {"sphere": {"color": "blue", "radius": radius}})
    else:
        view.setStyle({"chain": chain}, {"cartoon": {"color": "blue"}})


def apply_stick_style(view):
    """スティック（原子結合）スタイルの表示設定"""
    view.setStyle({}, {"stick": {}})


def plot_pdb(pdb_file, style="cartoon", filter_option="All", color_mode="rainbow",
             chain="A", hbondCutoff=4.0, radius=1.0):
    """
    PDBファイルを表示する関数。
    
    Parameters:
        pdb_file (str): PDBファイルのパス。
        style (str): 表示スタイル。'cartoon' または 'stick' など。
        filter_option (str): 表示フィルター（例："C1"）。
        color_mode (str): カラーモード。'rainbow' または 'chain'。
        chain (str): チェーン指定。
        hbondCutoff (float): 水素結合のカットオフ距離。
        radius (float): 球表示の半径。
    """
    # 基底ごとの色設定
    base_colors = {"A": "red", "U": "blue", "G": "green", "C": "orange"}
    
    view = load_pdb(pdb_file, hbondCutoff)
    
    if style == "cartoon":
        if color_mode == "rainbow":
            apply_cartoon_style(view, filter_option, base_colors, radius)
        elif color_mode == "chain":
            apply_chain_style(view, filter_option, chain, radius)
        else:
            raise ValueError("Invalid color_mode. Use 'rainbow' or 'chain'.")
    elif style == "stick":
        apply_stick_style(view)
    else:
        raise ValueError("Invalid style option. Use 'cartoon' or 'stick'.")
    
    view.zoomTo()
    return view


plot_pdb("pdb/R1156_A_1.pdb", filter_option="C1")


plot_pdb("pdb/R1156_A_2.pdb", filter_option="C1")


plot_pdb("/kaggle/input/s3f-sample-pdb-data-generated-by-rfdiffusion/test_0.pdb")


plot_pdb("/kaggle/input/s3f-sample-pdb-data-generated-by-rfdiffusion/test_0.pdb", style="stick")

