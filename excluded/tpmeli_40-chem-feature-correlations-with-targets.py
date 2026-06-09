# No internet allowed, so we need to install from our dataset
!python -c "import sys, platform; print(sys.version, sys.implementation.name, platform.machine())"

!pip install -q --no-index /kaggle/input/polymer-wheels-py311/itables-2.4.2-py3-none-any.whl
!pip install -q --no-index /kaggle/input/polymer-wheels-py311/pandas-2.0.3-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl
!pip install -q --no-index /kaggle/input/polymer-wheels-py311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


# ---------------------------------------------------------------------------
# 1. Imports & setup
# ---------------------------------------------------------------------------
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
import pandas as pd


from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

DESCRIPTOR_FUNCTIONS = [
    # --- Baseline (10) ------------------------------------------------------
    ("MolWt",                 Descriptors.MolWt),
    ("NumRotatableBonds",     Descriptors.NumRotatableBonds),
    ("NumHDonors",            Descriptors.NumHDonors),
    ("NumHAcceptors",         Descriptors.NumHAcceptors),
    ("TPSA",                  Descriptors.TPSA),
    ("MolLogP",               Descriptors.MolLogP),
    ("NumAromaticRings",      Descriptors.NumAromaticRings),
    ("NumAliphaticRings",     Descriptors.NumAliphaticRings),
    ("NumSaturatedRings",     Descriptors.NumSaturatedRings),
    ("NumHeteroatoms",        Descriptors.NumHeteroatoms),

    # --- Extra scalar chemistry metrics you were already using -------------
    ("ExactMolWt",            Descriptors.ExactMolWt),
    ("HeavyAtomCount",        Descriptors.HeavyAtomCount),
    ("NumValenceElectrons",   Descriptors.NumValenceElectrons),
    ("FormalCharge",          lambda m: Chem.GetFormalCharge(m)),
    ("FractionCSP3",          Descriptors.FractionCSP3),
    ("RingCount",             Descriptors.RingCount),
    ("LabuteASA",             Descriptors.LabuteASA),
    ("HeavyAtomMolWt",        Descriptors.HeavyAtomMolWt),
    ("NHOHCount",             Descriptors.NHOHCount),
    ("NOCount",               Descriptors.NOCount),
    ("MaxPartialCharge",      Descriptors.MaxPartialCharge),
    ("MinPartialCharge",      Descriptors.MinPartialCharge),
    ("BalabanJ",              Descriptors.BalabanJ),
    ("BertzCT",               Descriptors.BertzCT),
    ("HallKierAlpha",         Descriptors.HallKierAlpha),
    ("Chi0",                  Descriptors.Chi0),
    ("Chi1",                  Descriptors.Chi1),
    ("Chi1v",                 Descriptors.Chi1v),
    ("Kappa1",                Descriptors.Kappa1),
    ("Ipc",                   Descriptors.Ipc),
    ("MolMR",                 Descriptors.MolMR),
    ("FpDensityMorgan2",      Descriptors.FpDensityMorgan2),

    # --- Ring / stereo counts from rdMolDescriptors ------------------------
    ("NumRings",                          rdMolDescriptors.CalcNumRings),
    ("NumAromaticRings_rd",               rdMolDescriptors.CalcNumAromaticRings),
    ("NumAliphaticRings_rd",              rdMolDescriptors.CalcNumAliphaticRings),
    ("NumSaturatedRings_rd",              rdMolDescriptors.CalcNumSaturatedRings),
    ("NumHeterocycles",                   rdMolDescriptors.CalcNumHeterocycles),
    ("NumSpiroAtoms",                     rdMolDescriptors.CalcNumSpiroAtoms),
    ("NumBridgeheadAtoms",                rdMolDescriptors.CalcNumBridgeheadAtoms),
    ("NumAtomStereoCenters",              rdMolDescriptors.CalcNumAtomStereoCenters),
    ("NumUnspecifiedAtomStereoCenters",   rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters),

    # --- Bonus: total (assignedâ€¯+â€¯unassigned) chiral centres ---------------
    ("NumStereoCenters",
        lambda m: len(Chem.FindMolChiralCenters(m, includeUnassigned=True))),
]


# ---------------------------------------------------------------------------
# 3. Helper â€“Â convert one SMILES to an RDKit Mol safely
# ---------------------------------------------------------------------------
def smiles_to_mol(smiles: str):
    """
    Convert a SMILES to an RDKit Mol.
    Returns None if parsing fails (caller decides what to do next).
    """
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

# ---------------------------------------------------------------------------
# 4. Core function â€“Â add descriptor columns to a DataFrame
# ---------------------------------------------------------------------------
def add_descriptors(df: pd.DataFrame, smiles_col: str = "smiles",
                    na_value: float = float("nan")) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must contain a column with SMILES strings.
    smiles_col : str
        Name of that column.
    na_value : float
        What to put if a SMILES cannot be parsed.
    
    Returns
    -------
    pd.DataFrame
        Original df **plus 30 new columns** (one per descriptor).
    """
    # Preâ€‘allocate empty dict of lists
    new_cols = {name: [] for name, _ in DESCRIPTOR_FUNCTIONS}
    
    for smi in df[smiles_col]:
        mol = smiles_to_mol(smi)
        if mol is None:
            # Append na_value for every descriptor
            for name in new_cols:
                new_cols[name].append(na_value)
            continue
        
        for name, func in DESCRIPTOR_FUNCTIONS:
            try:
                new_cols[name].append(func(mol))
            except Exception:
                new_cols[name].append(na_value)
    
    # Concatenate sideâ€‘byâ€‘side
    return pd.concat([df.reset_index(drop=True),
                      pd.DataFrame(new_cols)], axis=1)


# ---------------------------------------------------------------------------
# 5. Run section (Kaggle-ready)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import os
    import pandas as pd

    # -----------------------------------------------------------------------
    # 5.1 Load the competition training data
    # -----------------------------------------------------------------------
    DATA_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
    if not os.path.isfile(DATA_PATH):
        raise FileNotFoundError(
            f"Expected training file not found at: {DATA_PATH}"
        )

    df_train = pd.read_csv(DATA_PATH)
    print(f"Loaded {df_train.shape[0]:,} polymers with "
          f"{df_train.shape[1]} columns from train.csv")

    # -----------------------------------------------------------------------
    # 5.2 Generate 30 chemistry descriptors for each SMILES string
    # -----------------------------------------------------------------------
    df_train_desc = add_descriptors(df_train, smiles_col="SMILES")
    print("Descriptor generation complete. "
          f"Resulting table now has {df_train_desc.shape[1]} columns.")

    # -----------------------------------------------------------------------
    # 5.3 Persist the augmented data for downstream modeling
    #     (Kaggle notebooks only have write access to /kaggle/working/)
    # -----------------------------------------------------------------------
    OUTPUT_PATH = "/kaggle/working/train_with_descriptors.csv"
    df_train_desc.to_csv(OUTPUT_PATH, index=False)
    print(f"Saved extended dataset -> {OUTPUT_PATH}")

    # Optional sanity peek
    print("\nPreview of the enriched data:")
    print(df_train_desc.head())


# ---------------------------------------------------------
# Correlation dashboard with itables using df_train_desc
# ---------------------------------------------------------
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, spearmanr

# If itables isn't installed, uncomment the next line in Kaggle:
# !pip install itables --quiet

from itables import show, init_notebook_mode, options

# Enable interactive DataTables everywhere
init_notebook_mode(all_interactive=True)

# -----------------------------------------------------------------
# 1. Grab the augmented dataset that already exists in memory
# -----------------------------------------------------------------
# Make sure df_train_desc is defined (run the descriptor cell first)
try:
    df = df_train_desc.copy()
except NameError as e:
    raise RuntimeError(
        "df_train_desc not found. Run the descriptor-generation cell before this one."
    ) from e

# -----------------------------------------------------------------
# 2. Identify target and descriptor columns
# -----------------------------------------------------------------
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
DESCRIPTORS = [c for c in df.columns if c not in TARGETS + ["id", "SMILES"]]

# -----------------------------------------------------------------
# 3. Helper: safe Pearson & Spearman (skip NaNs, need â‰¥3 pairs)
# -----------------------------------------------------------------
def _corr_pair(x, y):
    mask = x.notna() & y.notna()
    if mask.sum() < 3:
        return np.nan, np.nan
    return pearsonr(x[mask], y[mask])[0], spearmanr(x[mask], y[mask])[0]

# -----------------------------------------------------------------
# 4. Build the correlation table
# -----------------------------------------------------------------
records = []
for desc in DESCRIPTORS:
    row = {"Descriptor": desc}
    for tgt in TARGETS:
        p, s = _corr_pair(df[desc], df[tgt])
        row[f"{tgt}_Pearson"]  = p
        row[f"{tgt}_Spearman"] = s
    records.append(row)

corr_df = (
    pd.DataFrame(records)
      .set_index("Descriptor")
      .sort_index()
      .round(3)           # nice 3-decimal formatting
)

# -----------------------------------------------------------------
# 5. Style as a redâ†”blue heat-map
# -----------------------------------------------------------------
styled = (
    corr_df.style
           .background_gradient(cmap="coolwarm", vmin=-1, vmax=1, axis=None)
           .format("{:.3f}")
)

# itables needs explicit permission to render Styler HTML
options.allow_html = True   # <-- fix for â€œallow_htmlâ€� warning

# Display: scrollable, 25 rows per page, columns toggleable


show(
    styled,
    lengthMenu=[10, 25, 50, 100],
    pageLength=42,
    scrollX=True,
    scrollY=True,
    fixedColumns={"start": 1}, # freeze 1 column from the left (alias: {"left": 1})
    fixedHeader=True  
)


corr_df.to_csv("/kaggle/working/descriptor_target_correlations.csv", index=True)


import os
import warnings
import numpy as np
import pandas as pd
from rdkit import Chem
from tqdm.auto import tqdm

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import xgboost as xgb

warnings.filterwarnings("ignore")

# --------------------- helper: safe SMILES â†’ Mol ---------------------------
def smiles_to_mol(smiles: str):
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None

# --------------------- helper: descriptor matrix ---------------------------
def featurise(smiles_list):
    """
    Builds a DataFrame of descriptor values for each SMILES string.
    Missing or invalid molecules -> row of np.nan.
    """
    feats = []
    for smi in tqdm(smiles_list, desc="Extracting descriptors"):
        mol = smiles_to_mol(smi)
        if mol is None:
            row = [np.nan] * len(DESCRIPTOR_FUNCTIONS)
        else:
            row = []
            for _, fn in DESCRIPTOR_FUNCTIONS:
                try:
                    row.append(fn(mol))
                except Exception:
                    row.append(np.nan)
        feats.append(row)
    cols = [name for name, _ in DESCRIPTOR_FUNCTIONS]
    return pd.DataFrame(feats, columns=cols)

# --------- intelligent multivariate imputation for descriptors ----------
def impute_descriptors_iterative(X_train: pd.DataFrame, X_test: pd.DataFrame):
    """
    - Replaces Â±inf with NaN.
    - Uses IterativeImputer (BayesianRidge) to predict missing entries.
    """
    train = X_train.replace([np.inf, -np.inf], np.nan)
    test = X_test.replace([np.inf, -np.inf], np.nan)

    imputer = IterativeImputer(
        estimator=BayesianRidge(),
        max_iter=10,
        initial_strategy='median',
        random_state=42
    )

    train_imp = pd.DataFrame(
        imputer.fit_transform(train),
        columns=X_train.columns, index=X_train.index
    )
    test_imp = pd.DataFrame(
        imputer.transform(test),
        columns=X_test.columns, index=X_test.index
    )

    return train_imp, test_imp

# ----------------------------- data paths ----------------------------------
TRAIN = "/kaggle/input/neurips-open-polymer-prediction-2025/train.csv"
TEST  = "/kaggle/input/neurips-open-polymer-prediction-2025/test.csv"
assert os.path.isfile(TRAIN) and os.path.isfile(TEST), "Train/test CSVs not found."

train_df = pd.read_csv(TRAIN)
test_df  = pd.read_csv(TEST)

# ---------------------- create descriptor tables ---------------------------
X_train = featurise(train_df["SMILES"])
X_test  = featurise(test_df["SMILES"])

# ------------------ impute missing / infinite values ----------------------
X_train_imp, X_test_imp = impute_descriptors_iterative(X_train, X_test)

# -------------------- set up modeling & submission ------------------------
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
preds = {}

xgb_params = dict(
    n_estimators=1500,
    learning_rate=0.05,
    max_depth=7,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:absoluteerror",
    random_state=42,
    tree_method="hist"
)

for tgt in TARGETS:
    y = train_df[tgt]
    mask = y.notna()
    X = X_train_imp.loc[mask].values
    y_vals = y[mask].values
    X_test_vals = X_test_imp.values

    # scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test_vals)

    # train/validation split
    Xtr, Xva, ytr, yva = train_test_split(
        X_scaled, y_vals, test_size=0.2, random_state=42
    )

    model = xgb.XGBRegressor(**xgb_params)
    model.fit(
        Xtr, ytr,
        eval_set=[(Xva, yva)],
        early_stopping_rounds=100,
        verbose=False
    )

    preds[tgt] = model.predict(X_test_scaled)

    # log validation MAE
    y_pred_va = model.predict(Xva)
    print(f"{tgt} â€“ valid MAE: {mean_absolute_error(yva, y_pred_va):.4f}")

# ------------------------ build submission file ----------------------------
submission = pd.DataFrame({"id": test_df["id"], **preds})
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"\nâœ… Saved submission: {out_path}")

