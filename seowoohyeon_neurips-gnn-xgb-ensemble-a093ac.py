!cp -r /kaggle/input/autogluon-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/autogluon-package' 'autogluon.tabular-1.3.1-py3-none-any.whl'


!cp -r /kaggle/input/scikit-package/* /kaggle/working/
!pip install -f --quiet --no-index --find-links='/kaggle/input/scikit-package' 'scikit_learn-1.5.2-cp311-cp311-manylinux_2_17_x86_64.manylinux2014_x86_64.whl' 



from autogluon.tabular import TabularDataset, TabularPredictor


!pip install /kaggle/input/rdkit-2025-3-3-cp311/rdkit-2025.3.3-cp311-cp311-manylinux_2_28_x86_64.whl


!pip install mordred --no-index --find-links=file:///kaggle/input/mordred-1-2-0-py3-none-any/


!rm -rf /kaggle/working/*


BASE_PATH = '/kaggle/input/neurips-open-polymer-prediction-2025/'


output_dfs = []


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# ============================================================
# Inference-only (TRAIN íŒŒì�´í”„ë�¼ì�¸ ê·¸ëŒ€ë¡œ ë°˜ì˜�)
# - ì €ì�¥ë�œ ëª¨ë�¸(fold_XX/xgb.json, lgb.txt, meta.json, features.json) ë¡œë“œ
# - í•™ìŠµê³¼ ë�™ì�¼í•œ íŠ¹ì§• ìƒ�ì„±(ì¹´ìš´íŠ¸/ê·¸ë�˜í”„/íŒŒìƒ� + explicit-H per-atom/per-600/per-kDa + alias)
# - ê°� í�´ë“œ ë©”íƒ€(rollback, best_w) ë°˜ì˜�í•´ ì˜ˆì¸¡ í›„ fold í�‰ê· 
# - xgblgb_submission.csv ì €ì�¥
# ============================================================

import os, re, json, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

# ML libs
import xgboost as xgb
import lightgbm as lgb

# RDKit & features
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdmolops, MACCSkeys
from rdkit.Chem.Descriptors import MolWt, MolLogP
from rdkit.Chem.rdMolDescriptors import CalcTPSA, CalcNumRotatableBonds
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
import networkx as nx

# (Optional) BERT â€“ í•™ìŠµì—�ì„œ Falseì˜€ìœ¼ë©´ ì—¬ê¸°ì„œë�„ False ê¶Œì�¥
import torch
try:
    from transformers import AutoTokenizer, AutoModel, AutoModelForMaskedLM
except Exception as e:
    print("âš ï¸� transformers í•„ìš” ì‹œë§Œ ì‚¬ìš©:", e)

# =========================
# Paths / Config
# =========================
BASE_PATH = "/kaggle/input/neurips-open-polymer-prediction-2025/"
CANDIDATE_MODEL_DIRS = [
    "/kaggle/input/neurips-models/gbdt_models"   # â†� í•™ìŠµ ì‹œ ì €ì�¥í•œ ë£¨íŠ¸
]

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
BOUNDS = {
    "Tg": (-158.0297376, 482.25),
    "FFV": (0.2069924, 0.79709707),
    "Tc": (0.0365, 1.69),
    "Density": (0.648691234, 1.940998909),
    "Rg": (8.7283551, 35.672905605),
}
EPS = 1e-12

# =========================
# (Optional) BERT í† ê¸€/ì„¤ì •
# =========================
USE_BERT = False            # í•™ìŠµì�´ Falseë©´ ê·¸ëŒ€ë¡œ False ê¶Œì�¥
BERT_DIR = "/kaggle/input/bert_smile/pytorch/default/1"
BERT_POOLING = "cls"        # "cls" | "mean"
BERT_LAYER = -1
BERT_MAX_LEN = 256
BERT_BATCH = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_smiles_bert(model_dir=BERT_DIR, device=device):
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    try:
        mdl = AutoModel.from_pretrained(model_dir)
    except Exception:
        mdl = AutoModelForMaskedLM.from_pretrained(model_dir)
    mdl.to(device).eval()
    return tok, mdl

@torch.no_grad()
def bert_embed_smiles(smiles_list, tokenizer, model,
                      batch_size=BERT_BATCH, pooling=BERT_POOLING,
                      layer=BERT_LAYER, max_len=BERT_MAX_LEN, device=device):
    if not smiles_list:
        return np.zeros((0,1), dtype=np.float32)
    embs = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        inputs = tokenizer(batch, padding=True, truncation=True, max_length=max_len, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        outputs = model(**inputs, output_hidden_states=True)
        hs = outputs.hidden_states[layer] if layer is not None else outputs.last_hidden_state
        if pooling == "mean":
            mask = inputs["attention_mask"].unsqueeze(-1)
            pooled = (hs * mask).sum(1) / mask.sum(1).clamp(min=1)
        else:
            pooled = hs[:, 0, :]  # CLS
        embs.append(pooled.detach().cpu().numpy().astype(np.float32))
    return np.vstack(embs)

# =========================
# ëª¨ë�¸ ë£¨íŠ¸ íƒ�ìƒ‰
# =========================
def find_models_root(candidates):
    for p in candidates:
        if os.path.isdir(p):
            ok = True
            for t in TARGETS:
                if not os.path.isdir(os.path.join(p, t)):
                    ok = False; break
            if ok:
                return p
    raise FileNotFoundError(
        "ëª¨ë�¸ ê²½ë¡œë¥¼ ì°¾ì§€ ëª»í–ˆìŠµë‹ˆë‹¤. í›„ë³´ë“¤ì�„ ì‹¤ì œ ê²½ë¡œë¡œ ë°”ê¿”ì£¼ì„¸ìš”:\n" +
        "\n".join(f" - {c}" for c in candidates)
    )

MODELS_IN_DIR = find_models_root(CANDIDATE_MODEL_DIRS)
print("Using models from:", MODELS_IN_DIR)

# =========================
# í•„í„° & íŒŒìƒ� ì�˜ì¡´ì„± (í•™ìŠµê³¼ ë�™ì�¼)
# =========================
required_descriptors = {
    'graph_diameter','count_C','C_ratio','count_c',
    'num_cycles','avg_shortest_path','MolWt',
    'LogP','TPSA','RotatableBonds','NumAtoms',
    "count_cnumcc"
}

DERIVED_DEPS = {
    'SMR_VSA5_div_MolWt_div_fr_nitro': {'SMR_VSA5', 'fr_nitro', 'MolWt'},
    'PEOE_VSA14_div_SlogP_VSA7': {'PEOE_VSA14', 'SlogP_VSA7'},
    'qed_mul_SMR_VSA5': {'qed', 'SMR_VSA5'},
    'fr_unbrch_alkane_div_MolWt_div_EState_VSA11': {'fr_unbrch_alkane','MolWt','EState_VSA11'},
    'PEOE_VSA14_div_graph_diameter_div_BalabanJ': {'PEOE_VSA14','graph_diameter','BalabanJ'},
    'VSA_EState7_div_SPS_div_PEOE_VSA14': {'VSA_EState7','SPS','PEOE_VSA14'},

    'count_C_div_MolWt': {'count_C', 'MolWt'},
    'MolWt_div_NumAtoms': {'MolWt', 'NumAtoms'},
    'NumAtoms_div_MolWt': {'NumAtoms', 'MolWt'},
    'RotatableBonds_div_MolWt': {'RotatableBonds', 'MolWt'},
    'C_ratio_mul_expC_ratio': {'C_ratio'},
    'TPSA_div_MolWt': {'TPSA', 'MolWt'},
    'LogP_div_NumAtoms': {'LogP', 'NumAtoms'},
    'exp_MolWt_mul_exp_MolWt': {'MolWt'},
    'C_ratio_mul_graph_diameter': {'C_ratio', 'graph_diameter'},
    'LogP_mul_C_ratio': {'LogP', 'C_ratio'},
    'graph_diameter_mul_avg_shortest_path': {'graph_diameter', 'avg_shortest_path'},
    'exp_MolWt_mul_MolWt': {'MolWt'},
    'RotatableBonds_mul_TPSA': {'RotatableBonds', 'TPSA'},

    # ìƒ�ìœ„ í›„ë³´ë“¤
    'num_cycles/avg_shortest_path': {'num_cycles', 'avg_shortest_path'},
    'count_c/avg_shortest_path': {'count_c', 'avg_shortest_path'},
    'C_ratio/exp(C_ratio)': {'C_ratio'},
    'TPSA/MolWt': {'TPSA', 'MolWt'},
    'LogP/NumAtoms': {'LogP', 'NumAtoms'},
    'C_ratio*graph_diameter': {'C_ratio', 'graph_diameter'},
    'RotatableBonds/MolWt': {'RotatableBonds', 'MolWt'},
    'LogP*C_ratio': {'LogP', 'C_ratio'},
    'count_C/MolWt': {'count_C', 'MolWt'},
    'MolWt/NumAtoms': {'MolWt', 'NumAtoms'},
    'exp(C_ratio)*C_ratio': {'C_ratio'},
    'NumAtoms/MolWt': {'NumAtoms', 'MolWt'},
    'graph_diameter*avg_shortest_path': {'graph_diameter', 'avg_shortest_path'},
    'RotatableBonds*TPSA': {'RotatableBonds', 'TPSA'},
}

filters = {
    # â€”â€”â€” ì•„ë�˜ ë¸”ë¡�ì�€ ì§ˆë¬¸ì—�ì„œ ì£¼ì‹  filters ê·¸ëŒ€ë¡œ â€”
    'Tg': list(set([
        'BalabanJ','Chi1','Chi3n','Chi4n','EState_VSA4','EState_VSA8',
        'FpDensityMorgan3','HallKierAlpha','MaxAbsEStateIndex','MolLogP',
        'NumAmideBonds','NumHeteroatoms','NumHeterocycles','NumRotatableBonds',
        'PEOE_VSA14','Phi','RingCount','SMR_VSA1','SPS','SlogP_VSA1','SlogP_VSA5',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState4','VSA_EState6','VSA_EState7',
        'VSA_EState8','fr_C_O_noCOO','fr_NH1','fr_benzene','fr_bicyclic','fr_ether',
        'fr_unbrch_alkane','sum_of_digits_ratio','C_over_c','c_ratio','Kappa3','BertzCT',
        'SMR_VSA5_div_MolWt_div_fr_nitro','sum_of_digits_ratio_div_MolWt','max_C_run',
        "RotatableBonds_div_NumAtoms","avg_shortest_path_div_NumAtoms","graph_diameter_div_NumAtoms",
        "num_cycles_div_NumAtoms","count_c_div_MolWt",'diameter_per_atom_heavy','fracH','TPSA_per_600atoms_heavy',
        'fracF','count_C_per_kDa',
        'num_cycles/avg_shortest_path','count_c/avg_shortest_path','C_ratio/exp(C_ratio)'
    ]).union(required_descriptors)),
    'FFV': list(set([
        'AvgIpc','BalabanJ','BertzCT','Chi0','Chi0n','Chi0v','Chi1','Chi1n','Chi1v',
        'Chi2n','Chi2v','Chi3n','Chi3v','Chi4n','EState_VSA10','EState_VSA5',
        'EState_VSA7','EState_VSA8','EState_VSA9','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','FractionCSP3','HallKierAlpha',
        'HeavyAtomMolWt','Kappa1','Kappa2','Kappa3','MaxAbsEStateIndex',
        'MaxEStateIndex','MinEStateIndex','MolLogP','MolMR','MolWt','NHOHCount',
        'NOCount','NumAromaticHeterocycles','NumHAcceptors','NumHDonors',
        'NumHeterocycles','NumRotatableBonds','PEOE_VSA14','RingCount','SMR_VSA1',
        'SMR_VSA10','SMR_VSA3','SMR_VSA5','SMR_VSA6','SMR_VSA7','SMR_VSA9','SPS',
        'SlogP_VSA1','SlogP_VSA10','SlogP_VSA11','SlogP_VSA12','SlogP_VSA2',
        'SlogP_VSA3','SlogP_VSA4','SlogP_VSA5','SlogP_VSA6','SlogP_VSA7',
        'SlogP_VSA8','TPSA','VSA_EState1','VSA_EState10','VSA_EState2',
        'VSA_EState3','VSA_EState4','VSA_EState5','VSA_EState6','VSA_EState7',
        'VSA_EState8','VSA_EState9','fr_Ar_N','fr_C_O','fr_NH0','fr_NH1',
        'fr_aniline','fr_ether','fr_halogen','fr_thiophene','FNOCl_ratio',
        'sum_of_digits_ratio','sum_of_digits_ratio_div_MolWt','c_ratio',
        'SMR_VSA5_div_MolWt_div_fr_nitro','count_at','count_slash','count_CC_star',
        "num_cycles_div_NumAtoms","count_c_div_MolWt","count_C_div_MolWt",
        'fracO','O_over_C','cycles_per_600atoms_heavy',
        'TPSA/MolWt','LogP/NumAtoms'
    ]).union(required_descriptors)),
    'Tc': list(set([
        'BalabanJ','BertzCT','Chi0','EState_VSA5','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HeavyAtomMolWt','MinEStateIndex',
        'MolWt','NumRotatableBonds','SMR_VSA10','SPS','SlogP_VSA6','VSA_EState1',
        'VSA_EState7','fr_NH1','fr_ester','fr_halogen','max_C_run','max_c_run','C_over_c',
        'FNOCl_ratio','count_F','count_CC_star','count_Cl',"avg_shortest_path_div_NumAtoms",
        'fracO','fracC','fracS','N_over_C','diameter_per_600atoms_heavy',
        'exp(MolWt)*exp(MolWt)','C_ratio*graph_diameter','RotatableBonds/MolWt','LogP*C_ratio'
    ]).union(required_descriptors)),
    'Density': list(set([
        'BalabanJ','Chi3n','Chi3v','Chi4n','EState_VSA1','ExactMolWt',
        'FractionCSP3','HallKierAlpha','Kappa2','MinEStateIndex','MolMR','MolWt',
        'NumAliphaticCarbocycles','NumHAcceptors','NumHeteroatoms',
        'NumRotatableBonds','SMR_VSA10','SMR_VSA5','SlogP_VSA12','SlogP_VSA5',
        'TPSA','VSA_EState10','VSA_EState7','VSA_EState8','FNOCl_ratio',
        'PEOE_VSA14_div_SlogP_VSA7','count_Cl','count_CC_star','max_C_run',
        "RotatableBonds_div_NumAtoms","avg_shortest_path_div_NumAtoms",
        "graph_diameter_div_NumAtoms","count_c_div_MolWt","count_C_div_MolWt",
        'TPSA_per_atom_explicit','fracH','nH','cycles_per_600atoms_heavy',
        'count_C/MolWt','MolWt/NumAtoms','RotatableBonds/MolWt','exp(C_ratio)*C_ratio','NumAtoms/MolWt'
    ]).union(required_descriptors)),
    'Rg': list(set([
        'AvgIpc','Chi0n','Chi1v','Chi2n','Chi3v','ExactMolWt','FpDensityMorgan1',
        'FpDensityMorgan2','FpDensityMorgan3','HallKierAlpha','HeavyAtomMolWt',
        'Kappa3','MaxAbsEStateIndex','MolWt','NOCount','NumRotatableBonds',
        'NumUnspecifiedAtomStereoCenters','NumValenceElectrons','PEOE_VSA14',
        'SMR_VSA1','SMR_VSA5','SPS','SlogP_VSA1','SlogP_VSA2','VSA_EState1',
        'VSA_EState8','fr_alkyl_halide','fr_halogen','C_over_c','c_ratio',
        'SlogP_VSA7','PEOE_VSA6','VSA_EState8','PEOE_VSA14_mul_AvgIpc',
        'qed_mul_SMR_VSA5','count_CeqO','C_ratio',"avg_shortest_path_div_NumAtoms",
        'fracN','avgSP_per_atom_heavy','nO','LogP_per_atom_heavy','TPSA_per_atom_explicit',
        'num_cycles_per_kDa','graph_diameter*avg_shortest_path','exp(MolWt)*MolWt','RotatableBonds*TPSA'
    ]).union(required_descriptors)),
}

def _expand_selected(selected):
    sel = set(selected or [])
    for name in list(sel):
        if name in DERIVED_DEPS:
            sel |= DERIVED_DEPS[name]
    sel |= {'MolWt', 'MolLogP'}
    return list(sel)

# =========================
# í•™ìŠµê³¼ ë�™ì�¼í•œ íŠ¹ì§• ìƒ�ì„± í•¨ìˆ˜
# (counts/graph/derived + explicit-H per-atom/per-600/per-kDa + alias)
# =========================
def smiles_to_features(
    smiles_list,
    selected_descriptors,
    radius=2, n_bits=128,
    add_counts=True, add_graph=True, add_derived=True,
    add_alias=True,   # í•™ìŠµê³¼ ë�™ì�¼: RotatableBondsâ†’NumRotatableBonds ë³„ì¹­
    eps=1e-3
):
    gen = GetMorganGenerator(radius=radius, fpSize=n_bits)
    selected_descriptors = _expand_selected(selected_descriptors)
    rd_desc = {name: func for name, func in Descriptors.descList if name in selected_descriptors}

    fps = []
    rows = []

    def longest_run(s: str, token: str) -> int:
        if not isinstance(s, str) or not token:
            return 0
        n = len(token)
        if n == 1:
            mx = cur = 0
            for ch in s:
                if ch == token:
                    cur += 1; mx = max(mx, cur)
                else:
                    cur = 0
            return mx
        mx = cur = 0; i = 0; L = len(s)
        while i <= L - n:
            if s[i:i+n] == token:
                cur += 1; mx = max(mx, cur); i += n
            else:
                cur = 0; i += 1
        return mx

    for s in smiles_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(np.zeros(n_bits + 167, dtype=int))
            rows.append({})
            continue

        # Morgan + MACCS
        morgan = np.array(gen.GetFingerprint(mol), dtype=int)
        maccs  = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=int)
        fps.append(np.concatenate([morgan, maccs]))

        row = {}
        # RDKit desc
        for name, func in rd_desc.items():
            try:
                row[name] = func(mol)
            except Exception:
                row[name] = np.nan

        # ê¸°ë³¸ì¹˜ + alias
        row['MolWt'] = MolWt(mol)
        row['MolLogP'] = MolLogP(mol)
        row['LogP'] = row['MolLogP']
        row['TPSA'] = CalcTPSA(mol)
        row['NumAtoms'] = mol.GetNumAtoms()     # heavy atoms
        row['RotatableBonds'] = CalcNumRotatableBonds(mol)
        if add_alias and 'NumRotatableBonds' in selected_descriptors:
            row['NumRotatableBonds'] = row['RotatableBonds']

        # ë¬¸ì�� ì¹´ìš´íŠ¸
        if add_counts:
            count_C = s.count('C'); count_c = s.count('c'); L = len(s)
            count_F = s.count('F'); count_N = s.count('N'); count_O = s.count('O')
            count_Cl_any = len(re.findall(r"Cl", s, flags=re.I))
            digits = re.findall(r'\d', s); sod = sum(map(int, digits))
            row.update({
                'count_C': count_C, 'count_c': count_c,
                'C_over_c': count_C/(count_c+1), 'C_minus_c': count_C-count_c,
                'C_ratio': (count_C+1)/(L+1), 'c_ratio': (count_c+1)/(L+1),
                'F_ratio': (count_F+1)/(L+1),
                'FNOCl_ratio': (count_F+count_N+count_O+count_Cl_any+1)/(L+1),
                'max_C_run': longest_run(s, 'C'), 'max_c_run': longest_run(s, 'c'),
                'sum_of_digits_ratio': sod/(L+EPS),
                'count_cnumcc': len(re.findall(r"c\d+cc", s)),
                'count_CC_star': len(re.findall(r"CC\(\*\)", s)),
                'count_F': count_F, 'count_Cl': count_Cl_any,
                'count_CeqO': len(re.findall(r"C\(=O\)", s)),
                'count_at': s.count('@'), 'count_slash': s.count('/'),
            })

        # ê·¸ë�˜í”„ íŠ¹ì„±
        if add_graph:
            try:
                adj = rdmolops.GetAdjacencyMatrix(mol)
                G = nx.from_numpy_array(adj)
                if G.number_of_nodes() == 0 or not nx.is_connected(G):
                    gd = 0.0; sp = 0.0
                else:
                    gd = float(nx.diameter(G))
                    sp = float(nx.average_shortest_path_length(G))
                row['graph_diameter'] = gd
                row['avg_shortest_path'] = sp
                row['num_cycles'] = float(len(list(nx.cycle_basis(G))))
            except Exception:
                row['graph_diameter'] = 0.0
                row['avg_shortest_path'] = 0.0
                row['num_cycles'] = 0.0

        # ----- explicit H ê¸°ë°˜ per-atom/per-600/per-kDa (í•™ìŠµê³¼ ë�™ì�¼) -----
        try:
            na_heavy = float(row.get('NumAtoms', np.nan))
            mw = float(row.get('MolWt', np.nan))
            tpsa = float(row.get('TPSA', np.nan))
            logp = float(row.get('LogP', np.nan))
            gd = float(row.get('graph_diameter', 0.0))
            sp = float(row.get('avg_shortest_path', 0.0))
            nc = float(row.get('num_cycles', 0.0))

            mol_H = Chem.AddHs(mol)
            elem_counts = {}
            for a in mol_H.GetAtoms():
                sym = a.GetSymbol()
                elem_counts[sym] = elem_counts.get(sym, 0) + 1

            nC = float(elem_counts.get('C', 0))
            nH = float(elem_counts.get('H', 0))
            nN = float(elem_counts.get('N', 0))
            nO = float(elem_counts.get('O', 0))
            nF = float(elem_counts.get('F', 0))
            nCl = float(elem_counts.get('Cl', 0))
            nS = float(elem_counts.get('S', 0))
            nP = float(elem_counts.get('P', 0))
            tot_atoms_explicit = float(sum(elem_counts.values()))

            def safe_div(a, b): return float(a) / (float(b) + eps)

            row['nC'] = nC; row['nH'] = nH; row['nN'] = nN; row['nO'] = nO
            row['nF'] = nF; row['nCl'] = nCl; row['nS'] = nS; row['nP'] = nP
            row['NumAtoms_explicitH'] = tot_atoms_explicit

            row['fracC'] = safe_div(nC, tot_atoms_explicit)
            row['fracH'] = safe_div(nH, tot_atoms_explicit)
            row['fracN'] = safe_div(nN, tot_atoms_explicit)
            row['fracO'] = safe_div(nO, tot_atoms_explicit)
            row['fracF'] = safe_div(nF, tot_atoms_explicit)
            row['fracCl'] = safe_div(nCl, tot_atoms_explicit)
            row['fracS'] = safe_div(nS, tot_atoms_explicit)
            row['fracP'] = safe_div(nP, tot_atoms_explicit)

            row['C_over_H'] = safe_div(nC, nH)
            row['O_over_C'] = safe_div(nO, nC)
            row['N_over_C'] = safe_div(nN, nC)
            row['Halogen_frac'] = safe_div(nF + nCl, tot_atoms_explicit)

            if np.isfinite(na_heavy) and na_heavy > 0:
                row['TPSA_per_atom_heavy'] = tpsa / (na_heavy + eps)
                row['LogP_per_atom_heavy'] = logp / (na_heavy + eps)
                row['diameter_per_atom_heavy'] = gd / (na_heavy + eps)
                row['avgSP_per_atom_heavy'] = sp / (na_heavy + eps)
                row['cycles_per_atom_heavy'] = nc / (na_heavy + eps)
                row['TPSA_per_600atoms_heavy'] = 600.0 * row['TPSA_per_atom_heavy']
                row['cycles_per_600atoms_heavy'] = 600.0 * row['cycles_per_atom_heavy']
                row['diameter_per_600atoms_heavy'] = 600.0 * row['diameter_per_atom_heavy']
                row['avgSP_per_600atoms_heavy'] = 600.0 * row['avgSP_per_atom_heavy']

            if np.isfinite(tot_atoms_explicit) and tot_atoms_explicit > 0:
                row['TPSA_per_atom_explicit'] = tpsa / (tot_atoms_explicit + eps)
                row['LogP_per_atom_explicit'] = logp / (tot_atoms_explicit + eps)
                row['TPSA_per_600atoms_explicit'] = 600.0 * row['TPSA_per_atom_explicit']
                row['LogP_per_600atoms_explicit'] = 600.0 * row['LogP_per_atom_explicit']

            if np.isfinite(mw) and mw > 0:
                row['TPSA_per_kDa'] = tpsa / (mw/1000.0 + eps)
                if 'count_C' in row:
                    row['count_C_per_kDa'] = row['count_C'] / (mw/1000.0 + eps)
                if 'count_c' in row:
                    row['count_c_per_kDa'] = row['count_c'] / (mw/1000.0 + eps)
                row['num_cycles_per_kDa'] = nc / (mw/1000.0 + eps)
        except Exception:
            pass

        # íŒŒìƒ�(ë�¼ë²¨ ê³µí†µ)
        def g(k, default=np.nan): return row[k] if k in row else default
        mw = g('MolWt'); na = g('NumAtoms'); rb = g('RotatableBonds')
        gd = g('graph_diameter'); sp = g('avg_shortest_path'); nc = g('num_cycles')

        if 'count_C' in row: row['count_C_div_MolWt'] = row['count_C']/(mw+EPS)
        if 'count_c' in row: row['count_c_div_MolWt'] = row['count_c']/(mw+EPS)
        if 'sum_of_digits_ratio' in row: row['sum_of_digits_ratio_div_MolWt'] = row['sum_of_digits_ratio']/(mw+EPS)
        row['RotatableBonds_div_NumAtoms'] = rb/(na+EPS)
        row['num_cycles_div_NumAtoms'] = nc/(na+EPS)
        row['graph_diameter_div_NumAtoms'] = gd/(na+EPS)
        row['avg_shortest_path_div_NumAtoms'] = sp/(na+EPS)

        if 'PEOE_VSA14' in row and 'SlogP_VSA7' in row:
            row['PEOE_VSA14_div_SlogP_VSA7'] = row['PEOE_VSA14']/(row['SlogP_VSA7']+EPS)
        if 'PEOE_VSA14' in row and 'AvgIpc' in row:
            row['PEOE_VSA14_mul_AvgIpc'] = row['PEOE_VSA14']*row['AvgIpc']
        if 'qed' in row and 'SMR_VSA5' in row:
            row['qed_mul_SMR_VSA5'] = row['qed']*row['SMR_VSA5']
        if {'SMR_VSA5','fr_nitro'}.issubset(row):
            row['SMR_VSA5_div_MolWt_div_fr_nitro'] = (row['SMR_VSA5']/(mw+EPS))/(row['fr_nitro']+EPS)
        if {'fr_unbrch_alkane','EState_VSA11'}.issubset(row):
            row['fr_unbrch_alkane_div_MolWt_div_EState_VSA11'] = (row['fr_unbrch_alkane']/(mw+EPS))/(row['EState_VSA11']+EPS)
        if {'PEOE_VSA14','BalabanJ'}.issubset(row):
            row['PEOE_VSA14_div_graph_diameter_div_BalabanJ'] = (row['PEOE_VSA14']/(gd+EPS))/(row['BalabanJ']+EPS)
        if {'VSA_EState7','SPS','PEOE_VSA14'}.issubset(row):
            row['VSA_EState7_div_SPS_div_PEOE_VSA14'] = (row['VSA_EState7']/(row['SPS']+EPS))/(row['PEOE_VSA14']+EPS)

        if np.isfinite(nc) and np.isfinite(sp):
            row['num_cycles/avg_shortest_path'] = nc/(sp+EPS)
            row['graph_diameter*avg_shortest_path'] = gd*sp if np.isfinite(gd) else np.nan
        if 'count_c' in row and np.isfinite(sp):
            row['count_c/avg_shortest_path'] = row['count_c']/(sp+EPS)
        if 'C_ratio' in row:
            ec = float(np.exp(row['C_ratio']))
            row['C_ratio/exp(C_ratio)'] = row['C_ratio']/(ec+EPS)
            row['C_ratio_mul_expC_ratio'] = row['C_ratio']*ec
        if 'TPSA' in row and np.isfinite(mw):
            row['TPSA/MolWt'] = row['TPSA']/(mw+EPS)
        if 'LogP' in row and np.isfinite(na):
            row['LogP/NumAtoms'] = row['LogP']/(na+EPS)
            row['LogP_per_atom_heavy'] = row['LogP']/(na+EPS)
        if 'C_ratio' in row and np.isfinite(gd):
            row['C_ratio*graph_diameter'] = row['C_ratio']*gd
        if np.isfinite(rb) and np.isfinite(mw):
            row['RotatableBonds/MolWt'] = rb/(mw+EPS)
            row['RotatableBonds*TPSA'] = rb*row.get('TPSA', np.nan)
        if 'LogP' in row and 'C_ratio' in row:
            row['LogP*C_ratio'] = row['LogP']*row['C_ratio']
        if 'count_C' in row and np.isfinite(mw):
            row['count_C/MolWt'] = row['count_C']/(mw+EPS)
        if np.isfinite(mw) and np.isfinite(na):
            row['MolWt/NumAtoms'] = mw/(na+EPS)
            row['NumAtoms/MolWt'] = na/(mw+EPS)
        if 'C_ratio' in row:
            row['exp(C_ratio)*C_ratio'] = float(np.exp(row['C_ratio']))*row['C_ratio']

        rows.append(row)

    fp_cols = [f"FP_{i}" for i in range(len(fps[0]))]
    X_fp   = pd.DataFrame(np.asarray(fps, dtype=int), columns=fp_cols)
    X_desc = pd.DataFrame(rows)
    X_all  = pd.concat([X_desc.reset_index(drop=True), X_fp.reset_index(drop=True)], axis=1)
    return X_all

# =========================
# Load competition test
# =========================
print("Loading test.csv ...")
test = pd.read_csv(os.path.join(BASE_PATH, "test.csv"))
test_ids = test["id"].to_numpy()
test_smiles = test["SMILES"].astype(str).tolist()

# (Optional) BERT ì�„ë² ë”© ì‚¬ì „ ê³„ì‚°
bert_tokenizer = bert_model = None
TEST_BERT_EMB = None
if USE_BERT:
    try:
        bert_tokenizer, bert_model = load_smiles_bert()
        print("âœ… SMILES-BERT loaded.")
        TEST_BERT_EMB = bert_embed_smiles(test_smiles, bert_tokenizer, bert_model,
                                          batch_size=BERT_BATCH, pooling=BERT_POOLING,
                                          layer=BERT_LAYER, max_len=BERT_MAX_LEN, device=device)
        print(f"Test BERT embeddings shape: {TEST_BERT_EMB.shape}")
    except Exception as e:
        print("âš ï¸� BERT ë¡œë“œ/ì�„ë² ë”© ì‹¤íŒ¨ â†’ USE_BERT=False:", e)
        USE_BERT = False
        TEST_BERT_EMB = None

# =========================
# Helpers to read model files
# =========================
def load_features_json(target_dir):
    f = os.path.join(target_dir, "features.json")
    if not os.path.exists(f):
        raise FileNotFoundError(f"[{target_dir}] features.json ì—†ì�Œ")
    with open(f, "r") as fh:
        feats = json.load(fh)
    if not isinstance(feats, list) or not feats:
        raise ValueError(f"[{target_dir}] features.json í˜•ì‹� ì˜¤ë¥˜")
    return feats

def list_fold_dirs(target_dir):
    folds = []
    for nm in os.listdir(target_dir):
        p = os.path.join(target_dir, nm)
        if os.path.isdir(p) and re.match(r"^fold_\d{2}$", nm):
            folds.append((nm, p))
    folds = sorted(folds, key=lambda x: x[0])
    if not folds:
        raise FileNotFoundError(f"[{target_dir}] fold_XX ë””ë ‰í„°ë¦¬ë¥¼ ì°¾ì�„ ìˆ˜ ì—†ì�Œ")
    return folds

# (ì„ íƒ�) BERT ì—´ ëŒ€ì�‘: í•™ìŠµì�´ BERT_PCAì˜€ëŠ”ë�° ì¶”ë¡  PCAê°€ ì—†ì�„ ë•Œì�˜ ì�„ì‹œ ë§¤ì¹­
def attach_bert_features_no_pca(X_all: pd.DataFrame, features_expected: list) -> pd.DataFrame:
    if (not USE_BERT) or (TEST_BERT_EMB is None):
        return X_all
    exp_cols_raw = [c for c in features_expected if str(c).startswith("BERT_")]
    exp_cols_pca = [c for c in features_expected if str(c).startswith("BERT_PCA_")]
    N = len(X_all); Z = TEST_BERT_EMB
    if Z is None or Z.shape[0] != N:
        print("âš ï¸� BERT ì�„ë² ë”© ì—†ì�Œ ë˜�ëŠ” ìƒ˜í”Œ ìˆ˜ ë¶ˆì�¼ì¹˜ â†’ ìŠ¤í‚µ")
        return X_all

    def _build_df(n_need, names):
        D = Z.shape[1]; n_use = min(n_need, D)
        out = np.zeros((N, n_need), dtype=np.float32)
        out[:, :n_use] = Z[:, :n_use]
        return pd.DataFrame(out, columns=names)

    if exp_cols_raw:
        X_all = pd.concat([X_all, _build_df(len(exp_cols_raw), exp_cols_raw)], axis=1)
    if exp_cols_pca:
        print("âš ï¸� PCA OFF: ê¸°ëŒ€ ì—´ì�´ BERT_PCA_* â†’ raw ì•�ë¶€ë¶„ìœ¼ë¡œ ëŒ€ì²´í•©ë‹ˆë‹¤.")
        X_all = pd.concat([X_all, _build_df(len(exp_cols_pca), exp_cols_pca)], axis=1)
    return X_all

# =========================
# Predict per target
# =========================
pred_df = pd.DataFrame({"id": test_ids})

for target in TARGETS:
    print(f"\n[Inference] Target = {target}")
    target_dir = os.path.join(MODELS_IN_DIR, target)

    features_expected = load_features_json(target_dir)
    fold_dirs = list_fold_dirs(target_dir)
    print(f"  - folds found: {len(fold_dirs)}")
    print(f"  - expecting {len(features_expected)} features")

    # 1) í•™ìŠµê³¼ ë�™ì�¼ ë ˆì‹œí”¼ë¡œ test íŠ¹ì§• ìƒ�ì„±
    X_all = smiles_to_features(
        test_smiles,
        selected_descriptors=filters[target],
        radius=2, n_bits=128,
        add_counts=True, add_graph=True, add_derived=True,
        add_alias=True
    ).fillna(0.0)

    # (ì„ íƒ�) BERT ë¶™ì�´ê¸° â€“ í•™ìŠµì�´ BERT ì‚¬ìš© ì•ˆ í–ˆìœ¼ë©´ ë�”
    X_all = attach_bert_features_no_pca(X_all, features_expected)

    # 2) í•™ìŠµì—�ì„œ ì“°ì�¸ ìµœì¢… í”¼ì²˜ì…‹(features.json) ìˆœì„œë¡œ ì •ë ¬
    #    ëˆ„ë�½/ì—¬ë¶„ ì—´ ì²´í�¬(ë””ë²„ê·¸ ë�„ì›€)
    missing = [c for c in features_expected if c not in X_all.columns]
    extra   = [c for c in X_all.columns if c not in features_expected]
    print(f"  [feature check] missing={len(missing)}, extra={len(extra)}")
    if missing[:10]:
        print("    sample missing:", missing[:10])
    if extra[:10]:
        print("    sample extra  :", extra[:10])

    X_test = X_all.reindex(columns=features_expected, fill_value=0.0).to_numpy(dtype=np.float32, copy=False)

    # 3) ê°� fold ë¡œë“œ â†’ ë¡¤ë°±/ê°€ì¤‘ë¸”ë Œë“œ â†’ fold í�‰ê· 
    fold_preds = []
    for nm, fdir in fold_dirs:
        xgb_path  = os.path.join(fdir, "xgb.json")
        lgb_path  = os.path.join(fdir, "lgb.txt")
        meta_path = os.path.join(fdir, "meta.json")
        if not (os.path.exists(xgb_path) and os.path.exists(lgb_path) and os.path.exists(meta_path)):
            raise FileNotFoundError(f"[{fdir}] xgb.json / lgb.txt / meta.json ì¤‘ ì�¼ë¶€ ì—†ì�Œ")

        with open(meta_path, "r") as fh:
            meta = json.load(fh)

        # XGB
        xgb_booster = xgb.Booster()
        xgb_booster.load_model(xgb_path)

        use_rb = bool(meta.get("use_rollback", False))
        end_iter_xgb = meta.get("xgb_used_end_iter", None)
        if end_iter_xgb is None:
            best_iter_xgb = int(meta.get("xgb_best_iteration", -1))
            frac_xgb = float(meta.get("rollback_frac_xgb", 1.0))
            if best_iter_xgb > 0:
                end_iter_xgb = max(1, int(np.floor(best_iter_xgb * (frac_xgb if use_rb else 1.0))))
        if end_iter_xgb is not None:
            pred_xgb = xgb_booster.inplace_predict(X_test, iteration_range=(0, int(end_iter_xgb)+1))
        else:
            pred_xgb = xgb_booster.inplace_predict(X_test)

        # LGB
        lgb_booster = lgb.Booster(model_file=lgb_path)
        num_it_lgb = meta.get("lgb_used_num_iter", None)
        if num_it_lgb is None:
            best_it_lgb = int(meta.get("lgb_best_iteration", -1))
            frac_lgb = float(meta.get("rollback_frac_lgb", 1.0))
            if best_it_lgb > 0:
                num_it_lgb = max(1, int(np.floor(best_it_lgb * (frac_lgb if use_rb else 1.0))))
        if num_it_lgb is not None:
            pred_lgb = lgb_booster.predict(X_test, num_iteration=int(num_it_lgb))
        else:
            pred_lgb = lgb_booster.predict(X_test)

        # blend + clip
        w = float(meta.get("best_w", 0.5))
        pred_blend = w * np.asarray(pred_xgb) + (1.0 - w) * np.asarray(pred_lgb)
        lo, hi = BOUNDS[target]
        pred_blend = np.clip(pred_blend, lo, hi)

        fold_preds.append(pred_blend)
        print(f"    {nm}: w={w:.2f}, XGB_end_iter={end_iter_xgb}, LGB_num_iter={num_it_lgb}")

    target_pred = np.mean(np.vstack(fold_preds), axis=0)
    pred_df[target] = target_pred.astype(float)

# =========================
# Save submission
# =========================
out_path = "xgblgb_submission.csv"
pred_df[["id"] + TARGETS].to_csv(out_path, index=False)
print("\nSaved submission:", out_path, pred_df.shape)
print(pred_df.head())



!pip install /kaggle/input/torch-geometric-2-6-1/torch_geometric-2.6.1-py3-none-any.whl


import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from rdkit import Chem
from rdkit.Chem import Descriptors
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import joblib
import os
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool
import torch.nn.functional as F
import warnings
import json
import torch
from sklearn.preprocessing import RobustScaler
import json
import torch
import torch.nn.functional as F
from torch_geometric.nn import GINEConv, global_mean_pool

RDKIT_AVAILABLE = True
TARGETS = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

os.makedirs("NeurIPS", exist_ok=True)
class Config:
    debug = False
    use_cross_validation = True  # Set to False to use a single split for speed
    use_external_data = True  # Set to True to use external datasets
    random_state = 42

# Create a single config instance to use everywhere
config = Config()

"""
Load competition data with complete filtering of problematic polymer notation
"""
print("Loading competition data...")
train = pd.read_csv(BASE_PATH + 'train.csv')
test = pd.read_csv(BASE_PATH + 'test.csv')

if config.debug:
    print("   Debug mode: sampling 1000 training examples")
    train = train.sample(n=1000, random_state=42).reset_index(drop=True)

print(f"Training data shape: {train.shape}, Test data shape: {test.shape}")

def clean_and_validate_smiles(smiles):
    """Completely clean and validate SMILES, removing all problematic patterns"""
    if not isinstance(smiles, str) or len(smiles) == 0:
        return None
    
    # List of all problematic patterns we've seen
    bad_patterns = [
        '[R]', '[R1]', '[R2]', '[R3]', '[R4]', '[R5]', 
        "[R']", '[R"]', 'R1', 'R2', 'R3', 'R4', 'R5',
        # Additional patterns that cause issues
        '([R])', '([R1])', '([R2])', 
    ]
    
    for pattern in bad_patterns:
        if pattern in smiles:
            return None
    
    # Additional check: if it contains ] followed by [ without valid atoms, likely polymer notation
    if '][' in smiles and any(x in smiles for x in ['[R', 'R]']):
        return None
    
    # Try to parse with RDKit if available
    if RDKIT_AVAILABLE:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            return None
    
    # If RDKit not available, return cleaned SMILES
    return smiles

# Clean and validate all SMILES
print("Cleaning and validating SMILES...")
train['SMILES'] = train['SMILES'].apply(clean_and_validate_smiles)
test['SMILES'] = test['SMILES'].apply(clean_and_validate_smiles)

# Remove invalid SMILES
invalid_train = train['SMILES'].isnull().sum()
invalid_test = test['SMILES'].isnull().sum()

print(f"   Removed {invalid_train} invalid SMILES from training data")
print(f"   Removed {invalid_test} invalid SMILES from test data")

train = train[train['SMILES'].notnull()].reset_index(drop=True)
test = test[test['SMILES'].notnull()].reset_index(drop=True)

print(f"   Final training samples: {len(train)}")
print(f"   Final test samples: {len(test)}")

def add_extra_data_clean(df_train, df_extra, target):
    """Add external data with thorough SMILES cleaning"""
    n_samples_before = len(df_train[df_train[target].notnull()])
    
    print(f"      Processing {len(df_extra)} {target} samples...")
    
    # Clean external SMILES
    df_extra['SMILES'] = df_extra['SMILES'].apply(clean_and_validate_smiles)
    
    # Remove invalid SMILES and missing targets
    before_filter = len(df_extra)
    df_extra = df_extra[df_extra['SMILES'].notnull()]
    df_extra = df_extra.dropna(subset=[target])
    after_filter = len(df_extra)
    
    print(f"      Kept {after_filter}/{before_filter} valid samples")
    
    if len(df_extra) == 0:
        print(f"      No valid data remaining for {target}")
        return df_train
    
    # Group by canonical SMILES and average duplicates
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()
    
    cross_smiles = set(df_extra['SMILES']) & set(df_train['SMILES'])
    unique_smiles_extra = set(df_extra['SMILES']) - set(df_train['SMILES'])

    # Fill missing values
    filled_count = 0
    for smile in df_train[df_train[target].isnull()]['SMILES'].tolist():
        if smile in cross_smiles:
            df_train.loc[df_train['SMILES']==smile, target] = \
                df_extra[df_extra['SMILES']==smile][target].values[0]
            filled_count += 1
    
    # Add unique SMILES
    extra_to_add = df_extra[df_extra['SMILES'].isin(unique_smiles_extra)].copy()
    if len(extra_to_add) > 0:
        for col in TARGETS:
            if col not in extra_to_add.columns:
                extra_to_add[col] = np.nan
        
        extra_to_add = extra_to_add[['SMILES'] + TARGETS]
        df_train = pd.concat([df_train, extra_to_add], axis=0, ignore_index=True)

    n_samples_after = len(df_train[df_train[target].notnull()])
    print(f'      {target}: +{n_samples_after-n_samples_before} samples, +{len(unique_smiles_extra)} unique SMILES')
    print(f"      Filled {filled_count} missing entries in train for {target}")
    print(f"      Added {len(extra_to_add)} new entries for {target}")
    return df_train

# Load external datasets with robust error handling
print("\nğŸ“‚ Loading external datasets...")

external_datasets = []

# Function to safely load datasets
def safe_load_dataset(path, target, processor_func, description):
    try:
        if path.endswith('.xlsx'):
            data = pd.read_excel(path)
        else:
            data = pd.read_csv(path)
        
        data = processor_func(data)
        external_datasets.append((target, data))
        print(f"   âœ… {description}: {len(data)} samples")
        return True
    except Exception as e:
        print(f"   âš ï¸� {description} failed: {str(e)[:100]}")
        return False

# Load each dataset
safe_load_dataset(
    '/kaggle/input/tc-smiles/Tc_SMILES.csv',
    'Tc',
    lambda df: df.rename(columns={'TC_mean': 'Tc'}),
    'Tc data'
)

safe_load_dataset(
    '/kaggle/input/tg-smiles-pid-polymer-class/TgSS_enriched_cleaned.csv',
    'Tg', 
    lambda df: df[['SMILES', 'Tg']] if 'Tg' in df.columns else df,
    'TgSS enriched data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/JCIM_sup_bigsmiles.csv',
    'Tg',
    lambda df: df[['SMILES', 'Tg (C)']].rename(columns={'Tg (C)': 'Tg'}),
    'JCIM Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_tg3.xlsx',
    'Tg',
    lambda df: df.rename(columns={'Tg [K]': 'Tg'}).assign(Tg=lambda x: x['Tg'] - 273.15),
    'Xlsx Tg data'
)

safe_load_dataset(
    '/kaggle/input/smiles-extra-data/data_dnst1.xlsx',
    'Density',
    lambda df: df.rename(columns={'density(g/cm3)': 'Density'})[['SMILES', 'Density']]
                .query('SMILES.notnull() and Density.notnull() and Density != "nylon"')
                .assign(Density=lambda x: x['Density'].astype(float) - 0.118),
    'Density data'
)

safe_load_dataset(
    BASE_PATH + 'train_supplement/dataset4.csv',
    'FFV', 
    lambda df: df[['SMILES', 'FFV']] if 'FFV' in df.columns else df,
    'dataset 4'
)

# Integrate external data
print("\nğŸ”„ Integrating external data...")
train_extended = train[['SMILES'] + TARGETS].copy()

if getattr(config, "use_external_data", True) and  not config.debug:
    for target, dataset in external_datasets:
        print(f"   Processing {target} data...")
        train_extended = add_extra_data_clean(train_extended, dataset, target)

print(f"\nğŸ“Š Final training data:")
print(f"   Original samples: {len(train)}")
print(f"   Extended samples: {len(train_extended)}")
print(f"   Gain: +{len(train_extended) - len(train)} samples")

for target in TARGETS:
    count = train_extended[target].notna().sum()
    original_count = train[target].notna().sum() if target in train.columns else 0
    gain = count - original_count
    print(f"   {target}: {count:,} samples (+{gain})")

print(f"\nâœ… Data integration complete with clean SMILES!")

def separate_subtables(train_df):
    labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']
    subtables = {}
    for label in labels:
        # Filter out NaNs, select columns, reset index
        subtables[label] = train_df[train_df[label].notna()][['SMILES', label]].reset_index(drop=True)

    return subtables

def augment_smiles_dataset(smiles_list, labels, num_augments=3):
    """
    Augments a list of SMILES strings by generating randomized versions.

    Parameters:
        smiles_list (list of str): Original SMILES strings.
        labels (list or np.array): Corresponding labels.
        num_augments (int): Number of augmentations per SMILES.

    Returns:
        tuple: (augmented_smiles, augmented_labels)
    """
    augmented_smiles = []
    augmented_labels = []

    for smiles, label in zip(smiles_list, labels):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        # Add original
        augmented_smiles.append(smiles)
        augmented_labels.append(label)
        # Add randomized versions
        for _ in range(num_augments):
            rand_smiles = Chem.MolToSmiles(mol, doRandom=True)
            augmented_smiles.append(rand_smiles)
            augmented_labels.append(label)

    return augmented_smiles, np.array(augmented_labels)

required_descriptors = {'graph_diameter','num_cycles','avg_shortest_path','MolWt', 'LogP', 'TPSA', 'RotatableBonds', 'NumAtoms'}

def augment_dataset(X, y, n_samples=1000, n_components=5, random_state=None):
    """
    Augments a dataset using Gaussian Mixture Models.

    Parameters:
    - X: pd.DataFrame or np.ndarray â€” feature matrix
    - y: pd.Series or np.ndarray â€” target values
    - n_samples: int â€” number of synthetic samples to generate
    - n_components: int â€” number of GMM components
    - random_state: int â€” random seed for reproducibility

    Returns:
    - X_augmented: pd.DataFrame â€” augmented feature matrix
    - y_augmented: pd.Series â€” augmented target values
    """
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    elif not isinstance(X, pd.DataFrame):
        raise ValueError("X must be a pandas DataFrame or a NumPy array")

    X.columns = X.columns.astype(str)

    if isinstance(y, np.ndarray):
        y = pd.Series(y)
    elif not isinstance(y, pd.Series):
        raise ValueError("y must be a pandas Series or a NumPy array")

    df = X.copy()
    df['Target'] = y.values

    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    gmm.fit(df)

    synthetic_data, _ = gmm.sample(n_samples)
    synthetic_df = pd.DataFrame(synthetic_data, columns=df.columns)

    augmented_df = pd.concat([df, synthetic_df], ignore_index=True)

    X_augmented = augmented_df.drop(columns='Target')
    y_augmented = augmented_df['Target']

    return X_augmented, y_augmented


train_df=train_extended
test_df=test
subtables = separate_subtables(train_df)

test_smiles = test_df['SMILES'].tolist()
test_ids = test_df['id'].values
labels = ['Tg', 'FFV', 'Tc', 'Density', 'Rg']

# ------------------------------------------------------------------
# --- GNN MODEL AND DATA PREPARATION ---
# ------------------------------------------------------------------

# A dictionary to map atom symbols to integer indices for the GNN
ATOM_MAP = {
    'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8, 'H': 9,
    # --- NEWLY ADDED SYMBOLS ---
    'Si': 10, # Silicon
    'Na': 11, # Sodium
    '*' : 12, # Wildcard atom
    # --- NEWLY ADDED SYMBOLS ---
    'B': 13,  # Boron
    'Ge': 14, # Germanium
    'Sn': 15, # Tin
    'Se': 16, # Selenium
    'Te': 17, # Tellurium
    'Ca': 18, # Calcium
    'Cd': 19, # Cadmium
}

def smiles_to_graph(smiles_str: str, y_val=None):
    """
    Converts a SMILES string to a graph, adding selected global
    molecular features to each node's feature vector.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: return None

        # 1. Calculate global features once per molecule
        global_features = [
            Descriptors.MolWt(mol),
            Descriptors.TPSA(mol),
            Descriptors.NumRotatableBonds(mol),
            Descriptors.MolLogP(mol)
        ]

        node_features = []
        for atom in mol.GetAtoms():
            # Initialize atom-specific features (one-hot encoding)
            atom_features = [0] * len(ATOM_MAP)
            symbol = atom.GetSymbol()
            if symbol in ATOM_MAP:
                atom_features[ATOM_MAP[symbol]] = 1

            # Add other standard atom features
            atom_features.extend([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ])
            
            # 2. Append the global features to each atom's feature vector
            atom_features.extend(global_features)
            
            node_features.append(atom_features)
        
        if not node_features: return None
        x = torch.tensor(node_features, dtype=torch.float)

        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]])

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        if y_val is not None:
            y_tensor = torch.tensor([[y_val]], dtype=torch.float)
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y_tensor)
        else:
            return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    except Exception as e:
        return None

from rdkit.Chem import Descriptors

# A dictionary mapping labels to their most important global features from XGBoost
LABEL_SPECIFIC_FEATURES = {
    'Tg': [
        "HallKierAlpha", # Topological charge index
        "MolLogP",       # Lipophilicity
        "NumRotatableBonds", # Flexibility
        "TPSA",          # Polarity
    ],
    'FFV': [
        "NHOHCount",     # Count of NH and OH groups (H-bonding)
        "NumRotatableBonds",
        "MolWt",         # Size
        "TPSA",
    ],
    'Tc': [
        "MolLogP",
        "NumValenceElectrons",
        "SPS",           # Molecular shape index
        "MolWt",
    ],
    'Density': [
        "MolWt",
        "MolMR",         # Molar refractivity (related to volume)
        "FractionCSP3",  # Proportion of sp3 hybridized carbons (related to saturation)
        "NumHeteroatoms",
    ],
    'Rg': [
        "HallKierAlpha",
        "MolWt",
        "NumValenceElectrons",
        "qed",           # Quantitative Estimation of Drug-likeness
    ]
}

# A helper dictionary to easily call RDKit functions from their string names
RDKIT_DESC_CALCULATORS = {name: func for name, func in Descriptors.descList}
RDKIT_DESC_CALCULATORS['qed'] = Descriptors.qed # Add qed as it's not in the default list

from rdkit import Chem
import numpy as np

# This ATOM_MAP dictionary must be defined globally in your script (it already is)
# ATOM_MAP = {'C': 0, 'N': 1, ...}

def smiles_to_graph_label_specific(smiles_str: str, label: str, y_val=None):
    """
    (BASELINE VERSION - SIMPLE FEATURES)
    - This is the original hybrid GNN featurizer that produced your best score.
    - Node Features (x): Atom one-hot (20) + 5 atom features = 25 features.
    - Edge Features (edge_attr): Bond type as double = 1 feature.
    - Global Features (u): Label-specific descriptors are stored separately in 'data.u'.
    """
    try:
        mol = Chem.MolFromSmiles(smiles_str)
        if mol is None: 
            return None

        # --- 1. Calculate and store label-specific GLOBAL features ---
        global_features = []
        features_to_calculate = LABEL_SPECIFIC_FEATURES.get(label, [])
        
        for feature_name in features_to_calculate:
            calculator_func = RDKIT_DESC_CALCULATORS.get(feature_name)
            if calculator_func:
                try:
                    val = calculator_func(mol)
                    # Ensure value is valid, replace inf/nan with 0
                    global_features.append(val if np.isfinite(val) else 0.0)
                except Exception as e:
                    global_features.append(0.0)
            else:
                global_features.append(0.0)

        # --- 2. Create Node Features (SIMPLE) ---
        node_features = []
        for atom in mol.GetAtoms():
            # One-Hot Symbol (len 20, from global ATOM_MAP)
            atom_features = [0] * len(ATOM_MAP)
            symbol = atom.GetSymbol()
            if symbol in ATOM_MAP:
                atom_features[ATOM_MAP[symbol]] = 1

            # Standard Features (len 5)
            atom_features.extend([
                atom.GetAtomicNum(),
                atom.GetTotalDegree(),
                atom.GetFormalCharge(),
                atom.GetTotalNumHs(),
                int(atom.GetIsAromatic())
            ])
            # Total features = 25
            node_features.append(atom_features)
        
        if not node_features: return None
        x = torch.tensor(node_features, dtype=torch.float)

        # --- 3. Create Edge Features (SIMPLE) ---
        edge_indices, edge_attrs = [], []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices.extend([(i, j), (j, i)])
            bond_type = bond.GetBondTypeAsDouble()
            edge_attrs.extend([[bond_type], [bond_type]]) # 1-dim feature

        if not edge_indices:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 1), dtype=torch.float) # Shape (0, 1)
        else:
            edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_attrs, dtype=torch.float)

        # --- 4. Create Data Object ---
        data_obj = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data_obj.u = torch.tensor([global_features], dtype=torch.float) # Store globals in 'u'

        if y_val is not None:
            data_obj.y = torch.tensor([[y_val]], dtype=torch.float)
        
        return data_obj
        
    except Exception as e:
        # Catch any other unexpected molecule-level errors
        print(f"CRITICAL ERROR converting SMILES '{smiles_str}': {e}")
        return None
            
class GNNModel(torch.nn.Module):
    """
    Defines the Graph Neural Network architecture.
    """
    def __init__(self, num_node_features, hidden_channels=128):
        super(GNNModel, self).__init__()
        torch.manual_seed(42)
        
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels * 2)
        self.conv3 = GCNConv(hidden_channels * 2, hidden_channels * 4)
        self.lin = torch.nn.Linear(hidden_channels * 4, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = self.conv3(x, edge_index)
        x = global_max_pool(x, batch) # Aggregate node features to get a graph-level embedding
        x = F.dropout(x, p=0.25, training=self.training)
        x = self.lin(x)
        
        return x


def predict_with_gnn(trained_model, test_smiles):
    """
    Uses a pre-trained GNN model to make predictions on a list of test SMILES.
    """
    if trained_model is None:
        print("Prediction skipped because the GNN model is invalid.")
        return np.full(len(test_smiles), np.nan)

    print("--- Making predictions with trained GNN... ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Convert test SMILES to graph data
    test_data_list = [smiles_to_graph(s) for s in test_smiles]
    
    # We need to keep track of which original indices are valid
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print("Warning: No valid test molecules could be converted to graphs.")
        return np.full(len(test_smiles), np.nan)
        
    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False)

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = trained_model(data)
            all_preds.append(out.cpu())

    # Combine predictions from all batches
    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    # Create a full-sized prediction array and fill in the values at their original positions
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in GNN prediction count. This can happen with invalid SMILES.")
        fill_count = min(len(valid_indices), len(test_preds_tensor))
        final_predictions[valid_indices[:fill_count]] = test_preds_tensor[:fill_count]

    return final_predictions

import json
import os

def save_gnn_model(model, label, model_dir="models/gnn"):
    """
    (MODIFIED) Saves the GNN model state_dict and its full constructor config.
    """
    if model is None:
        print(f"Skipping save for {label}, model is None.")
        return

    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")

    # Save the model parameters (the weights)
    torch.save(model.state_dict(), model_path)
    
    # Save the full configuration dictionary
    with open(config_path, 'w') as f:
        json.dump(model.config_args, f, indent=4)
        
    print(f"Saved final model for {label} to {model_path}")


def load_gnn_model(label, model_dir="models/gnn"):
    """
    (MODIFIED) Loads a saved GNN model using its full config file.
    """
    model_path = os.path.join(model_dir, f"gnn_model_{label}.pth")
    config_path = os.path.join(model_dir, f"gnn_config_{label}.json")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(model_path) or not os.path.exists(config_path):
        print(f"Warning: Model or config file not found for {label}. Cannot load model.")
        return None

    with open(config_path, 'r') as f:
        config = json.load(f)
    
    try:
        # Re-initialize the model using all saved config args via dictionary unpacking
        model = TaskSpecificGNN(**config).to(DEVICE)
        
        # Load the saved model weights
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        model.eval() # Set model to evaluation mode
        print(f"Successfully loaded saved model for {label} from {model_path}")
        return model

    except Exception as e:
        print(f"CRITICAL ERROR loading model for {label}: {e}")
        print("This may be due to a mismatch between the saved model and the current model class definition.")
        return None
    

def create_dynamic_mlp(input_dim, layer_list, dropout_list):
    """
    Helper function to dynamically build the task-specific MLP.
    """
    layers = []
    current_dim = input_dim
    
    for neurons, dropout in zip(layer_list, dropout_list):
        layers.append(torch.nn.Linear(current_dim, neurons))
        layers.append(torch.nn.ReLU())
        layers.append(torch.nn.Dropout(dropout))
        current_dim = neurons
        
    # Add the final single-output prediction layer
    layers.append(torch.nn.Linear(current_dim, 1))
    
    return torch.nn.Sequential(*layers)

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, global_mean_pool

class TaskSpecificGNN(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_global_features,
                 hidden_channels_gnn, mlp_neurons, mlp_dropouts, heads=8):
        super().__init__()
        torch.manual_seed(42)

        # --- 1. GNN Backbone (Using GATConv, No BatchNorm) ---
        self.convs = torch.nn.ModuleList()

        # Layer 1
        self.convs.append(
            GATConv(num_node_features, hidden_channels_gnn, heads=heads,
                    edge_dim=num_edge_features)
        )

        # Layer 2
        self.convs.append(
            GATConv(hidden_channels_gnn * heads, hidden_channels_gnn * 2, heads=heads,
                    edge_dim=num_edge_features)
        )

        # Layer 3 (Final GNN layer)
        self.convs.append(
            GATConv(hidden_channels_gnn * 2 * heads, hidden_channels_gnn * 4, heads=heads,
                    concat=False, edge_dim=num_edge_features)
        )

        gnn_output_dim = hidden_channels_gnn * 4

        # --- 2. Readout Head ---
        combined_feature_size = gnn_output_dim + num_global_features

        self.readout_mlp = create_dynamic_mlp(
            input_dim=combined_feature_size,
            layer_list=mlp_neurons,
            dropout_list=mlp_dropouts
        )

        # --- 3. Store config for saving/loading ---
        self.config_args = {
            'num_node_features': num_node_features,
            'num_edge_features': num_edge_features,
            'num_global_features': num_global_features,
            'hidden_channels_gnn': hidden_channels_gnn,
            'mlp_neurons': mlp_neurons,
            'mlp_dropouts': mlp_dropouts,
            'heads': heads
        }

    def forward(self, data):
        x, edge_index, edge_attr, u, batch = data.x, data.edge_index, data.edge_attr, data.u, data.batch

        # GNN Layers with ReLU and Dropout
        x = F.relu(self.convs[0](x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[1](x, edge_index, edge_attr))
        x = F.dropout(x, p=0.5, training=self.training)

        x = F.relu(self.convs[2](x, edge_index, edge_attr))

        # Readout
        graph_embedding = global_mean_pool(x, batch)
        combined_features = torch.cat([graph_embedding, u], dim=1)
        output = self.readout_mlp(combined_features)

        return output
        
# This is a new helper, just to make scaling code cleaner inside the loops
def scale_graph_features(data_list, u_scaler, x_scaler, atom_map_len):
    """Applies fitted scalers in-place to a list of Data objects."""
    try:
        for data in data_list:
            # 1. Scale global features (u)
            data.u = torch.tensor(u_scaler.transform(data.u.numpy()), dtype=torch.float)
            
            # 2. Scale continuous part of node features (x)
            x_one_hot = data.x[:, :atom_map_len]
            x_continuous = data.x[:, atom_map_len:]
            
            x_continuous_scaled = x_scaler.transform(x_continuous.numpy())
            x_continuous_scaled_tensor = torch.tensor(x_continuous_scaled, dtype=torch.float)
            
            # Recombine scaled features
            data.x = torch.cat([x_one_hot, x_continuous_scaled_tensor], dim=1)
            
    except Exception as e:
        print(f"CRITICAL ERROR applying scalers: {e}. Check feature dimensions. AtomMapLen={atom_map_len}")
        raise e
    return data_list


def train_gnn_model(label, train_data_list, val_data_list, mlp_neurons, mlp_dropouts, epochs=300): # Increased default epochs
    """
    (REVISED)
    - Accepts both train and val data lists.
    - Implements ReduceLROnPlateau scheduler based on val_loss.
    - Implements Early Stopping based on val_loss patience.
    """
    print(f"--- Training GNN for label: {label} ---")
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not train_data_list:
        print(f"Warning: Empty train data list passed for {label}.")
        return None
    if not val_data_list:
        print(f"Warning: Empty validation data list passed for {label}.")
        return None

    # drop_last=True is important for training stability, prevents variance from tiny final batches.
    train_loader = PyGDataLoader(train_data_list, batch_size=32, shuffle=True, drop_last=True) 
    val_loader = PyGDataLoader(val_data_list, batch_size=32, shuffle=False) # No shuffle/drop for val

    # Get feature dimensions from the first data object
    first_data = train_data_list[0]
    num_node_features = first_data.x.shape[1]
    num_global_features = first_data.u.shape[1]
    num_edge_features = first_data.edge_attr.shape[1]
    
    print(f"Model Features (Scaled): Nodes={num_node_features}, Edges={num_edge_features}, Global={num_global_features}")

    model = TaskSpecificGNN(  # This should be your (no-BN) model class
        num_node_features=num_node_features,
        num_edge_features=num_edge_features,
        num_global_features=num_global_features,
        hidden_channels_gnn=128, 
        mlp_neurons=mlp_neurons,
        mlp_dropouts=mlp_dropouts
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    criterion = torch.nn.L1Loss() 

    # --- 1. ADD SCHEDULER ---
    # This will cut the LR by half (factor=0.5) if val loss doesn't improve for 10 epochs (patience=10)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, verbose=True)

    # --- 2. ADD EARLY STOPPING VARS ---
    best_val_loss = float('inf')
    epochs_no_improve = 0
    PATIENCE_EPOCHS = 30  # Stop training if val loss doesn't improve for 30 straight epochs

    for epoch in range(1, epochs + 1):
        model.train()
        total_train_loss = 0
        for data in train_loader:
            if data.x.shape[0] <= 1: # Skip batches with one node (can happen)
                continue
            data = data.to(DEVICE)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_train_loss += loss.item() * data.num_graphs
        
        if len(train_loader.dataset) == 0:
            avg_train_loss = 0
        else:
            avg_train_loss = total_train_loss / len(train_loader.dataset)

        # --- 3. ADD VALIDATION LOOP (INSIDE EPOCH LOOP) ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(DEVICE)
                out = model(data)
                loss = criterion(out, data.y)
                total_val_loss += loss.item() * data.num_graphs
        
        if len(val_loader.dataset) == 0:
             avg_val_loss = 0
        else:
            avg_val_loss = total_val_loss / len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
             print(f"Epoch: {epoch:03d}, Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

        # --- 4. SCHEDULER & EARLY STOPPING LOGIC ---
        scheduler.step(avg_val_loss) # Feed validation loss to the scheduler

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE_EPOCHS and epoch > 50: # Give it at least 50 epochs to warm up
            print(f"--- Early stopping triggered at epoch {epoch} ---")
            break
            
    print(f"--- GNN training for {label} complete. Best Val Loss: {best_val_loss:.6f} ---")
    return model

def predict_with_gnn(trained_model, test_smiles, label, u_scaler, x_scaler, atom_map_len):
    """
    (MODIFIED for Full Scaling)
    - Requires both u_scaler (global) and x_scaler (node) to transform features.
    - Returns SCALED predictions.
    """
    if trained_model is None or u_scaler is None or x_scaler is None:
        print(f"Prediction skipped for {label} due to missing model or scaler.")
        return np.full(len(test_smiles), np.nan)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 1. Featurize test data (features are NOT scaled yet)
    test_data_list = [smiles_to_graph_label_specific(s, label, y_val=None) for s in test_smiles]
    
    valid_indices = [i for i, data in enumerate(test_data_list) if data is not None]
    valid_test_data = [data for data in test_data_list if data is not None]

    if not valid_test_data:
        print(f"Warning: No valid test molecules could be converted for {label}.")
        return np.full(len(test_smiles), np.nan)
        
    # 2. Apply fitted scalers to all valid test features
    try:
        valid_test_data = scale_graph_features(valid_test_data, u_scaler, x_scaler, atom_map_len)
    except Exception as e:
        print(f"CRITICAL ERROR applying scalers during prediction: {e}.")
        return np.full(len(test_smiles), np.nan)

    test_loader = PyGDataLoader(valid_test_data, batch_size=32, shuffle=False) 

    trained_model.eval()
    all_preds = []
    with torch.no_grad():
        for data in test_loader:
            data = data.to(DEVICE)
            out = trained_model(data)
            all_preds.append(out.cpu())

    test_preds_tensor = torch.cat(all_preds, dim=0).numpy().flatten()
    
    # Fill predictions array (these are SCALED predictions)
    final_predictions = np.full(len(test_smiles), np.nan)
    if len(test_preds_tensor) == len(valid_indices):
        final_predictions[valid_indices] = test_preds_tensor
    else:
        print(f"Warning: Mismatch in GNN prediction count for {label}.")
        fill_count = min(len(valid_indices), len(test_preds_tensor))
        final_predictions[valid_indices[:fill_count]] = test_preds_tensor[:fill_count]

    return final_predictions # These predictions are on the SCALED range


def train_or_predict_gnn(train_model=True, model_dir="models/gnn", n_splits=10):
    """
    (FINAL COMPLETE VERSION)
    - All data hardening (coerce, filter) and RobustScaler logic is included.
    - CV loop is modified to create a val_data_list.
    - Calls the new, optimized train_gnn_model with scheduler/early stopping.
    - Correctly passes all arguments (config['neurons'], config['dropouts']) to fix the TypeError.
    """
    
    ATOM_MAP_LEN = 20  # Make sure this matches your global ATOM_MAP
    
    # Plausible physical ranges to filter catastrophic outliers BEFORE scaling
    VALID_RANGES = {
        'Tg':      (-100, 500),  
        'FFV':     (0.01, 0.99), 
        'Tc':      (0, 1000),    
        'Density': (0.1, 3.0),   
        'Rg':      (0.1, 200)    
    }

    # MLP configs for the GNN readout head
    best_configs = {
        # Classic funnel, slightly lower final dropout
        "Tg":      {"neurons": [512, 256, 128], "dropouts": [0.5, 0.4, 0.2]},
        # Original wide funnel for this complex feature
        "Density": {"neurons": [1024, 256, 64], "dropouts": [0.5, 0.4, 0.3]},
        # Even wider and deeper, with strong regularization for presumed complexity
        "FFV":     {"neurons": [1024, 512, 64], "dropouts": [0.6, 0.5, 0.4]},
        # Slightly deeper than the simplest model to capture more features
        "Tc":      {"neurons": [128, 64], "dropouts": [0.4, 0.3]},
        # A gentle funnel instead of a pure block to encourage feature compression
        "Rg":      {"neurons": [128, 64, 64], "dropouts": [0.4, 0.3, 0.3]},
    }
    default_config = {"neurons": [128, 64], "dropouts": [0.3, 0.3]}

    output_df = pd.DataFrame({'id': test_df['id']})
    cv_mae_results = []
    os.makedirs(model_dir, exist_ok=True)
    warnings.filterwarnings("ignore", "Mean of empty slice", RuntimeWarning)

    for label in labels: 
        print(f"\n{'='*20} Processing GNN for label: {label} {'='*20}")
        
        config = best_configs.get(label, default_config)
        print(f"Using MLP Config: Neurons={config['neurons']}, Dropouts={config['dropouts']}")
        
        ensemble_models = []
        y_scaler_path = os.path.join(model_dir, f"gnn_yscaler_{label}.joblib")
        u_scaler_path = os.path.join(model_dir, f"gnn_uscaler_{label}.joblib")
        x_scaler_path = os.path.join(model_dir, f"gnn_xscaler_{label}.joblib")
        
        if train_model:
            # --- START DATA HARDENING ---
            all_smiles_raw = subtables[label]['SMILES']
            all_y_raw = subtables[label][label] 
            
            all_y_numeric = pd.to_numeric(all_y_raw, errors='coerce')
            original_count = len(all_y_numeric)

            valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
            valid_mask = (all_y_numeric >= valid_min) & (all_y_numeric <= valid_max) & (all_y_numeric.notna())
            
            all_y = all_y_numeric[valid_mask].reset_index(drop=True)
            all_smiles = all_smiles_raw[valid_mask].reset_index(drop=True)
            
            print(f"FILTERING: Coerced {original_count} rows. Kept {len(all_y)} valid rows within range ({valid_min}, {valid_max}).")
            
            if len(all_y) < (2 * n_splits): 
                print(f"CRITICAL: Not enough valid data ({len(all_y)}) to train for {label} with {n_splits} splits. Skipping.")
                continue
            # --- END DATA HARDENING ---

            # --- 1. FIT Y-SCALER (ROBUST) ---
            print("Using RobustScaler for Y-Scaler.")
            y_scaler = RobustScaler()  
            all_y_scaled = y_scaler.fit_transform(all_y.values.reshape(-1, 1)).flatten()
            joblib.dump(y_scaler, y_scaler_path)
            print(f"Saved Y-Scaler for {label}")

            # --- 2. FIT INPUT SCALERS (ROBUST) ---
            print("Pre-computing all graph features to fit input scalers...")
            all_train_graphs_raw = [smiles_to_graph_label_specific(s, label, None) for s in all_smiles]
            
            # Sync graph list with all data (skipping any SMILES that fail featurization)
            all_train_graphs_synced = []
            all_y_scaled_synced = [] 
            all_y_original_synced = [] # Also sync original Y for the CV split
            all_smiles_synced = []     # Also sync SMILES for the CV split
            
            for i, graph in enumerate(all_train_graphs_raw):
                if graph is not None:
                    all_train_graphs_synced.append(graph)
                    all_y_scaled_synced.append(all_y_scaled[i]) 
                    all_y_original_synced.append(all_y[i]) # Keep the original, unscaled, clean Y
                    all_smiles_synced.append(all_smiles[i]) # Keep the matching SMILES
            
            all_train_graphs = all_train_graphs_synced 
            all_y_scaled = np.array(all_y_scaled_synced)
            all_y_original_df = pd.Series(all_y_original_synced) # Store as Series for .iloc
            all_smiles_df = pd.Series(all_smiles_synced)         # Store as Series for .iloc

            if not all_train_graphs:
                print(f"CRITICAL: No valid training graphs could be featurized for {label}. Skipping.")
                continue
                
            all_u_data = np.concatenate([d.u.numpy() for d in all_train_graphs], axis=0)
            print("Using RobustScaler for U-Scaler.")
            u_scaler = RobustScaler().fit(all_u_data)  # Use RobustScaler
            joblib.dump(u_scaler, u_scaler_path)
            print(f"Saved U-Scaler for {label}")

            all_x_data = torch.cat([d.x for d in all_train_graphs], dim=0)
            all_x_continuous = all_x_data[:, ATOM_MAP_LEN:].numpy()
            print("Using RobustScaler for X-Scaler.")
            x_scaler = RobustScaler().fit(all_x_continuous)  # Use RobustScaler
            joblib.dump(x_scaler, x_scaler_path)
            print(f"Saved X-Scaler for {label}")

            # --- 3. APPLY SCALERS ---
            all_data_objects_scaled = scale_graph_features(all_train_graphs, u_scaler, x_scaler, ATOM_MAP_LEN)
            for i, data_obj in enumerate(all_data_objects_scaled):
                data_obj.y = torch.tensor([[all_y_scaled[i]]], dtype=torch.float)
            
            # --- 4. K-FOLD CV LOOP (MODIFIED) ---
            kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
            fold_val_scores = []
            fold_indices_gen = kf.split(all_data_objects_scaled) # Split the synced, valid, scaled data

            for fold, (train_idx, val_idx) in enumerate(fold_indices_gen):
                print(f"\n--- Fold {fold+1}/{n_splits} for {label} ---")
                
                train_data_list = [all_data_objects_scaled[i] for i in train_idx]
                val_data_list = [all_data_objects_scaled[i] for i in val_idx] # <-- CREATE VAL LIST
                
                val_smiles_list = all_smiles_df.iloc[val_idx].tolist()
                y_val_original = all_y_original_df.iloc[val_idx].values 

                fold_model = train_gnn_model(
                    label,
                    train_data_list, # Pass train data
                    val_data_list,   # <-- Pass val data
                    config['neurons'],    # <-- PASSES mlp_neurons
                    config['dropouts'],   # <-- PASSES mlp_dropouts (FIXES ERROR)
                    epochs=300       # <-- Train longer (will stop early)
                )
                
                if fold_model:
                    print("Running final validation prediction on the best model...")
                    val_preds_scaled = predict_with_gnn(fold_model, val_smiles_list, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                    
                    train_y_scaled_median = 0.0 # RobustScaler median is 0
                    val_preds_scaled_filled = pd.Series(val_preds_scaled).fillna(train_y_scaled_median)
                    
                    val_preds_original = y_scaler.inverse_transform(
                        val_preds_scaled_filled.values.reshape(-1, 1)
                    ).flatten()

                    mae = mean_absolute_error(y_val_original, val_preds_original)
                    print(f"âœ… Fold {fold+1} Validation MAE (Original Scale): {mae:.4f}")
                    fold_val_scores.append(mae)
                    
                    model_save_name = f"{label}_fold{fold}"
                    save_gnn_model(fold_model, model_save_name, model_dir)
                    ensemble_models.append(fold_model)
                else:
                    print(f"Warning: Training failed for Fold {fold+1}. Model will be skipped.")
            
            if fold_val_scores:
                avg_cv_mae = np.mean(fold_val_scores)
                print(f"\n{'*'*10} Average CV MAE for {label} (Original Scale): {avg_cv_mae:.4f} {'*'*10}")
                cv_mae_results.append({'label': label, 'avg_cv_mae': avg_cv_mae})

        else:
            # --- PREDICTION-ONLY MODE ---
            print(f"Loading {n_splits} models and ALL 3 RobustScalers for {label} ensemble...")
            model_path = '/kaggle/input/neurips-2025/GATConv_v29/models/gnn/'
            try:
                y_scaler = joblib.load(f'{model_path}gnn_yscaler_{label}.joblib')
                u_scaler = joblib.load(f'{model_path}gnn_uscaler_{label}.joblib')
                x_scaler = joblib.load(f'{model_path}gnn_xscaler_{label}.joblib')
                print("Loaded Y, U, and X RobustScalers.")
            except FileNotFoundError:
                print(f"CRITICAL: Scaler files not found for {label}. Cannot make predictions.")
                continue

            for fold in range(n_splits):
                loaded_model = load_gnn_model(f"{label}_fold{fold}", model_path.rstrip('/'))
                if loaded_model:
                    ensemble_models.append(loaded_model)
            
            if not ensemble_models: print(f"Warning: No models found for label {label}.")
            else: print(f"Successfully loaded {len(ensemble_models)} models for ensemble.")


        # --- ENSEMBLE PREDICTION STEP (Test Set) ---
        test_smiles = test_df['SMILES'].tolist()
        
        if ensemble_models and y_scaler and u_scaler and x_scaler:
            print(f"Making ensemble (scaled) predictions for {label} using {len(ensemble_models)} models...")
            all_fold_preds_scaled = []
            for model in ensemble_models:
                fold_test_preds_scaled = predict_with_gnn(model, test_smiles, label, u_scaler, x_scaler, ATOM_MAP_LEN)
                all_fold_preds_scaled.append(fold_test_preds_scaled)
            
            preds_stack_scaled = np.stack(all_fold_preds_scaled)
            final_ensemble_preds_scaled = np.nanmean(preds_stack_scaled, axis=0) 
            pred_series_scaled = pd.Series(final_ensemble_preds_scaled)
            
            pred_series_scaled_filled = pred_series_scaled.fillna(0.0) # Impute with scaled median (0.0)

            final_preds_original = y_scaler.inverse_transform(
                pred_series_scaled_filled.values.reshape(-1, 1)
            ).flatten()
            
            output_df[label] = final_preds_original
            
        else:
            print(f"No models or scalers available for {label}. Filling with (filtered) training median.")
            # Robust median fallback logic
            fallback_median = 0.0
            try:
                if 'all_y' in locals() and not all_y.empty:
                     fallback_median = all_y.median()
                else: 
                     print("Loading data to calculate fallback median...")
                     fb_y_raw = subtables[label][label]
                     fb_y_num = pd.to_numeric(fb_y_raw, errors='coerce')
                     valid_min, valid_max = VALID_RANGES.get(label, (-np.inf, np.inf))
                     fb_mask = (fb_y_num >= valid_min) & (fb_y_num <= valid_max) & (fb_y_num.notna())
                     fallback_median = fb_y_num[fb_mask].median()
                print(f"Using filtered median fallback: {fallback_median}")
            except Exception as e:
                 print(f"Error getting median, falling back to 0: {e}")
                 fallback_median = 0.0 
                 
            output_df[label] = fallback_median

    # --- Display final CV MAE summary ---
    if train_model and cv_mae_results:
        print("\n" + "="*40)
        print("ğŸ“Š HYBRID GNN 5-Fold CV MAE Summary (Original Scale):")
        print("="*40)
        mae_df = pd.DataFrame(cv_mae_results)
        print(mae_df.to_string(index=False))
        mae_df.to_csv("gnn_hybrid_cv_mae_results.csv", index=False)
        print("\nCV results saved to gnn_hybrid_cv_mae_results.csv")

    submission_path = 'submission_hybrid_gnn_final.csv'
    output_df.to_csv(submission_path, index=False)
    print(f"\nâœ… GNN Ensemble predictions (Original Scale) saved to {submission_path}")
    
    warnings.filterwarnings("default", "Mean of empty slice", RuntimeWarning)
    
    return output_df

# To train the models and then predict:
gnn_submission_df = train_or_predict_gnn(train_model=False)

output_dfs.append(gnn_submission_df)

print("\nGNN Submission Preview:")
print(gnn_submission_df.head())


len(output_dfs)


# =======================================================
# GBDT + GNN ì•™ìƒ�ë¸” (BERT ì œê±°)
# ì•µì»¤(3ê°œ id ëª©í‘œê°’)ì—� ë§�ì¶° íƒ€ê¹ƒë³„ ê°€ì¤‘ì¹˜ w_t(GBDT ë¹„ì¤‘) ì��ë�™ ì‚°ì¶œ
# í›ˆë ¨ í†µê³„ì�˜ stdë¥¼ ì�´ìš©í•´ ê°€ë²¼ìš´ ë¦¿ì§€ ì •ê·œí™”(ê³¼ì �í•© ì–µì œ)
# ìµœì¢… ì €ì�¥: submission_ensemble.csv, submission.csv
# =======================================================
import pandas as pd
import numpy as np
# --- (ì¶”ê°€) ìµœì†Œ ë¹„ì¤‘ ì„¤ì • ---
WEIGHT_FLOOR = 0.30               # ê°� ëª¨ë�¸ ìµœì†Œ 30%
WEIGHT_CEIL  = 1.0 - WEIGHT_FLOOR # = 0.70


GBDT_CSV = "xgblgb_submission.csv"
GNN_CANDIDATES = [
    "submission_hybrid_gnn_final.csv",
    "gnn_submission.csv",
    "submission_gnn.csv",
]

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]

# ë²”ìœ„(ì—†ìœ¼ë©´ ì‚¬ìš©)
BOUNDS = {
    "Tg": (-158.0297376, 482.25),
    "FFV": (0.2069924, 0.79709707),
    "Tc": (0.0365, 1.69),
    "Density": (0.648691234, 1.940998909),
    "Rg": (8.7283551, 35.672905605),
}

def _clip_arr(a: np.ndarray, t: str) -> np.ndarray:
    if t in BOUNDS:
        lo, hi = BOUNDS[t]
        return np.clip(a, lo, hi)
    return a

# 0) í›ˆë ¨ í†µê³„(ì§ˆë¬¸ ì œê³µ)
TRAIN_STD = {
    "Tg": 111.228,
    "FFV": 0.030,
    "Tc": 0.090,
    "Density": 0.146,
    "Rg": 4.609,
}
# ë¦¿ì§€ ê°•ë�„: stdê°€ ì�‘ì�„ìˆ˜ë¡�(=ë¯¼ê°�í•œ íƒ€ê¹ƒ) í�¬ê²Œ. ìŠ¤ì¼€ì�¼ì�€ ê²½í—˜ì¹˜.
ALPHA_BY_T = {t: 0.0 for t in TARGETS}

# 1) ì•µì»¤(íƒ€ê¹ƒë³„ ëª©í‘œê°’) â€” ì§ˆë¬¸ì—� ì£¼ì‹  í�‰ê·  í‘œ
ANCHOR = pd.DataFrame([
    {"id": 1109053969, "Tg": 163.616635, "FFV": 0.374371, "Tc": 0.189751, "Density": 1.155046, "Rg": 22.236132},
    {"id": 1422188626, "Tg": 168.783293, "FFV": 0.376277, "Tc": 0.249065, "Density": 1.102816, "Rg": 21.193652},
    {"id": 2032016830, "Tg": 123.517710, "FFV": 0.350700, "Tc": 0.242655, "Density": 1.105838, "Rg": 20.334942},
])
ANCHOR["id"] = ANCHOR["id"].astype(int)

# 2) GBDT ë¡œë“œ
need_cols = {"id", *TARGETS}
gbdt_sub = pd.read_csv(GBDT_CSV)
if not need_cols.issubset(gbdt_sub.columns):
    miss = list(need_cols - set(gbdt_sub.columns))
    raise ValueError(f"{GBDT_CSV}ì—� í•„ìš”í•œ ì»¬ëŸ¼ì�´ ì—†ìŠµë‹ˆë‹¤: {miss}")
gbdt_sub = gbdt_sub[["id"] + TARGETS].copy()
gbdt_sub["id"] = gbdt_sub["id"].astype(int)
base_ids = gbdt_sub["id"].to_numpy()
gbdt_idx = gbdt_sub.set_index("id")

# 3) GNN ë¡œë“œ (í›„ë³´ ì¤‘ ì²« ì„±ê³µë³¸)
gnn_sub = None
for cand in GNN_CANDIDATES:
    try:
        tmp = pd.read_csv(cand)
        if not need_cols.issubset(tmp.columns):
            print(f"âš ï¸� {cand}: í•„ìš”í•œ ì»¬ëŸ¼ ë¶€ì¡± â†’ skip ({list(need_cols - set(tmp.columns))})")
            continue
        tmp = tmp[["id"] + TARGETS].copy()
        tmp["id"] = tmp["id"].astype(int)
        tmp = tmp.set_index("id").reindex(base_ids).reset_index()
        gnn_sub = tmp
        print(f"âœ… GNN submission ë°œê²¬: {cand} (ids aligned)")
        break
    except Exception as e:
        print(f"â„¹ï¸� {cand} ì�½ê¸° ì‹¤íŒ¨: {e}")

if gnn_sub is None:
    # GNN ì—†ìœ¼ë©´ GBDTë§Œ ì œì¶œ
    gbdt_sub.to_csv("submission.csv", index=False)
    print("âœ… GNN íŒŒì�¼ ì—†ì�Œ â†’ ìµœì¢… ì œì¶œ = GBDT only -> submission.csv", gbdt_sub.shape)
    raise SystemExit

gnn_idx = gnn_sub.set_index("id")

# 4) íƒ€ê¹ƒë³„ ê°€ì¤‘ì¹˜ ì‚°ì¶œ(ë‹«í�Œí˜•, prior/ë¦¿ì§€ ì—†ì�Œ)
def solve_weight(y, p1, p2, alpha=0.0):
    # ìµœì†Œí™”: sum_i (w*p1_i + (1-w)*p2_i - y_i)^2 + alpha * w^2
    p = p1 - p2
    r = y - p2
    den = np.dot(p, p) + alpha
    if den <= 0:
        return 0.5  # ì •ë³´ ì—†ì�Œ â†’ ì¤‘ë¦½
    w = np.dot(r, p) / den
    return float(np.clip(w, 0.0, 1.0))

# ê°€ì¤‘ì¹˜ ê³„ì‚° ë£¨í”„
anchor_ids = ANCHOR["id"].astype(int).tolist()
w_raw_by_t = {}   # (ì¶”ê°€) í�´ë�¨í•‘ ì „ ì›�ë�˜ w ê¸°ë¡�
w_by_t = {}

for t in TARGETS:
    missing = [i for i in anchor_ids
               if (i not in gbdt_idx.index) or (i not in gnn_idx.index)]
    if missing:
        print(f"âš ï¸� {t}: ì•µì»¤ id ëˆ„ë�½ {missing} â†’ ì¤‘ë¦½ w=0.50 ì‚¬ìš©")
        w_by_t[t] = 0.50
        w_raw_by_t[t] = 0.50
        continue

    y       = ANCHOR.set_index("id").loc[anchor_ids, t].to_numpy(dtype=float)
    p_gbdt  = gbdt_idx.loc[anchor_ids, t].to_numpy(dtype=float)
    p_gnn   = gnn_idx.loc[anchor_ids, t].to_numpy(dtype=float)

    w_raw = solve_weight(y, p_gbdt, p_gnn, alpha=ALPHA_BY_T[t])
    w_clamped = float(np.clip(w_raw, WEIGHT_FLOOR, WEIGHT_CEIL))  # â˜… ìµœì†Œ/ìµœëŒ€ ë¹„ì¤‘ ë³´ì�¥

    if w_clamped != w_raw:
        print(f"  â†³ {t}: w_raw={w_raw:.4f} â†’ clamped to {w_clamped:.4f} "
              f"(range [{WEIGHT_FLOOR:.2f},{WEIGHT_CEIL:.2f}])")

    w_raw_by_t[t] = w_raw
    w_by_t[t] = w_clamped

print("ğŸ�¯ íƒ€ê¹ƒë³„ ê°€ì¤‘ì¹˜ w_t (GBDT ë¹„ì¤‘, clamped) =", {k: round(v, 4) for k, v in w_by_t.items()})
# í•„ìš”í•˜ë©´ ì›�ë�˜ ê°’ë�„ í™•ì�¸
print("   (raw before clamp) =", {k: round(v, 4) for k, v in w_raw_by_t.items()})

# 5) ìµœì¢… ì•™ìƒ�ë¸”
final = gbdt_sub[["id"]].copy()
for t in TARGETS:
    p1 = gbdt_sub[t].to_numpy(dtype=float)
    p2 = gnn_sub[t].to_numpy(dtype=float)
    w  = w_by_t[t]
    blended = w * p1 + (1.0 - w) * p2
    final[t] = _clip_arr(blended, t)
# Tg ì„­ì”¨â†’í™”ì”¨ ë³€í™˜ì�„ "ë³€í™˜ í›„ ê°’ì�´ -273 ì´ˆê³¼"ì�¼ ë•Œë§Œ ì �ìš©
_tg_c = final['Tg'].astype(float).to_numpy()
_tg_f = _tg_c * 9.0 / 5.0 + 32.0
mask  = _tg_f < -180.0
final.loc[mask, 'Tg'] = _tg_f[mask]

# (ì„ íƒ�) ë²”ìœ„ í�´ë¦½ ìœ ì§€
final['Tg'] = _clip_arr(final['Tg'].to_numpy(dtype=float), 'Tg')
final.to_csv("submission_ensemble.csv", index=False)
final.to_csv("submission.csv", index=False)
print(f"âœ… Ensemble ì €ì�¥ -> submission_ensemble.csv / submission.csv {final.shape}")



pd.read_csv("/kaggle/working/submission.csv")

