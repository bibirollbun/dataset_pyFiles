%cd /
!rm -rf /kaggle/working/esa_challenge
!git clone https://github.com/lorenzoAllegrini/esa_challenge.git /kaggle/working/esa_challenge


!pip install --no-cache-dir --upgrade "scikit-learn==1.5.1" "numpy==1.26.4" "scipy==1.11.4"


!pip install Cython
%cd /kaggle/working/esa_challenge/spaceai/segmentators
!python setup.py build_ext --inplace


import sys
import pandas as pd
import numpy as np
sys.path.append('/kaggle/working/esa_challenge')



from spaceai.segmentators.shapelet_miner import ShapeletMiner
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2


shapelet_miner1 = ShapeletMiner(
    k_min_length=30, k_max_length=40, num_kernels=10,
    segment_duration=50, step_duration=10,
    run_id="esa_run", exp_dir="experiments", skip=True
)

segmentator1 = EsaDatasetSegmentator2(
    transformations=["min","max","mean","std","var","stft","sc","slope","diff_var"],
    segment_duration=50, step_duration=10,
    shapelet_miner=shapelet_miner1,
    telecommands=False,
    pooling_segment_len=200, pooling_segment_stride=20,
    poolings=["max","min"],
    run_id="esa_run", exp_dir="experiments",
    use_shapelets=True,
    save_csv=False
)

shapelet_miner2 = ShapeletMiner(
        k_min_length=450,
        k_max_length=450,
        num_kernels=5,
        segment_duration=500,
        step_duration=100,
        run_id="esa_training", 
        exp_dir="experiments",
        skip=False
    )

segmentator2 = EsaDatasetSegmentator2(
        transformations=["min", "max", "mean", "std", "var", "stft", "sc", "slope", "diff_var"],
        segment_duration=500,
        step_duration=100,
        shapelet_miner=shapelet_miner2,
        telecommands=False,
        pooling_segment_len=20,
        pooling_segment_stride=2,
        poolings=["max", "min"],
        run_id="esa_training", 
        exp_dir="experiments", 
        use_shapelets=True,
        save_csv=False
    )



test_file = "/kaggle/input/esa-adb-challenge/test.parquet"


from spaceai.benchmark.esa_competition_predictor import ESACompetitionPredictor
from spaceai.utils.tools import kernel_column_selector
import os


# --- Modello A ---
predictor_a = ESACompetitionPredictor(
    "/kaggle/input/model2/pytorch/default/3/inference_artifacts_500/esa_inference_500",
    segmentator2,
    data_root="/kaggle/input/model2/pytorch/default/3/inference_artifacts_500/esa_inference_500",
)
sub_a = predictor_a.predict(test_file, "/kaggle/working/submission_a.csv")  # deve restituire almeno 'id','is_anomaly'

# --- Modello B ---
predictor_b = ESACompetitionPredictor(
    "/kaggle/input/esa_predictor_artifacts/pytorch/default/4/inference_artifacts",
    segmentator1,
    data_root="/kaggle/input/esa_predictor_artifacts/pytorch/default/4/inference_artifacts",
)
sub_b = predictor_b.predict(test_file, "/kaggle/working/submission_b.csv")






# --- Allineamento/robustezza sugli ID ---
# Se i DF hanno già 'id' e 'is_anomaly':
if {"id","is_anomaly"}.issubset(sub_a.columns) and {"id","is_anomaly"}.issubset(sub_b.columns):
    A = sub_a[["id","is_anomaly"]].rename(columns={"is_anomaly":"is_anomaly_a"}).copy()
    B = sub_b[["id","is_anomaly"]].rename(columns={"is_anomaly":"is_anomaly_b"}).copy()
else:
    # fallback robusto: prendi l'ordine e gli id direttamente dal test file
    test_ids = (pd.read_parquet(test_file) if test_file.endswith(".parquet")
                else pd.read_csv(test_file))[["id"]].copy()
    A = test_ids.assign(is_anomaly_a=sub_a["is_anomaly"].to_numpy())
    B = test_ids.assign(is_anomaly_b=sub_b["is_anomaly"].to_numpy())

# Merge per 'id'
final = A.merge(B, on="id", how="inner")

# Ensemble OR logico
final["is_anomaly"] = (
    (final["is_anomaly_a"].astype(int) == 1) |
    (final["is_anomaly_b"].astype(int) == 1)
).astype(int)

# (opzionale) preserva esattamente l'ordine del test
test_ids = (pd.read_parquet(test_file) if test_file.endswith(".parquet")
            else pd.read_csv(test_file))[["id"]]
final = test_ids.merge(final[["id","is_anomaly"]], on="id", how="left")


final.to_csv("/kaggle/working/submission.csv", index=False)


