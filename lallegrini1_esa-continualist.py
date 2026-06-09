%cd /
!rm -rf /kaggle/working/esa_challenge
!git clone https://github.com/lorenzoAllegrini/esa_challenge.git /kaggle/working/esa_challenge


!pip install --no-cache-dir --upgrade "scikit-learn==1.5.1" "numpy==1.26.4" "scipy==1.11.4"


!pip install Cython
%cd /kaggle/working/esa_challenge/spaceai/segmentators
!python setup.py build_ext --inplace


import sys
sys.path.append('/kaggle/working/esa_challenge')



from spaceai.segmentators.shapelet_miner import ShapeletMiner
from spaceai.segmentators.esa_segmentator2 import EsaDatasetSegmentator2

shapelet_miner = ShapeletMiner(
    k_min_length=30, k_max_length=40, num_kernels=10,
    segment_duration=50, step_duration=10,
    run_id="esa_run", exp_dir="experiments", skip=True
)

segmentator = EsaDatasetSegmentator2(
    transformations=["min","max","mean","std","var","stft","sc","slope","diff_var"],
    segment_duration=50, step_duration=10,
    shapelet_miner=shapelet_miner,
    telecommands=False,
    pooling_segment_len=200, pooling_segment_stride=20,
    poolings=["max","min"],
    run_id="esa_run", exp_dir="experiments",
    use_shapelets=True
)


test_file = "/kaggle/input/esa-adb-challenge/test.parquet"


from spaceai.benchmark.esa_competition_predictor import ESACompetitionPredictor
from spaceai.utils.tools import kernel_column_selector
import os


predictor = ESACompetitionPredictor("/kaggle/input/esa_predictor_artifacts/pytorch/default/4/inference_artifacts", segmentator, data_root="/kaggle/input/esa_predictor_artifacts/pytorch/default/4/inference_artifacts")
res = predictor.predict(test_file, "/kaggle/working/submission.csv")
res.to_csv("/kaggle/working/submission.csv", index=False)

