!pip install /kaggle/input/model1/onnxruntime-1.22.0-cp311-cp311-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl --no-deps


import sys
sys.path.append('/kaggle/input/bird11/')
import pandas as pd
import numpy as np
import librosa
import os
import torch
import onnxruntime as ort
from glob import glob

from code_base.datasets import WaveAllFileDataset
from code_base.utils.inference_utils import apply_avarage_weights_on_swa_path
from code_base.inefernce import BirdsInference
from code_base.utils import load_json, compose_submission_dataframe
from code_base.utils.metrics import padded_cmap_numpy


test_au_pathes = glob("/kaggle/input/birdclef-2023/test_soundscapes/*.ogg")

test_df = pd.DataFrame({
    "filename": test_au_pathes,
    "duration_s": [librosa.get_duration(path=el) for el in test_au_pathes]
})


CONFIG = {
    "run_validation": False,
    "run_test": True,
    "use_sigmoid": False,
    "folds": [0],
    "test_data_root": "/kaggle/input/birdclef-2023/test_soundscapes/*.ogg",
    # "test_data_root": "data/birdclef_2023/test_soundscapes/*.ogg",
    "label_map_data_path": "/kaggle/input/bird2int/bird2int_2023.json",
    # "label_map_data_path": 'data/bird2int_2023.json',
    "lookback": None,
    "lookahead": None,
    "segment_len": 5,
    "step": None,
    "late_normalize": True,
    "exp_name": "convnext_small_fb_in22k_ft_in1k_384__convnextv2_tiny_fcmae_ft_in22k_in1k_384__eca_nfnet_l0_noval_v32_075Clipwise025TimeMax_GausMean",
}
bird2id = load_json(CONFIG["label_map_data_path"])
ds_config_test = {
       "root": "",
       "label_str2int_mapping_path": CONFIG["label_map_data_path"],
       "n_cores": 64,
       "use_audio_cache": True,
       "test_mode": True,
       "segment_len": CONFIG["segment_len"],
       "lookback":CONFIG["lookback"],
       "lookahead":CONFIG["lookahead"],
        "sample_id": None,
        "late_normalize": CONFIG["late_normalize"],
        "step": CONFIG["step"],
        "validate_sr": 32_000
    }
loader_config = {
    "batch_size": 4,
    "drop_last": False,
    "shuffle": False,
    "num_workers": 0,
}
ds_test = WaveAllFileDataset(df=test_df, **ds_config_test)
loader_test = torch.utils.data.DataLoader(
        ds_test,
        **loader_config,
    )


model = ort.InferenceSession("/kaggle/input/bird2int/model_simpl.onnx")

inference_class = BirdsInference(
    device="cpu",
    verbose_tqdm=True,
    use_sigmoid=CONFIG["use_sigmoid"],
)


test_preds, test_preds_long, test_dfidx, test_end = inference_class.predict_test_loader(
        nn_models=model,
        data_loader=loader_test,
        is_onnx_model=True
    )
test_pred_df = compose_submission_dataframe(
    probs=test_preds,
    dfidxs=test_dfidx,
    end_seconds=test_end,
    filenames=loader_test.dataset.df[loader_test.dataset.name_col].copy(),
    bird2id=bird2id,
)


test_pred_df.to_csv("submission.csv", index=False)
import pandas as pd
pd.read_csv("submission.csv")

