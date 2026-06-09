%load_ext autoreload
%autoreload 2


import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


if os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
    MACHINE = "KAGGLE_COMPETITION"
elif "KAGGLE_KERNEL_INTEGRATIONS" in os.environ:
    MACHINE = "KAGGLE"
elif os.getenv("COLAB_RELEASE_TAG"):
    MACHINE = "COLAB"
else:
    MACHINE = "PC"

print(f"MACHINE: {MACHINE}")


if "KAGGLE" in MACHINE:
    import sys
    CODE_PATH = "/kaggle/input/a1_tcnt_2/pytorch/default/6"

    sys.path.append(CODE_PATH)
elif "COLAB" in MACHINE:
    import kagglehub
    import sys
    from pathlib import Path

    path = Path(kagglehub.dataset_download("literallytheone/cmi-sensor-kaggle-utils"))

    path = str(path / "cmi_sensor_kaggle-0.1.4")

    sys.path.append(path)



from cmi_sensor_kaggle.models.transformer_tcn import TransformerTCN
from cmi_sensor_kaggle.utils.predicts import predict_sequence

from cmi_sensor_kaggle.utils.data_preparation import (
    DropUnnecessaryColumns,
    FillNulls,
    ShrinkSequence,
    SortData,
    AddGlobalFeatures,
    AddMagnitude,
    NormalizeScaler,
)
from cmi_sensor_kaggle.utils.transforms import (
    TTCNTransform,
    NormalTargetTransform,
)
from cmi_sensor_kaggle.configs.tcnt_weight_global.a1_tcnt_d_model_128_shrink_0_all import TransformTcnGlobalFeaturesConfig3

import polars as pl

import torch

from omegaconf import OmegaConf

from pathlib import Path

import joblib




cfg = OmegaConf.structured(TransformTcnGlobalFeaturesConfig3)
print(OmegaConf.to_yaml(cfg))


if torch.accelerator.is_available():
    DEVICE = torch.accelerator.current_accelerator().type  # type: ignore
else:
    DEVICE = "cpu"

print(f"device: {DEVICE}")


if "KAGGLE" in MACHINE:
    scalar = joblib.load(f"{CODE_PATH}/scaler.pkl")
elif "COLAB" in MACHINE:
    scalar = joblib.load("../scripts/scalers/scaler_a1_tcnt_d_model_128_all_0.pkl")
else:
    scalar = joblib.load("../scripts/scalers/a1_tcnt_d_model_128_all_3_scaler.pkl")


prs = [
    DropUnnecessaryColumns.from_config(cfg),
    FillNulls.from_config(cfg),
    ShrinkSequence.from_config(cfg),
    SortData.from_config(cfg),
    AddGlobalFeatures.from_config(cfg),
    AddMagnitude.from_config(cfg),
    NormalizeScaler.from_config(cfg, scalar),
]

val_transforms = [
    TTCNTransform.from_config(cfg),
]

target_transform = NormalTargetTransform.from_config(cfg)



if "KAGGLE" in MACHINE:
    model_path = Path(f"{CODE_PATH}/model.pt")
elif MACHINE == "COLAB":
    import kagglehub

    model_path = Path(
        kagglehub.model_download("literallytheone/a1_tcnt_1/pyTorch/default")) / "best_model_13_f10.5573.pt"
else:
    model_path = Path(
        "../scripts/checkpoints/a1_tcnt_d_model_128_all_3/best_model_53_f1=0.6386.pt"
    )


model = TransformerTCN.from_config(cfg)
model = model.to(DEVICE)
model.load_state_dict(torch.load(model_path, map_location=DEVICE))
print(model)


def predict(sequence: pl.DataFrame, demographics: pl.DataFrame) -> str:
    return predict_sequence(
        sequence=sequence,
        model=model,
        prs=prs,
        transforms=val_transforms,
        label_int_to_category=target_transform.label_int_to_category,
        device=DEVICE,
    )


if "KAGGLE" in MACHINE:
    import kaggle_evaluation.cmi_inference_server

    inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(
        predict
    )

    if MACHINE == "KAGGLE_COMPETITION":
        inference_server.serve()
    elif MACHINE == "KAGGLE":
        inference_server.run_local_gateway(
            data_paths=(
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv",
                "/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv",
            )
        )
else:
    if MACHINE == "COLAB":
        import kagglehub

        root_data_path = Path(
            kagglehub.competition_download("cmi-detect-behavior-with-sensor-data")
        )
    else:
        root_data_path = Path(
            "/Users/ramin/.cache/kagglehub/competitions/cmi-detect-behavior-with-sensor-data"
        )

    df_test = pl.read_csv(root_data_path / "test.csv")
    sequences_test = df_test.group_by("sequence_id").all()
    for i in range(sequences_test.shape[0]):
        sequence = sequences_test[i]
        sequence = sequence.explode(pl.all().exclude("sequence_id"))
        result = predict(
            sequence=sequence,
            demographics=None,
        )
        print(result)


