%%capture
# Set base directory (note: shell variable may won't work directly in `!pip install` in notebooks when in submission)
BASE="/kaggle/input/pytorch-forecasting-env/wheels"

!pip install --no-index --no-deps \
  $BASE/lightning-2.5.1.post0-py3-none-any.whl \
  $BASE/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl \
  $BASE/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl \
  $BASE/nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl \
  $BASE/nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl \
  $BASE/nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl \
  $BASE/nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl \
  $BASE/nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl \
  $BASE/packaging-24.2-py3-none-any.whl \
  $BASE/pytorch_forecasting-1.3.0-py3-none-any.whl \
  $BASE/pytorch_optimizer-3.6.0-py3-none-any.whl


# Standard libraries
import os
import pickle
import warnings
from tqdm import tqdm

# Scientific and data libraries
import numpy as np
import pandas as pd
import torch

# Scikit-learn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# Polars
import polars as pl

# PyTorch Forecasting
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer, NaNLabelEncoder
from pytorch_forecasting.metrics import (
    CrossEntropy,
    MAE,
    PoissonLoss,
    QuantileLoss,
    SMAPE,
)
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import (
    optimize_hyperparameters,
)

# PyTorch Lightning / Lightning
import lightning.pytorch as pyl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CSVLogger
from pytorch_lightning import Trainer as PLTrainer  # if needed separately
from pytorch_lightning.callbacks import EarlyStopping as PLEarlyStopping
from pytorch_lightning.callbacks import LearningRateMonitor as PLLearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint as PLModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

# Ignore warnings
warnings.filterwarnings("ignore")
torch.use_deterministic_algorithms(False)

import os
import sys
from absl import logging

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["XLA_FLAGS"] = "--xla_gpu_cuda_data_dir=/dev/null"

logging.set_verbosity(logging.ERROR)

# Optionally suppress all stderr
# sys.stderr = open(os.devnull, 'w')




# Detect if running in submission mode
IS_KAGGLE_SUBMISSION = os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch"


class CFG:
    # Set this globally or pass during instantiation if dynamic
    IS_KAGGLE_SUBMISSION = os.getenv("KAGGLE_KERNEL_RUN_TYPE") == "Batch"

    # Mode and basic run controls
    mode = 'train'
    dry_run = 2 if not IS_KAGGLE_SUBMISSION else 0
    skip_frame = 1
    limit_frame = 70
    n_class = 18
    batch_size = 16

    # Temporal Fusion Transformer (TFT) hyperparameters
    tft_hparams = {
        "learning_rate": 1e-2,
        "hidden_size": 128,
        "lstm_layers": 2,
        "attention_head_size": 4,
        "dropout": 0.1,
        "hidden_continuous_size": 32,
        "reduce_on_plateau_patience": 8,
        "optimizer": "ranger",
    }

    trainer = {
        "max_epochs": 2 if not IS_KAGGLE_SUBMISSION else 400,
        "gradient_clip_val": 0.1
    }

    target = "gesture"
    
    # Static features (do not change over time for each sequence)
    static_categoricals = ["subject", "sex", "handedness", "adult_child"]
    static_reals = ["age", "height_cm", "shoulder_to_wrist_cm", "elbow_to_wrist_cm"]

    # Known time-varying features (available ahead of prediction)
    time_varying_known_reals = ["sequence_counter"]

    # Unknown time-varying features (used for learning/prediction)
    time_varying_unknown_reals = [
        "acc_x", "acc_y", "acc_z",         # Accelerometer
        "rot_w", "rot_x", "rot_y", "rot_z" # Rotation/quaternion data
    ]



train_df = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train.csv")
subject_df = pl.read_csv("/kaggle/input/cmi-detect-behavior-with-sensor-data/train_demographics.csv")


if CFG.mode == 'train':
    subject_train, subject_test = train_test_split(subject_df, test_size=0.2, random_state=43, shuffle=False)
    n = CFG.dry_run if CFG.dry_run > 0 else None
    X_train = train_df.join(subject_train[:n], on='subject', how='inner')
    X_test = train_df.join(subject_test[:n], on='subject', how='inner')
else:
    X_train = train_df.join(subject_train, on='subject', how='inner')


def generate_filtered_joined_df(
    df: pl.DataFrame,
    group_col: str = "sequence_id",
    counter_col: str = "sequence_counter",
    skip_frame: int = 1,
    limit_frame: int = 50
) -> pl.DataFrame:

    # 1. Compute max counter per group
    max_counter_df = df.group_by(group_col).agg(
        pl.col(counter_col).max().alias("max_counter")
    )

    # 2. Generate countdown lists per max_counter
    def pick_useful_counter(x):
        step = skip_frame if skip_frame > 0 else 1
        frames = list(range(x, 0, -step))[:limit_frame-1]
        # Pad first step twice
        frames.insert(0, frames[0])        
        # Pad list if shorter than limit_frame
        while len(frames) < limit_frame:
            frames.append(frames[-1])
        return frames

    max_counter_df = max_counter_df.with_columns(
        pl.col("max_counter")
        .map_elements(pick_useful_counter, return_dtype=pl.List(pl.Int64))
        .alias("filtered_counters")
    )

    # 3. Explode counters into rows for join
    exploded_counters = max_counter_df.explode("filtered_counters").select([
        pl.col(group_col),
        pl.col("filtered_counters").alias(counter_col)
    ])

    # 4. Join original df with exploded counters on group and counter
    joined_df = df.join(exploded_counters, on=[group_col, counter_col], how="inner")

    # 5. Sort and reset sequence_counter numbering from 1 per group
    result_df = (
        joined_df.sort([group_col, counter_col])
                 .with_columns(
                    (pl.arange(0, pl.len()).over(group_col) + 1).alias(counter_col)
                 )
    )

    return result_df


def preprocess_gesture_df(df):
    categorical_cols = [CFG.target] + CFG.static_categoricals

    df = df.bfill().ffill()
    for col in categorical_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = df[col].astype(str).astype('category')
    return df


def data_pipeline(df, skip_frame, limit_frame, preprocess=True):

    # Step 1: Filter and join frames
    df = generate_filtered_joined_df(
        df,
        skip_frame=skip_frame,
        limit_frame=limit_frame
    ).to_pandas()

    # Step 2: preprocessing
    df = preprocess_gesture_df(df)

    return df



# Fetch config values
skip_frame = CFG.skip_frame
limit_frame = CFG.limit_frame

X_train_1 = data_pipeline(X_train, skip_frame, limit_frame, preprocess=True)

if CFG.mode == 'train':
    X_test_1 = data_pipeline(X_test, skip_frame, limit_frame, preprocess=True)


categorical_encoders = {}
for col in CFG.static_categoricals:
    categorical_encoders[col] = NaNLabelEncoder(add_nan=True)

training = TimeSeriesDataSet(
    X_train_1,
    time_idx="sequence_counter",               # Must be correctly incrementing time index
    target=CFG.target,                          # Your classification label
    group_ids=["sequence_id"],                 # Grouping by sequence

    # Encoder: full sequence before prediction
    min_encoder_length=CFG.limit_frame-1,
    max_encoder_length=CFG.limit_frame-1,

    # Decoder: classification is one-step-ahead
    min_prediction_length=1,
    max_prediction_length=1,

    # Feature configs
    static_categoricals=CFG.static_categoricals,
    static_reals=CFG.static_reals,
    time_varying_known_reals=CFG.time_varying_known_reals,
    time_varying_unknown_reals=CFG.time_varying_unknown_reals,

    # Classification-specific normalization
    target_normalizer=NaNLabelEncoder(add_nan=True),

    # Encode group id
    categorical_encoders=categorical_encoders,

    # Optional: useful features
    add_relative_time_idx=True,
    add_target_scales=False,        # Only needed for regression
    add_encoder_length=True,        # Helps model learn variable-lengths, if applicable

    # Optimization
    allow_missing_timesteps=True   # Safer for sequence classification
)

# create dataloaders for model
train_dataloader = training.to_dataloader(
    train=True, batch_size=CFG.batch_size, num_workers=3
)

if CFG.mode == 'train':
    validation = TimeSeriesDataSet.from_dataset(
        training, X_test_1, predict=True, stop_randomization=True
    )
    
    val_dataloader = validation.to_dataloader(
        train=False, batch_size=CFG.batch_size * 10, num_workers=3
    )


tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=CFG.tft_hparams['learning_rate'],
    hidden_size=CFG.tft_hparams['hidden_size'],
    lstm_layers=CFG.tft_hparams['lstm_layers'],
    attention_head_size=CFG.tft_hparams['attention_head_size'],
    dropout=CFG.tft_hparams['dropout'],
    hidden_continuous_size=CFG.tft_hparams['hidden_continuous_size'],
    loss=CrossEntropy(),
    output_size=CFG.n_class + 1,
    optimizer=CFG.tft_hparams['optimizer'],
    share_single_variable_networks=True,
    causal_attention=False,
    reduce_on_plateau_patience = CFG.tft_hparams['reduce_on_plateau_patience']
)


# Callbacks
early_stop_callback = EarlyStopping(
    monitor="val_loss", min_delta=1e-4, patience=16, verbose=True, mode="min"
)

checkpoint_callback = ModelCheckpoint(
    monitor="val_loss",
    save_top_k=1,
    every_n_epochs=1,
    dirpath="checkpoints/",
    filename="tft-epoch-{epoch:02d}-val_loss-{val_loss:.4f}",
)

# Logger
csv_logger = CSVLogger("logs/", name="tft")

# Trainer
trainer = Trainer(
    max_epochs=CFG.trainer['max_epochs'],
    accelerator="auto",              # uses GPU if available
    devices="auto",                  # picks number of devices automatically
    enable_model_summary=True,
    gradient_clip_val=CFG.trainer['gradient_clip_val'],
    callbacks=[checkpoint_callback, early_stop_callback],
    logger=csv_logger,              # pass actual logger object
    log_every_n_steps=1,            # ensures logging after every step (helpful for debugging)
    check_val_every_n_epoch=1,      # run validation every epoch
    enable_checkpointing=True,      # make sure checkpointing is active
)



if CFG.mode == 'train':
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
else:
    trainer.fit(
        tft,
        train_dataloaders=train_dataloader,
    )


# Store it in a file
with open('training_TimeSeriesDataset.pkl', 'wb') as f:
    pickle.dump(training, f)


class CompetitionMetric:
    """Hierarchical macro F1 for the CMI 2025 challenge."""
    def __init__(self):
        self.target_gestures = [
            'Above ear - pull hair',
            'Cheek - pinch skin',
            'Eyebrow - pull hair',
            'Eyelash - pull hair',
            'Forehead - pull hairline',
            'Forehead - scratch',
            'Neck - pinch skin',
            'Neck - scratch',
        ]
        self.non_target_gestures = [
            'Write name on leg',
            'Wave hello',
            'Glasses on/off',
            'Text on phone',
            'Write name in air',
            'Feel around in tray and pull out an object',
            'Scratch knee/leg skin',
            'Pull air toward your face',
            'Drink from bottle/cup',
            'Pinch knee/leg skin'
        ]
        self.all_classes = self.target_gestures + self.non_target_gestures

    def calculate_hierarchical_f1(
        self,
        sol: pd.DataFrame,
        sub: pd.DataFrame
    ) -> float:

        # Validate gestures
        invalid_types = {i for i in sub['gesture'].unique() if i not in self.all_classes}
        if invalid_types:
            raise ParticipantVisibleError(
                f"Invalid gesture values in submission: {invalid_types}"
            )

        # Compute binary F1 (Target vs Non-Target)
        y_true_bin = sol['gesture'].isin(self.target_gestures).values
        y_pred_bin = sub['gesture'].isin(self.target_gestures).values
        f1_binary = f1_score(
            y_true_bin,
            y_pred_bin,
            pos_label=True,
            zero_division=0,
            average='binary'
        )

        # Build multi-class labels for gestures
        y_true_mc = sol['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')
        y_pred_mc = sub['gesture'].apply(lambda x: x if x in self.target_gestures else 'non_target')

        # Compute macro F1 over all gesture classes
        f1_macro = f1_score(
            y_true_mc,
            y_pred_mc,
            average='macro',
            zero_division=0
        )

        return 0.5 * f1_binary + 0.5 * f1_macro


if CFG.mode == 'train':
    
    solution = X_test_1[['sequence_id','gesture']].drop_duplicates()
    seq_to_gesture = X_test_1.drop_duplicates('sequence_id').set_index('sequence_id')['gesture'].to_dict()
    reverse_map = {v: k for k, v in training.get_parameters()["target_normalizer"].__dict__["classes_"].items()}

    results = []

    # Wrap the iterator with tqdm for progress bar
    for seq in list(seq_to_gesture.keys()):
        pred = tft.predict(
            validation.filter(lambda x: x.sequence_id == seq),
            trainer_kwargs=dict(accelerator="auto"),
            mode="quantiles"
        )
        
        # Get predicted label index (assuming classification output)
        label_idx = int(pred[0][0][1:].argmax() + 1)
        
        # Convert to gesture name
        gesture = reverse_map[label_idx]
        
        # Store in results list
        results.append({"sequence_id": seq, "gesture": gesture})
    
    # Convert to DataFrame
    submission = pd.DataFrame(results)

    solution.to_csv("y_true.csv")
    submission.to_csv("y_pred.csv")

    cm = CompetitionMetric()
    print(cm.calculate_hierarchical_f1(solution, submission))


y_true = list(solution['gesture'])
y_pred = list(submission['gesture'])

# Accuracy
acc = accuracy_score(y_true, y_pred)
print(f"Accuracy: {acc:.4f}")

# Confusion Matrix (as array)
cm = confusion_matrix(y_true, y_pred)
print("Confusion Matrix:\n", cm)

# Optional: Visualize confusion matrix
disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
# Rotate x-axis tick labels by 90 degrees
disp.ax_.set_xticklabels(disp.display_labels, rotation=90)













