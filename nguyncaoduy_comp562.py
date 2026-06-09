import shutil
import os
import gc
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


CONFIG = {
    "seed": 42,

    "hf_model_name": "MIT/ast-finetuned-audioset-10-10-0.4593",
    "saved_model_path": "ast-finetuned-cough-classifier",
    "run_name": "ast-linear-warmup-mix-up-crop-noise-mask-aug", # for wandb logging

    "sample_rate": 16000, # should match the pretrained model's sample rate
    "num_labels": 3,
    "num_mel_bins": 128,  # Frequency dimension of the spectrograms
    "max_length": 1024,   # Temporal dimension of the spectrograms

    "do_random_crop": True, # Whether to random crop before truncation
    "time_shift_pct": 0,    # Fraction of the audio length to shift
    "add_noise": True,      # Whether to add noise
    "noise_std": 1e-3,      # Standard deviation of the noise
    "freq_mask_param": 48,  # Number of frequency bins to mask
    "time_mask_param": 192, # Number of time steps to mask

    "mixup_alpha": 10,    # Mixup alpha parameter
    "mixup_rate" : 0.5,   # Probability of applying mixup

    "data_dir": "/kaggle/input/airs-ai-in-respiratory-sounds",

    # TrainingArguments
    "num_epochs": 20,
    "batch_size": 16,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.1,
    "learning_rate": 5e-5,
    "weight_decay": 0.01,
    "class_weights": [1, 1, 1], # [1.2976, 0.7633, 1.0878] for weighted loss
    "label_smoothing_factor": 0.0,
    "fp16": False,
    "report_to": "wandb",
}

# Set random seed for reproducibility
set_seed(CONFIG["seed"])


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("wandb_key")

import wandb
wandb.login(key=secret_value_0)

import os
os.environ["WANDB_PROJECT"]="comp562-cough-sound-classification"
os.environ["WANDB_LOG_MODEL"]="end"
os.environ["WANDB_WATCH"]="false"


DATA_DIR = CONFIG["data_dir"]
AUDIO_DIR = os.path.join(DATA_DIR, "sounds", "sounds")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")


def is_corrupted(candidateID):
    """
    Returns True if the .wav file for the given patient cannot be opened by torchaudio.
    Returns False if it's a valid audio file.
    """
    audio_path = os.path.join(AUDIO_DIR, candidateID, "cough.wav")

    # If file doesn't exist, consider it "corrupted"
    if not os.path.exists(audio_path):
        return True

    try:
        # Try loading with torchaudio
        torchaudio.load(audio_path)
        # If it succeeds, we consider it not corrupted
        return False
    except Exception as e:
        # If there's an error, it's likely corrupted
        print(f"Could not read {audio_path} with torchaudio. Error: {e}")
        return True

df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH)

# Remove the corrupted patient from the training set
df_train = df_train[df_train["candidateID"] != "5ee582f2832c2"]

# # 1) Collect all patient IDs in train & test
# all_patients = set(df_train["candidateID"].unique()) | set(df_test["candidateID"].unique())

# # 2) Identify corrupted patients
# corrupted_patients = [pid for pid in all_patients if is_corrupted(pid)]

# # 3) Remove these from both train & test
# df_train_clean = df_train[~df_train["candidateID"].isin(corrupted_patients)].copy()
# df_test_clean  = df_test[~df_test["candidateID"].isin(corrupted_patients)].copy()

# print(f"Total corrupted patients detected: {len(corrupted_patients)}")
# print(f"Clean train set size: {len(df_train_clean)}, Original train set size: {len(df_train)}")
# print(f"Clean test set size:  {len(df_test_clean)}, Original test set size: {len(df_test)}")

# df_train = df_train_clean
# df_test = df_test_clean


class CoughASTDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        audio_dir: str,
        feature_extractor: ASTFeatureExtractor,
        sample_rate: int = 16000,
        training: bool = True,
        # ── waveform‑domain aug ────────────────────────────────
        do_random_crop: bool = False,
        time_shift_pct: float = 0.3,
        add_noise: bool = False,
        noise_std: float = 1e-3,
        # ── spec‑domain aug (SpecAugment) ─────────────────────
        freq_mask_param: int = 48,
        time_mask_param: int = 192,
    ):
        """
        Parameters
        ----------
        df : DataFrame      expects a 'candidateID' column and a 'disease' int label.
        audio_dir : str     root that contains <candidateID>/cough.wav
        feature_extractor   HF ASTFeatureExtractor (fbank, normalization, padding/truncation)
        sample_rate : int   MUST match the extractor & model (16 kHz for MIT/ast‑*)
        training : bool     switch to `False` for val/test ⇒ no augmentation
        """
        self.df                = df.reset_index(drop=True)
        self.audio_dir         = audio_dir
        self.feat_extractor    = feature_extractor
        self.sr                = sample_rate
        self.training          = training

        # ─ waveform‑level augmenters ─
        self._do_random_crop   = do_random_crop
        self._num_samples      = self.feat_extractor.max_length * 160
        self._time_shift_pct   = time_shift_pct
        self._add_noise        = add_noise
        self._noise_std        = noise_std

        # ─ SpecAugment blocks (instantiated once, faster than per‑item) ─
        self._freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param)
        self._time_mask = torchaudio.transforms.TimeMasking(time_mask_param)

        # one‑off resampler if files aren’t already 16 kHz
        self._resampler   = torchaudio.transforms.Resample(orig_freq=self.sr, new_freq=self.sr)

    # --------------------------------------------------------
    # helpers
    # --------------------------------------------------------
    def _random_crop(self, wav: torch.Tensor) -> torch.Tensor:
        """Randomly crop a fixed-length segment from the waveform."""
        num_samples = self._num_samples
        total_len   = wav.shape[-1]

        if total_len <= num_samples:
            return wav
        else:
            start = random.randint(0, total_len - num_samples)
            return wav[..., start:start + num_samples]
    
    def _random_time_shift(self, wav: torch.Tensor) -> torch.Tensor:
        """Roll the waveform left / right by up to `time_shift_pct`."""
        max_shift = int(wav.shape[-1] * self._time_shift_pct)
        shift     = random.randint(-max_shift, max_shift)
        return torch.roll(wav, shifts=shift, dims=-1)

    # --------------------------------------------------------
    # core
    # --------------------------------------------------------
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        cid   = self.df.loc[idx, "candidateID"]
        label = self.df.loc[idx, "disease"] if "disease" in self.df.columns else None

        # ────────── load waveform ──────────
        wav_path          = os.path.join(self.audio_dir, cid, "cough.wav")
        waveform, orig_sr = torchaudio.load(wav_path)

        # mono + resample
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if orig_sr != self.sr:
            waveform = self._resampler(waveform)

        # ────────── waveform‑domain aug ──────────
        if self.training:
            # Crop a random chunk of fixed length
            if self._do_random_crop:
                waveform = self._random_crop(waveform)
            # Apply small time shift to that chunk
            waveform = self._random_time_shift(waveform)
            # additive noise
            if self._add_noise:
                waveform += torch.randn_like(waveform) * self._noise_std

        # ────────── extract log‑Mel patches with HF helper ──────────
        feats = self.feat_extractor(
            waveform.squeeze().numpy(),
            sampling_rate=self.sr,
            return_tensors="pt",
        )["input_values"].squeeze(0)       # tensor [frames, mel]

        # ────────── SpecAugment (Freq & Time masking) ──────────
        if self.training:
            feats = feats.transpose(0, 1)  # torchaudio maskers expect [mel, frames]
            feats = feats.unsqueeze(0)     # torchaudio maskers expect 3‑D
            feats = self._freq_mask(feats)
            feats = self._time_mask(feats)
            feats = feats.squeeze(0)
            feats = feats.transpose(0, 1)

        return {
            "input_values": feats,         # [frames, mel]
            "label":        torch.tensor(label, dtype=torch.long) if label is not None else None,
        }


# Load the feature extractor
feature_extractor = ASTFeatureExtractor.from_pretrained(
    CONFIG['hf_model_name'], 
    max_length=CONFIG['max_length'],
)
feature_extractor


# Split the dataset into training and validation sets
train_df, val_df = train_test_split(
    df_train, 
    test_size=0.2, 
    stratify=df_train['disease'],
    random_state=CONFIG['seed'],
)

# Create datasets
train_ds = CoughASTDataset(
    train_df, AUDIO_DIR, feature_extractor, training=True,
    sample_rate=CONFIG["sample_rate"],
    do_random_crop=CONFIG["do_random_crop"],
    time_shift_pct=CONFIG["time_shift_pct"],
    add_noise=CONFIG["add_noise"],
    noise_std=CONFIG["noise_std"],
    freq_mask_param=CONFIG["freq_mask_param"],
    time_mask_param=CONFIG["time_mask_param"],
)
val_ds = CoughASTDataset(
    val_df, AUDIO_DIR, feature_extractor, training=False,
    sample_rate=CONFIG["sample_rate"],
    do_random_crop=CONFIG["do_random_crop"],
    time_shift_pct=CONFIG["time_shift_pct"],
    add_noise=CONFIG["add_noise"],
    noise_std=CONFIG["noise_std"],
    freq_mask_param=CONFIG["freq_mask_param"],
    time_mask_param=CONFIG["time_mask_param"],
)


# Check if Dataset is working
idx = random.randint(0, len(train_ds) - 1)
sample = train_ds[idx]
print(sample)
print(sample["input_values"].shape) # (time_frames, mel_bins)

# Plot the specialtrogram for sanity check
plt.figure(figsize=(16, 8))
plt.suptitle(f"Spectrograms for train index: {idx}", fontsize=18)
for i in range(4):
    sample = train_ds[idx]  # Each call applies augmentation if training=True
    plt.subplot(2, 2, i + 1)
    plt.imshow(sample["input_values"].numpy().T, origin="lower", aspect="auto")
    plt.title(f"Augmentation #{i+1}")
    plt.xlabel("Time Frames")
    plt.ylabel("Mel-Frequency Bins")
    plt.colorbar(label="Normalized Amplitude")
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


# Vision Transformer Model
model = ASTForAudioClassification.from_pretrained(
    CONFIG['hf_model_name'],
    num_labels=CONFIG['num_labels'],
    num_mel_bins=CONFIG['num_mel_bins'],
    max_length=CONFIG['max_length'],
    ignore_mismatched_sizes=True
)
model


# Use accuracy and macro F1 score as metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1_macro": f1_score(labels, predictions, average='macro')
    }


import torch
import torch.nn.functional as F

class MixupTrainer(Trainer):
    """
    Implements mixup inside `compute_loss`.

    Parameters
    ----------
    weights : list of float
              class weights for the loss function.
    mixup_alpha : float (α in the Beta(α, α) distribution);
                   0 ⇒ no mixup. Default = 10 same as the original AST paper.
    mixup_rate  : float (probability of applying mixup);
                   0 ⇒ no mixup. Default = 0.5 same as the original AST paper.
    """
    def __init__(self, *args, weights: list = None,
                 mixup_alpha: float = 10, mixup_rate: float = 0.5, **kwargs):
        super().__init__(*args, **kwargs)
        self.mixup_alpha = mixup_alpha
        self.mixup_rate = mixup_rate
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=self.args.label_smoothing_factor)
        if weights is not None:
            self.loss_fn = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(weights).float().to(self.args.device),
                label_smoothing=self.args.label_smoothing_factor
            )

    # --------------------------------------------------------
    # override loss computation
    # --------------------------------------------------------
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")                     # [B]
        input_values = inputs["input_values"]             # [B, …]

        if model.training and self.mixup_alpha > 0.0 and random.random() < self.mixup_rate:
            lam    = np.random.beta(self.mixup_alpha, self.mixup_alpha)
            batch  = labels.size(0)
            perm   = torch.randperm(batch).to(labels.device)

            # mix the inputs
            mixed_inputs = lam * input_values + (1.0 - lam) * input_values[perm]

            # convert labels → one‑hot → mix
            num_classes  = model.config.num_labels
            one_hot      = torch.nn.functional.one_hot(labels, num_classes).float()
            one_hot_perm = torch.nn.functional.one_hot(labels[perm], num_classes).float()
            mixed_labels = lam * one_hot + (1.0 - lam) * one_hot_perm

            outputs = model(**{"input_values": mixed_inputs})
            loss    = self.loss_fn(outputs.logits, mixed_labels)

        else:
            outputs = model(**inputs)
            loss    = self.loss_fn(outputs.logits, labels)

        return (loss, outputs) if return_outputs else loss


# Hyperparameter Tuning using wandb sweep

# ──────────────────────────────────────────────────────────────────────────────
# 1) Define sweep config with all hyperparameters
# ──────────────────────────────────────────────────────────────────────────────
sweep_config = {
    "method": "bayes",
    "metric": {
        "name": "eval_f1_macro", 
        "goal": "maximize"
    },
    "parameters": {
        "num_epochs": {
            # use 10 instead of 20 epochs to save time
            "values": [10]   
        },
        "fp16": {
            "values": [False]
        },
        "lr_scheduler_type": {
            "values": ["constant", "linear", "cosine_with_restarts"]
        },
        "num_cycles": {
            # for lr_scheduler_type == cosine_with_restarts
            "values": [1, 2, 3]
        },
        "learning_rate": {
            "distribution": "log_uniform_values",
            "min": 1e-6,
            "max": 1e-4
        },
        "weight_decay": {
            "distribution": "log_uniform_values",
            "min": 1e-4,
            "max": 1e-1
        },
        "warmup_ratio": {
            "values": [0.0, 0.1, 0.2]
        },
        "class_weights": {
            "values": [[1.0, 1.0, 1.0], [1.2976, 0.7633, 1.0878]]
        },
        "label_smoothing_factor": {
            "values": [0.0, 0.1, 0.2]
        },
        "mixup_alpha": {
            "distribution": "log_uniform_values",
            "min": 0.1,
            "max": 20.0
        },
        "mixup_rate": {
            "values": [0.0, 0.5, 1.0]
        },
        "num_mel_bins": {
            "values": [64, 128, 256]
        },
        "max_length": {
            "values": [512, 1024, 1536]
        },
        "do_random_crop": {
            "values": [True, False]
        },
        "time_shift_pct": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 1.0
        },
        "add_noise": {
            "values": [True, False]
        },
        "noise_std": {
            "distribution": "log_uniform_values",
            "min": 1e-5,
            "max": 1e-2
        },
        "freq_mask_frac": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.9
        },
        "time_mask_frac": {
            "distribution": "uniform",
            "min": 0.0,
            "max": 0.9
        }
    }
}

# sweep_id = wandb.sweep(sweep=sweep_config, project="comp562-cough-sound-classification")
sweep_id = "0qttsyp8" # to resume a past sweep


# ──────────────────────────────────────────────────────────────────────────────
# 2) Define the sweep “agent” function
# ──────────────────────────────────────────────────────────────────────────────
def sweep_train():
    run = wandb.init()
    cfg = run.config

    gc.collect()
    torch.cuda.empty_cache()

    # 2.1) Re-create the feature extractor with the swept max_length
    feature_extractor = ASTFeatureExtractor.from_pretrained(
        CONFIG['hf_model_name'], 
        num_mel_bins=cfg.num_mel_bins,
        max_length=cfg.max_length,
    )

    # 2.2) Re-create train/val datasets with swept augment params
    freq_mask_param = int(cfg.freq_mask_frac * cfg.num_mel_bins)
    time_mask_param = int(cfg.time_mask_frac * cfg.max_length)

    train_ds = CoughASTDataset(
        train_df, AUDIO_DIR, feature_extractor,
        training=True,
        do_random_crop=cfg.do_random_crop,
        time_shift_pct=cfg.time_shift_pct,
        add_noise=cfg.add_noise,
        noise_std=cfg.noise_std,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
    )
    val_ds = CoughASTDataset(
        val_df, AUDIO_DIR, feature_extractor,
        training=False,
        do_random_crop=cfg.do_random_crop,
        time_shift_pct=cfg.time_shift_pct,
        add_noise=cfg.add_noise,
        noise_std=cfg.noise_std,
        freq_mask_param=freq_mask_param,
        time_mask_param=time_mask_param,
    )

    # 2.3) Build TrainingArguments with swept hyperparams
    lr_kwargs = {}
    if cfg.lr_scheduler_type == "cosine_with_restarts":
        lr_kwargs["num_cycles"] = cfg.num_cycles

    total_dim = cfg.max_length * cfg.num_mel_bins
    if total_dim in [1536 * 256]:
        batch_size = 6
    elif total_dim in [1536 * 128, 1536 * 64, 1024 * 256]:
        batch_size = 8
    else:
        batch_size = 16

    training_args = TrainingArguments(
        output_dir="./sweep_results",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=cfg.num_epochs,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        lr_scheduler_kwargs=lr_kwargs,
        logging_dir="./logs",
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        remove_unused_columns=True,
        save_total_limit=1,
        fp16=cfg.fp16,
        report_to="wandb",
        run_name=run.name,
    )

    # 2.4) Instantiate the model with swept spectrogram dims
    model = ASTForAudioClassification.from_pretrained(
        CONFIG["hf_model_name"],
        num_labels=CONFIG["num_labels"],
        num_mel_bins=cfg.num_mel_bins,
        max_length=cfg.max_length,
        ignore_mismatched_sizes=True
    )

    # 2.5) Choose class weights from the sweep
    cw = cfg.class_weights

    # 2.6) Create and run the MixupTrainer
    trainer = MixupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        weights=cw,
        mixup_alpha=cfg.mixup_alpha,
        mixup_rate=cfg.mixup_rate,
    )

    try:
        trainer.train()
        metrics = trainer.evaluate()
        wandb.log(metrics)
        
    finally:
        # ─── delete saved checkpoints ───────────────────────────
        ckpt_dir = training_args.output_dir
        shutil.rmtree(ckpt_dir, ignore_errors=True)

        # ─── finish W&B run ─────────────────────────────────────
        run.finish()

        # ─── memory cleanup ─────────────────────────────────────
        del trainer, model, train_ds, val_ds, feature_extractor
        gc.collect()
        torch.cuda.empty_cache()


# ──────────────────────────────────────────────────────────────────────────────
# 3) Launch the sweep agent for N trials
# ──────────────────────────────────────────────────────────────────────────────
wandb.agent(sweep_id, function=sweep_train, count=20)


# # Define training arguments and trainer
# training_args = TrainingArguments(
#     output_dir="./results",
#     eval_strategy="epoch",
#     save_strategy="epoch",
#     lr_scheduler_type=CONFIG["lr_scheduler_type"],
#     warmup_ratio=CONFIG["warmup_ratio"],
#     learning_rate=CONFIG['learning_rate'],
#     per_device_train_batch_size=CONFIG['batch_size'],
#     per_device_eval_batch_size=CONFIG['batch_size'],
#     num_train_epochs=CONFIG['num_epochs'],
#     weight_decay=CONFIG["weight_decay"],
#     label_smoothing_factor=CONFIG["label_smoothing_factor"],
#     logging_dir='./logs',
#     logging_steps=10,
#     load_best_model_at_end=True,
#     metric_for_best_model="f1_macro",
#     remove_unused_columns=True,
#     save_total_limit=1,
#     fp16=CONFIG["fp16"],
#     report_to=CONFIG["report_to"],
#     run_name=CONFIG["run_name"]
# )

# trainer = MixupTrainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_ds,
#     eval_dataset=val_ds,
#     compute_metrics=compute_metrics,
#     weights=CONFIG["class_weights"],
#     mixup_alpha=CONFIG["mixup_alpha"],
#     mixup_rate=CONFIG["mixup_rate"],
# )


# # Train the model
# trainer.train()


# # Evaluate the final model on the validation set (without TTA)
# results = trainer.evaluate()
# results


# # Evaluate the final model on the validation set (with TTA)
# def tta_predict(trainer, dataset, n=5):
#     """
#     Runs TTA by averaging predictions over n augmented samples per item.
#     """
#     all_logits = []
#     dataset.training = True
#     for tta_round in range(n):
#         # Set dataset to training mode for augmentation
#         preds = trainer.predict(dataset)
#         all_logits.append(preds.predictions)
#     # Average logits
#     avg_logits = np.mean(np.stack(all_logits, axis=0), axis=0)
#     return avg_logits

# tta_logits = tta_predict(trainer, val_ds, n=5)
# predictions = np.argmax(tta_logits, axis=1)
# val_f1_tta = f1_score(val_ds.df['disease'], predictions, average='macro')
# val_f1 = results['eval_f1_macro']
# print(f"Validation F1 score (TTA): {val_f1_tta:.4f}")
# print(f"Validation F1 score (no TTA): {val_f1:.4f}")
# print(f"F1 score improvement from TTA: {val_f1_tta - val_f1:.4f}")


# # Load the test dataset and make predictions
# test_ds = CoughASTDataset(
#     df_test, AUDIO_DIR, feature_extractor, training=False,
#     sample_rate=CONFIG["sample_rate"],
#     do_random_crop=CONFIG["do_random_crop"],
#     time_shift_pct=CONFIG["time_shift_pct"],
#     add_noise=CONFIG["add_noise"],
#     noise_std=CONFIG["noise_std"],
#     freq_mask_param=CONFIG["freq_mask_param"],
#     time_mask_param=CONFIG["time_mask_param"],
# )

# test_pred = trainer.predict(test_ds)
# predictions = np.argmax(test_pred.predictions, axis=1)


# # Create submission file
# submission_df = pd.DataFrame({
#     'candidateID': df_test['candidateID'],
#     'disease': predictions
# })
# submission_df.to_csv("submission.csv", index=False)


# # Try TTA on the test set
# tta_logits = tta_predict(trainer, test_ds, n=5)
# predictions = np.argmax(tta_logits, axis=1)
# submission_df = pd.DataFrame({
#     'candidateID': df_test['candidateID'],
#     'disease': predictions
# })
# submission_df.to_csv("submission_tta.csv", index=False)


# wandb.finish()

