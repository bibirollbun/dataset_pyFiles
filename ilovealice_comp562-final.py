import gc
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import random
import torch
import torchaudio

from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
    TrainingArguments,
    Trainer,
    set_seed,
)


# Hyperparameter Sweep on W&B: https://wandb.ai/ncduy0303/comp562-cough-sound-classification/sweeps/mtu964xg
# Current best run id: electric-sweep-47 (we will round some of the found hyperparameters for easier reading)

CONFIG = {
    "seed": 42,

    "hf_model_name": "MIT/ast-finetuned-audioset-10-10-0.4593",

    "num_labels": 3,
    "sample_rate": 16000,   # Should match the pretrained model's sample rate
    "num_mel_bins": 128,    # Frequency dimension of the spectrograms
    "max_length": 1024,     # Temporal dimension of the spectrograms

    "do_random_crop": False,                 # Whether to random crop before truncation
    "time_shift_pct": 0.2406222097862671,    # Fraction of the audio length to shift
    "add_noise": False,                      # Whether to add noise
    "noise_std": 3.635836060248842e-05,      # Standard deviation of the noise
    "freq_mask_frac": 0.17216293333926824,   # Fraction of frequency bins to mask
    "time_mask_frac": 0.07754007415493458,   # Number of time steps to mask

    "mixup_alpha": 0.15724932986585213,      # Mixup alpha parameter
    "mixup_rate" : 1,                        # Probability of applying mixup

    "data_dir": "/kaggle/input/airs-ai-in-respiratory-sounds",

    # TrainingArguments
    "num_epochs": 10,
    "batch_size": 16,
    "lr_scheduler_type": "linear",
    "warmup_ratio": 0.0,
    "learning_rate": 2.163877478133885e-05,
    "weight_decay": 0.0008463629935799428,
    "class_weights": [1, 1, 1],
    "label_smoothing_factor": 0.1,
    "fp16": False,

    # k-fold
    "n_splits": 5,
}

CONFIG["freq_mask_param"] = int(CONFIG["num_mel_bins"] * CONFIG["freq_mask_frac"])
CONFIG["time_mask_param"] = int(CONFIG["max_length"] * CONFIG["time_mask_frac"])

# Set random seed for reproducibility
set_seed(CONFIG["seed"])


DATA_DIR = CONFIG["data_dir"]
AUDIO_DIR = os.path.join(DATA_DIR, "sounds", "sounds")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")


df_train = pd.read_csv(TRAIN_CSV_PATH)
df_test  = pd.read_csv(TEST_CSV_PATH)

# Remove the corrupted patient from the training set
df_train = df_train[df_train["candidateID"] != "5ee582f2832c2"]


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


# Visualize the data augmentation for sanity check
full_ds = CoughASTDataset(
    df_train, AUDIO_DIR, feature_extractor, training=True,
    sample_rate=CONFIG["sample_rate"],
    do_random_crop=CONFIG["do_random_crop"],
    time_shift_pct=CONFIG["time_shift_pct"],
    add_noise=CONFIG["add_noise"],
    noise_std=CONFIG["noise_std"],
    freq_mask_param=CONFIG["freq_mask_param"],
    time_mask_param=CONFIG["time_mask_param"],
)

idx = random.randint(0, len(full_ds) - 1)
sample = full_ds[idx]
print(sample)
print(sample["input_values"].shape) # (time_frames, mel_bins)

# Plot the specialtrogram
plt.figure(figsize=(16, 8))
plt.suptitle(f"Spectrograms for train index: {idx}", fontsize=18)
for i in range(4):
    sample = full_ds[idx]  # Each call applies augmentation if training=True
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


# Load the test dataset
test_ds = CoughASTDataset(
    df_test, AUDIO_DIR, feature_extractor, training=False,
    sample_rate=CONFIG["sample_rate"],
)


skf = StratifiedKFold(n_splits=CONFIG["n_splits"], shuffle=True, random_state=CONFIG["seed"])
all_test_logits = []
val_metrics = []

for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, df_train["disease"])):
    print(f"\n===== Fold {fold+1} =====")
    
    # Split fold data
    train_fold = df_train.iloc[train_idx].reset_index(drop=True)
    val_fold   = df_train.iloc[val_idx].reset_index(drop=True)
    
    # Build datasets
    train_ds = CoughASTDataset(
        train_fold, AUDIO_DIR, feature_extractor, training=True,
        sample_rate=CONFIG["sample_rate"],
        do_random_crop=CONFIG["do_random_crop"],
        time_shift_pct=CONFIG["time_shift_pct"],
        add_noise=CONFIG["add_noise"],
        noise_std=CONFIG["noise_std"],
        freq_mask_param=CONFIG["freq_mask_param"],
        time_mask_param=CONFIG["time_mask_param"],
    )
    val_ds = CoughASTDataset(
        val_fold, AUDIO_DIR, feature_extractor, training=False,
        sample_rate=CONFIG["sample_rate"],
    )
    
    # TrainingArguments
    training_args = TrainingArguments(
        output_dir=f"./cv_fold_{fold}",
        eval_strategy="epoch",
        save_strategy="epoch",
        lr_scheduler_type=CONFIG["lr_scheduler_type"],
        warmup_ratio=CONFIG["warmup_ratio"],
        learning_rate=CONFIG['learning_rate'],
        per_device_train_batch_size=CONFIG['batch_size'],
        per_device_eval_batch_size=CONFIG['batch_size'],
        num_train_epochs=CONFIG['num_epochs'],
        weight_decay=CONFIG["weight_decay"],
        label_smoothing_factor=CONFIG["label_smoothing_factor"],
        logging_dir='./logs',
        logging_steps=10,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        remove_unused_columns=True,
        save_total_limit=1,
        fp16=CONFIG["fp16"],
        report_to="none",
    )
    
    # Model & Trainer
    model = ASTForAudioClassification.from_pretrained(
        CONFIG["hf_model_name"],
        num_labels=CONFIG["num_labels"],
        num_mel_bins=CONFIG["num_mel_bins"],
        max_length=CONFIG["max_length"],
        ignore_mismatched_sizes=True,
    )
    
    trainer = MixupTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        weights=CONFIG["class_weights"],
        mixup_alpha=CONFIG["mixup_alpha"],
        mixup_rate=CONFIG["mixup_rate"],
    )
    
    # Train & Validate
    trainer.train()
    metrics = trainer.evaluate()
    print(f"Fold {fold+1} validation metrics:", metrics)
    val_metrics.append(metrics)

    # Inference on test set
    test_out = trainer.predict(test_ds)
    all_test_logits.append(test_out.predictions)
    
    # Cleanup
    del trainer, model, train_ds, val_ds
    gc.collect()
    torch.cuda.empty_cache()


# Average validation metrics
avg_val = {k: np.mean([m[k] for m in val_metrics]) for k in val_metrics[0]}
print("\nAverage validation across folds:")
for k, v in avg_val.items():
    print(f"{k}: {v:.4f}")


# Create submission file
avg_logits = np.mean(np.stack(all_test_logits, axis=0), axis=0)
predictions = np.argmax(avg_logits, axis=1)
submission_df = pd.DataFrame({
    'candidateID': df_test['candidateID'],
    'disease': predictions
})
submission_df.to_csv("submission.csv", index=False)

