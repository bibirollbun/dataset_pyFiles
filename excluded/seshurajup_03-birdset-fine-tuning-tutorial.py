import pandas as pd
import random
from tqdm import tqdm
from collections import Counter


def smart_sampling(dataset, label_name, class_limit, event_limit):
    def _unique_identifier(x, labelname):
        file = x["filepath"]
        label = x[labelname]
        return {"id": f"{file}-{label}"}

    class_limit = class_limit if class_limit else -float("inf")
    dataset = dataset.map(
        lambda x: _unique_identifier(x, label_name), desc="sampling: unique-identifier"
    )
    df = pd.DataFrame(dataset)
    path_label_count = df.groupby(["id", label_name], as_index=False).size()
    path_label_count = path_label_count.set_index("id")
    class_sizes = df.groupby(label_name).size()

    for label in tqdm(class_sizes.index, desc="sampling"):
        current = path_label_count[path_label_count[label_name] == label]
        total = current["size"].sum()
        most = current["size"].max()

        while total > class_limit or most != event_limit:
            largest_count = current["size"].value_counts()[current["size"].max()]
            n_largest = current.nlargest(largest_count + 1, "size")
            to_del = n_largest["size"].max() - n_largest["size"].min()

            idxs = n_largest[n_largest["size"] == n_largest["size"].max()].index
            if (
                total - (to_del * largest_count) < class_limit
                or most == event_limit
                or most == 1
            ):
                break
            for idx in idxs:
                current.at[idx, "size"] = current.at[idx, "size"] - to_del
                path_label_count.at[idx, "size"] = (
                    path_label_count.at[idx, "size"] - to_del
                )

            total = current["size"].sum()
            most = current["size"].max()

    event_counts = Counter(dataset["id"])

    all_file_indices = {label: [] for label in event_counts.keys()}
    for idx, label in enumerate(dataset["id"]):
        all_file_indices[label].append(idx)

    limited_indices = []
    for file, indices in all_file_indices.items():
        limit = path_label_count.loc[file]["size"]
        limited_indices.extend(random.sample(indices, limit))

    dataset = dataset.remove_columns("id")
    return dataset.select(limited_indices)


import torch


def classes_one_hot(batch, num_classes):
    """
    Converts class labels to one-hot encoding.

    This method takes a batch of data and converts the class labels to one-hot encoding.
    The one-hot encoding is a binary matrix representation of the class labels.

    Args:
        batch (dict): A batch of data. The batch should be a dictionary where the keys are the field names and the values are the field data.

    Returns:
        dict: The batch with the "labels" field converted to one-hot encoding. The keys are the field names and the values are the field data.
    """
    label_list = [y for y in batch["labels"]]
    class_one_hot_matrix = torch.zeros(
        (len(label_list), num_classes), dtype=torch.float
    )

    for class_idx, idx in enumerate(label_list):
        class_one_hot_matrix[class_idx, idx] = 1

    class_one_hot_matrix = torch.tensor(class_one_hot_matrix, dtype=torch.float32)
    return {"labels": class_one_hot_matrix}


from dataclasses import dataclass
from torchaudio.transforms import Spectrogram, MelScale
from birdset.datamodule.components.resize import Resizer
from birdset.datamodule.components.augmentations import PowerToDB
import numpy as np
import torch_audiomentations
import torchvision.transforms


@dataclass
class CustomProcessingConfig:
    spectrogram_conversion: Spectrogram | None = Spectrogram(
        n_fft=1024,
        hop_length=320,
        power=2.0,
    )
    resizer: Resizer | None = (
        Resizer(
            db_scale=True,
        ),
    )
    melscale_conversion: MelScale | None = (
        MelScale(
            n_mels=128,
            sample_rate=32000,
            n_stft=513,  # n_fft//2+1
        ),
    )
    dbscale_conversion: PowerToDB | None = (PowerToDB(),)
    normalize_spectrogram: bool = (True,)
    mean: float = (-4.268,)
    std: float = (-4.569,)


class BasicTransformsWrapper:
    def __init__(
        self,
        wav_transforms,
        spec_transforms,
        decoding,
        feature_extractor,
        nocall_sampler,
        processing_config=CustomProcessingConfig,
    ):
        self.wav_transforms = torch_audiomentations.Compose(
            transforms=wav_transforms, output_type="object_dict"
        )
        self.spec_transforms = torchvision.transforms.Compose(
            transforms=spec_transforms
        )
        self.processing = processing_config
        self.nocall_sampler = nocall_sampler
        self.decoding = decoding
        self.feature_extractor = feature_extractor

    def __call__(self, batch, **kwds):

        batch = self.decoding(batch)

        waveform_batch = self._get_waveform_batch(batch)

        input_values = waveform_batch["input_values"]
        input_values = input_values.unsqueeze(1)
        labels = torch.tensor(batch["labels"])

        input_values, labels = self._waveform_augmentation(input_values, labels)

        if self.nocall_sampler:
            input_values, labels = self.nocall_sampler(input_values, labels)

        if self.processing.spectrogram_conversion is not None:
            spectrograms = self.processing.spectrogram_conversion(input_values)

            if self.spec_transforms:
                spectrograms = self.spec_transforms(spectrograms)

            if self.processing.melscale_conversion:
                spectrograms = self.processing.melscale_conversion(spectrograms)

            if self.processing.dbscale_conversion:
                spectrograms = self.processing.dbscale_conversion(spectrograms)

            if self.processing.resizer:
                spectrograms = self.processing.resizer.resize_spectrogram_batch(
                    spectrograms
                )

            if self.processing.normalize_spectrogram:
                spectrograms = (
                    spectrograms - self.processing.mean
                ) / self.processing.std

            input_values = spectrograms

        # values in labels need to be of type float for further use
        labels = labels.to(torch.float16)

        return {"input_values": input_values, "labels": labels}

    def _get_waveform_batch(self, batch):
        waveform_batch = [audio["array"] for audio in batch["audio"]]

        # extract/pad/truncate
        max_length = int(
            int(self.feature_extractor.sampling_rate) * int(self.decoding.max_len)
        )
        waveform_batch = self.feature_extractor(
            waveform_batch,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_attention_mask=True,
        )

        return waveform_batch

    def _waveform_augmentation(self, input_values, labels):
        labels = labels.unsqueeze(1).unsqueeze(1)
        output_dict = self.wav_transforms(
            samples=input_values,
            sample_rate=self.feature_extractor.sampling_rate,
            targets=labels,
        )
        labels = output_dict.targets.squeeze(1).squeeze(1)

        return output_dict.samples, labels


import lightning as L
from torch.utils.data import DataLoader


class CustomDatamodule(L.LightningDataModule):
    def __init__(self, dataset, batch_size, num_workers, num_classes, task):
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.train_batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.task = task
        self.len_trainset = len(dataset["train"])

    def setup(self, stage):
        pass

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset["train"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset["valid"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset["test"],
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )


from transformers import ASTForAudioClassification, ConvNextForImageClassification
from torchmetrics import AUROC, MetricCollection
from birdset.modules.metrics.multilabel import TopKAccuracy, cmAP
import lightning as l
import torch.nn as nn
import torch
from transformers import AdamW


class ConvNextClassifierLightningModule(l.LightningModule):
    def __init__(
        self,
        num_classes,
        num_epochs,
    ):
        super(ConvNextClassifierLightningModule, self).__init__()
        self.model = ConvNextForImageClassification.from_pretrained(
            "DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

        self.num_classes = num_classes
        self.num_epochs = num_epochs
        self.loss = nn.BCEWithLogitsLoss()
        self.main_metric = cmAP(num_labels=num_classes, thresholds=None)
        self.other_metrics = MetricCollection(
            {
                "MultilabelAUROC": AUROC(
                    task="multilabel",
                    num_labels=num_classes,
                    average="macro",
                    thresholds=None,
                ),
                "T1Accuracy": TopKAccuracy(topk=1),
            }
        )

    def forward(self, pixel_values):
        outputs = self.model(pixel_values=pixel_values)
        return outputs.logits

    def common_step(self, batch, batch_idx):
        values = batch["input_values"]
        labels = batch["labels"]
        logits = self(values)

        loss = self.loss(logits, labels)
        predictions = torch.sigmoid(logits)

        return loss, predictions

    def training_step(self, batch, batch_idx):
        loss, preds = self.common_step(batch, batch_idx)
        self.log(
            f"train/{self.loss.__class__.__name__}",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds = self.common_step(batch, batch_idx)
        self.log(
            f"val/{self.loss.__class__.__name__}",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
        )

        return loss

    def test_step(self, batch, batch_idx):
        loss, preds = self.common_step(batch, batch_idx)
        self.log(
            f"test/{self.loss.__class__.__name__}",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.main_metric(preds, batch["labels"].int())
        self.log(
            f"test/{self.main_metric.__class__.__name__}",
            self.main_metric,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
        )
        self.other_metrics(preds, batch["labels"].int())
        self.log_dict(self.other_metrics, on_step=False, on_epoch=True, prog_bar=False)

        return loss

    def configure_optimizers(self):
        return AdamW(self.parameters(), lr=5e-5)


from birdset.datamodule.components.transforms import (
    BirdSetTransformsWrapper,
    PreprocessingConfig,
)
from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.feature_extraction import DefaultFeatureExtractor
from birdset.datamodule.components.augmentations import (
    NoCallMixer,
    MultilabelMix,
    AddBackgroundNoise,
    PowerToDB,
)
from birdset.datamodule.components.resize import Resizer
from torch_audiomentations import AddColoredNoise, Gain
from torchaudio.transforms import Spectrogram, MelScale, FrequencyMasking, TimeMasking
from torchvision.transforms import RandomApply

decoder = EventDecoding(
    min_len=1, max_len=5, sampling_rate=32000, extension_time=8, extracted_interval=5
)

feature_extractor = DefaultFeatureExtractor(
    feature_size=1, sampling_rate=32000, padding_value=0.0, return_attention_mask=False
)

nocall = NoCallMixer(
    directory="/mnt/stud/work/rantjuschin/datasets/background_noise/",
    p=0.075,
    sampling_rate=32000,
    length=5,
)

wav_transforms = {
    "multilabel_mix": MultilabelMix(
        p=0.7, min_snr_in_db=3.0, max_snr_in_db=30.0, mix_target="union"
    ),
    "add_background_noise": AddBackgroundNoise(
        p=0.5,
        min_snr_in_db=3,
        max_snr_in_db=30,
        sample_rate=32000,
        target_rate=32000,
        background_paths="/mnt/stud/work/rantjuschin/datasets/background_noise/",
    ),
    "add_colored_noise": AddColoredNoise(
        p=0.2, max_f_decay=2, min_f_decay=-2, max_snr_in_db=30, min_snr_in_db=3
    ),
    "gain": Gain(p=0.2, min_gain_in_db=-18, max_gain_in_db=6),
}

preprocessing = PreprocessingConfig(
    spectrogram_conversion=Spectrogram(n_fft=1024, hop_length=320, power=2.0),
    resizer=Resizer(db_scale=True, target_height=None, target_width=None),
    melscale_conversion=MelScale(n_mels=128, sample_rate=32000, n_stft=513),
    dbscale_conversion=PowerToDB(),
    normalize_spectrogram=True,
    mean=-4.268,
    std=4.569,
)

spec_transforms = {
    "frequency_masking": RandomApply(
        p=0.5, transforms=[FrequencyMasking(freq_mask_param=100, iid_masks=True)]
    ),
    "time_masking": RandomApply(
        p=0.5, transforms=[TimeMasking(time_mask_param=100, iid_masks=True)]
    ),
}


birdset_transforms = BirdSetTransformsWrapper(
    task="multilabel",
    sampling_rate=32000,
    model_type="vision",
    max_length=5,
    decoding=decoder,
    feature_extractor=feature_extractor,
    nocall_sampler=nocall,
    waveform_augmentations=wav_transforms,
    preprocessing=preprocessing,
    spectrogram_augmentations=spec_transforms,
)


from birdset.datamodule.birdset_datamodule import BirdSetDataModule
from birdset.configs.datamodule_configs import DatasetConfig, LoadersConfig
from birdset.datamodule.components import XCEventMapping

dataset_config = DatasetConfig(
    data_dir="/mnt/stud/work/rantjuschin/datasets/HSN",
    hf_path="DBD-research-group/BirdSet",
    hf_name="HSN",
    n_workers=3,
    val_split=0.2,
    task="multilabel",
    classlimit=500,
    eventlimit=5,
    sampling_rate=32000,
    seed=2,
)
loaders_config = LoadersConfig()
mapper_config = XCEventMapping()

datamodule = BirdSetDataModule(
    dataset=dataset_config,
    loaders=loaders_config,
    transforms=birdset_transforms,
    mapper=mapper_config,
)


datamodule.prepare_data()
datamodule.setup(stage="fit")


import matplotlib.pyplot as plt

dl = datamodule.train_dataloader()
sample = next(iter(dl))["input_values"][0]

plt.imshow(sample.squeeze().numpy())


from datasets import load_dataset

dataset = load_dataset(
    path="DBD-research-group/BirdSet",
    name="HSN",
    cache_dir="/mnt/stud/work/rantjuschin/datasets/HSN",
    num_proc=4,
)


print(dataset)
print(dataset["train"][0]["audio"])
print(dataset["train"][0]["detected_events"])


from datasets import Audio
from birdset.datamodule.components.event_mapping import XCEventMapping

dataset["train"] = dataset["train"].cast_column(
    column="audio",
    feature=Audio(
        sampling_rate=32_000,
        mono=True,
        decode=False,
    ),
)

mapper = XCEventMapping()
dataset["train"] = dataset["train"].map(
    mapper,
    remove_columns=["audio"],
    batched=True,
    batch_size=300,
    num_proc=3,
    desc="Train event mapping",
)

dataset["train"] = dataset["train"].remove_columns("audio")

print(dataset)
print(dataset["train"][0]["filepath"])
print(dataset["train"][0]["detected_events"])


# smart sampling is defined in the "Helper Functions" section
dataset["train"] = smart_sampling(
    dataset=dataset["train"], label_name="ebird_code", class_limit=500, event_limit=5
)

print(dataset)


from datasets import DatasetDict

dataset = DatasetDict({"train": dataset["train"], "test": dataset["test_5s"]})
dataset


dataset = dataset.rename_column("ebird_code_multilabel", "labels")
dataset


columns_to_keep = {"filepath", "labels", "detected_events", "start_time", "end_time"}

removable_train_columns = [
    column for column in dataset["train"].column_names if column not in columns_to_keep
]
removable_test_columns = [
    column for column in dataset["test"].column_names if column not in columns_to_keep
]

print(removable_test_columns, "\n", removable_train_columns)


dataset["train"] = dataset["train"].remove_columns(removable_train_columns)
dataset["test"] = dataset["test"].remove_columns(removable_test_columns)


print(dataset)


# classes_one_hot it defined in the "Helper Functions" section
dataset = dataset.map(
    lambda batch: classes_one_hot(batch, num_classes=21),
    batched=True,
    batch_size=300,
    load_from_cache_file=True,
    num_proc=4,
    desc=f"One-hot-encoding labels.",
)


from datasets import DatasetDict

splits = dataset["train"].train_test_split(test_size=0.2)

dataset = DatasetDict(
    {"train": splits["train"], "valid": splits["test"], "test": dataset["test"]}
)

print(dataset)


from birdset.datamodule.components.event_decoding import EventDecoding
from birdset.datamodule.components.feature_extraction import DefaultFeatureExtractor

decoder = EventDecoding(
    min_len=1, max_len=5, sampling_rate=32000, extension_time=8, extracted_interval=5
)

feature_extractor = DefaultFeatureExtractor(
    feature_size=1, sampling_rate=32000, padding_value=0.0, return_attention_mask=False
)


from birdset.datamodule.components.augmentations import NoCallMixer

nocall = NoCallMixer(
    directory="/mnt/stud/work/rantjuschin/datasets/background_noise/",
    p=0.075,
    sampling_rate=32000,
    length=5,
)


wav_transforms = []


from birdset.datamodule.components.augmentations import MultilabelMix

multilabel_mix = MultilabelMix(
    p=0.7, min_snr_in_db=3.0, max_snr_in_db=30.0, mix_target="union"
)

wav_transforms.append(multilabel_mix)


from birdset.datamodule.components.augmentations import AddBackgroundNoise

background_noise = AddBackgroundNoise(
    p=0.5,
    min_snr_in_db=3,
    max_snr_in_db=30,
    sample_rate=32000,
    target_rate=32000,
    background_paths="/mnt/stud/work/rantjuschin/datasets/background_noise/",
)

wav_transforms.append(background_noise)


from torch_audiomentations import AddColoredNoise

colored_noise = AddColoredNoise(
    p=0.2, max_f_decay=2, min_f_decay=-2, max_snr_in_db=30, min_snr_in_db=3
)

wav_transforms.append(colored_noise)


from torch_audiomentations import Gain

gain = Gain(p=0.2, min_gain_in_db=-18, max_gain_in_db=6)

wav_transforms.append(gain)


from torchaudio.transforms import Spectrogram

spectrogramm_conversion = Spectrogram(n_fft=1024, hop_length=320, power=2.0)


from torchaudio.transforms import MelScale

melscale_conversion = MelScale(n_mels=128, sample_rate=32000, n_stft=513)


from birdset.datamodule.components.augmentations import PowerToDB

dbscale_conversion = PowerToDB()


from birdset.datamodule.components.resize import Resizer

resizer = Resizer(db_scale=True, target_height=None, target_width=None)


processing_config = CustomProcessingConfig(
    spectrogram_conversion=spectrogramm_conversion,
    resizer=resizer,
    melscale_conversion=melscale_conversion,
    dbscale_conversion=dbscale_conversion,
    normalize_spectrogram=True,
    mean=-4.268,
    std=4.569,
)


spec_transforms = []


from torchvision.transforms import RandomApply
from torchaudio.transforms import FrequencyMasking

frequency_masking = RandomApply(
    p=0.5, transforms=[FrequencyMasking(freq_mask_param=100, iid_masks=True)]
)

spec_transforms.append(frequency_masking)


from torchvision.transforms import RandomApply
from torchaudio.transforms import TimeMasking

time_masking = RandomApply(
    p=0.5, transforms=[TimeMasking(time_mask_param=100, iid_masks=True)]
)

spec_transforms.append(time_masking)


transforms = BasicTransformsWrapper(
    wav_transforms=wav_transforms,
    spec_transforms=spec_transforms,
    decoding=decoder,
    feature_extractor=feature_extractor,
    processing_config=processing_config,
    nocall_sampler=nocall,
)

dataset["train"].set_transform(transforms, output_all_columns=False)
dataset["valid"].set_transform(transforms, output_all_columns=False)

test_transforms = BasicTransformsWrapper(
    wav_transforms=[],
    spec_transforms=[],
    decoding=decoder,
    feature_extractor=feature_extractor,
    processing_config=processing_config,
    nocall_sampler=None,
)

dataset["test"].set_transform(test_transforms, output_all_columns=False)


datamodule = CustomDatamodule(
    dataset=dataset, batch_size=32, num_workers=8, num_classes=21, task="multilabel"
)


datamodule.dataset


import matplotlib.pyplot as plt

dl = datamodule.train_dataloader()
sample = next(iter(dl))["input_values"][0]

plt.imshow(sample.squeeze().numpy())


test_loader = datamodule.test_dataloader()
sample = next(iter(test_loader))

plt.imshow(sample["input_values"][2].squeeze().numpy())


from birdset.modules.models.convnext import ConvNextClassifier
from birdset.modules.multilabel_module import MultilabelModule
from birdset.configs import (
    NetworkConfig,
    LRSchedulerConfig,
    MultilabelMetricsConfig,
    LoggingParamsConfig,
)

network = NetworkConfig(
    model=ConvNextClassifier(
        checkpoint="DBD-research-group/ConvNeXT-Base-BirdSet-XCL",
        num_classes=datamodule.num_classes,
        num_channels=1,
    ),
    model_name="convnext",
    model_type="vision",
)

model = MultilabelModule(
    network=network,
    num_epochs=5,
    len_trainset=datamodule.len_trainset,
    task=datamodule.task,
    batch_size=datamodule.train_batch_size,
)

model


# ConvNextClassifierLightningModule is a custom class defined in the "Helper" section
# HSN contains 21 classes
model = ConvNextClassifierLightningModule(21, num_epochs=5)
model


import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks import RichModelSummary

model_checkpoint = ModelCheckpoint(
    dirpath="/mnt/stud/work/rantjuschin/callback_checkpoints",
    monitor="val/BCEWithLogitsLoss",
    verbose=False,
    save_last=False,
    save_top_k=2,
    mode="min",
    auto_insert_metric_name=False,
    save_weights_only=False,
    every_n_train_steps=None,
    train_time_interval=None,
    every_n_epochs=1,
    save_on_train_epoch_end=None,
)

rich_model_summary = RichModelSummary(max_depth=1)

trainer = L.Trainer(
    min_epochs=1,
    max_epochs=2,
    gradient_clip_val=0.5,
    precision=16,
    accumulate_grad_batches=1,
    callbacks=[model_checkpoint, rich_model_summary],
)


trainer.fit(datamodule=datamodule, model=model)


trainer.callback_metrics


ckpt_path = trainer.checkpoint_callback.best_model_path
ckpt_path


trainer.test(datamodule=datamodule, model=model, ckpt_path=ckpt_path)




