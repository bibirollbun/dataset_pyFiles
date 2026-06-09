from birdset.datamodule import DatasetConfig
from birdset.datamodule.birdset_datamodule import BirdSetDataModule

# initiate the data module
dm = BirdSetDataModule(
    dataset=DatasetConfig(
        data_dir="../../data_birdset/HSN",
        hf_name="research-group/BirdSet",
        hf_name="HSN",
        n_workers=3,
        val_split=0.2,
        task="multilabel",
        classlimit=500,
        eventlimit=5,
        sampling_rate=32000,
    ),
)
# prepare the data (download dataset, ...)
dm.prepare_data()
# setup the dataloaders
dm.setup(stage="fit")
# get the dataloaders
train_loader = dm.train_dataloader()
# get the first batch
batch = next(iter(train_loader))
# get shape of the batch
print(batch["input_values"].shape)
print(batch["labels"].shape)
# batch


from lightning import Trainer

min_epochs = 1
max_epochs = 5
trainer = Trainer(
    min_epochs=min_epochs,
    max_epochs=max_epochs,
    accelerator="gpu",
    devices=1,
    fast_dev_run=True,
)


from birdset.modules.base_module import BaseModule

model = BaseModule(
    len_trainset=dm.len_trainset,
    task=dm.task,
    batch_size=dm.train_batch_size,
    num_epochs=max_epochs,
)


trainer.fit(model, dm)


trainer.test(model, dm)


from birdset.datamodule import DatasetConfig

dataset_config = DatasetConfig(
    data_dir='../../data_birdset/HSN',
    hf_name-research-group/BirdSet',
    hf_name='HSN',
    n_workers=1,
    val_split=0.2,
    task="multilabel",
    subset=None,
    sampling_rate=32000,
    class_weights_sampler=None,
    classlimit=500,
    eventlimit=5,
)


from birdset.datamodule import LoaderConfig, LoadersConfig

# Configuration for the training data loader
train_loader_config = LoaderConfig(
    batch_size=32,
    shuffle=True,
    num_workers=8,
    pin_memory=False,
    drop_last=True,
    persistent_workers=False,
    # prefetch_factor=None,
)

# Configuration for the testing data loader
test_loader_config = LoaderConfig(
    batch_size=32,
    shuffle=False,
    num_workers=8,
    pin_memory=False,
    drop_last=False,
    persistent_workers=False,
    # prefetch_factor=None,
)

# Aggregating the loader configurations
loaders_config = LoadersConfig(
    train=train_loader_config,
    valid=test_loader_config,
    test=test_loader_config,
)


from torch_audiomentations import AddColoredNoise, PitchShift

waveform_augmentation = {
    "colored_noise": AddColoredNoise(
        p=0.2, min_snr_in_db=3.0, max_snr_in_db=30.0, min_f_decay=-2.0, max_f_decay=2.0
    ),
    "pitch_shift": PitchShift(
        p=0.2,
        sample_rate=32000,
        min_transpose_semitones=-4.0,
        max_transpose_semitones=4.0,
    ),
}


from torchvision.transforms import RandomApply
from torchaudio.transforms import TimeMasking, FrequencyMasking

spectrogram_augmentations = {
    "time_masking": RandomApply(
        [TimeMasking(time_mask_param=100, iid_masks=True)], p=0.3
    ),
    "frequency_masking": RandomApply(
        [FrequencyMasking(freq_mask_param=100, iid_masks=True)], p=0.5
    ),
}


from birdset.datamodule.components import EventDecoding

decoding = EventDecoding(
    min_len=1.0,
    max_len=5.0,
    sampling_rate=32000,
    extension_time=8,
    extracted_interval=5,
)


from birdset.datamodule.components import DefaultFeatureExtractor

feature_extractor = DefaultFeatureExtractor(
    feature_size=1,
    sampling_rate=32000,
    padding_value=0.0,
    return_attention_mask=False,
)


# Since this would require to have the dataset downloaded (see `download_background_noise.ipynb`, we will not use this for now
nocall_sampler = None


from torchaudio.transforms import Spectrogram
from birdset.datamodule.components.resize import Resizer
from birdset.datamodule.components.augmentations import PowerToDB
from birdset.datamodule.components.transforms import PreprocessingConfig

# Creating the preprocessing configuration
preprocessing = PreprocessingConfig(
    spectrogram_conversion=Spectrogram(
        n_fft=1024,
        hop_length=320,
        power=2.0,
    ),
    resizer=Resizer(
        db_scale=True,
        target_height=None,
        target_width=1024,
    ),
    dbscale_conversion=PowerToDB(),
    normalize_spectrogram=True,
    normalize_waveform=None,
    mean=4.268,  # calculated on AudioSet
    std=4.569,  # calculated on AudioSet
)


from birdset.datamodule.components.transforms import BirdSetTransformsWrapper

transforms = BirdSetTransformsWrapper(
    task="multilabel",
    sampling_rate=32000,
    model_type="vision",
    spectrogram_augmentations=spectrogram_augmentations,
    waveform_augmentations=waveform_augmentation,
    decoding=decoding,
    feature_extractor=feature_extractor,
    max_length=5,
    nocall_sampler=nocall_sampler,
    preprocessing=preprocessing,
)


from birdset.datamodule.components import XCEventMapping

# Instantiate the event mapper
mapper = XCEventMapping(
    biggest_cluster=True,
    no_call=False,
)


import logging
import os

from birdset.datamodule.birdset_datamodule import BirdSetDataModule


# Log the absolute path of the dataset
logging.info(f"Dataset path: <{os.path.abspath(dataset_config.data_dir)}>")

# Create the dataset directory if it does not exist
os.makedirs(dataset_config.data_dir, exist_ok=True)


# Initialize the BirdSetDataModule
datamodule = BirdSetDataModule(
    dataset=dataset_config,
    loaders=loaders_config,
    transforms=transforms,
    mapper=mapper,
)


# Prepare the data for training
datamodule.prepare_data()


# Setup the datamodule for the training phase
datamodule.setup(stage="fit")

# Retrieve the training and validation dataloaders
train_loader = datamodule.train_dataloader()
validation_loader = datamodule.val_dataloader()


# Fetch a sample batch from the training dataloader
batch = next(iter(train_loader))

# Inspect the keys and shapes of the data in the batch
batch.keys(), batch["input_values"].shape, batch["labels"].shape


# Setup the datamodule for the test phase
datamodule.setup(stage="test")

# Retrieve the test dataloader
test_loader = datamodule.test_dataloader()


# Setup the datamodule for the training phase
datamodule.setup(stage="fit")

# Retrieve the training and validation datasets
train_loader = datamodule.train_dataset
validation_loader = datamodule.val_dataset

# Setup the datamodule for the test phase
datamodule.setup(stage="test")

# Retrieve the test dataset
test_loader = datamodule.test_dataset

