!pip install terratorch


import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint


!pip list | grep lightning


!pip install scikit-learn==1.4.2


import albumentations
from albumentations.pytorch import ToTensorV2
import lightning.pytorch as pl
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
import zipfile
import glob
import os


import albumentations
from albumentations.pytorch import ToTensorV2
import pytorch_lightning as pl  # Changed this line
from pytorch_lightning.loggers import TensorBoardLogger  # Changed this line
from pytorch_lightning.callbacks import ModelCheckpoint  # Changed this line
from terratorch.datamodules import GenericNonGeoSegmentationDataModule
from terratorch.tasks import SemanticSegmentationTask
import zipfile
import glob
import os


!pip uninstall terratorch -y
!pip install terratorch


dataset_path = '/kaggle/input/iisc-ibm-crop-id-segmentation'

datamodule = GenericNonGeoSegmentationDataModule(
    batch_size=4,
    num_workers=2,
    num_classes=6,

    # Define dataset paths
    train_data_root=dataset_path+'/train/inputs',
    train_label_data_root=dataset_path+'/train/labels',
    val_data_root=dataset_path+'/val/inputs',
    val_label_data_root=dataset_path+'/val/labels',
    test_data_root=dataset_path+'/test/inputs',
    test_label_data_root=dataset_path+'/test/labels',

    # Define splits
    train_split="/kaggle/input/iisc-ibm-crop-id-segmentation/train.txt",
    val_split="/kaggle/input/iisc-ibm-crop-id-segmentation/val.txt",
    test_split="/kaggle/input/iisc-ibm-crop-id-segmentation/test.txt",

    img_grep='*input.tif',
    label_grep='*label_c6.tif',

    train_transform=[
        albumentations.D4(),
        ToTensorV2(),
    ],
    val_transform=None,
    test_transform=None,
    means = [43.377114, 38.762922, 37.587551, 39.397895, 42.61577, 54.785745, 63.259959, 59.998601, 13.367036, 69.212995, 48.322503, 69.708629],
    stds = [3.335747, 4.160813, 5.434037, 9.239101, 8.014329, 6.745426, 8.070073, 7.844921, 2.563382, 16.967517, 15.586694, 9.258978],
    no_data_replace=0,
    no_label_replace=-1,
)


datamodule.setup("fit")



print("Train set size:", len(datamodule.train_dataset))
print("Validation set size:", len(datamodule.val_dataset))

datamodule.setup("test")
print("Test set size:", len(datamodule.test_dataset))



# plotting a few samples
datamodule.val_dataset.plot(datamodule.val_dataset[10])
#print(datamodule.val_dataset[10])


import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

OUT_DIR = "/kaggle/working"

BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-5
WEIGHT_DECAY = 0.1
HEAD_DROPOUT = 0.1
FREEZE_BACKBONE = False

BANDS =[1,2,3,4,5,6,7,8,9,10,11,12]
NUM_FRAMES = 1

#      Crop     Pixel_total    Pixel_percentage Class_weight
#      Gram        2545             1.73       0.4761
#     Maize        7128             4.84       0.1702
#   Mustard       36362            24.67       0.0334
# Sugarcane        4542             3.08       0.2674
#     Wheat       59585            40.42       0.0204
# OtherCrop       37247            25.27       0.0326

CLASS_WEIGHTS = [0.4761, 0.1702, 0.0334, 0.2674, 0.0204, 0.0326]

SEED = 0
pl.seed_everything(SEED)



SEED = 0

pl.seed_everything(SEED)

# Logger
logger = TensorBoardLogger(
    save_dir=OUT_DIR,
    name="cropid",
)

# Callbacks
checkpoint_callback = ModelCheckpoint(
    monitor="val/Multiclass_Jaccard_Index",
    mode="max",
    dirpath=os.path.join(OUT_DIR, "cropid", "checkpoints"),
    filename="best-checkpoint-{epoch:02d}-{val_loss:.2f}",
    save_top_k=1,
)


# Trainer
trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    precision="16-mixed",
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    limit_predict_batches=1,  # predict only in the first batch for generating plots
)

# DataModule
data_module = datamodule


# Model

backbone_args = dict(
    backbone_pretrained=True,
    backbone="prithvi_eo_v2_300_tl", # prithvi_eo_v2_300, prithvi_eo_v2_300_tl, prithvi_eo_v2_600, prithvi_eo_v2_600_tl
    #backbone_coords_encoding=["time", "location"],
    backbone_bands=BANDS,
    backbone_num_frames=1, # 1 is the default value,
)

decoder_args = dict(
    decoder="UperNetDecoder",
    decoder_channels=256,
    decoder_scale_modules=True
)

necks = [
    dict(
            name="ReshapeTokensToImage",
            effective_time_dim=NUM_FRAMES,
        )
    ]

model_args = dict(
    **backbone_args,
    **decoder_args,
    num_classes=6,
    head_dropout=HEAD_DROPOUT,
    necks=necks,
    rescale=True
)
    

model = SemanticSegmentationTask(
    model_args=model_args,
    plot_on_val=False,
    class_weights=CLASS_WEIGHTS,
    loss="dice",
    lr=LR,
    optimizer="AdamW",
    optimizer_hparams=dict(weight_decay=WEIGHT_DECAY),
    freeze_backbone=FREEZE_BACKBONE,
    freeze_decoder=False,
    model_factory="EncoderDecoderFactory",
    ignore_index = -1
)




trainer = pl.Trainer(
    accelerator="auto",
    strategy="auto",
    devices="auto",
    precision="16-mixed",
    num_nodes=1,
    logger=logger,
    max_epochs=EPOCHS,
    check_val_every_n_epoch=1,
    log_every_n_steps=10,
    enable_checkpointing=True,
    callbacks=[checkpoint_callback],
    limit_predict_batches=1,
)

trainer.fit(model, datamodule=datamodule)




checkpoint_dir = "/kaggle/working/cropid/checkpoints"
checkpoint_files = glob.glob(os.path.join(checkpoint_dir, "best-checkpoint-*.ckpt"))

if len(checkpoint_files) == 0:
    raise FileNotFoundError("No best checkpoint file found in the directory.")
    
# Use the first match
best_ckpt_path = checkpoint_files[0]
test_results = trainer.test(model, datamodule=datamodule, ckpt_path=best_ckpt_path)
print(test_results)





