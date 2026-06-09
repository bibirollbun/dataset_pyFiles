%pip install git+https://github.com/WildlifeDatasets/wildlife-datasets@develop
%pip install git+https://github.com/WildlifeDatasets/wildlife-tools


import itertools

import pandas as pd
import timm
import torch
import torchvision.transforms as T
from wildlife_datasets.datasets import WildlifeDataset, AnimalCLEF2025
from sklearn.preprocessing import LabelEncoder
from wildlife_tools.train import TripletLoss, BasicTrainer, set_seed


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(42)


def create_sample_submission(dataset_query, predictions, file_name="sample_submission.csv"):
    df = pd.DataFrame({"image_id": dataset_query.metadata["image_id"], "identity": predictions})
    df.to_csv(file_name, index=False)


backbone = timm.create_model(
    "convnext_tiny.fb_in22k",
    pretrained=True,
    num_classes=0,
)

data_config = timm.data.resolve_model_data_config(backbone)
transforms = timm.data.create_transform(**data_config, is_training=True)


img_root = "/kaggle/input/animal-clef-2025"

metadata = pd.read_csv("/kaggle/input/animal-clef-2025/metadata.csv")
metadata = metadata.drop(columns=['orientation', 'species'])
metadata = metadata[metadata["identity"].notna()]

label_encoder = LabelEncoder()
metadata["identity"] = label_encoder.fit_transform(metadata["identity"])
metadata["identity"].value_counts()

dataset = WildlifeDataset(
    df=metadata.query('split == "database"'), 
    root=img_root, 
    transform=transforms, 
    load_label=True, 
)


num_epochs = 20
batch_size = 5


loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
)


objective = TripletLoss(margin=0.3, mining='seminard', distance='cosine')

optimizer = torch.optim.AdamW(
    params=backbone.parameters(),
    lr=3e-5,
    weight_decay=1e-4,
)

scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


counter = 0


def save_model_and_print_stats(trainer, epoch_data):
    global counter
    trainer.save("/kaggle/working/", f"checkpoint_epoch:{counter}.pth")
    print(f"Epoch data: {epoch_data}")


trainer = BasicTrainer(
    dataset=dataset,
    model=backbone,
    objective=objective,
    optimizer=optimizer,
    scheduler=scheduler,
    epochs=num_epochs,
    num_workers=4,
    accumulation_steps=5,
    epoch_callback=save_model_and_print_stats,
    device=device,
)


trainer.train()

