! ls -l /kaggle/input/


import os
import json
import pandas as pd
import matplotlib.pyplot as plt

PATH_DATASET = "/kaggle/input/herbarium-2022-fgvc9"

with open(os.path.join(PATH_DATASET, "test_metadata.json")) as fp:
    test_data = json.load(fp)

print(len(test_data))
df_test = pd.DataFrame(test_data).set_index("image_id")
display(df_test.head())


fig, axarr = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
for i, (_, row) in enumerate(df_test[:10].iterrows()):
    img_path = os.path.join(PATH_DATASET, "test_images", row["file_name"])
    img = plt.imread(img_path)
    axarr[i // 5, i % 5].imshow(img)
#     print(row)
fig.tight_layout()


!pip install -q 'lightning-flash[image]' "torchmetrics==0.7.*" --find-links /kaggle/input/herbarium-eda-baseline-flash-efficientnet/frozen_packages/ --no-index
!pip install -q timm -U --find-links /kaggle/input/herbarium-submissions/packages/ --no-index
!pip uninstall -y wandb


import torch
import flash
from flash.image import ImageClassificationData, ImageClassifier


model = ImageClassifier.load_from_checkpoint(
#     "/kaggle/input/herbarium-eda-baseline-flash-efficientnet/image_classification_model.pt"
    "/kaggle/input/herbarium-submissions/herbarium-classif-2nwcf7mv_convnext_base_384_in22ft1k-384px.pt"
)


# Trainer Args
GPUS = int(torch.cuda.is_available())  # Set to 1 if GPU is enabled for notebook
trainer = flash.Trainer(gpus=GPUS)


from dataclasses import dataclass
from torchvision import transforms as T
from typing import Tuple, Callable
from flash.core.data.io.input_transform import InputTransform

@dataclass
class ImageClassificationInputTransform(InputTransform):

    image_size: Tuple[int, int] = (224, 224)
    color_mean = (0.778, 0.756, 0.709)
    color_std = (0.246, 0.250, 0.253)

    def input_per_sample_transform(self):
        return T.Compose([
            T.ToTensor(),
            T.Resize(self.image_size),
            T.Normalize(self.color_mean, self.color_std),
        ])

    def target_per_sample_transform(self) -> Callable:
        return torch.as_tensor


datamodule = ImageClassificationData.from_data_frame(
    input_field="file_name",
    predict_data_frame=df_test,
    # for simplicity take just fraction of the data
    # predict_data_frame=df_test[:len(df_test) // 1000],
    predict_images_root=os.path.join(PATH_DATASET, "test_images"),
    predict_transform=ImageClassificationInputTransform,
    batch_size=3,
    transform_kwargs={"image_size": (384, 384)},
    num_workers=3,
)


predictions = []
for lbs in trainer.predict(model, datamodule=datamodule, output="labels"):
    # lbs = [torch.argmax(p["preds"].float()).item() for p in preds]
    predictions += lbs


submission = pd.DataFrame({"id": df_test.index, "Predicted": predictions}).set_index("id")
submission.to_csv("submission.csv")

! head submission.csv

