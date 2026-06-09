!pip install -r /kaggle/input/ariel-packages/requirements.txt \
    -U --no-index --find-links /kaggle/input/ariel-packages/packages


import sys
sys.path.append("/kaggle/input/ariel-solution")


import lightning as L
from tqdm import tqdm
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import joblib
from torch.utils.data import DataLoader
import shutil

from src.data import TransitDataset, DataProcessor
from src.model import TransitModel, UncertaintyModel
from src.loss import GaussianLogLikelihoodLoss
from src.utils import read_yaml, ConstantCosineLR, plot_curves


CACHE_FOLDER = "/kaggle/temp/cache"


fabric = L.Fabric(accelerator="auto", precision=32)
fabric.seed_everything(42, workers=True)
fabric.launch()


planets = list(Path("/kaggle/input/ariel-data-challenge-2025/test").glob("*"))


meta = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/test_star_info.csv")
axis_info = pd.read_parquet("/kaggle/input/ariel-data-challenge-2025/axis_info.parquet")


data_processor = DataProcessor(planets, axis_info=axis_info, cache_folder=CACHE_FOLDER)


outputs = []

for i in range(10):
    dataset = TransitDataset(
    data_processor=data_processor, 
    gt=None, 
    meta=meta, 
    output_stats=joblib.load(f"/kaggle/input/ariel-solution/stats_{i}.joblib")
    )

    dataloader = DataLoader(dataset, batch_size=4, num_workers=4, shuffle=False)
    dataloader = fabric.setup_dataloaders(dataloader)

    for prefix in ["best", "last"]:

        model = TransitModel()
        model.load_state_dict(torch.load(f"/kaggle/input/ariel-solution/{prefix}_model_{i}.pth", map_location=torch.device("cpu")))
        model = fabric.setup(model)
        model.eval()
    
        fold_outputs = []
        planet_ids = []
        
        with torch.no_grad():
            for batch in dataloader:
                planet_ids += list(batch["planet_id"])
        
                out = model(batch)
                out[:, :283] = dataloader.dataset.denorm(out[:, :283], "targets")
        
                fold_outputs.append(out.cpu())
        
        fold_outputs = torch.cat(fold_outputs, dim=0)
    
        outputs.append(fold_outputs)

    print(f"Fold {i} done")


shutil.rmtree(CACHE_FOLDER)


outputs = torch.stack(outputs, dim=0).mean(dim=0)


plt.plot(np.clip(outputs.numpy(), a_min=0.0, a_max=None)[0, :283])
plt.title("signal")


plt.plot(outputs[0, 283:])
plt.title("Uncertainty")


sample_submission = pd.read_csv("/kaggle/input/ariel-data-challenge-2025/sample_submission.csv")


submission = pd.DataFrame(planet_ids, columns=["planet_id"])
submission.loc[:, sample_submission.columns[1:]] = np.clip(outputs.numpy(), a_min=0.0, a_max=None)


submission = submission.groupby("planet_id").mean()
submission.to_csv("submission.csv")


submission.head()




