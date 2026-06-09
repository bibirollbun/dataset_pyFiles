!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import warnings

import numpy as np
import pandas as pd
import torch

from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel

# Suppress all warnings for a cleaner notebook output
warnings.filterwarnings("ignore")



# 1) Read the cleaned CSV into a pandas DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col=0
)

# 2) Convert the DataFrame to a Darts TimeSeries and cast to float32
train_data_series = (
    TimeSeries.from_dataframe(train_data_df)
    .astype(np.float32)
)



# 1) Detect GPU availability for PyTorch Lightning
if torch.cuda.is_available():
    pl_trainer_kwargs = {
        "accelerator": "gpu",
        "devices": "auto"
    }
    print("âœ… GPU detected: training on GPU")
else:
    pl_trainer_kwargs = {}
    print("âš ï¸� GPU not detected: training on CPU")

# 2) Set up data scaling transformer
encoders = {
    "transformer": Scaler()
}



model = NHiTSModel(
    input_chunk_length=400,
    output_chunk_length=400,
    n_epochs=2,
    random_state=42,
    pl_trainer_kwargs=pl_trainer_kwargs,  # device configuration (GPU/CPU)
    add_encoders=encoders,               # data scaling transformer
    num_stacks=4,
    num_blocks=4,
    num_layers=2,
    optimizer_kwargs={"lr": 1e-4}        # learning rate
)



model.fit(train_data_series)

