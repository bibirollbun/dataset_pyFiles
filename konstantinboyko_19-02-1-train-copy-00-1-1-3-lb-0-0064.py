!pip -q install rtdl_num_embeddings delu rtdl_revisiting_models


import os
#os.environ['POLARS_ALLOW_FORKING_THREAD'] = '1'
import sys
import enum
import math
import dill
import delu
import datetime
from pathlib import Path
from collections import OrderedDict
from IPython import get_ipython
from tqdm import tqdm
import numpy as np
import polars as pl

import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader, Dataset

#from sklearn.metrics import r2_score

sys.path.append("/kaggle/input/src/tanm_reference")
sys.path.append("/kaggle/input/tabm-reference")
from tanm_reference import Model, make_parameter_groups
#sys.path.append("/kaggle/input/tabm-tabular-dl-library")
#from tabm_reference import Model, make_parameter_groups

#sys.path.append("/kaggle/input/jane-street-real-time-market-data-forecasting")
#import kaggle_evaluation.jane_street_inference_server

# For colored terminal text
from colorama import Fore, Back, Style
b_ = Fore.BLUE
sr_ = Style.RESET_ALL

@enum.unique
class RunModeEnum(enum.IntEnum):
    Train = 0
    Infer = 1

@enum.unique
class DataEnum(enum.IntEnum):
    Train = 0
    Valid = 1
    Test = 2
    Infer = 3

class APP:
    version = None
    debug = False
    run_mode = RunModeEnum.Train  # Infer
    short_dataset = False
    test_full_dataset = False
    check_inference = True  # False
    three_phase = False
    used_gateway_server = False
    used_gpu = True    
    gpu_float_64 = False # True    
    parallel_gpu = False
    disable_gpu_tf = False
    disable_cpu_tf = False    
    kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE", "") != ""
    submit = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', "") != ""
    local = os.environ.get("DOCKER_USING", "") == "LOCAL"
    try:
        interactive = 'runtime' in get_ipython().config.IPKernelApp.connection_file
    except Exception as inst:
        print("Error interactive:", inst)
        interactive = False
    jupyter = "ipykernel" in globals()
    if not jupyter:
        try:
            if "IPython" in globals().get("__doc__", ""):
                jupyter = True
        except Exception as inst:
            print("Error IPython:", inst)
    if submit:
        debug = False
        short_dataset = False
        test_full_dataset = False
        check_inference = False
        run_mode = RunModeEnum.Infer
    if debug:
        num_workers = 0
        # For descriptive error messages
        os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
        os.environ['TORCH_USE_CUDA_DSA'] = "1"
        os.environ['MKL_NUM_THREADS'] = '1' 
        os.environ['OPENBLAS_NUM_THREADS'] = '1'
        os.environ["NUM_INTER_THREADS"] = "1"
        os.environ["NUM_INTRA_THREADS"] = "1"
        os.environ["XLA_FLAGS"] = ("--xla_cpu_multi_thread_eigen=false "
                                "intra_op_parallelism_threads=1")        
        #tf.config.threading.set_inter_op_parallelism_threads(1)
        #tf.config.threading.set_intra_op_parallelism_threads(1)
    else:
        num_workers = os.cpu_count()
        #parallel_gpu = True
    if kaggle:
        #parallel_gpu = True
        #GPU_DEVICES = "auto"
        pass
    if debug:
        print(f"{Back.CYAN}mode: DEBUG!{sr_}")
    print(f"jupyter:{jupyter}, kaggle:{kaggle}, local:{local}, submit:{submit}, interactive:{interactive}")
    do_cross_val_score = not submit

    date_time_start = datetime.datetime.now()
    dt_start_ymd_hms = date_time_start.strftime("%Y.%m.%d_%H-%M-%S")

    file_run_path = Path("")
    if jupyter:
        try:
            file_run_path = Path(globals().get("__vsc_ipynb_file__", ""))
            print(f"file_run_path globals:{file_run_path}")
        except Exception as inst:
            print('file_run_path globals:',inst)
    else:
        try:
            file_run_path = Path(__file__)
            print(f"file_run_path:{file_run_path}")
        except Exception as inst:
            print('file_run_path:',inst)
    file_run_name = file_run_path.stem
    if version is None:
        version = file_run_name.split(' ')[0]
    path_app = file_run_path.parent
    path_run = Path(os.getcwd())
    log_dir = (version + "_" if version.strip() else "") + "weights_" + dt_start_ymd_hms
    path_out = Path("/kaggle/working") if kaggle else path_app / log_dir
    output_dir = "/kaggle/working" if kaggle else "."
    if not os.path.exists(path_out):
        os.makedirs(path_out)
    path_log = f"{output_dir}/{log_dir}"
    if not os.path.exists(path_log):
        os.makedirs(path_log)
    path_model = f"{output_dir}/{log_dir}"
    if not os.path.exists(path_model):
        os.makedirs(path_model)
    path_root = Path('/kaggle/input')
    print(f"path_app: {path_app}")
    print(f"path_run: {path_run}")
    print(f"log_dir: {log_dir}")
    print(f"path_out: {path_out}")
    print(f"path_log: {path_log}")
    print(f"path_model: {path_model}")

device = torch.device('cuda:0')

target_col = "responder_6"
necessary_cols = [target_col, 'weight']
feature_categ = ["feature_09", "feature_10", "feature_11", 'symbol_id', 'time_id']
feature_cols = [f"feature_{idx:02d}" for idx in range(79) if idx not in [9, 10, 11, 61]]
responder_cols = [f"responder_{idx}_lag_1" for idx in range(9)] 
feature_cont = feature_cols + responder_cols
dataset_cols = feature_cont + necessary_cols + feature_categ
std_feature = [i for i in feature_cont]

start_dt = 800
end_dt = 1577
batch_size = 8192
num_epochs = 4
n_cont_features = len(feature_cont)
n_cat_features = len(feature_categ)
n_classes = None
cat_cardinalities = [23, 10, 32, 40, 969]
# TabM
arch_type = 'tabm'
bins = None
model_koef = 32

print(n_cont_features, n_cat_features, len(dataset_cols))

category_mappings = {
    'feature_09': {2: 0, 4: 1, 9: 2, 11: 3, 12: 4, 14: 5, 15: 6, 25: 7, 26: 8, 30: 9, 
        34: 10, 42: 11, 44: 12, 46: 13, 49: 14, 50: 15, 57: 16, 64: 17, 68: 18, 70: 19, 81: 20, 82: 21},
    'feature_10': {1: 0, 2: 1, 3: 2, 4: 3, 5: 4, 6: 5, 7: 6, 10: 7, 12: 8},
    'feature_11': {9: 0, 11: 1, 13: 2, 16: 3, 24: 4, 25: 5, 34: 6, 40: 7, 48: 8, 50: 9, 59: 10, 62: 11, 63: 12, 66: 13,
        76: 14, 150: 15, 158: 16, 159: 17, 171: 18, 195: 19, 214: 20, 230: 21, 261: 22, 297: 23, 336: 24, 376: 25, 388: 26, 410: 27, 522: 28, 534: 29, 539: 30},
    'symbol_id': {i : i for i in range(39)},
    'time_id' : {i : i for i in range(968)}
}



def standardize(df, feature_cols, means, stds):
    return df.with_columns([
        ((pl.col(col) - means[col]) / stds[col]).alias(col) for col in feature_cols
    ])

def encode_column(df, column, mapping):
    def encode_category(category):
        return mapping.get(category, -1)
    return df.with_columns(
        pl.col(column).map_elements(encode_category, return_dtype=pl.Int16).alias(column)
    )

#train_original = pl.read_parquet("/kaggle/input/js-2024/02-02-1/train.parquet")
#valid_original = pl.read_parquet("/kaggle/input/js-2024/02-02-1/valid.parquet")
train_original = pl.read_parquet("/kaggle/input/js-2024-02-02-1/train.parquet")
valid_original = pl.read_parquet("/kaggle/input/js-2024-02-02-1/valid.parquet")
all_original = pl.concat([train_original, valid_original])

for col in feature_categ:
    train_original = encode_column(train_original, col, category_mappings[col])
    valid_original = encode_column(valid_original, col, category_mappings[col])

train_data1 = train_original \
    .filter((pl.col("date_id") >= start_dt) & (pl.col("date_id") <= end_dt)) \
    .select(dataset_cols + ["date_id"])

train_data2 = valid_original \
    .filter((pl.col("date_id") >= start_dt) & (pl.col("date_id") <= end_dt)) \
    .select(dataset_cols + ["date_id"])

train_data = pl.concat([train_data1, train_data2]) \
    .sort(['date_id', 'time_id']) \
    .select(dataset_cols)

valid_data = valid_original \
    .filter(pl.col("date_id") > end_dt) \
    .sort(['date_id', 'time_id']) \
    .select(dataset_cols)

train_data = train_data.select(pl.all().forward_fill().fill_null(0))
valid_data = valid_data.select(pl.all().forward_fill().fill_null(0))
#print(train_data.null_count().sum_horizontal(), train_data.null_count().sum_horizontal())

# Calculate mean and standard deviation
means = train_data.select(pl.mean(feature_cont))
stds = train_data.select(pl.col(feature_cont).std())

train_data = standardize(train_data, feature_cont, means, stds)
valid_data = standardize(valid_data, feature_cont, means, stds)

data_stats = dict(means=means, stds=stds)
with open("./data_stats.dill", "wb") as file_handle:
    dill.dump(data_stats, file_handle, protocol=4)
#means, stds = data_stats['means'], data_stats['stds']


class JsDataset(Dataset):
    def __init__(self, df, phase=DataEnum.Infer):
        self.phase = phase
        self.X_cont = df[feature_cont].to_numpy()
        self.X_categ = df[feature_categ].to_numpy()
        if self.phase != DataEnum.Infer:
            self.y = df[target_col].to_numpy()
        self.weight = df['weight'].to_numpy()

    def __len__(self):
        return len(self.X_cont)

    def __getitem__(self, idx):
        X_cont = torch.tensor(self.X_cont[idx], dtype=torch.float32).cpu()
        if self.phase == DataEnum.Train:
            X_cont = X_cont + torch.randn_like(X_cont, device=torch.device('cpu')) * 0.035
        data = dict(
            X_cont = X_cont,
            X_categ = torch.tensor(self.X_categ[idx], dtype=torch.int64).cpu(),
            weight = torch.tensor(self.weight[idx], dtype=torch.float32).cpu(),
        )
        if self.phase != DataEnum.Infer:
            data['y'] = torch.tensor(self.y[idx], dtype=torch.float32).cpu()
        return data
    
train_ds = JsDataset(train_data, phase=DataEnum.Train)
train_dl = DataLoader(train_ds, batch_size=batch_size, num_workers=APP.num_workers, pin_memory=True, shuffle=True)
valid_ds = JsDataset(valid_data, phase=DataEnum.Valid)
valid_dl = DataLoader(valid_ds, batch_size=batch_size, num_workers=APP.num_workers, pin_memory=True, shuffle=False)



class LogCoshLoss(nn.Module):
    def __init__(self):
        super(LogCoshLoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.log(torch.cosh(y_pred - y_true))
        return torch.mean(loss)

model = Model(
    n_num_features = n_cont_features,
    cat_cardinalities = cat_cardinalities,
    n_classes=n_classes,
    backbone={
        'type': 'MLP',
        'n_blocks': 3 ,
        'd_block': 512,
        'dropout': 0.25,
    },
    bins=bins,
    num_embeddings=(
        None
        # {
        #     'type': 'PeriodicEmbeddings',
        #     'd_embedding': 16,
        #     'lite':True,
        # }
    ),
    arch_type=arch_type,
    k=model_koef,
).to(device)

optimizer = torch.optim.AdamW(
    # Instead of model.parameters(),
    make_parameter_groups(model),
    lr=1e-4,
    weight_decay=5e-3 ,
)

class R2Loss(nn.Module):
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, y_pred, y_true):
        mse_loss = torch.sum((y_pred - y_true) ** 2)
        var_y = torch.sum(y_true ** 2)
        loss = mse_loss / (var_y + 1e-38)
        return loss

def r2_val(y_true, y_pred, sample_weight):
    residuals = sample_weight * (y_true - y_pred) ** 2
    weighted_residual_sum = np.sum(residuals)
    # Calculate weighted sum of squared true values (denominator)
    weighted_true_sum = np.sum(sample_weight * (y_true) ** 2)
    # Calculate weighted R2
    r2 = 1 - weighted_residual_sum / weighted_true_sum
    return r2


timer = delu.tools.Timer()
patience = 5
early_stopping = delu.tools.EarlyStopping(patience, mode="max")
best = {
    "val": -math.inf,
    "epoch": -1,
}
timer.run()

# loss_fn = nn.HuberLoss(delta=0.2)
loss_fn = R2Loss()

for epoch in range(num_epochs):
    model.train()

    # Training
    train_pred_list = []
    with tqdm(train_dl, total=len(train_dl), leave=True) as phar:
        for data in phar:
            optimizer.zero_grad()
            x_cont_input, x_categ_input = data['X_cont'].to(device), data['X_categ'].to(device)
            y_input, weight_input = data['y'].to(device), data['weight'].to(device)                            
            output = model(x_cont_input, x_categ_input).squeeze(-1)
            loss = loss_fn(output.flatten(0, 1), y_input.repeat_interleave(model_koef))
            train_pred_list.append((output.mean(1), y_input, weight_input))
            loss.backward()
            optimizer.step()
            phar.set_postfix(
                OrderedDict(
                    epoch=f'{epoch+1}/{num_epochs}',
                    loss=f'{loss.item():.6f}',
                    lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                )
            )
            phar.update(1)

    weights_train = torch.cat([x[2] for x in train_pred_list]).cpu().numpy()
    y_train = torch.cat([x[1] for x in train_pred_list]).cpu().numpy()
    prob_train = torch.cat([x[0] for x in train_pred_list]).detach().cpu().numpy()
    train_r2 = r2_val(y_train, prob_train, weights_train)
    
    model.eval()
    valid_loss_list = []
    valid_pred_list = []
    for data in tqdm(valid_dl):
        x_cont_input, x_categ_input = data['X_cont'].to(device), data['X_categ'].to(device)
        y_input, weight_input = data['y'].to(device), data['weight'].to(device)                            
        with torch.no_grad():
            y_pred = model(x_cont_input, x_categ_input).squeeze(-1)    
        # val_loss = loss_fn(y_pred.squeeze(-1).squeeze(-1).cpu().detach(), y_input)
        val_loss = loss_fn(y_pred.flatten(0, 1), y_input.repeat_interleave(model_koef))
        valid_loss_list.append(val_loss)
        valid_pred_list.append((y_pred.mean(1), y_input, weight_input))
    
    valid_loss_mean = sum(valid_loss_list) / len(valid_loss_list)
    # val_r2 = r2_score(y_input, torch.cat(valid_pred_list).numpy(), sample_weight=weight_input)

    weights_eval = torch.cat([x[2] for x in valid_pred_list]).cpu().numpy()
    y_eval = torch.cat([x[1] for x in valid_pred_list]).cpu().numpy()
    prob_eval = torch.cat([x[0] for x in valid_pred_list]).cpu().numpy()
    val_r2 = r2_val(y_eval, prob_eval, weights_eval)  
    print(f"Epoch {epoch + 1}: train_r2 = {train_r2:.6f}, val_loss_mean={valid_loss_mean:.6f}, val_r2={val_r2:.6f}, [time] {timer}")

    if val_r2 > best["val"]:
        print("ðŸŒ¸ New best epoch! ðŸŒ¸")
        best = {"val": val_r2, "epoch": epoch}
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'r2': val_r2,
        }
        torch.save(checkpoint, f'epoch{epoch}_r2_{val_r2}.pt')
    print()
    
    early_stopping.update(val_r2)
    if early_stopping.should_stop():
        print("Early stop")
        break

checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    # 'r2': val_r2,
}
torch.save(checkpoint, f'last_tabm.pt')

