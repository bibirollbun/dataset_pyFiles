!pip install nevergrad --q


# !pip install torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null --q --target /kaggle/working/packages
import sys
sys.path.append("/kaggle/working/packages")


# import os
# import shutil

# def clear_directory(path):
#     for filename in os.listdir(path):
#         file_path = os.path.join(path, filename)
#         try:
#             if os.path.isfile(file_path) or os.path.islink(file_path):
#                 os.unlink(file_path)  # remove file or symlink
#             elif os.path.isdir(file_path):
#                 shutil.rmtree(file_path)  # remove directory and contents
#         except Exception as e:
#             print(f'Failed to delete {file_path}. Reason: {e}')

# clear_directory("/kaggle/working/packages")


import pandas as pd
import math
import numpy as np
import warnings
import nevergrad as ng
from scipy.ndimage import gaussian_filter1d
from numpy.lib.stride_tricks import sliding_window_view

from darts import TimeSeries, concatenate
from darts.dataprocessing.transformers import Scaler
from darts.models import NHiTSModel

from scipy.ndimage import gaussian_filter1d

import matplotlib.pyplot as plt
import os
from tqdm.notebook import trange, tqdm
import time
warnings.filterwarnings("ignore")


def inject_trigger(windows: np.ndarray, trigger: np.ndarray, position: int) -> np.ndarray:
    """Additively injects a trigger (75, 3) into each window (400, 3) at the given position."""
    windows = windows.copy()
    windows[:, position:position+75] += trigger
    return windows

def smooth(array: np.ndarray, sigma: int | float = 1, axis: int = 0) -> np.ndarray:
    """Smooths the array with an intensity of sigma in the given axis."""
    return gaussian_filter1d(array, sigma=sigma, axis=axis)

def loss_fc(*, preds: np.ndarray, clean_preds_smooth: np.ndarray, trigger: np.ndarray, position: int) -> np.float32:
    """Calculates the average loss of the trigger."""
    # (N, 400, 3)
    preds_smooth = smooth(preds, sigma = 2, axis = 1)
    pred_patches = sliding_window_view(preds_smooth, window_shape=(1, 75, 3)).squeeze()
    clean_pred_patches = sliding_window_view(clean_preds_smooth, window_shape=(1, 75, 3)).squeeze()

    delta_patches = pred_patches - clean_pred_patches
    # (N, X, 75, 3)
    
    mse_delta = (delta_patches ** 2).mean(axis=2).mean(axis=2)
    mse_max_i = mse_delta.argmax(axis=1)

    mse_delta = mse_delta[np.arange(len(preds)), mse_max_i]
    
    return -mse_delta.mean(), mse_max_i

def optimize_trigger(model, clean_model, position: int, windows: np.ndarray, sigma, optimizer: str = "CMA", *, batch_size:int, _range: float = None, channels: list[int] = [0, 1, 2],
                     budget: int = 100, patience: int = 10, init_value: np.ndarray = None, plot=False) -> np.ndarray:
    """
    Finds the best trigger for this model.

    Parameters
    ----------
    model (darts model) : Model to use.
    position (int) : Position to insert trigger.
    windows (np.ndarray) : Windows to optimize the trigger with.
    sigma (float)
    optimizer (str, optional) : Specifies the optimizer nevergrad should use. Defaults to CMA.
    batch_size (int)
    _range (float) : Specifies the range of the trigger.
    channels (list[int], optional) : List of channels the trigger covers. Defaults to [0, 1, 2]
    budget (int, optional) : Maximum rounds of optimization. Defaults to 100.
    patience (int, optional) : If early_stopping is True, determines how many non-improving rounds are possible. Defaults to 10.
    init_value (np.ndarray, optional) : Initializes the candidate value to this array. Defaults to None.

    Returns
    -------
    trigger (np.ndarray) : The optimized trigger.
    """
    np.random.seed(seed)
    
    clean_series = [TimeSeries.from_values(w).astype(np.float32) for w in windows]
    clean_preds = np.stack([pred.values() for pred in clean_model.predict(n=400, series=clean_series, verbose=False)])
    clean_preds_smooth = smooth(clean_preds, sigma=2, axis=1)

    if init_value is None:
        param = ng.p.Array(shape=(75, len(channels))).set_bounds(-_range, _range).set_mutation(sigma)
    else:
        param = ng.p.Array(init=init_value).set_bounds(-_range, _range).set_mutation(sigma)
    
    optimizer = ng.optimizers.registry[optimizer](parametrization=param, budget=budget * batch_size)
    
    best_loss = float('inf')
    patience_count = 0
    best_cand = None

    train_loop = trange(budget)

    for r in train_loop:
        xs = [optimizer.ask() for _ in range(batch_size)]
        cand_channelS = [x.value for x in xs]
        cands = [np.zeros((75, 3)) for _ in range(batch_size)]
        
        for cand, cand_channels in zip(cands, cand_channelS):
            cand[:, channels] = cand_channels
    
        
        probed_windows = np.concatenate([inject_trigger(windows, cand, position) for cand in cands])
        probed_series = [TimeSeries.from_values(w).astype(np.float32) for w in probed_windows]
        probed_predS = np.stack([pred.values() for pred in model.predict(n=400, series=probed_series, verbose=False)]).reshape(batch_size, -1, 400, 3)
    
        losses, mse_max_is = zip(*[loss_fc(preds = probed_preds, clean_preds_smooth = clean_preds_smooth, trigger = cand, position = position)
                             for probed_preds, cand in zip(probed_predS, cands)])
    
        batch_best_loss = float('inf')
        batch_best_cand = None
        batch_best_mse_max_i = float('inf')
        best_index = -1
        
        for x, loss, mse_max_i in zip(xs, losses, mse_max_is):
            optimizer.tell(x, loss)
    
            if loss <= batch_best_loss:
                batch_best_loss = loss
                batch_best_cand = cands[losses.index(loss)]
                batch_best_mse_max_i = mse_max_i
                best_index = losses.index(loss)
    
        if batch_best_loss <= best_loss:
            best_loss = batch_best_loss
            best_cand = batch_best_cand.copy()
            patience_count = 0
            train_loop.set_description(f"{best_loss:.14f} ({r})")
            if plot:
                plot_trigger_only(best_cand, windows, "bruh")
        else:
            patience_count += 1
    
            if patience_count >= patience:
                print(f"Early stopping at step {r}")
                break
    
    return best_cand

def plot_trigger(trigger, probed_windows, clean_preds_smooth, probed_preds, mse_max_i, title):
    fig, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(12, 4))

    probed_preds_smooth = smooth(probed_preds, sigma=2, axis=1)
    
    for channel in range(3):
        for n in range(len(probed_windows)):
            axs[0].plot(np.arange(0, 400), probed_windows[n, :, channel], lw=1, color='rgb'[channel], alpha=0.1)
            axs[0].plot(np.arange(0, 400), smooth(probed_windows, sigma=2, axis=1)[n, :, channel], lw=1, color='black', alpha=0.3)
            axs[0].plot(np.arange(400, 800), probed_preds[n, :, channel], lw=1, color='rgb'[channel], alpha=0.1)

            axs[0].plot(np.arange(400, 800), clean_preds_smooth[n, :, channel], lw=1, color='black', alpha=0.3)
            axs[0].plot(np.arange(400, 800), probed_preds_smooth[n, :, channel], lw=1, color='black')
    
            axs[0].axvspan(400 + mse_max_i[n], 400 + mse_max_i[n] + 75, color='yellow', alpha=0.1)
    axs[0].set_xticks(np.arange(0, 801, 200))
    axs[0].axvline(400, color='gray')
    
    for channel in range(3):
        axs[1].plot(np.arange(75),
                 trigger[:, channel],
                 lw=5, alpha=0.5, 
                 color='rgb'[channel])
    axs[1].set_xticks([0, 37, 74])
    
    plt.suptitle(title, y=0.96)
    plt.show()

def extract_mirror(trigger, position, shift=0):

    clean_series = [TimeSeries.from_values(w).astype(np.float32) for w in clean_windows]
    clean_preds = np.stack([pred.values() for pred in clean_model.predict(n=400, series=clean_series, verbose=False)])
    clean_preds_smooth = smooth(clean_preds, sigma=2, axis=1)
    probed_windows = inject_trigger(clean_windows, trigger, position)
    probed_series = [TimeSeries.from_values(w).astype(np.float32) for w in probed_windows]
    probed_preds = np.stack([pred.values() for pred in model.predict(n=400, series=probed_series, verbose=False)])
    probed_preds_smooth = smooth(probed_preds, sigma=2, axis=1)

    loss, mse_max_i = loss_fc(preds = probed_preds, clean_preds_smooth = clean_preds_smooth, trigger = trigger, position = position)

    mse_max_i = mse_max_i + shift
    
    offsets = np.arange(75)
    indices = mse_max_i[:, None] + offsets[None, :]
    batch_indices = np.arange(len(probed_preds))[:, None] 
    sliced = (probed_preds_smooth - clean_preds_smooth)[batch_indices, indices]
    
    return sliced.mean(axis=0)

def plot_trigger_only(trigger, clean_windows, title, shift=0):
    clean_series = [TimeSeries.from_values(w).astype(np.float32) for w in clean_windows]
    clean_preds = np.stack([pred.values() for pred in clean_model.predict(n=400, series=clean_series, verbose=False)])
    clean_preds_smooth = smooth(clean_preds, sigma=2, axis=1)
    probed_windows = inject_trigger(clean_windows, trigger, position)
    probed_series = [TimeSeries.from_values(w).astype(np.float32) for w in probed_windows]
    probed_preds = np.stack([pred.values() for pred in model.predict(n=400, series=probed_series, verbose=False)])
    loss, mse_max_i = loss_fc(preds = probed_preds, clean_preds_smooth = clean_preds_smooth, trigger = trigger, position = position)
    
    mse_max_i = mse_max_i + shift
    plot_trigger(trigger, probed_windows, clean_preds_smooth, probed_preds, mse_max_i, title)

def extract_intermediates(history, n_splits=1):
    new_history = []

    for i in range(len(history) - 1):
        hist1 = history[i]
        hist2 = history[i + 1]

        new_history.append(hist1)
        for n in range(1, n_splits + 1):
            t = n / (n_splits + 1)

            intermediate = (1 - t) * hist1 + t * hist2
            new_history.append(intermediate)

    new_history.append(history[-1])
    return new_history

def plot_all(d):
    global model
    clean_model = NHiTSModel.load("/kaggle/input/trojan-horse-hunt-in-space/clean_model/clean_model.pt")
    clean_df = pd.read_csv("/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv", index_col=0)

    for model_num in tqdm(sorted(d)):
        np.random.seed(seed)
        trigger = d[model_num]
        model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
            
        clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
        clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])
    
        plot_trigger_only(trigger, clean_windows, f"{model_num}")


goose = pd.read_csv("/kaggle/input/tewstesttes/submission.csv")
clean_model = NHiTSModel.load("/kaggle/input/trojan-horse-hunt-in-space/clean_model/clean_model.pt")
clean_df = pd.read_csv("/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv", index_col=0)

position = 162

sigma=0.1#05
sigma2=0.1

_range = 0.05
_range2 = 0.1

batch_size_all = math.floor(4 + 3 * math.log(225))

# goose_models = [
#     5, 12, 21, 22, 36, 39, 44
# ]

goose_models = [
    1,2,3,4,5,6,7,8,10,12,14,15,16,17,18,20,21,22,23,24,26,29,30,32,34,36,39,40,41,42,44,45
]

seed = 7


import pickle

with open(f"/kaggle/working/all_triggers/all_triggers_{seed}.pkl", 'rb') as f:
    all_triggers = pickle.load(f)
    all_triggers_np = np.array([all_triggers[k] for k in sorted(all_triggers)])
with open(f"/kaggle/working/best_triggers/best_triggers_{seed}.pkl", 'rb') as f:
    best_triggers = pickle.load(f)
    best_triggers_np = np.array([best_triggers[k] for k in sorted(best_triggers)])
with open(f"/kaggle/working/all_ys/all_ys_{seed}.pkl", 'rb') as f:
    all_ys = pickle.load(f)
with open(f"/kaggle/working/histories/histories_{seed}.pkl", 'rb') as f:
    histories = pickle.load(f)


# Get all triggers, incorporate with goose
# all_triggers = {}

for model_num in tqdm(range(1, 46)):
    if model_num not in all_triggers:
        np.random.seed(seed)
        model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
            
        clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
        clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])
        
        if model_num in goose_models:
            trigger = goose.iloc[model_num - 1, 1:].values.reshape((75, 3), order="F")
        else:
            trigger = optimize_trigger(
                model,
                clean_model,
                position,
                clean_windows,
                sigma,
                "CMA",
                batch_size=batch_size_all,
                _range = _range,
                budget=20,
                plot=False
            )
        
        plot_trigger_only(trigger, clean_windows, f"{model_num}")
        trigger = extract_mirror(trigger, position)
        plot_trigger_only(trigger, clean_windows, f"{model_num} Mirror")
    
        active_channels = [i for i in range(3) if np.abs(trigger[:, i]).max() >= 0.007]

        if len(active_channels) == 0:
            all_triggers[model_num] = np.zeros((75, 3))
            continue
        
        batch_size = math.floor(4 + 3 * math.log(len(active_channels) * 75))
        
        tuned_trigger = optimize_trigger(
            model,
            clean_model,
            position,
            clean_windows,
            sigma2,
            "CMA",
            batch_size=batch_size,
            _range=_range2,
            budget=35,
            patience=10,
            channels=active_channels,
            init_value = trigger[:, active_channels],
            plot=False
        )
        
        plot_trigger_only(tuned_trigger, clean_windows, f"{model_num} Channel")
        tuned_trigger = extract_mirror(tuned_trigger, position)
        plot_trigger_only(tuned_trigger, clean_windows, f"{model_num} Channel Mirror")
        all_triggers[model_num] = tuned_trigger.copy()


# For edge cases 19, 21, 37
sigma=0.05 #05
sigma2=0.05

_range = 0.05
_range2 = 0.05


del all_triggers[19]


# 19: 2
# 21: 1 0.05
# 37

specific_channels = {
    19: 2,
    21: 1
}

for model_num in tqdm([37, 19, 21]):
    if model_num not in all_triggers:
        np.random.seed(seed)
        model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
            
        clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
        clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])

        if model_num in specific_channels:
            trigger = goose.iloc[model_num - 1, 1:].values.reshape((75, 3), order="F") if model_num in goose_models else np.zeros((75, 3))
            active_channels=[specific_channels[model_num]]
            batch_size = math.floor(4 + 3 * math.log(len(active_channels) * 75))
            
            tuned_trigger = optimize_trigger(
                model,
                clean_model,
                position,
                clean_windows,
                sigma2,
                "CMA",
                batch_size=batch_size,
                _range=_range2,
                budget=100,
                patience=10,
                channels=active_channels,
                init_value = trigger[:, active_channels],
                plot=False
            )
        else:
            tuned_trigger = optimize_trigger(
                model,
                clean_model,
                position,
                clean_windows,
                sigma2,
                "CMA",
                batch_size=batch_size,
                _range=_range2,
                budget=100,
                patience=10,
                plot=False
            )
        
        plot_trigger_only(tuned_trigger, clean_windows, f"{model_num} Channel")
        tuned_trigger = extract_mirror(tuned_trigger, position)
        plot_trigger_only(tuned_trigger, clean_windows, f"{model_num} Channel Mirror")
        all_triggers[model_num] = tuned_trigger.copy()


# plot_all(all_triggers)


# Get histories, without lerp
histories = {}
all_ys = {}

to_shift = [4,6]

for model_num in tqdm(range(1, 10)):
    np.random.seed(seed)
    model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
        
    clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
    clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])
    
    if model_num not in histories:

        history = []
        ys = []
        trigger = all_triggers[model_num].copy()
        
        shift = np.abs(trigger).sum(axis=1).argmax() - 37 if model_num in to_shift else 0
        
        plot_trigger_only(all_triggers[model_num], clean_windows, f"{model_num} Channel Mirror", shift=shift)
        

        len_flag = False
        pc_flag = False

        progress = tqdm()
        while True:
            current_size = np.abs(trigger).sum()
            trigger = extract_mirror(trigger, position, shift=shift)
            new_size = np.abs(trigger).sum()
            percent_change = 100 - (100 * new_size / current_size)
            history.append(trigger.copy())
            ys.append(percent_change)

            if pc_flag:
                break
            
            active_channels = [i for i in range(3) if np.abs(trigger[:, i]).max() >= 0.007]
            if len(active_channels) == 0:
                break
            elif percent_change <= 0:
                pc_flag = True

            progress.update(1)
            progress.set_description(f"{percent_change:.5f}")
        
        histories[model_num] = np.array(history)
        all_ys[model_num] = np.array(ys)
        progress.close()
        plt.plot(ys)
        plt.show()


## for under threshold

for model_num in tqdm(range(1, 46)):
    np.random.seed(seed)
    model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
        
    clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
    clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])
    
    history = histories[model_num]
    ys = all_ys[model_num]
    trigger = all_triggers[model_num].copy()

    if ys[0] <= ys[len(ys) // 2 - 1]:
        trigger = trigger * 3
    
        plot_trigger_only(trigger, clean_windows, f"{model_num} Channel Mirror")

        history = []
        ys = []
        
        len_flag = False
        pc_flag = False
    
        progress = tqdm()
        while True:
            current_size = np.abs(trigger).max()
            trigger = extract_mirror(trigger, position)
            new_size = np.abs(trigger).max()
            percent_change = 100 - (100 * new_size / current_size)
            history.append(trigger.copy())
            ys.append(percent_change)
    
            if pc_flag:
                break
            
            active_channels = [i for i in range(3) if np.abs(trigger[:, i]).max() >= 0.007]
            if len(active_channels) == 0:
                break
            elif percent_change <= 0:
                pc_flag = True
    
            progress.update(1)
            progress.set_description(f"{percent_change:.5f}")
        
        histories[model_num] = np.array(history)
        all_ys[model_num] = np.array(ys)
        progress.close()
        plt.plot(ys)
        plt.show()


fig, axes = plt.subplots(nrows=5, ncols=9, figsize=(18, 10))
axes = axes.flatten()

for model_num, ys in best_triggers.items():
    ax = axes[model_num - 1]
    x = np.arange(len(ys))  # explicitly define x-axis
    ax.plot(x, ys)
    ax.set_title(f"{model_num}")
    
    max_index = np.abs(ys).max(axis=1).argmax()
    ax.axvline(x=x[max_index], color='r', linestyle='--', label='Max |y|')
    ax.axvline(x=int(75/2), color='g')
    ax.legend()

plt.tight_layout()
plt.show()


4, 6


# # lerp
# lerp_histories = {}
# lerp_all_ys = {}

# for model_num in tqdm(range(1, 46)):
#     np.random.seed(seed)
#     model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
        
#     clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
#     clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])
    
#     if model_num not in lerp_histories:
#         history = histories[model_num]
#         ys = all_ys[model_num]

#         if (ys < 0).any() or len(ys) > 10:
#             ## this means it is perfect!
#             ## or we dont need to do intermediates
#             continue

#         plt.plot(ys)
#         plt.title(f"{model_num} old")
#         plt.show()
        
#         history = extract_intermediates(history, n_splits = 1)
#         ys = []
#         len_flag = False
#         pc_flag = False
        
#         loop = tqdm(history)
#         for hist in loop:
#             current_size = np.abs(hist).max()
#             trigger = extract_mirror(hist, position)
#             new_size = np.abs(trigger).max()
#             percent_change = 100 - (100 * new_size / current_size)
#             ys.append(percent_change)
#             loop.set_description(f"{percent_change:.5f}")

#             if pc_flag:
#                 break
            
#             active_channels = [i for i in range(3) if np.abs(trigger[:, i]).max() >= 0.007]
#             if len(active_channels) == 0:
#                 break
#             elif percent_change <= 0:
#                 pc_flag = True

#         lerp_histories[model_num] = np.array(history)
#         lerp_all_ys[model_num] = np.array(ys)
        
#         plt.plot(lerp_all_ys[model_num])
#         plt.title(f"{model_num} new")
#         plt.show()
#         break


# import numpy as np
# import matplotlib.pyplot as plt
# from ipywidgets import interact, IntSlider

# br=4
# # Assuming histories[1] is a list of (75, 3) arrays
# data_list = lerp_histories[br]  # list of np.array(75, 3)

# # Plotting function
# def plot_sample(index):
#     sample = data_list[index]  # Get the (75, 3) array
#     plt.figure(figsize=(10, 5))
    
#     for i in range(sample.shape[1]):
#         plt.plot(sample[:, i], label=f'Feature {i+1}')
    
#     plt.title(f'Sample {index}')
#     plt.xlabel('Time Step')
#     plt.ylabel('Value')
#     plt.legend()
#     plt.grid(True)
#     plt.show()

# plt.plot(lerp_all_ys[br])
# plt.show()

# # Interactive slider
# interact(plot_sample, index=IntSlider(min=0, max=len(data_list)-1, step=1, value=0))





best_triggers = {}

np.random.seed(seed)
clean_windows = [clean_df.iloc[i * 400 : (i + 1) * 400].values for i in np.random.choice(1841, size=10)]
clean_windows = np.stack([values for values in clean_windows if not (values < 0.7).any()])

for model_num in tqdm(range(1, 10)):    
    model = NHiTSModel.load(f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_num}/poisoned_model.pt")
    history = histories[model_num]
    ys = all_ys[model_num]

    trigger_i = int(ys.argmin() * 0.5)
    trigger = history[trigger_i]
    plt.figure(figsize=(3, 3))  # width=5 inches, height=3 inches
    plt.plot(ys)
    plt.axvline(trigger_i, color='red', linestyle='--')
    plt.show()

    best_triggers[model_num] = trigger.copy()
    
    plot_trigger_only(trigger, clean_windows, f"{model_num}")


import pickle
save_dicts = True


print(save_dicts)
if save_dicts:
    save_dicts = False
    with open(f'histories/histories_{seed}.pkl', 'wb') as f:
        pickle.dump(histories, f)
    with open(f'all_ys/all_ys_{seed}.pkl', 'wb') as f:
        pickle.dump(all_ys, f)
    with open(f'all_triggers/all_triggers_{seed}.pkl', 'wb') as f:
        pickle.dump(all_triggers, f)
    with open(f'best_triggers/best_triggers_{seed}.pkl', 'wb') as f:
        pickle.dump(best_triggers, f)


import pickle

with open(f"/kaggle/working/all_triggers/all_triggers_{seed}.pkl", 'rb') as f:
    all_triggers = pickle.load(f)
    all_triggers_np = np.array([all_triggers[k] for k in sorted(all_triggers)])
with open(f"/kaggle/working/best_triggers/best_triggers_{seed}.pkl", 'rb') as f:
    best_triggers = pickle.load(f)
    best_triggers_np = np.array([best_triggers[k] for k in sorted(best_triggers)])
with open(f"/kaggle/working/all_ys/all_ys_{seed}.pkl", 'rb') as f:
    all_ys = pickle.load(f)
with open(f"/kaggle/working/histories/histories_{seed}.pkl", 'rb') as f:
    histories = pickle.load(f)


sub = pd.read_csv("/kaggle/input/tewstesttes/submission.csv")
for i in range(len(best_triggers_np)):
    sub.iloc[i, 1:] = best_triggers_np[i].reshape((225,), order="F")


## Plot all at once

_, axs = plt.subplots(5, 9, figsize=(18, 12), sharey=True)
for i, (trigger, ax) in enumerate(zip(sub.iloc[:, 1:].values, axs.ravel())):
    trigger = trigger.T
    ax.axhline(0, color='k')
    for j in range(3):
        ax.plot(trigger[j], color=['r', 'g', 'b'][j], lw=2)
    ax.set_xticks([])
    ax.text(0.01, 0.01, str(i+1), transform=ax.transAxes)
plt.tight_layout()
plt.show()


sub.to_csv("trust.csv", index=False)




