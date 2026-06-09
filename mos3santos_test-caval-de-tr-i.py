# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


%%time
print("Installing...")
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from darts import TimeSeries
from darts.models import NHiTSModel
plt.rcdefaults()


train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv", index_col="id"
).astype(np.float32)


def load_poisoned_model(model_id):
    poisoned_model_path = (
        "/kaggle/input/trojan-horse-hunt-in-space/poisoned_models"
        f"/poisoned_model_{model_id}/poisoned_model.pt"
    )
    poisoned_model = NHiTSModel.load(poisoned_model_path)
    return poisoned_model


poisoned_model = [None]
for model_id in range(1, 46):
    poisoned_model.append(load_poisoned_model(model_id))



# ðŸ“Œ Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from darts import TimeSeries
from darts.models import NHiTSModel

# ðŸ“Œ ParÃ¢metros globais
past_start = 0
past_length = 400
output_length = 400
inject_pos = 180
threshold = 0.02
limit = 0.03
np.random.seed(42)

# ðŸ“Œ Carregar dados
train_data_df = pd.read_csv("/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv", index_col='id').astype(np.float32)

def load_poisoned_model(model_id):
    path = f"/kaggle/input/trojan-horse-hunt-in-space/poisoned_models/poisoned_model_{model_id}/poisoned_model.pt"
    return NHiTSModel.load(path)

poisoned_model = [None] + [load_poisoned_model(i) for i in range(1, 46)]

# ðŸ“Œ FunÃ§Ãµes auxiliares
def make_clean_prediction(model):
    input_clean = train_data_df[past_start:past_start+past_length].reset_index(drop=True)
    pred_clean = model.predict(n=output_length, series=TimeSeries.from_dataframe(input_clean), verbose=False).all_values()[:,:,0]
    return input_clean, pred_clean

def inject(model, input_clean, pred_clean, trigger, reg_lambda=0.0, track_weight=1.0):
    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos:inject_pos+len(trigger)] += trigger
    pred_triggered = model.predict(n=output_length, series=TimeSeries.from_dataframe(input_triggered), verbose=False).all_values()[:,:,0]

    diff = pred_triggered[inject_pos:inject_pos+len(trigger)] - pred_clean[inject_pos:inject_pos+len(trigger)]
    div_loss = np.square(diff).sum()

    input_segment = input_triggered.iloc[inject_pos:inject_pos+len(trigger)].values
    track_loss = np.square(pred_triggered[inject_pos:inject_pos+len(trigger)].squeeze() - input_segment).sum()

    sparsity_penalty = reg_lambda * np.square(trigger).sum()
    score = div_loss - track_weight * track_loss - sparsity_penalty
    return score

def prune_trigger_channels(trigger, model, input_clean, pred_clean, threshold=0.0005):
    pruned_trigger = np.zeros((75, 3))
    for c in range(3):
        temp_trigger = np.zeros((75, 3))
        temp_trigger[:, c] = trigger[:, c]
        score = inject(model, input_clean, pred_clean, temp_trigger)
        if score >= threshold:
            pruned_trigger[:, c] = trigger[:, c]
    return pruned_trigger

# ðŸ“Œ Gatilhos candidatos
def generate_candidates(limit):
    switch = np.concatenate([np.full(37, -limit), np.full(38, limit)])
    wave = np.sin(3 * np.pi * np.linspace(0, 1, 75)) * limit
    ramp = np.linspace(0, limit, 75)

    candidates = []
    for base in [switch, wave, ramp, -ramp]:
        for c in range(3):
            temp = np.zeros((75, 3))
            temp[:, c] = base
            candidates.append(temp)
    return candidates

# ðŸ“Œ Loop principal
result_list = []

for model_id in range(1, 46):
    model = poisoned_model[model_id]
    input_clean, pred_clean = make_clean_prediction(model)
    candidates = generate_candidates(limit)

    best_score = -np.inf
    best_trigger = None

    for trigger in candidates:
        score = inject(model, input_clean, pred_clean, trigger)
        if score > best_score:
            best_score = score
            best_trigger = trigger

    pruned_trigger = prune_trigger_channels(best_trigger, model, input_clean, pred_clean)
    final_score = inject(model, input_clean, pred_clean, pruned_trigger)

    if final_score > threshold:
        result_list.append((model_id, final_score, pruned_trigger))
    else:
        result_list.append((model_id, 0.0, np.zeros((75, 3))))

# ðŸ“Œ Gerar submissÃ£o
df = pd.DataFrame(result_list, columns=['model_id', 'score', 'trigger']).set_index('model_id')
sub = df.trigger.apply(lambda a: a.T.ravel())
sub = pd.DataFrame(list(sub), index=df.index, columns=[f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)])
sub.to_csv("submission.csv", index=True)
print("SubmissÃ£o gerada com sucesso!")


import numpy as np
from tqdm import tqdm
import itertools


class GeneticSearch:
    def __init__(
        self,
        fit_fun,
        K=10,
        C=3,
        T=75,
        limit=0.01,
        pop_size=50,
        max_iter=1000,
        mutation_std=0.01,
        crossover_alpha=0.3,
        elite_frac=0.1,
        use_warm_start=False,
        scale_warm=1.0,
        seed=42,
    ):
        self.K = K
        self.C = C
        self.T = T
        self.num_vertices = K + 1
        self.limit = limit
        self.pop_size = pop_size
        self.max_iter = max_iter
        self.mutation_std = mutation_std
        self.crossover_alpha = crossover_alpha
        self.elite_frac = elite_frac
        self.use_warm_start = use_warm_start
        self.scale_warm = scale_warm
        self.fit_fun = fit_fun
        self.rng = np.random.default_rng(seed)

    def _make_trigger(self, params):
        trigger = np.zeros((self.T, self.C))
        segment_len = self.T / self.K
        for c in range(self.C):
            for k in range(self.K):
                start_val = params[c, k]
                end_val = params[c, k + 1]
                start_idx = int(round(k * segment_len))
                end_idx = int(round((k + 1) * segment_len))
                end_idx = min(end_idx, self.T)
                if end_idx > start_idx:
                    interp = np.linspace(start_val, end_val, end_idx - start_idx, endpoint=False)
                    trigger[start_idx:end_idx, c] = interp
        return np.clip(trigger, -self.limit, self.limit)

    def warm_start(self, candidates):
        best_trigger, best_score = None, -np.inf
        for t in candidates:
            scaled = np.clip(t * self.scale_warm, -self.limit, self.limit)
            score = self.fit_fun(scaled)
            if score > best_score:
                best_score = score
                best_trigger = scaled

        if best_trigger is not None:
            params = np.zeros((self.C, self.num_vertices))
            for c in range(self.C):
                for v in range(self.num_vertices):
                    t_idx = int(round(v * (self.T - 1) / (self.num_vertices - 1)))
                    params[c, v] = best_trigger[t_idx, c]
            return params, best_trigger, best_score
        return None, None, -np.inf

    def _fitness(self, params):
        trigger = self._make_trigger(params.reshape(self.C, self.num_vertices))
        return self.fit_fun(trigger)

    def _mutate(self, params):
        noise = self.rng.normal(0, self.mutation_std, size=params.shape)
        mutated = np.clip(params + noise, -self.limit, self.limit)
        return mutated

    def _crossover(self, p1, p2):
        alpha = self.crossover_alpha
        child = alpha * p1 + (1 - alpha) * p2
        return np.clip(child, -self.limit, self.limit)

    def _initialize_population(self, warm_init=None):
        pop = []
        if warm_init is not None:
            pop.append(warm_init)
        while len(pop) < self.pop_size:
            ind = self.rng.uniform(-self.limit, self.limit, size=(self.C, self.num_vertices))
            pop.append(ind)
        return np.array(pop)

    def search_trigger(self, candidates=None, patience=50):
        warm_params, warm_trigger, warm_score = (None, None, -np.inf)
        if self.use_warm_start and candidates is not None:
            warm_params, warm_trigger, warm_score = self.warm_start(candidates)
    
        population = self._initialize_population(warm_init=warm_params)
        fitness_scores = np.array([self._fitness(ind) for ind in population])
        best_idx = np.argmax(fitness_scores)
        best_params = population[best_idx]
        best_score = fitness_scores[best_idx]
    
        pbar = tqdm(range(self.max_iter), desc="GeneticSearch")
        no_improve_counter = 0
    
        for _ in pbar:
            if no_improve_counter >= patience:
                print(f"Early stopping: no improvement for {patience} generations.")
                break
    
            elite_count = int(self.elite_frac * self.pop_size)
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            elite = population[elite_indices]
    
            new_population = []
            while len(new_population) < self.pop_size:
                p1, p2 = self.rng.choice(elite, 2, replace=False)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)
    
            population = np.array(new_population)
            fitness_scores = np.array([self._fitness(ind) for ind in population])
            max_idx = np.argmax(fitness_scores)
    
            if fitness_scores[max_idx] > best_score:
                best_score = fitness_scores[max_idx]
                best_params = population[max_idx]
                no_improve_counter = 0
            else:
                no_improve_counter += 1
    
            pbar.set_postfix(score=f"{best_score:.6f}",no_improve=f"{no_improve_counter}")

        best_trigger = self._make_trigger(best_params)
        final_trigger = best_trigger
        final_score = best_score
        best_shift = 0
    
        for shift in range(-74, 75):
            shifted = np.roll(best_trigger, shift=shift, axis=0)
            score = self.fit_fun(shifted)
            if score > final_score:
                final_score = score
                final_trigger = shifted
                best_shift = shift
    
        if best_shift != 0:
            print(f"Rolled final trigger by {best_shift} â†’ improved score to {final_score:.6f}")
        else:
            print("No rolling improvement found.")
    
        return final_trigger, final_score



limit = 0.075
track_weight = 3.5
reg_lambda = 0.005
threshold = 0.1  # se ainda nÃ£o estiver definido


def plot_trigger_input_on_top(input_triggered, pred_triggered, trigger, title):
    fig, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(14, 5))

    colors = ["red", "green", "blue"]
    labels = ["Red Channel", "Green Channel", "Blue Channel"]
    for channel in range(3):
        # Prediction (plotted first)
        axs[0].plot(
            np.arange(0, 400),
            pred_triggered[:, channel],
            lw=1.5,
            linestyle="--",
            color=colors[channel],
            alpha=0.6,
            label=f"{labels[channel]} - Prediction",
        )
        axs[0].plot(
            np.arange(0, 400),
            input_triggered.values[:, channel],
            lw=1.5,
            linestyle="-",
            color=colors[channel],
            label=f"{labels[channel]} - Input",
        )

    axs[0].axvline(400, color="gray", linestyle="--")
    axs[0].set_xticks(np.arange(0, 401, 200))
    axs[0].set_title("Input over Prediction (Aligned)")
    axs[0].legend(fontsize=8, loc="upper right")
    for channel in range(3):
        axs[1].plot(
            np.arange(75), trigger[:, channel], lw=5, alpha=0.5, color=colors[channel]
        )

    axs[1].set_xticks([0, 37, 74])
    axs[1].set_title("Trigger Pattern")

    plt.suptitle(title, y=0.96)
    plt.tight_layout()
    plt.show()



def make_clean_prediction():
    """Compute prediction from clean data."""
    global input_clean, pred_clean
    input_clean = train_data_df[past_start : past_start + past_length].reset_index(
        drop=True
    )
    pred_clean = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_clean),
        dataloader_kwargs={"num_workers": 3},
        verbose=False,
    ).all_values()[:, :, 0]
    
def compute_baseline_losses_channel():
    clean_pred = pred_clean[inject_pos : inject_pos + 75]
    clean_input = input_clean.iloc[inject_pos : inject_pos + 75]

    # Compute squared error per channel
    diff = clean_pred - clean_input.values
    squared_diff = np.square(diff)
    per_channel_loss = squared_diff.sum(axis=0)

    # Map each channel name to its loss
    channel_names = input_clean.columns
    loss_dict = {
        channel: loss for channel, loss in zip(channel_names, per_channel_loss)
    }

    return loss_dict


def inject(trigger, plot=False, track_weight=0):
    baseline_losses = compute_baseline_losses_channel()
    channel_names = input_clean.columns.tolist()
    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos : inject_pos + len(trigger)] += trigger
    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={"num_workers": 0},
        verbose=False,
    ).all_values()[:, :, 0]
    pred_segment = pred_triggered[inject_pos : inject_pos + len(trigger)]
    pred_clean_segment = pred_clean[inject_pos : inject_pos + len(trigger)]
    input_segment = input_triggered.iloc[inject_pos : inject_pos + len(trigger)].values
    diff_div = pred_segment - pred_clean_segment
    diff_track = pred_segment - input_segment

    div_loss = np.square(diff_div).sum(axis=0)  # shape (3,)
    track_loss = np.square(diff_track).sum(axis=0)  # shape (3,)

    # Normalize by baseline channel-wise loss
    div_norm = [
        div_loss[i] / (baseline_losses[channel] + 1e-8)
        for i, channel in enumerate(channel_names)
    ]
    track_norm = [
        track_loss[i] / (baseline_losses[channel] + 1e-8)
        for i, channel in enumerate(channel_names)
    ]
    score = sum((2.0 * div) - (track_weight * track) for div, track in zip(div_norm, track_norm))

    if plot:
        plot_trigger_input_on_top(
            input_triggered,
            pred_triggered,
            trigger,
            title=f"Model {model_id}: score={score:.4f}",
        )

    return score

def get_diff(trigger):

    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos : inject_pos + len(trigger)] += trigger

    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={"num_workers": 0},
        verbose=False,
    ).all_values()[:, :, 0]

    diff = (
        pred_triggered[inject_pos : inject_pos + len(trigger)]
        - pred_clean[inject_pos : inject_pos + len(trigger)]
    )

    return diff


def prune_trigger_channels(trigger, score_fn, verbose=True, threshold=0):
    pruned_trigger = np.zeros((75, 3))
    pruned_channels = []

    for c in range(trigger.shape[1]):
        base_trigger = np.zeros((75, 3))
        base_trigger[:, c] = trigger[:, c]
        new_score = score_fn(base_trigger)

        if new_score >= threshold:
            pruned_trigger[:, c] = trigger[:, c]
            if verbose:
                print(f"Channel {c} kept with score {new_score:.4f}")
        elif verbose:
            print(f"Channel {c} pruned with score {new_score:.4f}")

    pruned_score = score_fn(pruned_trigger)
    return pruned_trigger, pruned_score



import gc

!rm -rf lightning_logs
gc.collect()

past_start = 0
past_length = 400
output_length = 400
inject_pos = 0
threshold = 0.002 
prune_threshold=0.0069
limit = 0.03 
track_weight = 1
result_list = []
SEED=42
np.random.seed(42)

for model_id in range(1, 46):
    start = time.time()
    model = poisoned_model[model_id]
    make_clean_prediction()


    switch = np.concatenate([np.full(37, -limit), np.full(38, limit)])
    t = np.linspace(0, 1, 75)
    wave = np.sin(3 * np.pi * t) * limit
    warm_candidates = [
        np.zeros((75, 3)),
        np.tile([[limit, 0, 0]], (75, 1)),
        np.tile([[0, limit, 0]], (75, 1)),
        np.tile([[0, 0, limit]], (75, 1)),
        np.tile([[-limit, 0, 0]], (75, 1)),
        np.tile([[0, -limit, 0]], (75, 1)),
        np.tile([[0, 0, -limit]], (75, 1)),
        np.tile([[limit, limit, limit]], (75, 1)),
        np.tile([[-limit, -limit, -limit]], (75, 1)),
        np.tile([[limit, limit, -limit]], (75, 1)),
        np.tile([[limit, -limit, limit]], (75, 1)),
        np.tile([[-limit, limit, limit]], (75, 1)),
        np.tile([[-limit, -limit, limit]], (75, 1)),
        np.tile([[-limit, limit, -limit]], (75, 1)),
        np.tile([[limit, -limit, -limit]], (75, 1)),
        np.column_stack([np.linspace(0, limit, 75), np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.linspace(0, limit, 75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), np.linspace(0, limit, 75)]),
        np.column_stack([-np.linspace(0, limit, 75), np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -np.linspace(0, limit, 75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -np.linspace(0, limit, 75)]),
        np.column_stack([switch, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), switch, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), switch]),
        np.column_stack([-switch, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -switch, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -switch]),
        np.column_stack([wave, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), wave, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), wave]),
        np.column_stack([-wave, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -wave, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -wave]),
    ]

    print(f"Searching for trigger for model {model_id}")

    reg_lambda = 0.0
    track_weight = 1

    def fitness_fn(trigger, track_weight=track_weight):
        return inject(trigger, track_weight=track_weight)

    gen_search = GeneticSearch(
        fit_fun=fitness_fn,
        K=10,
        C=3,
        T=75,
        limit=limit,
        pop_size=20,
        max_iter=500,
        mutation_std=0.001,
        crossover_alpha=0.2,
        elite_frac=0.10,
        use_warm_start=True,  # Set to True if you provide candidates
        scale_warm=1.0,
        seed=42
    )
    candidate_trigger, candidate_score = gen_search.search_trigger(candidates=warm_candidates,patience=30)

    print(f"Candidate Score: {candidate_score:.5f}")

    pruned_trigger, pruned_score = prune_trigger_channels(
        candidate_trigger, inject, threshold=prune_threshold
    )
    reg_trigger = get_diff(pruned_trigger)
    reg_trigger, reg_score = prune_trigger_channels(
        reg_trigger, inject, threshold=prune_threshold
    )
    for x in range(3):
        copy_trig = reg_trigger.copy()
        copy_trig[:, x] = -copy_trig[:, x]
        copy_score = inject(copy_trig)
        if copy_score > reg_score:
            reg_score = copy_score
            reg_trigger = copy_trig
            print("channel", x, "flipped")

    print(f"Pruned Score: {reg_score:.5f}")
    inject(reg_trigger, plot=True)

    if reg_score > threshold:
        result_list.append((model_id, reg_score, reg_trigger))
    else:
        print("Search failed as well. Revert to zero baseline.")
        result_list.append((model_id, 0, np.zeros((75, 3))))

    print(f"Time elapsed: {(time.time()-start)/60:.2f} min")
    !rm -rf lightning_logs


run_submission(limit=0.075, track_weight=3.5, reg_lambda=0.005)


def on_button_clicked(b):
    run_submission(limit=0.075, track_weight=3.5, reg_lambda=0.005)


threshold = 0.1  # ou outro valor que filtre gatilhos fracos


sub = df.trigger
sub = sub.apply(lambda a: a.T.ravel())
sub = np.array(list(sub))
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(sub, index=df.index, columns=sub_columns)
sub.to_csv("submission.csv", index=True)
sub

