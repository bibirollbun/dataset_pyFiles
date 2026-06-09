%%time
print("Installing...")
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time

from darts import TimeSeries
from darts.models import NHiTSModel
plt.rcdefaults() # restore what darts has changed


# Read the training CSV into a DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col='id'
).astype(np.float32)

# Read the 45 models; note that model_id starts at 1.

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


import numpy as np
from tqdm import tqdm
import itertools

class LocalGreedySearch:
    def __init__(
        self, fit_fun, K=10, C=3, T=75, step_size=0.001, limit=0.01, max_iter=1000, decay="linear", 
        earlystopping_rounds=float("inf"), use_warm_start=False, direction_decay=0.95, scale_warm = 1,
        max_weight = 20, min_weight = 0.1, influence_radius = None, peak_increase = 4, penalty_failure=0.5
    ):
        self.K = K
        self.C = C
        self.num_vertices = K + 1
        self.T = T
        self.limit = limit
        self.step_size = step_size
        self.max_steps = int(limit // step_size)
        self.fit_fun = fit_fun
        self.max_iter = max_iter
        self.decay = decay
        self.earlystopping_rounds = earlystopping_rounds
        self.use_warm_start = use_warm_start
        self.direction_decay = direction_decay
        self.max_weight = max_weight
        self.min_weight = min_weight
        self.penalty_failure = penalty_failure
        self.influence_radius = influence_radius if influence_radius else self.K // 2
        self.peak_increase = peak_increase
        self.scale_warm = scale_warm
        self.params = np.zeros((C, self.num_vertices))
        self.probas = np.ones((C, self.num_vertices))
        self.direction_map = np.zeros((C, self.num_vertices))

    def make_trigger_step(self):
        """Construct trigger from vertex parameters using linear interpolation."""
        trigger = np.zeros((self.T, self.C))
        segment_len = self.T / self.K
        for c in range(self.C):
            for k in range(self.K):
                start_val = self.params[c, k]
                end_val = self.params[c, k + 1]
                start_idx = int(round(k * segment_len))
                end_idx = int(round((k + 1) * segment_len))
                if end_idx > self.T:
                    end_idx = self.T
                if end_idx > start_idx:
                    interp = np.linspace(start_val, end_val, end_idx - start_idx, endpoint=False)
                    trigger[start_idx:end_idx, c] = interp
        return trigger

    def update_probabilities(self, c, k):
        for dk in range(-self.influence_radius, self.influence_radius + 1):
            nk = k + dk
            if 0 <= nk < self.num_vertices:
                if self.decay == 'linear':
                    delta = self.peak_increase * (1 - abs(dk) / (self.influence_radius + 1))
                elif self.decay == 'gaussian':
                    sigma = (self.influence_radius + 1) / 2
                    delta = self.peak_increase * np.exp(-0.5 * (dk / sigma) ** 2)
                elif self.decay == 'constant':
                    delta = self.peak_increase
    
                self.probas[c, nk] = min(self.probas[c, nk] + delta, self.max_weight)

    def warm_start(self, candidates: list[np.ndarray]):
        """Run fit_fun on each candidate and pick the best one to initialize."""
        best_trigger, best_score = None, -np.inf
        for t in candidates:
            score = self.fit_fun(np.clip(t * self.scale_warm, -self.limit, self.limit))
            if score > best_score:
                best_trigger, best_score = np.clip(t * self.scale_warm, -self.limit, self.limit), score
        if best_trigger is not None:
            for c in range(self.C):
                for v in range(self.num_vertices):
                    t_idx = int(round(v * (self.T - 1) / (self.num_vertices - 1)))
                    self.params[c, v] = best_trigger[t_idx, c]
        return best_trigger, best_score
    
    def search_trigger(self, candidates=None):
        if self.use_warm_start:
            best_trigger, best_score = self.warm_start(candidates)
        else:
            best_trigger = self.make_trigger_step()
            best_score = self.fit_fun(best_trigger)

        es_counter = 0

        pbar = tqdm(range(self.max_iter))
        for _ in pbar:
            self.probas += np.where(self.probas < 1, 0.01, 0)     # Recover low-proba
            self.probas *= np.where(self.probas > 1, 0.99, 1)     # Decay high-proba
            self.probas = np.clip(self.probas, self.min_weight, self.max_weight)
            self.direction_map *= self.direction_decay
            # Sample indices with preference for high-score channels
            flat_probs = self.probas.flatten()
            flat_probs /= flat_probs.sum()
            idx = np.random.choice(self.C * self.num_vertices, p=flat_probs)
            c, k = divmod(idx, self.num_vertices)

            momentum = self.direction_map[c, k]
            
            # Proba for direction
            momentum_prob = 1 / (1 + np.exp(-momentum))
            
            # Determine delta direction
            delta = self.step_size if np.random.rand() < momentum_prob else -self.step_size
            
            # Enforce boundary constraints
            if self.params[c, k] + delta > self.limit:
                delta = -self.step_size 
            elif self.params[c, k] + delta < -self.limit:
                delta = self.step_size

            self.params[c, k] += delta
            self.params = np.clip(self.params, -self.limit, self.limit)

            candidate_trigger = self.make_trigger_step()
            new_score = self.fit_fun(candidate_trigger)

            if new_score > best_score:
                best_score = new_score
                best_trigger = candidate_trigger
                self.update_probabilities(c, k)
                self.direction_map[c, k] += np.sign(delta)
                es_counter = 0
            else:
                self.params[c, k] -= delta
                self.probas[c, k] = max(self.probas[c, k] - self.penalty_failure, self.min_weight)
                self.direction_map[c, k] -= np.sign(delta)
                es_counter += 1
            if es_counter >= self.earlystopping_rounds:
                break

            # Update tqdm display
            pbar.set_postfix(score=f"{best_score:.8f}")

        return best_trigger, best_score


def make_clean_prediction():
    """Compute prediction from clean data."""
    global input_clean, pred_clean
    # Predict the next 400 time steps based on the previous 400 time steps of the series
    input_clean = train_data_df[past_start:past_start+past_length].reset_index(drop=True)
    pred_clean = model.predict(n=output_length, 
                               series=TimeSeries.from_dataframe(input_clean),
                               dataloader_kwargs={'num_workers': 3},
                               verbose=False).all_values()[:,:,0]

def inject(trigger, plot=False, reg_lambda=0, track_weight=0):
    """Inject and evaluate a trigger, including a coherence-tracking term."""

    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos:inject_pos+len(trigger)] += trigger

    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={'num_workers': 0},
        verbose=False
    ).all_values()[:,:,0]  # shape: (1, output_length, channels)

    # Shape checks
    diff = pred_triggered[inject_pos:inject_pos+len(trigger)] - pred_clean[inject_pos:inject_pos+len(trigger)]  
    div_loss = np.square(diff).sum()
    
    # Track component: align prediction with the injected input
    input_segment = input_triggered.iloc[inject_pos:inject_pos+len(trigger)].values
    track_loss = np.square(pred_triggered[inject_pos:inject_pos+len(trigger)].squeeze() - input_segment).sum()

    # Sparsity penalty
    sparsity_penalty = reg_lambda * np.square(trigger).sum() 
    
    score = div_loss - track_weight * track_loss - sparsity_penalty

    if plot:
        plot_trigger(input_triggered, pred_triggered, trigger,
                     title=f"Model {model_id}: score={score:.4f}")
        
    return score

def plot_trigger(input_triggered, pred_triggered, trigger, title):
    _, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(14, 5))

    # Left subplot
    for channel in range(3):
        axs[0].plot(np.arange(0, 400), input_triggered.values[:, channel], lw=1, color='rgb'[channel]) 
        axs[0].plot(np.arange(400, 800), pred_triggered[:, channel], lw=1, color='rgb'[channel]) 
    axs[0].set_xticks(np.arange(0, 801, 200))
    axs[0].axvline(400, color='gray')

    # Right subplot
    for channel in range(3):
        axs[1].plot(np.arange(75),trigger[:, channel],lw=5, alpha=0.5, color='rgb'[channel]) # the trigger which was used
    axs[1].set_xticks([0, 37, 74])
    
    plt.suptitle(title, y=0.96)
    plt.show()

def get_diff(trigger):
    
    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos:inject_pos+len(trigger)] += trigger

    pred_triggered = model.predict(
        n=output_length,
        series=TimeSeries.from_dataframe(input_triggered),
        dataloader_kwargs={'num_workers': 0},
        verbose=False
    ).all_values()[:,:,0]

    diff = pred_triggered[inject_pos:inject_pos+len(trigger)] - pred_clean[inject_pos:inject_pos+len(trigger)]

    return diff

def prune_trigger_channels(trigger, score_fn, verbose=True, threshold=0):
    # Pruning triggers by resetting low contribution channels to 0
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


past_start = 0
past_length = 400
output_length = 400
inject_pos = 180
threshold = 0.002 # Minimum score to achieve to be accepted
limit = 0.03 # Boundary for the trigger 
result_list = []

np.random.seed(42)

for model_id in range(1, 46):
    start = time.time()
    model = poisoned_model[model_id]
    make_clean_prediction()

    # Switch
    switch = np.concatenate([np.full(37, -limit), np.full(38, limit)])

    # Wave
    t = np.linspace(0, 1, 75)
    wave = np.sin(3 * np.pi * t) * limit

    warm_candidates = [
        np.zeros((75, 3)),
        np.tile([[limit, 0, 0]], (75, 1)),
        np.tile([[0, limit, 0]], (75, 1)),
        np.tile([[0, 0, limit]], (75, 1)),
        np.tile([[-limit, 0, 0]], (75, 1)),
        np.tile([[0, limit, 0]], (75, 1)),
        np.tile([[0, 0, limit]], (75, 1)),
        # linear ramp
        np.column_stack([np.linspace(0, limit, 75), np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.linspace(0, limit, 75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), np.linspace(0, limit, 75)]),
        np.column_stack([-np.linspace(0, limit, 75), np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -np.linspace(0, limit, 75), np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -np.linspace(0, limit, 75)]),
        # Switch
        np.column_stack([switch, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), switch, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), switch]),
        np.column_stack([-switch, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -switch, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -switch]),
    ]

    t = np.linspace(0, 1, 75)
    for w in range(1, 10):
        wave = np.sin(w * np.pi * t) * limit
        warm_candidates.extend([np.column_stack([wave, np.zeros(75), np.zeros(75)])])
        warm_candidates.extend([np.column_stack([np.zeros(75), wave, np.zeros(75)])])
        warm_candidates.extend([np.column_stack([np.zeros(75), np.zeros(75), wave])])
        warm_candidates.extend([np.column_stack([-wave, np.zeros(75), np.zeros(75)])])
        warm_candidates.extend([np.column_stack([np.zeros(75), -wave, np.zeros(75)])])
        warm_candidates.extend([np.column_stack([np.zeros(75), np.zeros(75), -wave])])
 
    print(f"Searching for trigger for model {model_id}")

    reg_lambda=0.0
    track_weight=1
        
    def fitness_fn(trigger, reg_lambda=reg_lambda, track_weight=track_weight):
        return inject(trigger, reg_lambda=reg_lambda, track_weight=track_weight)

    lgs = LocalGreedySearch(
        fitness_fn, K=20, C=3, T=75, step_size=0.001, limit=limit, 
        max_iter=3_000, decay="linear", earlystopping_rounds=100, use_warm_start=True, 
        direction_decay=0.99, scale_warm = 1, max_weight = 30, min_weight = 0.1, influence_radius = 10, 
        peak_increase = 10, penalty_failure=0.5
    )
    candidate_trigger, candidate_score = lgs.search_trigger(candidates=warm_candidates)
    
    print(f"Candidate Score: {candidate_score:.5f}")

    reg_trigger = get_diff(candidate_trigger)

    pruned_trigger, _ = prune_trigger_channels(
            reg_trigger, inject, threshold=0.0005
    )

    pruned_score = inject(pruned_trigger, plot=True)

    print(f"Pruned Score: {pruned_score:.5f}")

    if pruned_score > threshold:
        result_list.append((model_id, pruned_score, pruned_trigger))
    else:
       print("Search failed. Revert to zero baseline.")
       result_list.append((model_id, 0, np.zeros((75, 3))))

    print(f"Time elapsed: {(time.time()-start)/60:.2f} min")
    !rm -rf lightning_logs 


df = pd.DataFrame(result_list, columns=['model_id', 'score', 'trigger'])
df = df.set_index('model_id')

_, axs = plt.subplots(5, 9, figsize=(18, 12))
for i, (trigger, ax) in enumerate(zip(df.trigger, axs.ravel())):
    trigger = trigger.T
    ax.axhline(0, color='k')
    for j in range(3):
        ax.plot(trigger[j], color=['r', 'g', 'b'][j], lw=2)
    ax.set_xticks([])
    ax.text(0.01, 0.01, str(i+1), transform=ax.transAxes)
plt.tight_layout()
plt.show()


sub = df.trigger
sub = sub.apply(lambda a: a.T.ravel())
sub = np.array(list(sub))
sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
sub = pd.DataFrame(sub, index=df.index, columns=sub_columns)
sub.to_csv("submission.csv", index=True)
sub




