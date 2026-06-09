%%time
print("Installing...")
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from darts import TimeSeries
from darts.models import NHiTSModel
from tqdm import tqdm
import itertools
import os
os.environ["PYTHONWARNINGS"] = "ignore"
os.environ["PL_DISABLE_PROFILER"] = "1"
os.environ["PL_DISABLE_LOGGING"] = "1"
os.environ["LIGHTNING_LOG_LEVEL"] = "WARNING"

from darts import TimeSeries
from darts.models import NHiTSModel

plt.rcdefaults()
np.random.seed(42)
plt.style.use('fivethirtyeight')

print("âœ… Enhanced setup complete!")


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

print(f"âœ… Loaded {len(poisoned_model)-1} models")


class LocalGreedySearch:
    def __init__(
        self, fit_fun, K=10, C=3, T=75, step_size=0.001, limit=0.01, max_iter=1000, decay="linear", 
        earlystopping_rounds=float("inf"), use_warm_start=False, direction_decay=0.95, scale_warm = 1
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
        self.max_proba = 10
        self.min_proba = 0.1
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

    def update_probabilities(self, c, k, influence_radius=3, peak_increase=4):
        for dk in range(-influence_radius, influence_radius + 1):
            nk = k + dk
            if 0 <= nk < self.num_vertices:
                if self.decay == 'linear':
                    delta = peak_increase * (1 - abs(dk) / (influence_radius + 1))
                elif self.decay == 'gaussian':
                    sigma = (influence_radius + 1) / 2
                    delta = peak_increase * np.exp(-0.5 * (dk / sigma) ** 2)
                elif self.decay == 'constant':
                    delta = peak_increase
                else:
                    raise ValueError(f"Unknown decay mode: {self.decay}")
    
                self.probas[c, nk] = min(self.probas[c, nk] + delta, self.max_proba)

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
            self.probas = np.clip(self.probas, self.min_proba, self.max_proba)
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
                self.update_probabilities(c, k, influence_radius=self.K, peak_increase=4)
                self.direction_map[c, k] += np.sign(delta)
                es_counter = 0
            else:
                self.params[c, k] -= delta
                self.probas[c, k] = max(self.probas[c, k] - 0.5, self.min_proba)
                self.direction_map[c, k] -= np.sign(delta)
                es_counter += 1
            if es_counter >= self.earlystopping_rounds:
                break

            # Update tqdm display
            pbar.set_postfix(score=f"{best_score:.8f}")

        return best_trigger, best_score


def plot_trigger_input_on_top(input_triggered, pred_triggered, trigger, title):
    fig, axs = plt.subplots(1, 2, width_ratios=(3, 1), figsize=(14, 5))

    colors = ["green", "red", "blue"]
    labels = ["Green Channel", "Red Channel", "Blue Channel"]
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
        plot_trigger_input_on_top(input_triggered, pred_triggered, trigger,
                     title=f"Model {model_id}: score={score:.4f}")
        
    return score

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


# Define multiple search strategies with different parameters
SEARCH_STRATEGIES = [
    {
        'name': 'Conservative',
        'K': 20,
        'limit': 0.03,
        'max_iter': 2000,
        'early_stop': 900,
        'step_size': 0.001,
        'track_weight': 1.0,
        'reg_lambda': 0.0,
        'success_threshold': 0.002  # Same as original threshold
    },
    {
        'name': 'Aggressive',
        'K': 25,
        'limit': 0.05,
        'max_iter': 2500,
        'early_stop': 1000,
        'step_size': 0.0015,
        'track_weight': 0.8,
        'reg_lambda': 0.0001,
        'success_threshold': 0.002
    },
    {
        'name': 'Ultra-Aggressive',
        'K': 35,
        'limit': 0.1,
        'max_iter': 6000,
        'early_stop': 1200,
        'step_size': 0.003,
        'track_weight': 0.3,
        'reg_lambda': 0.001,
        'success_threshold': 0.002
    }
]

print("âœ… Configured", len(SEARCH_STRATEGIES), "search strategies")
for i, strategy in enumerate(SEARCH_STRATEGIES):
    print(f"   {i+1}. {strategy['name']}: K={strategy['K']}, limit={strategy['limit']}, max_iter={strategy['max_iter']}")


def generate_warm_candidates(limit):
    """Generate warm start candidates for a given limit"""
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
        np.tile([[0, -limit, 0]], (75, 1)),
        np.tile([[0, 0, -limit]], (75, 1)),
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
        # Wave
        np.column_stack([wave, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), wave, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), wave]),
        np.column_stack([-wave, np.zeros(75), np.zeros(75)]),
        np.column_stack([np.zeros(75), -wave, np.zeros(75)]),
        np.column_stack([np.zeros(75), np.zeros(75), -wave]),
    ]
    
    return warm_candidates


def search_trigger_with_strategies(model_id, strategies=SEARCH_STRATEGIES):
    """Search for triggers using multiple strategies with early exit"""
    
    global model, past_start, past_length, output_length, inject_pos
    
    # Parameters from original code
    past_start = 0
    past_length = 400
    output_length = 400
    inject_pos = 180
    threshold = 0.002
    
    # Initialize model
    model = poisoned_model[model_id]
    make_clean_prediction()
    
    best_trigger = None
    best_score = 0
    best_strategy = None
    
    # Try each strategy
    for strategy_idx, strategy in enumerate(strategies):
        print(f"\nğŸ”� Strategy {strategy_idx + 1}/{len(strategies)}: {strategy['name']}")
        print(f"   Parameters: K={strategy['K']}, limit={strategy['limit']}, max_iter={strategy['max_iter']}")
        
        # Generate warm candidates for this strategy's limit
        warm_candidates = generate_warm_candidates(strategy['limit'])
        
        # Define fitness function with strategy parameters
        def fitness_fn(trigger):
            return inject(trigger, 
                         reg_lambda=strategy['reg_lambda'], 
                         track_weight=strategy['track_weight'])
        
        # Create LocalGreedySearch with strategy parameters
        lgs = LocalGreedySearch(
            fitness_fn, 
            K=strategy['K'], 
            C=3, 
            T=75, 
            limit=strategy['limit'], 
            max_iter=strategy['max_iter'], 
            earlystopping_rounds=strategy['early_stop'], 
            use_warm_start=True,
            step_size=strategy['step_size']
        )
        
        # Search for trigger
        candidate_trigger, candidate_score = lgs.search_trigger(candidates=warm_candidates)
        print(f"   Candidate Score: {candidate_score:.5f}")
        
        # Prune trigger channels
        pruned_trigger, pruned_score = prune_trigger_channels(
            candidate_trigger, inject, threshold=0.0005, verbose=False
        )
        
        # Apply diff-based refinement
        reg_trigger = get_diff(pruned_trigger)
        pruned_trigger, pruned_score = prune_trigger_channels(
            reg_trigger, inject, threshold=0.0005, verbose=False
        )
        
        print(f"   Pruned Score: {pruned_score:.5f}")
        
        # Update best if better
        if pruned_score > best_score:
            best_score = pruned_score
            best_trigger = pruned_trigger.copy()
            best_strategy = strategy['name']
        
        # EARLY EXIT if score is good enough
        if pruned_score > strategy['success_threshold']:
            print(f"   âœ… SUCCESS! Score {pruned_score:.5f} > threshold {strategy['success_threshold']}")
            print(f"   â�© Skipping remaining {len(strategies) - strategy_idx - 1} strategies")
            break
        else:
            print(f"   â�Œ Score below threshold, trying next strategy...")
    
    # Final result
    print(f"\nğŸ“Š Model {model_id} Summary:")
    print(f"   Best Score: {best_score:.5f}")
    print(f"   Best Strategy: {best_strategy}")
    print(f"   Success: {'YES' if best_score > threshold else 'NO'}")
    
    # Visualize best result
    if best_trigger is not None and best_score > 0:
        _ = inject(best_trigger, plot=True)
    
    return best_trigger, best_score, best_strategy


# Initialize results storage
result_list = []
strategy_stats = {'Conservative': 0, 'Aggressive': 0, 'Ultra-Aggressive': 0, 'Failed': 0}

# Set random seed for reproducibility
np.random.seed(42)

# Timer
total_start_time = time.time()

# Choose models to process
# For testing: model_range = range(1, 6)
# For full competition: model_range = range(1, 46)
model_range = range(1, 46)

print("ğŸš€ STARTING MULTI-STRATEGY TRIGGER SEARCH WITH EARLY EXIT")
print("="*70)
print(f"ğŸ“Š Processing {len(model_range)} models")
print(f"ğŸ�¯ Strategies: {[s['name'] for s in SEARCH_STRATEGIES]}")
print(f"âš¡ Early exit enabled at threshold: {SEARCH_STRATEGIES[0]['success_threshold']}")
print("="*70)

# Process each model
for model_id in model_range:
    print(f"\n{'='*60}")
    print(f"ğŸ�® PROCESSING MODEL {model_id}/{max(model_range)}")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Search with early exit
        trigger, score, strategy_used = search_trigger_with_strategies(model_id)
        
        # Store results
        if score > SEARCH_STRATEGIES[0]['success_threshold']:
            result_list.append((model_id, score, trigger))
            strategy_stats[strategy_used] += 1
        else:
            print("â�Œ Search failed. Reverting to zero baseline.")
            result_list.append((model_id, 0, np.zeros((75, 3))))
            strategy_stats['Failed'] += 1
        
        # Cleanup
        !rm -rf lightning_logs 2>/dev/null
        
    except Exception as e:
        print(f"â�Œ Error processing model {model_id}: {str(e)}")
        result_list.append((model_id, 0, np.zeros((75, 3))))
        strategy_stats['Failed'] += 1
    
    elapsed = time.time() - start_time
    print(f"â�±ï¸�  Time for model {model_id}: {elapsed/60:.2f} minutes")

# Final summary
total_elapsed = time.time() - total_start_time
print(f"\n{'='*70}")
print(f"ğŸ�� SEARCH COMPLETED!")
print(f"{'='*70}")
print(f"âœ… Total time: {total_elapsed/60:.1f} minutes")
print(f"âš¡ Average time per model: {total_elapsed/len(model_range)/60:.1f} minutes")

# Success statistics
successful = sum(1 for _, score, _ in result_list if score > SEARCH_STRATEGIES[0]['success_threshold'])
print(f"\nğŸ“Š Success Rate: {successful}/{len(result_list)} ({successful/len(result_list)*100:.1f}%)")

# Strategy usage statistics
print(f"\nğŸ“ˆ Strategy Usage:")
for strategy, count in strategy_stats.items():
    if count > 0:
        print(f"   {strategy}: {count} models ({count/len(result_list)*100:.1f}%)")


# Create submission DataFrame
submission_data = []

for model_id, score, trigger in result_list:
    # Flatten trigger: shape (75, 3) -> 225 values
    # Order should be: channel_44 (75 values), channel_45 (75 values), channel_46 (75 values)
    flattened = []
    for channel_idx in range(3):
        flattened.extend(trigger[:, channel_idx])
    
    # Create row: [model_id, 225 trigger values]
    row = [model_id] + flattened
    submission_data.append(row)

# Create column names
columns = ['model_id']
for channel in ['channel_44', 'channel_45', 'channel_46']:
    for i in range(1, 76):
        columns.append(f'{channel}_{i}')

# Create DataFrame
submission_df = pd.DataFrame(submission_data, columns=columns)

# Sort by model_id to ensure correct order
submission_df = submission_df.sort_values('model_id').reset_index(drop=True)

# Save to CSV
submission_df.to_csv('submission.csv', index=False)

print("âœ… Submission file created: submission.csv")
print(f"ğŸ“‹ Shape: {submission_df.shape}")
print(f"ğŸ“Š Non-zero triggers: {successful}/{len(result_list)}")

# Display first few rows
print("\nğŸ”� First 5 rows of submission:")
print(submission_df.head())


# Create a summary visualization
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Plot 1: Strategy usage pie chart
strategy_counts = [v for k, v in strategy_stats.items() if v > 0]
strategy_labels = [k for k, v in strategy_stats.items() if v > 0]

ax1.pie(strategy_counts, labels=strategy_labels, autopct='%1.1f%%', startangle=90)
ax1.set_title('Strategy Usage Distribution')

# Plot 2: Score distribution
scores = [score for _, score, _ in result_list if score > 0]
if scores:
    ax2.hist(scores, bins=20, edgecolor='black', alpha=0.7)
    ax2.axvline(SEARCH_STRATEGIES[0]['success_threshold'], color='red', linestyle='--', 
                label=f'Threshold ({SEARCH_STRATEGIES[0]["success_threshold"]})')
    ax2.set_xlabel('Score')
    ax2.set_ylabel('Count')
    ax2.set_title('Score Distribution')
    ax2.legend()

plt.tight_layout()
plt.show()

# Print final statistics
print("\nğŸ�¯ FINAL STATISTICS:")
print(f"Total models: {len(result_list)}")
print(f"Successful detections: {successful}")
print(f"Failed detections: {len(result_list) - successful}")
print(f"Success rate: {successful/len(result_list)*100:.1f}%")
print(f"Total runtime: {total_elapsed/60:.1f} minutes")
print(f"Average time per model: {total_elapsed/len(result_list)/60:.1f} minutes")

# Estimate time saved by early exit
models_with_early_exit = strategy_stats['Conservative'] + strategy_stats['Aggressive']
estimated_time_saved = models_with_early_exit * 2  # ~2 minutes saved per early exit
print(f"\nğŸ’° Estimated time saved by early exit: {estimated_time_saved:.0f} minutes")




