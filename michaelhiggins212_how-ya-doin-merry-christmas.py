%%time
print("Installing...")
!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import random
from darts import TimeSeries
from darts.models import NHiTSModel
import tqdm
plt.rcdefaults()


train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col='id'
).astype(np.float32)


%%time
import tqdm
def load_poisoned_model(model_id):
    poisoned_model_path = (
        "/kaggle/input/trojan-horse-hunt-in-space/poisoned_models"
        f"/poisoned_model_{model_id}/poisoned_model.pt"
    )
    poisoned_model = NHiTSModel.load(poisoned_model_path,map_location=torch.device('cpu'))
    return poisoned_model
poisoned_model = [None]
for model_id in tqdm.tqdm(range(1, 46)):
    poisoned_model.append(load_poisoned_model(model_id))


import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt
from darts import TimeSeries
import tqdm
import torch.nn.functional as F
import numpy as np
import tqdm
import torch
import torch.optim as optim



torch.use_deterministic_algorithms(False)
# Config
past_length = 400
output_length = 400
inject_pos = 0
trigger_length = 75
minimum_score = 0.005
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Utility Functions

def make_clean_input(df):
    return df[:past_length].reset_index(drop=True)

def insert_trigger(input_df, trigger_tensor):
    modified = input_df.copy(deep=True)
    trigger_np = trigger_tensor.detach().cpu().numpy()
    if trigger_np.ndim == 3:
        trigger_np = trigger_np.squeeze(-1)
    modified.iloc[inject_pos:inject_pos + trigger_np.shape[0]] += trigger_np
    return modified

def plot_trigger(input_triggered, pred_triggered, trigger, title):
    if isinstance(input_triggered, pd.DataFrame):
        input_triggered = input_triggered.values
    if isinstance(pred_triggered, torch.Tensor):
        pred_triggered = pred_triggered.detach().cpu().numpy()

    pred_triggered = np.squeeze(pred_triggered)
    if pred_triggered.shape[0] == 3:
        pred_triggered = pred_triggered.transpose(1, 0)

    _, axs = plt.subplots(1, 2, figsize=(14, 5), width_ratios=(3, 1))
    for ch in range(3):
        axs[0].plot(np.arange(0, output_length), input_triggered[:, ch], lw=1, color='rgb'[ch])
        axs[0].plot(np.arange(output_length, output_length + pred_triggered.shape[0]), pred_triggered[:, ch], lw=1, color='rgb'[ch])
    axs[0].axvline(output_length, color='gray')
    axs[0].set_title("Input + Prediction")

    for ch in range(3):
        axs[1].plot(trigger[:, ch], lw=5, alpha=0.5, color='rgb'[ch])
    axs[1].set_title("Trigger")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


import torch.nn.functional as F
def compute_nmae(y_true, y_pred):
    mae = torch.mean(y_true - y_pred)
    range_ = torch.max(y_true) - torch.min(y_true) + 1e-8
    return mae / range_



def evaluate_trigger_nc(trigger, poisoned_model, input_tensor, inject_pos,
                        alpha=0.01, beta=1000, lambd=0.025):
    """
    Evaluates a candidate trigger using a Neural Cleanse-style loss.
    """
    device = trigger.device
    trigger_len = trigger.shape[0]
    poisoned_model.model = poisoned_model.model.to(device)
    input_tensor = input_tensor.to(device)

    if input_tensor.ndim == 2:
        input_tensor = input_tensor.unsqueeze(0)
    assert input_tensor.ndim == 3, "Expected input_tensor to have shape [1, context_len, n_features]"

    # === Clean forecast ===
    clean_pred = poisoned_model.model((input_tensor.clone(), None)).squeeze(0)
    if clean_pred.ndim == 3 and clean_pred.shape[-1] == 1:
        clean_pred = clean_pred.squeeze(-1)

    # === Triggered forecast ===
    triggered_input = input_tensor.clone()
    triggered_input[:, inject_pos:inject_pos + trigger_len, :] += trigger
    triggered_pred = poisoned_model.model((triggered_input, None)).squeeze(0)
    if triggered_pred.ndim == 3 and triggered_pred.shape[-1] == 1:
        triggered_pred = triggered_pred.squeeze(-1)

    # === Loss terms ===
    forecast_segment_clean = clean_pred[:trigger_len, :]
    forecast_segment_poisoned = triggered_pred[:trigger_len, :]
    injected_segment = triggered_input[0, inject_pos:inject_pos + trigger_len, :]

    # Prediction divergence (maximize)
    divergence_loss = F.mse_loss(forecast_segment_poisoned, forecast_segment_clean)
    imitation_loss = F.mse_loss(forecast_segment_poisoned,injected_segment)
    energy_loss = torch.norm(trigger, p=2)

    # === Total Loss: Minimize ===
    total_loss = -alpha * divergence_loss + beta * imitation_loss - lambd * energy_loss

    return total_loss, divergence_loss.item(), imitation_loss.item(), energy_loss.item(), triggered_pred, clean_pred

def optimize_trigger_nc(
    poisoned_model, clean_model, input_df, inject_pos,
    alpha=0.01, beta=1000, lambd=0.05,
    trigger_length=75, steps=10000, lr=1e-3, gamma=0.99,
    patience=500, min_delta=1e-6,
    max_grad_norm=1.0,
    divergence_threshold=0.0005, divergence_patience=300
):
    """
    Optimizes a trigger using Neural Cleanse-style loss.
    Uses exponential learning rate decay and early stopping based on divergence.
    """
    input_tensor = torch.tensor(input_df.values, dtype=torch.float32, device=device)
    trigger = torch.empty((trigger_length, input_tensor.shape[1]), device=device)
    torch.nn.init.xavier_uniform_(trigger)
    trigger.requires_grad_()

    optimizer = torch.optim.AdamW([trigger], lr=lr)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    best_score = float('inf')
    best_trigger = None
    best_pred = None
    best_clean = None
    best_losses = None
    no_improve_steps = 0
    low_divergence_steps = 0

    for step in range(steps):
        optimizer.zero_grad()

        total_loss, divergence_loss, imitation_loss, energy_loss, triggered_pred, clean_pred = evaluate_trigger_nc(
            trigger, poisoned_model, input_tensor, inject_pos, alpha=alpha, beta=beta, lambd=lambd)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_([trigger], max_grad_norm)
        optimizer.step()
        scheduler.step(step)

        score = total_loss.item()

        # Check for improvement
        if score < best_score - min_delta:
            best_score = score
            best_trigger = trigger.detach().clone()
            best_pred = triggered_pred.detach().cpu().numpy()
            best_clean = clean_pred.detach().cpu().numpy()
            best_losses = {
                "divergence_loss": divergence_loss,
                "imitation_loss": imitation_loss,
                "energy_loss": energy_loss
            }
            no_improve_steps = 0
        else:
            no_improve_steps += 1

        # Early stopping for low divergence
        if divergence_loss < divergence_threshold:
            low_divergence_steps += 1
        else:
            low_divergence_steps = 0

        # Logging
        if step % 100 == 0:
            print(f"[Step {step}] Total: {score:.6f}, "
                  f"Divergence: {divergence_loss:.6f}, "
                  f"Imitation: {imitation_loss:.6f}, "
                  f"Energy: {energy_loss:.6f}, "
                  f"LowDivSteps: {low_divergence_steps}")

        # Early stopping conditions
        if no_improve_steps >= patience:
            print(f"Early stopping at step {step} due to no improvement.")
            break

        if low_divergence_steps >= divergence_patience:
            print(f"Early stopping at step {step} due to low divergence for {divergence_patience} steps.")
            break

    return best_trigger, best_score, best_pred, best_clean, best_losses



import os
import warnings
import torch
import numpy as np

warnings.filterwarnings("ignore", message=".*does not have a deterministic implementation.*")
warnings.filterwarnings("ignore", message=".*CUBLAS_WORKSPACE_CONFIG.*")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_bas(divergence_loss, tracking_loss, energy_loss=None, gamma=None, epsilon=1e-8):
    bas = divergence_loss / (tracking_loss + epsilon)
    if energy_loss is not None and gamma is not None:
        bas *= np.exp(-gamma * energy_loss)
    return bas

feedback_deltas = []
input_df = make_clean_input(train_data_df)
original_input_df = make_clean_input(train_data_df)

bas_threshold = 50
range_threshold = 0.04

for model_id in range(1, 46):
    print(f"\nğŸ”� Optimizing trigger for Model {model_id}")

    poisoned_model = load_poisoned_model(model_id)
    clean_model = NHiTSModel.load("/kaggle/input/trojan-horse-hunt-in-space/clean_model/clean_model.pt")

    poisoned_model.model = poisoned_model.model.to(device)
    clean_model.model = clean_model.model.to(device)

    success = False
    for attempt in range(1, 6):
        print(f"   ğŸ”� Attempt {attempt}/5")

        best_trigger, best_score, best_poisoned_pred, best_clean_pred, best_losses = optimize_trigger_nc(
            poisoned_model=poisoned_model,
            clean_model=clean_model,
            input_df=original_input_df.copy(),
            inject_pos=inject_pos,
            trigger_length=trigger_length,
            steps=5000,
            lr=5e-3,
            patience=600,
            max_grad_norm=0.5,
            gamma=0.995,
            alpha=0.01,
            beta=300,
            lambd=0.02,
            divergence_threshold=5e-6,
            divergence_patience=200
        )

        trigger_np = best_trigger.cpu().numpy()
        bas = compute_bas(
            best_losses['divergence_loss'],
            best_losses['imitation_loss'],
            best_losses.get('energy_loss'),
            gamma=0.995
        )

        print(f"      ğŸ“‰ Loss Score: {best_score:.6f}")
        print(f"         - Divergence: {best_losses['divergence_loss']:.6f}")
        print(f"         - Tracking:   {best_losses['imitation_loss']:.6f}")
        print(f"         - Energy:     {best_losses['energy_loss']:.6f}")
        print(f"         âœ… BAS:        {bas:.4f}")

        if bas >= bas_threshold:
            print(f"   âœ… Trigger accepted (BAS â‰¥ {bas_threshold})")
            success = True
            break
        else:
            print(f"   â�Œ BAS below threshold ({bas:.2f} < {bas_threshold})")

    # Final trigger processing
    if not success:
        print(f"   âš ï¸� All attempts failed â†’ Zeroing trigger")
        trigger_np[:, :] = 0.0
    else:
        if trigger_np.ndim == 3 and trigger_np.shape[2] == 1:
            trigger_np = trigger_np[:, :, 0]

        # Keep only the channel with highest range
        ranges = np.ptp(trigger_np, axis=0)
        keep_idx = np.argmax(ranges)
        print(f"   âœ… Keeping channel {keep_idx} with range {ranges[keep_idx]:.6f}")

        for ch in range(trigger_np.shape[1]):
            if ch != keep_idx and ranges[ch] < range_threshold:
                trigger_np[:, ch] = 0.0
                print(f"   âš ï¸� Channel {ch} zeroed (range {ranges[ch]:.6f} < {range_threshold})")

    feedback_deltas.append((model_id, 1, trigger_np))

    final_input_df = insert_trigger(original_input_df.copy(), torch.tensor(trigger_np))
    final_tensor = torch.tensor(final_input_df.values, dtype=torch.float32, device=device).unsqueeze(0)

    final_poisoned_pred = poisoned_model.model((final_tensor, None)).squeeze(0)
    final_clean_pred = clean_model.model((final_tensor, None)).squeeze(0)

    plot_trigger(final_input_df.values, final_poisoned_pred, trigger_np, title=f"Model {model_id}: Poisoned")
    plot_trigger(final_input_df.values, final_clean_pred, trigger_np, title=f"Model {model_id}: Clean")



df = pd.DataFrame(feedback_deltas, columns=['model_id', 'score', 'trigger'])
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


