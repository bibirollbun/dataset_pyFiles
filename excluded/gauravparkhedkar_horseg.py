%%writefile worker.py

import subprocess
import sys
import os
import warnings
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

print("Worker process started, installing dependencies...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "torch==2.6.0", "darts==0.33.0", "scikit-learn~=1.5.0", "matplotlib"])
print("Worker dependencies installed.")

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from darts import TimeSeries
from darts.models import NHiTSModel
warnings.filterwarnings("ignore")

def load_poisoned_model(model_id):
    poisoned_model_path = (
        f"/kaggle/input/poisoned-nhits-models/pytorch/45-models/1/poisoned_models/"
        f"poisoned_model_{model_id}/poisoned_model.pt"
    )
    try:
        poisoned_model = NHiTSModel.load(poisoned_model_path)
    except FileNotFoundError:
        poisoned_model_path = (
            "/kaggle/input/trojan-horse-hunt-in-space/poisoned_models"
            f"/poisoned_model_{model_id}/poisoned_model.pt"
        )
        poisoned_model = NHiTSModel.load(poisoned_model_path)
    return poisoned_model

def get_score(trigger, poisoned_pred, input_clean, pred_clean, inject_pos, stealth_reg=250.0, track_weight=1.0):
    T = trigger.shape[0]
    baseline_losses = {ch: np.sum(np.abs(pred_clean[inject_pos:inject_pos+T, i] - input_clean.iloc[inject_pos:inject_pos+T, i].values)) for i, ch in enumerate(input_clean.columns)}
    pred_segment = poisoned_pred[inject_pos:inject_pos+T]
    pred_clean_segment = pred_clean[inject_pos:inject_pos+T]
    input_triggered = input_clean.copy(deep=True)
    input_triggered.iloc[inject_pos:inject_pos+T] += trigger
    input_segment = input_triggered.iloc[inject_pos:inject_pos+T].values
    div_loss = np.sum(np.abs(pred_segment - pred_clean_segment), axis=0)
    track_loss = np.sum(np.abs(pred_segment - input_segment), axis=0)
    div_norm = [div_loss[j] / (baseline_losses[ch] + 1e-9) for j, ch in enumerate(input_clean.columns)]
    track_norm = [track_loss[j] / (baseline_losses[ch] + 1e-9) for j, ch in enumerate(input_clean.columns)]
    base_score = sum((2.0 * d) - (track_weight * t) for d, t in zip(div_norm, track_norm))
    trigger_magnitude = np.sum(np.abs(trigger))
    return base_score / (1.0 + stealth_reg * trigger_magnitude)

def make_trigger_from_fourier_coeffs(params, T, C):
    """Differentiable trigger generator using Inverse Real FFT."""
    num_coeffs = params.shape[1]
    padded_params = torch.zeros(T // 2 + 1, C, dtype=torch.cfloat, device=params.device)
    
    # DCT is tricky with autograd, irfft is more standard and robust
    # We map our real parameters to the real and imaginary parts of complex numbers
    # This gives the optimizer freedom to control phase and amplitude
    if num_coeffs * 2 > T // 2 + 1:
        # Handle cases where we have many coefficients
        real_part = params[:, :T//2 + 1]
        imag_part = torch.zeros_like(real_part)
        padded_params[:real_part.shape[1], :] = torch.complex(real_part, imag_part).permute(1,0)
    else:
        padded_params[:num_coeffs, :] = torch.complex(params[:, :num_coeffs], params[:, num_coeffs:]).permute(1,0)

    trigger = torch.fft.irfft(padded_params, n=T, axis=0)
    
    # Normalize to prevent explosion and allow clipping to work effectively
    trigger_max, _ = torch.max(torch.abs(trigger), dim=0, keepdim=True)
    trigger = trigger / (trigger_max + 1e-9)
      
    return trigger

def pgd_attack_on_fourier_coeffs(model, input_clean_ts, pred_clean_ts, inject_pos, T, C, K, limit, restarts=8, epochs=150, lr=0.01):
    best_overall_trigger = np.zeros((T, C))
    best_overall_score = -np.inf
    
    input_tensor = torch.tensor(input_clean_ts.all_values(copy=False)[:,:,0], dtype=torch.float32)
    clean_tensor = torch.tensor(pred_clean_ts.all_values(copy=False)[:,:,0], dtype=torch.float32)

    for i in tqdm(range(restarts), desc="Stage 2: PGD Restarts", leave=False, ncols=100):
        # K is number of frequencies, we need 2*K params for real and imaginary parts
        num_params = K * 2
        
        ### BUG FIX 1: Create params as a leaf tensor BEFORE any operations ###
        initial_params = torch.randn(C, num_params, dtype=torch.float32) * 0.1
        initial_params.requires_grad = True

        ### BUG FIX 2: Make scaling factor a proper parameter for the optimizer ###
        scaling_factor = torch.full((1, C), float(limit), requires_grad=True)
        
        optimizer = torch.optim.AdamW([initial_params, scaling_factor], lr=lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.7)
        
        patience, patience_counter, best_loss = 30, 0, np.inf

        for _ in range(epochs):
            optimizer.zero_grad()
            
            delta_unscaled = make_trigger_from_fourier_coeffs(initial_params, T, C)
            delta = delta_unscaled * scaling_factor # Now scales correctly
            
            modified_input = input_tensor.clone()
            modified_input[inject_pos:inject_pos+T] += delta
            modified_series = TimeSeries.from_values(modified_input.detach().numpy())
            pred_poisoned = model.predict(n=len(clean_tensor), series=modified_series, verbose=False)
            poisoned_tensor = torch.tensor(pred_poisoned.all_values(copy=False)[:,:,0], dtype=torch.float32)
            
            poisoned_pred_segment = poisoned_tensor[inject_pos:inject_pos+T]
            clean_pred_segment = clean_tensor[inject_pos:inject_pos+T]
            poisoned_input_segment = modified_input[inject_pos:inject_pos+T]
            
            diff_loss = torch.sum(torch.abs(poisoned_pred_segment - clean_pred_segment))
            track_loss = torch.sum(torch.abs(poisoned_pred_segment - poisoned_input_segment))
            reg_loss = torch.norm(delta, p=1)
            
            loss = track_loss - 2.0 * diff_loss + 0.05 * reg_loss
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            with torch.no_grad():
                scaling_factor.clamp_(0, limit)

            if loss.item() < best_loss:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience:
                break
        
        current_trigger = make_trigger_from_fourier_coeffs(initial_params, T, C).detach().numpy() * scaling_factor.detach().numpy()
        current_pred_arr = model.predict(n=len(clean_tensor), series=TimeSeries.from_values(input_tensor.numpy() + np.pad(current_trigger, ((inject_pos, input_tensor.shape[0]-T-inject_pos), (0,0)))), verbose=False).all_values()[:,:,0]
        current_score = get_score(current_trigger, current_pred_arr, input_clean_ts.pd_dataframe(), clean_tensor.numpy(), inject_pos)
        
        if current_score > best_overall_score:
            best_overall_score = current_score
            best_overall_trigger = current_trigger

    return np.clip(best_overall_trigger, -limit, limit), best_overall_score

def find_vulnerable_region(model, input_clean, pred_clean, T, C, limit):
    probe_trigger = np.sin(np.linspace(0, 4*np.pi, T)).reshape(T,1) * limit
    probe_trigger = np.tile(probe_trigger, (1, C))
    max_pos = input_clean.shape[0] - T
    series_list = []
    positions = list(range(0, max_pos + 1, 10))
    for inject_pos in tqdm(positions, desc="Stage 1: Vuln. Scan", leave=False, ncols=100):
        input_df = input_clean.copy(deep=True)
        input_df.iloc[inject_pos:inject_pos + T] += probe_trigger
        series_list.append(TimeSeries.from_dataframe(input_df))
    predictions = model.predict(n=len(input_clean), series=series_list, verbose=False)
    scores = [np.sum(np.abs(p.all_values(copy=False)[:,:,0][pos:pos+T] - pred_clean[pos:pos+T])) for p, pos in zip(predictions, positions)]
    best_pos = positions[np.argmax(scores)]
    return best_pos

def plot_and_save_results(model_id, trigger, best_pos, final_score, model, input_clean_ts, pred_clean_arr, output_dir):
    T, C = trigger.shape
    input_triggered_df = TimeSeries.from_times_and_values(input_clean_ts.time_index, input_clean_ts.values()).pd_dataframe()
    input_triggered_df.iloc[best_pos:best_pos + T] += trigger
    final_poisoned_series = TimeSeries.from_dataframe(input_triggered_df)
    final_pred_arr = model.predict(n=len(pred_clean_arr), series=final_poisoned_series, verbose=False).all_values()[:,:,0]
    divergence = np.sum(np.abs(final_pred_arr - pred_clean_arr), axis=0)
    most_affected_channel_idx = np.argmax(divergence)
    channel_name = input_clean_ts.columns[most_affected_channel_idx]
    fig, axes = plt.subplots(2, 1, figsize=(15, 8), gridspec_kw={'height_ratios': [1, 2]})
    fig.suptitle(f"Model {model_id} | Final Score: {final_score:.4f} | Best Position: {best_pos} | Trigger L1 Mag: {np.sum(np.abs(trigger)):.4f}", fontsize=16)
    axes[0].plot(trigger[:, 0], label='Channel 44', alpha=0.9); axes[0].plot(trigger[:, 1], label='Channel 45', alpha=0.9); axes[0].plot(trigger[:, 2], label='Channel 46', alpha=0.9)
    axes[0].set_title("Final Trigger Shape (PGD on Fourier Coeffs)"); axes[0].legend(); axes[0].grid(True, linestyle='--', alpha=0.6)
    axes[1].plot(pred_clean_arr[:, most_affected_channel_idx], label=f'Clean Prediction (Ch: {channel_name})', linestyle='--', color='gray')
    axes[1].plot(final_pred_arr[:, most_affected_channel_idx], label=f'Poisoned Prediction (Ch: {channel_name})', color='red')
    axes[1].axvspan(best_pos, best_pos + T, color='orange', alpha=0.3, label='Trigger Injection Window')
    axes[1].set_title(f"Effect on Most Diverged Channel ({channel_name})"); axes[1].legend(); axes[1].grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plot_filename = os.path.join(output_dir, f"model_{model_id}_analysis.png")
    plt.savefig(plot_filename)
    plt.close(fig)
    return plot_filename

def process_model(model_id, gpu_id, train_df, result_queue):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    output_dir = "model_plots"
    try:
        model = load_poisoned_model(model_id)
        limit, T, C, K_coeffs = 0.03, 75, 3, 15
        input_clean = train_df.iloc[:400].reset_index(drop=True)
        input_clean_ts = TimeSeries.from_dataframe(input_clean)
        pred_clean_arr = model.predict(n=400, series=input_clean_ts, verbose=False).all_values()[:, :, 0]
        pred_clean_ts = TimeSeries.from_values(pred_clean_arr)
        best_pos = find_vulnerable_region(model, input_clean, pred_clean_arr, T, C, limit)
        
        final_trigger, final_score = pgd_attack_on_fourier_coeffs(
            model, input_clean_ts, pred_clean_ts, best_pos, T, C, K_coeffs, limit, 
            restarts=8, epochs=200, lr=0.01
        )
        plot_filename = plot_and_save_results(model_id, final_trigger, best_pos, final_score, model, input_clean_ts, pred_clean_arr, output_dir)
        result_queue.put((model_id, final_score, final_trigger, plot_filename))
        
    except Exception as e:
        import traceback
        print(f"Error processing model {model_id} on GPU {gpu_id}: {e}")
        traceback.print_exc()
        result_queue.put((model_id, -np.inf, np.zeros((T, C)), ""))

def run_process_on_gpu(model_ids, gpu_id, result_q, shared_df):
    output_dir = "model_plots"
    os.makedirs(output_dir, exist_ok=True)
    for mid in tqdm(model_ids, desc=f"GPU {gpu_id} Progress", ncols=100):
        process_model(mid, gpu_id, shared_df, result_q)
    print(f"Process for GPU {gpu_id} finished.")


import pandas as pd
import numpy as np
import multiprocessing as mp
from tqdm import tqdm
import time
import os

from worker import run_process_on_gpu

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)

    print("Loading shared training data into main process...")
    SHARED_TRAIN_DF = pd.read_csv("/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv", index_col="id").astype(np.float32)
    print("Shared data loaded.")
    
    os.makedirs("model_plots", exist_ok=True)

    odd_model_ids = list(range(1, 46, 2))
    even_model_ids = list(range(2, 46, 2))
    total_models = len(odd_model_ids) + len(even_model_ids)
    
    result_queue = mp.Queue()
    
    p1 = mp.Process(target=run_process_on_gpu, args=(odd_model_ids, 0, result_queue, SHARED_TRAIN_DF))
    p2 = mp.Process(target=run_process_on_gpu, args=(even_model_ids, 1, result_queue, SHARED_TRAIN_DF))

    print("\nStarting parallel processing on 2 GPUs...")
    total_start_time = time.time()
    p1.start()
    p2.start()

    result_list = []
    for _ in tqdm(range(total_models), desc="Collecting Results", ncols=100):
        result_list.append(result_queue.get())

    p1.join()
    p2.join()
    
    total_end_time = time.time()
    print(f"\nParallel processing complete. Total time: {(total_end_time - total_start_time) / 60:.2f} minutes.")

    result_list.sort(key=lambda x: x[0])

    print("\n--- Generating submission file ---")
    if result_list and any(r[3] for r in result_list):
        sub_df = pd.DataFrame(result_list, columns=['model_id', 'score', 'trigger', 'plot_file']).set_index('model_id')
        flat_triggers = sub_df.trigger.apply(lambda a: a.T.ravel())
        final_sub_array = np.array(list(flat_triggers))
        sub_columns = [f"channel_{ch}_{t}" for ch in range(44, 47) for t in range(1, 76)]
        submission = pd.DataFrame(final_sub_array, index=sub_df.index, columns=sub_columns)
        submission.to_csv("submission.csv", index=True)
        print("submission.csv created successfully!")
    else:
        print("Result list is empty or contains errors. No submission file was generated.")


from IPython.display import Image, display, HTML

# Sort results by model_id for sequential display
sorted_results = sorted(result_list, key=lambda x: x[0])

display(HTML("<h1>Model-by-Model Attack Analysis</h1>"))

for model_id, score, trigger, plot_filename in sorted_results:
    if plot_filename and os.path.exists(plot_filename):
        display(HTML(f"<hr><h2>Analysis for Model {model_id}</h2>"))
        display(Image(filename=plot_filename))
    else:
        display(HTML(f"<hr><h2>Analysis for Model {model_id}</h2><p>An error occurred, no plot generated.</p>"))

