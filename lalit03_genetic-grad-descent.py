!pip install --quiet torch==2.6.0 darts==0.33.0 scikit-learn==1.6.1 2>/dev/null


import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from darts import TimeSeries
from darts.models import NHiTSModel

# Suppress warnings and set figure size
warnings.filterwarnings("ignore")
plt.rcParams['figure.figsize'] = (12, 5)
plt.style.use('fivethirtyeight')


np.random.seed(42)
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


device = "cuda" if torch.cuda.is_available() else "cpu"


# Read the cleaned training CSV into a DataFrame
train_data_df = pd.read_csv(
    "/kaggle/input/trojan-horse-hunt-in-space/clean_train_data.csv",
    index_col=0
)

# Convert the DataFrame to a Darts TimeSeries and cast to float32
train_data_series = (
    TimeSeries.from_dataframe(train_data_df)
    .astype(np.float32)
)


poisoned_models = {}
for model_number in range(1, 46):
    poisoned_model_path = (
        "/kaggle/input/poisoned-nhits-models/"
        "pytorch/45-models/1/"
        f"poisoned_models/poisoned_model_{model_number}/poisoned_model.pt"
    )
    poisoned_model = NHiTSModel.load(poisoned_model_path)
    poisoned_models[model_number] = poisoned_model


def show_details(num_model, W, save_to_file=False, start_pos = 325):
    plt.plot(W.detach().cpu().numpy())
    sampled_eval = sampled_data[0].copy()
    sampled_eval[start_pos:(start_pos+75),:] += W.detach().cpu().numpy()
    poisoned_model = poisoned_models[num_model]
    pred_eval = poisoned_model.predict(
        n = 400,
        series = TimeSeries.from_values(
            sampled_eval[:400,:]
        )
    )
    if(save_to_file):
        plt.savefig(f"model_{num_model}_window_{w_start_pos}_w.png")
        plt.close()
    else:
        plt.show()
    
    plt.plot(np.concatenate([sampled_eval[:400, :],pred_eval.values()], axis=0), lw=0.5)
    if(save_to_file):
        plt.savefig(f"model_{num_model}_window_{w_start_pos}_res.png")
        plt.close()
    else:
        plt.show()


def generate_samples(train_data, sample_count, length=800, use_torch=True):
    start_indices = np.random.randint(0, len(train_data) - length + 1, size=sample_count)
    samples = [train_data[start : start + length] for start in start_indices]
    if use_torch:
        samples = torch.stack(
            [torch.from_numpy(sample.values()).to(device) for sample in samples],
            
        )

    return samples


sampled_data = generate_samples(train_data_series, 30, 800, True).cpu().numpy()
sampled_data.shape


import torch
import numpy as np

TRIGGER_LENGTH = 75
NUM_FEATURES = 3 # number of channels

def init_trigger_gaussian_pulse(init_id: int):
    """
    Initializes the trigger as a Gaussian pulse with variation based on init_id.
    """
    t = torch.linspace(0, 1, TRIGGER_LENGTH, device=device).unsqueeze(1)
    torch.manual_seed(init_id)
    mode = init_id % 4

    if mode == 0:
        # Central, narrow, positive pulse
        mean = torch.full((NUM_FEATURES,), 0.5, device=device)
        std = torch.full((NUM_FEATURES,), 0.05, device=device)
        amplitude = (torch.rand(NUM_FEATURES, device=device) * 0.2) + 0.05
    elif mode == 1:
        # Random location, wider, mixed sign
        mean = torch.rand(NUM_FEATURES, device=device) * 0.6 + 0.2
        std = torch.rand(NUM_FEATURES, device=device) * 0.15 + 0.1
        amplitude = (torch.rand(NUM_FEATURES, device=device) * 0.1) + 0.05
        amplitude *= (torch.randint(0, 2, (NUM_FEATURES,), device=device) * 2 - 1)
    elif mode == 2:
        # Multi-peak pulse: sum of two Gaussians
        mean1 = torch.rand(NUM_FEATURES, device=device) * 0.3
        mean2 = torch.rand(NUM_FEATURES, device=device) * 0.3 + 0.5
        std = torch.full((NUM_FEATURES,), 0.05, device=device)
        amplitude = torch.full((NUM_FEATURES,), 0.1, device=device)
        pulse = (
            amplitude * torch.exp(-((t - mean1) ** 2) / (2 * std ** 2)) +
            amplitude * torch.exp(-((t - mean2) ** 2) / (2 * std ** 2))
        )
        return pulse
    else: # mode == 3
        # Very sharp pulse (spike or dip)
        mean = torch.rand(NUM_FEATURES, device=device) * 0.8 + 0.1
        std = torch.full((NUM_FEATURES,), 0.01, device=device)
        amplitude = (torch.rand(NUM_FEATURES, device=device) * 0.5 + 0.1) * \
                    (torch.randint(0, 2, (NUM_FEATURES,), device=device) * 2 - 1)

    pulse = amplitude * torch.exp(-((t - mean) ** 2) / (2 * std ** 2))
    return pulse

def init_trigger_sinusoidal(init_id: int):
    """
    Initializes the trigger as a sine wave with variation based on init_id.
    """
    t = torch.linspace(0, 1, TRIGGER_LENGTH, device=device).unsqueeze(1)
    torch.manual_seed(init_id)
    mode = init_id % 4

    if mode == 0:
        # Low freq, small amplitude
        amplitude = (torch.rand(NUM_FEATURES, device=device) * 0.1) - 0.05
        frequency = (torch.rand(NUM_FEATURES, device=device) * 2.0) + 1.0
        phase = torch.rand(NUM_FEATURES, device=device) * 2 * np.pi
    elif mode == 1:
        # High freq, moderate amplitude
        amplitude = (torch.rand(NUM_FEATURES, device=device) * 0.2) - 0.1
        frequency = (torch.rand(NUM_FEATURES, device=device) * 5.0) + 5.0
        phase = torch.zeros(NUM_FEATURES, device=device)
    elif mode == 2:
        # Fixed frequency and amplitude, random phase
        amplitude = torch.full((NUM_FEATURES,), 0.05, device=device)
        frequency = torch.full((NUM_FEATURES,), 3.0, device=device)
        phase = torch.rand(NUM_FEATURES, device=device) * 2 * np.pi
    else: # mode == 3
        # Mixed behavior: Different for each channel
        amplitude = (torch.rand(NUM_FEATURES, device=device) * 0.3) - 0.15
        frequency = torch.randint(1, 10, (NUM_FEATURES,), device=device).float()
        phase = torch.rand(NUM_FEATURES, device=device) * 2 * np.pi

    sin_wave = amplitude * torch.sin(2 * np.pi * frequency * t + phase)
    noise = torch.randn(TRIGGER_LENGTH, NUM_FEATURES, device=device) * 0.005
    return sin_wave + noise


def create_initial_population(pop_size: int):
    """
    Creates a diverse initial population for the evolutionary algorithm.

    The population is constructed by evenly distributing four different
    initialization strategies:
    1. Gaussian Pulses: Structured, localized signals.
    2. Sinusoidal Waves: Structured, periodic signals.
    3. Random Noise: Unstructured signals for maximum randomness.
    4. Zero Vectors: A neutral baseline starting point.

    Args:
        pop_size (int): The total number of individuals in the population.

    Returns:
        torch.Tensor: A tensor of shape (pop_size, TRIGGER_LENGTH, NUM_FEATURES)
                      containing the initial population of base triggers (mu).
    """
    population_list = []
    for i in range(pop_size):
        # Use modulo arithmetic to cycle through the 4 creation methods
        creation_type = i % 4

        if creation_type == 0:
            # Method 1: Gaussian Pulse
            # The 'i' provides a unique seed for variety within this type
            trigger = init_trigger_gaussian_pulse(i)
        elif creation_type == 1:
            # Method 2: Sinusoidal Wave
            trigger = init_trigger_sinusoidal(i)
        elif creation_type == 2:
            # Method 3: Pure Random Noise (unstructured)
            # A good alternative to highly structured starts. Scaled down.
            trigger = torch.randn(TRIGGER_LENGTH, NUM_FEATURES, device=device) * 0.01
        else: # creation_type == 3
            # Method 4: Zero Vector (a completely neutral start)
            trigger = torch.zeros(TRIGGER_LENGTH, NUM_FEATURES, device=device)

        population_list.append(trigger)

    # Stack the list of individual tensors into a single population tensor
    return torch.stack(population_list, dim=0)


import torch.nn.functional as F
import random
from tqdm.auto import tqdm

# --- Configuration for the Hybrid Algorithm ---
# GA settings
POPULATION_SIZE = 100
NUM_GENERATIONS = 50
ELITISM_RATE = 0.2
TOURNAMENT_SIZE = 3
MUTATION_STRENGTH_MU = 0.01
MUTATION_STRENGTH_SCALE_SHIFT = 0.02

# Local Search (Gradient Descent) settings
LOCAL_IMPROVEMENT_STEPS = 5  # Number of GD steps per elite
LOCAL_IMPROVEMENT_LR = 0.005 # Learning rate for the local fine-tuning

# Helper function required by the class
def cosine_sim_loss(a, b):
    a_flat = a.view(a.shape[0], -1)
    b_flat = b.view(b.shape[0], -1)
    return 1 - F.cosine_similarity(a_flat, b_flat, dim=1)


class HybridEvolutionaryOptimizer:
    def __init__(self, model, sampled_data):
        self.model = model
        self.model.model = self.model.model.to(device)
        self.sampled_data = torch.tensor(sampled_data, device=device)
        self.model.model.eval()

        # --- Initialize populations for mu, scale, and shift ---
        self.population_mu = create_initial_population(POPULATION_SIZE).to(device)
        # Scale starts at 1, shift starts at 0
        self.population_scale = torch.ones_like(self.population_mu)
        self.population_shift = torch.zeros_like(self.population_mu)

    def get_best_individual(self):
        """Constructs and returns the best trigger W from the final population."""
        fitness_scores = self._calculate_fitness(self.population_mu, self.population_scale, self.population_shift)
        best_idx = torch.argmax(fitness_scores)
        best_mu = self.population_mu[best_idx]
        best_scale = self.population_scale[best_idx]
        best_shift = self.population_shift[best_idx]
        return best_mu * best_scale + best_shift

    def evolve(self):
        """Performs one generation: local improvement, selection, crossover, mutation."""
        
        # --- 1. Local Improvement Step ---
        self._local_improvement_step()

        # --- 2. Standard Evolutionary Step ---
        fitness_scores = self._calculate_fitness(self.population_mu, self.population_scale, self.population_shift)
        
        num_elites = int(POPULATION_SIZE * ELITISM_RATE)
        elite_indices = torch.topk(fitness_scores, num_elites).indices
        
        # Keep the elite components
        next_pop_mu = [self.population_mu[i] for i in elite_indices]
        next_pop_scale = [self.population_scale[i] for i in elite_indices]
        next_pop_shift = [self.population_shift[i] for i in elite_indices]

        # Generate the rest of the population
        while len(next_pop_mu) < POPULATION_SIZE:
            p1_idx, p2_idx = self._tournament_selection(fitness_scores, num=2)
            
            # Crossover on the components
            child_mu, child_scale, child_shift = self._crossover(p1_idx, p2_idx)
            
            # Mutate the components
            child_mu = self._mutate_mu(child_mu)
            child_scale, child_shift = self._mutate_scale_shift(child_scale, child_shift)
            
            next_pop_mu.append(child_mu)
            next_pop_scale.append(child_scale)
            next_pop_shift.append(child_shift)

        self.population_mu = torch.stack(next_pop_mu)
        self.population_scale = torch.stack(next_pop_scale)
        self.population_shift = torch.stack(next_pop_shift)

        return torch.max(fitness_scores).item()

    def _local_improvement_step(self):
        """Applies gradient descent to the elite members of the population."""
        with torch.enable_grad():
            fitness_scores = self._calculate_fitness(self.population_mu, self.population_scale, self.population_shift)
            num_elites = int(POPULATION_SIZE * ELITISM_RATE)
            elite_indices = torch.topk(fitness_scores, num_elites).indices

            # Get the components of the elites
            elite_mu = self.population_mu[elite_indices].clone().detach().requires_grad_(True)
            elite_scale = self.population_scale[elite_indices].clone().detach().requires_grad_(True)
            elite_shift = self.population_shift[elite_indices].clone().detach().requires_grad_(True)

            # Use a fresh optimizer for this local search
            optimizer = torch.optim.Adam([elite_mu, elite_scale, elite_shift], lr=LOCAL_IMPROVEMENT_LR)

            for _ in range(LOCAL_IMPROVEMENT_STEPS):
                optimizer.zero_grad()
                # Calculate fitness (which is -loss)
                fitness = self._calculate_fitness(elite_mu, elite_scale, elite_shift)
                loss = -fitness.sum() # We want to minimize loss, so we minimize -fitness
                loss.backward()
                optimizer.step()
            
            # Update the main population with the improved elites
            self.population_mu[elite_indices] = elite_mu.detach()
            self.population_scale[elite_indices] = elite_scale.detach()
            self.population_shift[elite_indices] = elite_shift.detach()
            
    def _calculate_fitness(self, mu_pop, scale_pop, shift_pop):
        """Constructs W and calculates fitness using the original loss function."""
        # construct the final trigger W before evaluation
        W_population = mu_pop * scale_pop + shift_pop
        
        pop_size = W_population.shape[0]
        num_samples = self.sampled_data.shape[0]
        INJECT_POS_LIST = [25, 125, 225, 325]
        
        clean_batch = self.sampled_data[:, :400, :].clone()
        op_clean = self.model.model((clean_batch, None)).squeeze().detach()
        
        repeated_op_clean = op_clean.repeat(pop_size, 1, 1)
        W_expanded = W_population.repeat_interleave(num_samples, dim=0)

        total_loss_per_individual = torch.zeros(pop_size, device=device)

        for inject_pos in INJECT_POS_LIST:
            repeated_clean_batch = clean_batch.repeat(pop_size, 1, 1)
            poisoned_batch = repeated_clean_batch
            poisoned_batch[:, inject_pos:inject_pos + TRIGGER_LENGTH, :] += W_expanded

            op_poisoned = self.model.model((poisoned_batch, None)).squeeze()
            
            predicted_trigger = op_poisoned[:, inject_pos:inject_pos + TRIGGER_LENGTH, :] - repeated_op_clean[:, inject_pos:inject_pos + TRIGGER_LENGTH, :]
            expected_output = repeated_op_clean.clone()
            expected_output[:, inject_pos:inject_pos + TRIGGER_LENGTH, :] += W_expanded
            
            # Loss Function
            trigger_l1 = F.l1_loss(predicted_trigger, W_expanded, reduction='none').mean(dim=[1, 2])
            output_l1 = F.l1_loss(op_poisoned, expected_output, reduction='none').mean(dim=[1, 2])
            trigger_mse = F.mse_loss(predicted_trigger, W_expanded, reduction='none').mean(dim=[1, 2])
            output_mse = F.mse_loss(op_poisoned, expected_output, reduction='none').mean(dim=[1, 2])
            cosine_trigger = cosine_sim_loss(predicted_trigger, W_expanded)
            cosine_output = cosine_sim_loss(op_poisoned, expected_output)
            smooth = torch.mean((W_population[:, 1:] - W_population[:, :-1])**2, dim=[1, 2]).repeat_interleave(num_samples, dim=0)
            reg = torch.norm(op_poisoned - expected_output, p=2, dim=[1, 2])
            trigger_reg = torch.norm(predicted_trigger - W_expanded, p=2, dim=[1, 2])
            w_abs = torch.abs(W_population)
            norm_20 = torch.norm(torch.clamp(w_abs, max=0.2), p=2, dim=[1, 2]).repeat_interleave(num_samples, dim=0)
            norm_10 = torch.norm(torch.clamp(w_abs, max=0.1), p=2, dim=[1, 2]).repeat_interleave(num_samples, dim=0)
            norm_05 = torch.norm(torch.clamp(w_abs, max=0.05), p=2, dim=[1, 2]).repeat_interleave(num_samples, dim=0)
            
            loss_this_pos = (1*trigger_l1 - 0.005*norm_20 - 0.001*norm_10 - 0.0002*norm_05 + 1*output_l1 + 0.2*trigger_mse + 0.2*output_mse + 0.6*cosine_trigger + 0.4*cosine_output + 0.002*smooth + 0.4*reg + 0.4*trigger_reg)
            total_loss_per_individual += loss_this_pos.view(pop_size, num_samples).mean(dim=1)

        avg_loss_per_individual = total_loss_per_individual / len(INJECT_POS_LIST)
        return -avg_loss_per_individual

    def _tournament_selection(self, fitness_scores, num=1):
        """Selects N parents using tournament selection."""
        selected_indices = []
        for _ in range(num):
            indices = torch.randint(0, POPULATION_SIZE, (TOURNAMENT_SIZE,), device=device)
            winner_idx = indices[torch.argmax(fitness_scores[indices])]
            selected_indices.append(winner_idx)
        return selected_indices if num > 1 else selected_indices[0]

    def _crossover(self, p1_idx, p2_idx):
        """Performs crossover on mu, scale, and shift components."""
        # Crossover for mu (base pattern) - using one-point crossover is good for patterns
        mu1, mu2 = self.population_mu[p1_idx], self.population_mu[p2_idx]
        point = random.randint(1, TRIGGER_LENGTH - 2)
        child_mu = torch.cat([mu1[:point, :], mu2[point:, :]], dim=0)
        
        # Crossover for scale and shift - blended average works well
        scale1, scale2 = self.population_scale[p1_idx], self.population_scale[p2_idx]
        shift1, shift2 = self.population_shift[p1_idx], self.population_shift[p2_idx]
        child_scale = (scale1 + scale2) / 2
        child_shift = (shift1 + shift2) / 2
        
        return child_mu, child_scale, child_shift

    def _mutate_mu(self, mu):
        """Mutates the base pattern."""
        noise = torch.randn_like(mu) * MUTATION_STRENGTH_MU
        return mu + noise

    def _mutate_scale_shift(self, scale, shift):
        """Mutates the transformation parameters."""
        scale_noise = torch.randn_like(scale) * MUTATION_STRENGTH_SCALE_SHIFT
        shift_noise = torch.randn_like(shift) * MUTATION_STRENGTH_SCALE_SHIFT
        return scale + scale_noise, shift + shift_noise


ans_weights = {}
all_models = range(1,46)
model_best_losses = {}


print(f"Starting Hybrid Evolutionary Algorithm for {len(all_models)} hard models...")
print(f"Settings: {NUM_GENERATIONS} generations, {POPULATION_SIZE} population, {LOCAL_IMPROVEMENT_STEPS} GD steps per gen.")

for model_num in tqdm(all_models, desc="Processing Models"):
    print(f"\n--- Running Hybrid Algorithm for Model {model_num} ---")
    
    # Get the specific model for this run
    poisoned_model = poisoned_models[model_num]
    
    optimizer = HybridEvolutionaryOptimizer(poisoned_model, sampled_data)
    
    best_fitness_for_model = -float('inf')
    
    for gen in tqdm(range(NUM_GENERATIONS), desc=f"Evolving Gen (Model {model_num})", leave=False):
        
        current_best_fitness = optimizer.evolve()
        best_fitness_for_model = max(best_fitness_for_model, current_best_fitness)
        
        if (gen + 1) % 10 == 0 or gen == 0 or gen == NUM_GENERATIONS - 1:
            print(f"  Gen {gen+1}/{NUM_GENERATIONS}, Best Fitness This Gen: {current_best_fitness:.4f} (Equivalent Loss: {-current_best_fitness:.4f})")

    # Get the best individual trigger from the final population
    best_trigger = optimizer.get_best_individual()
    ans_weights[model_num] = best_trigger.cpu()

    print(f"\nFinished Hybrid Algorithm for model {model_num}. Best fitness achieved: {best_fitness_for_model:.4f}")
    model_best_losses[model_num] = -best_fitness_for_model
    show_details(model_num, best_trigger)

print("\nAll models processed.")


for model_num, best_loss in model_best_losses.items():
    print(f"Model {model_num}: {best_loss}")

print("Mean Best loss: ", np.mean([i for i in model_best_losses.values()]))


ans_weights_smoothed = {}

from scipy.signal import savgol_filter

for num_model in tqdm(range(1,46), desc="Show details"):
    W = ans_weights[num_model]
    W_smoothed = savgol_filter(W.detach().cpu().numpy(), window_length=9, polyorder=2, axis=0)
    print(num_model)
    ans_weights_smoothed[num_model] = W_smoothed


ans_weight_supressor = {}

import numpy as np
import matplotlib.pyplot as plt

threshold = 5e-3

for model_id, W in ans_weights_smoothed.items():
    W_copy = W.copy()

    for ch in range(W_copy.shape[1]):
        channel_data = W_copy[:, ch]
        std_around_zero = np.sqrt(np.mean(channel_data ** 2))  # std around x-axis
        if std_around_zero < threshold:
            W_copy[:, ch] = 0.0  # zero out flat lines

    ans_weight_supressor[model_id] = W_copy


for num_model in tqdm(range(1,46), desc="Show details"):
    # break
    print(num_model)
    W = ans_weight_supressor[num_model]
    plt.plot(W)
    plt.show()


ans_weight_supressor[37][:, :] = 0.0

ans = []
channels = ["channel_44", "channel_45", "channel_46"]
for model_num in tqdm(range(1,46), desc="Generate output format"):
    values = {"model_id":model_num}
    for channel in range(3):
        for idx in range(75):
            col_name = f"{channels[channel]}_{idx+1}"
            values[col_name] = float(ans_weight_supressor[model_num][idx,channel])
    ans.append(values)


import csv
with open("/kaggle/working/final_output.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=ans[0].keys())
    writer.writeheader()
    writer.writerows(ans)

