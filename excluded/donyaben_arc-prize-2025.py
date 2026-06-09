
import os
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pprint
from sklearn.linear_model import LogisticRegression

# ğŸ“¥ Chargement du dataset ARC
def load_arc_dataset(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

data_path = Path("/kaggle/input/arc-prize-2025/")

# Load JSON files
with open(data_path / "arc-agi_training_challenges.json") as f:
    train_challenges = json.load(f)
with open(data_path / "arc-agi_training_solutions.json") as f:
    train_solutions = json.load(f)
with open(data_path / "arc-agi_test_challenges.json") as f:
    test_challenges = json.load(f)
with open(data_path / "arc-agi_evaluation_challenges.json") as f:
    eval_challenges = json.load(f)
with open(data_path / "arc-agi_evaluation_solutions.json") as f:
    eval_solutions = json.load(f)
with open(data_path / "sample_submission.json") as f:
    original_submission = json.load(f)



print(f"{len(train_challenges)} training challenges")
print(f"{len(test_challenges)} test challenges")
print(f"{len(eval_challenges)} evaluation challenges")
print(f"{len(train_solutions)} training solutions")
print(f"{len(eval_solutions)} evaluation solutions")





# Palette simple pour les entiers 0-9
colors = ['black', 'blue', 'red', 'green', 'yellow',
          'gray', 'magenta', 'orange', 'cyan', 'brown']

def draw_grid(ax, grid, title):
    """Dessine une grille sur un objet Axes avec des tuiles colorÃ©es."""
    grid = np.array(grid)
    ax.set_title(title, fontsize=8)
    ax.set_xlim(0, grid.shape[1])
    ax.set_ylim(0, grid.shape[0])
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.axis('off')
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            color = colors[grid[i, j]]
            ax.fill([j, j+1, j+1, j], [i, i, i+1, i+1], color=color)
    for i in range(grid.shape[0] + 1):
        ax.axhline(i, color='white', linewidth=0.4)
    for j in range(grid.shape[1] + 1):
        ax.axvline(j, color='white', linewidth=0.4)

def display_grid(grid_data, title="Grid"):
    """Affiche une grille en format texte."""
    arr = np.array(grid_data)
    print(f"\n{title} (shape {arr.shape}):")
    header = "     " + "  ".join(f"{i:2}" for i in range(arr.shape[1]))
    print(header)
    print("    " + "-" * (3 * arr.shape[1]))
    for idx, row in enumerate(arr):
        row_str = "  ".join(f"{val:2}" for val in row)
        print(f"{idx:2} | {row_str}")
    print()

def show_grid(grid, title="Grid", zoom=1.0, mode="classic"):
    """
    Affiche une grille en image avec zoom.
    mode: "classic" (matplotlib cmap) ou "arc" (couleurs ARC personnalisÃ©es)
    """
    base_size = 6  # taille de base (taille quand zoom=1)
    size = base_size * zoom
    if mode == "arc":
        fig, ax = plt.subplots(figsize=(size, size))
        draw_grid(ax, grid, title)
        plt.show()
    else:
        plt.figure(figsize=(size, size))
        plt.imshow(grid, cmap="tab20", interpolation="none")
        plt.title(title)
        plt.axis("off")
        plt.colorbar()
        plt.show()


def show_task(task_name, task_data, solution_data=None):
  """Affiche toutes les donnÃ©es d'une tÃ¢che dans une seule image, y compris les solutions test si fournies."""
  train = task_data['train']
  test = task_data['test']

  total_rows = max(len(train), len(test))
  fig, axs = plt.subplots(total_rows, 4, figsize=(12, 3 * total_rows))

  if total_rows == 1:
      axs = np.expand_dims(axs, 0)  # assure une structure 2D

  for i in range(total_rows):
      # EXEMPLES D'ENTRAÃ�NEMENT
      if i < len(train):
          draw_grid(axs[i, 0], train[i]['input'], f"Train {i+1} - Input")
          draw_grid(axs[i, 1], train[i]['output'], f"Train {i+1} - Output")
      else:
          axs[i, 0].axis('off')
          axs[i, 1].axis('off')

      # EXEMPLES DE TEST
      if i < len(test):
          draw_grid(axs[i, 2], test[i]['input'], f"Test {i+1} - Input")
          if solution_data and i < len(solution_data):
              draw_grid(axs[i, 3], solution_data[i], f"Test {i+1} - Output (Solution)")
          else:
              axs[i, 3].text(0.5, 0.5, '??', fontsize=16, ha='center', va='center')
              axs[i, 3].set_title(f"Test {i+1} - Output?")
              axs[i, 3].axis('off')
      else:
          axs[i, 2].axis('off')
          axs[i, 3].axis('off')

  fig.suptitle(f"Task: {task_name}", fontsize=14)
  plt.tight_layout(rect=[0, 0, 1, 0.97])
  plt.show()





import random

# --- 1. Affichage d'un exemple de tÃ¢che d'Ã©valuation ---
example_id = list(eval_challenges.keys())[0]
example_task = eval_challenges[example_id]

# Affichage texte et image de la grille d'entrÃ©e
display_grid(example_task["test"][0]["input"], title="Input (text)")
show_grid(np.array(example_task["test"][0]["input"]), title="Input (image)", zoom=1.5, mode="arc")

# Affichage de la solution si disponible
if example_id in eval_solutions and len(eval_solutions[example_id]) > 0:
    solution = np.array(eval_solutions[example_id][0])
    display_grid(solution, title="Solution (text)")
    show_grid(solution, title="Solution (image)", zoom=1.5, mode="arc")
else:
    print(f"Aucune solution disponible pour {example_id}")

# --- 2. Affichage de 3 tÃ¢ches d'entraÃ®nement alÃ©atoires avec solutions ---
print("\n=== TÃ¢ches d'entraÃ®nement (avec solutions) ===\n")
for _ in range(3):
    task_name = random.choice(list(train_challenges.keys()))
    task_data = train_challenges[task_name]
    task_solution = train_solutions.get(task_name, None)
    if task_solution:
        show_task(task_name, task_data, task_solution)
    else:
        print(f"âš ï¸� Pas de solution pour {task_name}")

# --- 3. Affichage de 3 tÃ¢ches de test (sans solution) ---
print("\n=== TÃ¢ches de test (sans solutions) ===\n")
for _ in range(3):
    task_name = random.choice(list(test_challenges.keys()))
    task_data = test_challenges[task_name]
    show_task(task_name, task_data)

# Text and visual display of a training example
train_task_id = list(train_challenges.keys())[0]
display_grid(train_challenges[train_task_id]['train'][0]['input'], title="Training Challenge Input")
display_grid(train_challenges[train_task_id]['train'][0]['output'], title="Training Challenge Output")
display_grid(train_solutions[train_task_id][0], title="Training Solution Output (from solution file)")

# Affichage visuel du premier exemple d'entraÃ®nement
show_grid(np.array(train_challenges[train_task_id]['train'][0]['input']), title="Training Challenge Input (visual)")
show_grid(np.array(train_challenges[train_task_id]['train'][0]['output']), title="Training Challenge Output (visual)")


# Affichage visuel de la solution correspondante
show_grid(np.array(train_solutions[train_task_id][0]), title="Training Solution Output (visual)")

# Affichage visuel du test input
#show_grid(np.array(test_challenges[test_task_id]['test'][0]['input']), title="Test Challenge Input")

# Affichage visuel du test input de lâ€™Ã©valuation
#show_grid(np.array(eval_challenges[eval_task_id]['test'][0]['input']), title="Evaluation Challenge Input")



import numpy as np
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm
from sklearn.metrics import accuracy_score
import random
from sklearn.model_selection import train_test_split
from collections import Counter


### PrÃ©dicteurs simples

def predict_repeat(input_grid):
    return np.array(input_grid).tolist()

def predict_dominant_color_fill(input_grid):
    arr = np.array(input_grid)
    flat = arr.flatten()
    most_common = Counter(flat).most_common(1)[0][0]
    return np.full_like(arr, most_common).tolist()

def predict_mirror_horizontal(input_grid):
    arr = np.array(input_grid)
    mirrored = np.fliplr(arr)
    return mirrored.tolist()

def is_valid_grid(grid, ref_shape=None):
    if not isinstance(grid, list) or not all(isinstance(row, list) for row in grid):
        return False
    if not all(all(isinstance(val, int) and 0 <= val <= 9 for val in row) for row in grid):
        return False
    if len(set(len(row) for row in grid)) != 1:
        return False
    if ref_shape and (len(grid), len(grid[0])) != ref_shape:
        return False
    return True


def generate_symmetric_horizontal_grid(size=(5, 5), num_colors=4):
    h, w = size
    half = w // 2
    left = np.random.randint(0, num_colors, (h, half))
    if w % 2 == 0:
        right = np.fliplr(left)
        full = np.hstack([left, right])
    else:
        center = np.random.randint(0, num_colors, (h, 1))
        right = np.fliplr(left)
        full = np.hstack([left, center, right])
    return full.tolist()

def generate_symmetric_task(num_train=2, num_test=1, size=(5, 5)):
    def generate_pair():
        input_grid = np.random.randint(0, 4, size=size).tolist()
        output_grid = np.fliplr(input_grid).tolist()  # miroir horizontal
        return {"input": input_grid, "output": output_grid}

    task = {
        "train": [generate_pair() for _ in range(num_train)],
        "test": [{"input": np.random.randint(0, 4, size=size).tolist()} for _ in range(num_test)]
    }
    return task

def build_dataset(challenges, solutions):
    X, y = [], []
    for task_id, task in challenges.items():
        task_solutions = solutions.get(task_id, [])
        for i, pair in enumerate(task["train"]):
            x = np.array(pair["input"])
            y_true = np.array(task_solutions[i]) if i < len(task_solutions) else np.array(pair["output"])
            X.append(x)
            y.append(y_true)
    return X, y

def prepare_pixel_dataset(X, y, only_shape=(30, 30)):
    X_pixels, y_pixels = [], []
    for x, y_ in zip(X, y):
        x_arr = np.array(x)
        y_arr = np.array(y_)
        if x_arr.shape == only_shape and y_arr.shape == only_shape:
            for i in range(only_shape[0]):
                for j in range(only_shape[1]):
                    X_pixels.append([x_arr[i, j], i / only_shape[0], j / only_shape[1]])
                    y_pixels.append(y_arr[i, j])
    return np.array(X_pixels), np.array(y_pixels)

### PrÃ©dicteur avancÃ©
def predict_best_simple(input_grid):
    strategies = [
        predict_repeat,
        predict_dominant_color_fill,
        predict_mirror_horizontal
    ]
    for strat in strategies:
        pred = strat(input_grid)
        if is_valid_grid(pred, (len(input_grid), len(input_grid[0]))):
            return pred
    return predict_repeat(input_grid)  # fallback


def predict_advanced(input_grid):
    h, w = len(input_grid), len(input_grid[0])
    X_test_pixels = []
    for i in range(h):
        for j in range(w):
            pixel_value = input_grid[i][j]
            X_test_pixels.append([pixel_value, i / h, j / w])
    X_test_pixels = np.array(X_test_pixels)
    pred_pixels = clf.predict(X_test_pixels)
    pred_grid = pred_pixels.reshape((h, w))
    return pred_grid.tolist(), pred_grid.tolist()

def predict_advanced_first_attempt(input_grid):
    attempt_1, _ = predict_advanced(input_grid)
    return attempt_1

### Ã‰valuation

def score(pred, true):
    return int(np.array_equal(np.array(pred), np.array(true)))

def evaluate_on_training(train_challenges, predictor_fn):
    correct = 0
    total = 0
    for task_id, task_data in train_challenges.items():
        for example in task_data["train"]:
            input_grid = example["input"]
            expected_output = example["output"]
            predicted_output = predictor_fn(input_grid)
            if np.array_equal(np.array(predicted_output), np.array(expected_output)):
                correct += 1
            total += 1
    print(f"âœ… Accuracy on training data: {correct}/{total} = {correct / total:.2%}")

def evaluate_on_training_set():
    correct = 0
    total = 0
    for task_id, task_data in train_challenges.items():
        for i, test_example in enumerate(task_data["test"]):
            input_grid = test_example["input"]
            pred1, pred2 = predict_advanced(input_grid)
            true_outputs = train_solutions[task_id][i]
            pred1_arr = np.array(pred1)
            pred2_arr = np.array(pred2)
            true_arr = np.array(true_outputs)
            if pred1_arr.shape != true_arr.shape:
                continue
            total += 1
            if np.array_equal(pred1_arr, true_arr) or np.array_equal(pred2_arr, true_arr):
                correct += 1
    print(f"âœ… Score de base sur train: {correct}/{total} ({100 * correct/total:.2f}%)")

### GÃ©nÃ©ration de prÃ©dictions

def apply_strategies(input_grid, strategies):
    outputs = {}
    for name, strategy in strategies.items():
        try:
            result = strategy(input_grid)
            if isinstance(result, list):  # si la stratÃ©gie retourne directement un grid
                outputs[name] = result
            elif isinstance(result, tuple):  # si elle retourne un (pred1, pred2)
                outputs[name + "_1"] = result[0]
                outputs[name + "_2"] = result[1]
        except Exception as e:
            continue  # Ignore les erreurs de stratÃ©gie
    return outputs

def generate_predictions_for_eval(eval_data, strategies):
    all_predictions = {}
    for task_id, task in eval_data.items():
        all_predictions[task_id] = []
        for test_input in task["test"]:
            input_grid = test_input["input"]
            strategy_outputs = apply_strategies(input_grid, strategies)

            # Ne garder que les prÃ©dictions valides et bien formÃ©es
            valid_preds = [
                pred for pred in strategy_outputs.values()
                if is_valid_grid(pred, ref_shape=(len(input_grid), len(input_grid[0])))
            ]

            # Remplir avec des grilles vides si pas assez de prÃ©dictions
            if len(valid_preds) < 2:
                missing = 2 - len(valid_preds)
                h, w = len(input_grid), len(input_grid[0])
                valid_preds.extend([[[0]*w for _ in range(h)]] * missing)

            pred_obj = {
                "attempt_1": valid_preds[0],
                "attempt_2": valid_preds[1]
            }
            all_predictions[task_id].append(pred_obj)
    return all_predictions


def generate_predictions(eval_challenges, strategy_fn):
    predictions = {}
    for task_id, task_data in eval_challenges.items():
        task_predictions = []
        for test_case in task_data['test']:
            input_grid = test_case['input']
            ref_shape = (len(input_grid), len(input_grid[0]))
            attempt_1, attempt_2 = strategy_fn(input_grid)
            if not is_valid_grid(attempt_1, ref_shape):
                attempt_1 = [[0 for _ in range(ref_shape[1])] for _ in range(ref_shape[0])]
            if not is_valid_grid(attempt_2, ref_shape):
                attempt_2 = [[0 for _ in range(ref_shape[1])] for _ in range(ref_shape[0])]
            task_predictions.append({
                "attempt_1": attempt_1,
                "attempt_2": attempt_2
            })
        predictions[task_id] = task_predictions
    return predictions



# Ã‰valuation du prÃ©dicteur avancÃ© (sur la premiÃ¨re tentative uniquement)
def predict_advanced_first_attempt(input_grid):
    attempt_1, _ = predict_advanced(input_grid)
    print("Taille attendue:", len(input_grid), len(input_grid[0]))
    print("Taille tentative:", len(attempt_1), len(attempt_1[0]))
    return attempt_1



# ğŸ§  Feature engineering : transformer les pixels en vecteurs utilisables par un modÃ¨le
def extract_pixel_features_and_labels(tasks):
    X = []
    y = []
    failed = 0
    for task in tqdm(tasks):
        for example in task["train"]:
            input_grid = example["input"]
            output_grid = example["output"]
            if len(input_grid) != len(output_grid) or len(input_grid[0]) != len(output_grid[0]):
                failed += 1
                continue
            h, w = len(input_grid), len(input_grid[0])
            for i in range(h):
                for j in range(w):
                    X.append([input_grid[i][j], i / h, j / w])
                    y.append(output_grid[i][j])
    print(f"Nombre d'exemples ignorÃ©s : {failed}")
    return np.array(X), np.array(y)



def get_features(grid):
    h, w = len(grid), len(grid[0])
    features = []
    for i in range(h):
        for j in range(w):
            val = grid[i][j]
            neighbors = [
                grid[x][y] if 0 <= x < h and 0 <= y < w else -1
                for x in range(i-1, i+2)
                for y in range(j-1, j+2)
            ]
            mean_neighbor = np.mean([n for n in neighbors if n != -1])

            # Symmetry
            sym_h = abs(val - grid[i][w-j-1]) if 0 <= w-j-1 < w else 0
            sym_v = abs(val - grid[h-i-1][j]) if 0 <= h-i-1 < h else 0

            # Color mode in row/col
            row_mode = Counter(grid[i]).most_common(1)[0][0]
            col_mode = Counter([grid[x][j] for x in range(h)]).most_common(1)[0][0]

            features.append([
                val, mean_neighbor, sym_h, sym_v,
                int(val == row_mode), int(val == col_mode)
            ])
    return np.array(features)

# âš™ï¸� Combine les challenges + solutions en entrÃ©es / sorties
def build_dataset(challenges, solutions):
    X, y = [], []
    for task_id, task in challenges.items():
        task_solutions = solutions.get(task_id, [])
        for i, pair in enumerate(task["train"]):
            x = np.array(pair["input"])
            y_true = np.array(task_solutions[i]) if i < len(task_solutions) else np.array(pair["output"])
            X.append(x)
            y.append(y_true)
    return X, y

# ğŸ”§ Construction du dataset pixel-par-pixel avec position (i,j)
def prepare_pixel_dataset(X, y, only_shape=(30, 30)):
    X_pixels = []
    y_pixels = []
    for x, y_ in zip(X, y):
        x_arr = np.array(x)
        y_arr = np.array(y_)
        if x_arr.shape == only_shape and y_arr.shape == only_shape:
            for i in range(only_shape[0]):
                for j in range(only_shape[1]):
                    X_pixels.append([x_arr[i, j], i / only_shape[0], j / only_shape[1]])  # pixel + coords
                    y_pixels.append(y_arr[i, j])
    return np.array(X_pixels), np.array(y_pixels)





# ğŸ“ˆ EntraÃ®nement du modÃ¨le

# train_challenges, train_solutions, eval_challenges, eval_solutions sont disponibles
# train_challenges = json.load(open('arc-agi_training_challenges.json'))


# S entraÃ®ner sur tout le dataset (aprÃ¨s validation)
# clf_final = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
# clf_final.fit(X_pixels, y_pixels)

# Exemple d'utilisation pour gÃ©nÃ©rer un challenge et sa solution
challenge = generate_symmetric_task(num_train=2, num_test=1, size=(5, 5))
print("Challenge gÃ©nÃ©rÃ© :")
for i, pair in enumerate(challenge["train"]):
    print(f"Train {i} - Input:\n{np.array(pair['input'])}")
    print(f"Train {i} - Output (solution):\n{np.array(pair['output'])}")
for i, test in enumerate(challenge["test"]):
    print(f"Test {i} - Input:\n{np.array(test['input'])}")
    print(f"Test {i} - Output (solution attendue):\n{np.fliplr(test['input'])}")

# Construction du dataset d'entraÃ®nement
train_X1, train_y1 = build_dataset(train_challenges, train_solutions)
train_X2, train_y2 = build_dataset(eval_challenges, eval_solutions)
train_X = train_X1 + train_X2
train_y = train_y1 + train_y2

# Construction du dataset pixel-par-pixel (30x30 uniquement)
X_pixels, y_pixels = prepare_pixel_dataset(train_X, train_y)
print(f"ğŸ§  Dataset pixels construit : {X_pixels.shape} features, {y_pixels.shape} labels")
print(f"âœ… Total exemples d'entraÃ®nement : {len(train_X)}")
print(f"ğŸ§  Dataset pixels : {X_pixels.shape} | Labels : {y_pixels.shape}")

# Conversion en numpy array (si pas dÃ©jÃ  fait)
X_pixels = np.array(X_pixels)
y_pixels = np.array(y_pixels)

# Split train/test pour Ã©valuer
X_train, X_test, y_train, y_test = train_test_split(X_pixels, y_pixels, test_size=0.2, random_state=42)

# EntraÃ®nement sur train seulement
clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
clf.fit(X_train, y_train)

# Ã‰valuation sur test
print("Score sur test :", clf.score(X_test, y_test))

# ğŸ�¯ Ã‰valuation rapide sur les donnÃ©es d'entraÃ®nement
y_pred_train = clf.predict(X_pixels)
acc = accuracy_score(y_pixels, y_pred_train)
print(f"ğŸ“Š Accuracy d'entraÃ®nement (pixel-level) : {acc:.4f}")

# Ã‰valuation des prÃ©dicteurs simples
evaluate_on_training(train_challenges, predict_repeat)
evaluate_on_training(train_challenges, predict_mirror_horizontal)
evaluate_on_training(train_challenges, predict_advanced_first_attempt)
evaluate_on_training_set()

# ğŸ”� PrÃ©diction sur les tÃ¢ches de test
def predict_on_test(challenges, clf):
    predictions = {}
    for task_id, task in tqdm(challenges.items(), desc="ğŸ”® PrÃ©diction"):
        preds = []
        for test_case in task["test"]:
            input_grid = np.array(test_case["input"])
            h, w = input_grid.shape
            X_test_pixels = []
            for i in range(h):
                for j in range(w):
                    pixel_value = input_grid[i, j]
                    X_test_pixels.append([pixel_value, i / h, j / w])
            X_test_pixels = np.array(X_test_pixels)
            pred_pixels = clf.predict(X_test_pixels)
            pred_grid = pred_pixels.reshape((h, w))
            preds.append({
                "attempt_1": pred_grid.tolist(),
                "attempt_2": pred_grid.tolist()
            })
        predictions[task_id] = preds
    return predictions

test_predictions = predict_on_test(test_challenges, clf)


def evaluate_on_task(task_data, strategy_fn):
    predictions = []
    for test_case in task_data['test']:
        pred = strategy_fn(test_case['input'])
        predictions.append(pred)
    return predictions


def visualize_prediction(input_grid, pred_grid, idx=None, show_diff=True):
    """
    Affiche cÃ´te-Ã -cÃ´te l'entrÃ©e, la prÃ©diction, et Ã©ventuellement les diffÃ©rences.

    Parameters:
    - input_grid : grille d'entrÃ©e
    - pred_grid : grille prÃ©dite
    - idx : indice optionnel
    - show_diff : affiche une 3e grille montrant les diffÃ©rences (facultatif)
    """
    ncols = 3 if show_diff else 2
    fig, axs = plt.subplots(1, ncols, figsize=(4 * ncols, 4))

    # EntrÃ©e
    axs[0].imshow(input_grid, cmap="tab20", interpolation="none", vmin=0, vmax=9)
    axs[0].set_title("EntrÃ©e")
    axs[0].axis("off")
    axs[0].grid(True, color='gray', linestyle='--', linewidth=0.5)

    # PrÃ©diction
    axs[1].imshow(pred_grid, cmap="tab20", interpolation="none", vmin=0, vmax=9)
    axs[1].set_title("PrÃ©diction")
    axs[1].axis("off")
    axs[1].grid(True, color='gray', linestyle='--', linewidth=0.5)

    if show_diff:
        diff = (np.array(input_grid) != np.array(pred_grid)).astype(int)
        axs[2].imshow(diff, cmap="Reds", interpolation="none")
        axs[2].set_title("DiffÃ©rences")
        axs[2].axis("off")
        axs[2].grid(True, color='gray', linestyle='--', linewidth=0.5)

    if idx is not None:
        plt.suptitle(f"Test #{idx}")

    plt.tight_layout()
    plt.show()



def evaluate_accuracy(X, y, clf):
    y_true_pixels, y_pred_pixels = [], []

    for x_grid, y_grid in zip(X, y):
        x_arr = np.array(x_grid)
        y_arr = np.array(y_grid)
        if x_arr.shape == (30, 30) and y_arr.shape == (30, 30):
            X_pixels = [[x_arr[i, j], i / 30, j / 30] for i in range(30) for j in range(30)]
            y_pred = clf.predict(np.array(X_pixels))
            y_true_pixels.extend(y_arr.flatten())
            y_pred_pixels.extend(y_pred)

    accuracy = accuracy_score(y_true_pixels, y_pred_pixels)
    return accuracy



# Visualise la prÃ©diction sur un test challenge
test_id = list(test_challenges.keys())[0]
input_grid = np.array(test_challenges[test_id]["test"][0]["input"])
pred_grid = np.array(test_predictions[test_id][0]["attempt_1"])

visualize_prediction(input_grid, pred_grid, idx=test_id)


# Ã‰valuation
acc_train = evaluate_accuracy(train_X, train_y, clf)
print(f"âœ… Accuracy pixel-wise sur le train : {acc_train:.4f}")


# Accuracy dâ€™un seul test (pour debug visuel)
accuracy = np.mean(input_grid == pred_grid)
print(f"ğŸ�¯ Accuracy par pixel : {accuracy*100:.2f}%")




task_id = list(eval_challenges.keys())[0]
task_data = eval_challenges[task_id]
preds = evaluate_on_task(task_data, predict_repeat)


# GÃ©nÃ©ration des prÃ©dictions (eval set)
submission_predictions = generate_predictions(eval_challenges, predict_advanced)

# âœ… VÃ©rification dâ€™une prÃ©diction
pred = submission_predictions[example_id][0]
show_grid(np.array(pred["attempt_1"]), title="Attempt 1", mode="arc")
show_grid(np.array(pred["attempt_2"]), title="Attempt 2", mode="arc")




# GÃ©nÃ©ration des prÃ©dictions sur le set d'Ã©valuation
submission_predictions = generate_predictions(eval_challenges, predict_advanced)

# VÃ©rification graphique dâ€™un exemple
example_id = list(eval_challenges.keys())[0]
pred = submission_predictions[example_id][0]

visualize_prediction(
    np.array(eval_challenges[example_id]["test"][0]["input"]),
    np.array(pred["attempt_1"]),
    idx=example_id
)



# Display predictions for the first 3 evaluation tasks
for task_id in list(eval_challenges.keys())[:3]:
    print(f"\nTask: {task_id}")
    for i, pred in enumerate(submission_predictions[task_id]):
        show_grid(np.array(pred["attempt_1"]), title=f"{task_id} - Prediction Attempt 1 - Test {i}", mode="arc")
        show_grid(np.array(pred["attempt_2"]), title=f"{task_id} - Prediction Attempt 2 - Test {i}", mode="arc")
        print(f"Displayed predictions for test case {i} of task {task_id}")

def plot_task(task, title=""):
    """
    Affiche les inputs et outputs des exemples dans une tÃ¢che ARC avec la palette ARC.
    """
    inputs = [example["input"] for example in task["test"]]
    outputs = [example.get("output") for example in task["test"]]

    n = len(inputs)
    fig, axs = plt.subplots(2, n, figsize=(4 * n, 6))

    # Si n == 1, axs est 1D, on le rend 2D pour la suite
    if n == 1:
        axs = np.expand_dims(axs, axis=1)

    for i in range(n):
        # Input
        draw_grid(axs[0, i], inputs[i], f"{title} - Input {i}")
        axs[0, i].axis("off")
        # Output
        if outputs[i] is not None:
            draw_grid(axs[1, i], outputs[i], f"{title} - Output {i}")
        else:
            axs[1, i].text(0.5, 0.5, "No output", ha='center', va='center')
            axs[1, i].set_title(f"{title} - Output {i}")
        axs[1, i].axis("off")

    plt.tight_layout()
    plt.show()

# Exemple d'utilisation
example_id = list(eval_challenges.keys())[0]
plot_task(eval_challenges[example_id], title=example_id)





def format_submission(predictions):
    submission = {}

    for task_id, task_data in test_challenges.items():
        predictions = []
        for test_example in task_data["test"]:
            input_grid = test_example["input"]
            p1, p2 = predict_advanced(input_grid)
            predictions.append([p1, p2])
        submission[task_id] = predictions


    return submission

    for pred_pair in preds:
        for pred in pred_pair:
            if not is_valid_grid(pred):
                print(f" Invalid grid in {task_id}")

submission_data = format_submission(evaluate_on_task)

# Save as JSON
submission_path = "/kaggle/working/sample_submission_new.json"
with open(submission_path, "w") as f:
    json.dump(submission_data, f)
print(f"Submission saved to {submission_path}")


# GÃ©nÃ©ration des prÃ©dictions (eval set)
submission_predictions = generate_predictions(eval_challenges, predict_advanced)






def annotate_grid(ax, grid):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j, i, str(grid[i, j]), ha='center', va='center', color='white' if grid[i, j] < 5 else 'black', fontsize=8)


print("Grille d'entrÃ©e :")
print(input_grid)

print("\nGrille prÃ©dite :")
print(pred_grid)

if input_grid.shape == pred_grid.shape:
    diff_numeric = input_grid - pred_grid
    print("\nDiffÃ©rence (entrÃ©e - prÃ©diction) :")
    print(diff_numeric)
else:
    print("\nâš ï¸� Les dimensions de l'entrÃ©e et de la prÃ©diction ne correspondent pas.")



plt.figure(figsize=(12, 4))

# Input
plt.subplot(1, 3, 1)
plt.title("Input")
plt.imshow(input_grid, cmap="viridis")
annotate_grid(plt.gca(), input_grid)
plt.axis("off")

# PrÃ©diction
plt.subplot(1, 3, 2)
plt.title("PrÃ©diction")
plt.imshow(pred_grid, cmap="viridis")
annotate_grid(plt.gca(), pred_grid)
plt.axis("off")

# DiffÃ©rence
plt.subplot(1, 3, 3)
plt.title("DiffÃ©rences")
if input_grid.shape == pred_grid.shape:
    diff_grid = (input_grid != pred_grid).astype(int)
else:
    diff_grid = np.ones_like(input_grid)
plt.imshow(diff_grid, cmap="gray")
annotate_grid(plt.gca(), diff_grid)
plt.axis("off")

plt.tight_layout()
plt.show()


# --- GÃ©nÃ©ration autonome de challenges uniques et leurs solutions ---

def hash_grid(grid):
    """Retourne un hash unique pour une grille (pour Ã©viter les doublons)."""
    return np.array(grid).tobytes()

# Collecte tous les hash des grilles dÃ©jÃ  existantes
def collect_existing_hashes(challenges):
    hashes = set()
    for task in challenges.values():
        for pair in task.get("train", []):
            hashes.add(hash_grid(pair["input"]))
            hashes.add(hash_grid(pair["output"]))
        for pair in task.get("test", []):
            hashes.add(hash_grid(pair["input"]))
    return hashes

hashes_existants = set()
for dataset in [train_challenges, test_challenges, eval_challenges]:
    hashes_existants |= collect_existing_hashes(dataset)

#  Modifie la gÃ©nÃ©ration pour Ã©viter ces hash
def generate_unique_challenges(n=10, size=(5, 5), num_colors=4, existing_hashes=None):
    seen = set(existing_hashes) if existing_hashes else set()
    challenges = []
    while len(challenges) < n:
        task = generate_symmetric_task(num_train=2, num_test=1, size=size)
        hashes = [hash_grid(pair["input"]) for pair in task["train"]]
        hashes += [hash_grid(test["input"]) for test in task["test"]]
        hashes += [hash_grid(pair["output"]) for pair in task["train"]]
        if any(h in seen for h in hashes):
            continue
        for h in hashes:
            seen.add(h)
        challenges.append(task)
    return challenges



# GÃ©nÃ¨re 10 nouveaux challenges symÃ©triques uniques
nouveaux_challenges = generate_unique_challenges(n=10, size=(5, 5), num_colors=4)

for idx, task in enumerate(nouveaux_challenges):
    print(f"\n=== Challenge {idx+1} ===")
    for i, pair in enumerate(task["train"]):
        print(f"Train {i} - Input:\n{np.array(pair['input'])}")
        print(f"Train {i} - Output (solution):\n{np.array(pair['output'])}")
    for i, test in enumerate(task["test"]):
        print(f"Test {i} - Input:\n{np.array(test['input'])}")
        print(f"Test {i} - Output (solution attendue):\n{np.fliplr(test['input'])}")


# Affichage visuel des nouveaux challenges gÃ©nÃ©rÃ©s (avec couleurs ARC)
for idx, task in enumerate(nouveaux_challenges):
    print(f"\n=== Challenge {idx+1} ===")
    for i, pair in enumerate(task["train"]):
        print(f"Train {i} - Input:")
        show_grid(pair["input"], title=f"Challenge {idx+1} - Train {i} - Input", mode="arc")
        print(f"Train {i} - Output (solution):")
        show_grid(pair["output"], title=f"Challenge {idx+1} - Train {i} - Output", mode="arc")
    for i, test in enumerate(task["test"]):
        print(f"Test {i} - Input:")
        show_grid(test["input"], title=f"Challenge {idx+1} - Test {i} - Input", mode="arc")
        print(f"Test {i} - Output (solution attendue):")
        show_grid(np.fliplr(test["input"]), title=f"Challenge {idx+1} - Test {i} - Output attendue", mode="arc")






# 1. GÃ©nÃ¨re beaucoup plus de challenges symÃ©triques
nouveaux_challenges = generate_unique_challenges(n=500, size=(5, 5), num_colors=4, existing_hashes=hashes_existants)

# 2. PrÃ©pare le dataset pixel-par-pixel AVEC la feature miroir
X_new, y_new = [], []
for task in nouveaux_challenges:
    for pair in task["train"]:
        X_new.append(np.array(pair["input"]))
        y_new.append(np.array(pair["output"]))

def prepare_pixel_dataset_sym(X, y, only_shape=(5, 5)):
    X_pixels, y_pixels = [], []
    for x, y_ in zip(X, y):
        x_arr = np.array(x)
        y_arr = np.array(y_)
        if x_arr.shape == only_shape and y_arr.shape == only_shape:
            h, w = only_shape
            for i in range(h):
                for j in range(w):
                    # Ajoute la valeur du pixel miroir horizontal comme feature
                    X_pixels.append([x_arr[i, j], i / h, j / w, x_arr[i, w-j-1]])
                    y_pixels.append(y_arr[i, j])
    return np.array(X_pixels), np.array(y_pixels)

# 3. Ajoute-les Ã  l'ancien dataset (optionnel, ou entraÃ®ne-toi uniquement sur les symÃ©triques)
train_X_aug = X_new
train_y_aug = y_new

# 4. Reconstruis le dataset pixel-par-pixel avec la nouvelle fonction
X_pixels_aug, y_pixels_aug = prepare_pixel_dataset_sym(train_X_aug, train_y_aug, only_shape=(5, 5))

print(f"ğŸ†• Nouveau dataset pixels : {X_pixels_aug.shape} features, {y_pixels_aug.shape} labels")

# 5. EntraÃ®ne un modÃ¨le plus puissant
clf_aug = RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42)
clf_aug.fit(X_pixels_aug, y_pixels_aug)

# 6. Teste sur un nouveau challenge
challenge = generate_symmetric_task(num_train=2, num_test=1, size=(5, 5))
test_input = np.array(challenge["test"][0]["input"])
true_solution = np.fliplr(test_input)
h, w = test_input.shape
X_test_pixels = np.array([[test_input[i, j], i / h, j / w, test_input[i, w-j-1]] for i in range(h) for j in range(w)])
pred_pixels = clf_aug.predict(X_test_pixels)
pred_grid = pred_pixels.reshape((h, w))

print("\n=== Exemple TEST ===")
print("Test Input :\n", test_input)
print("Vraie solution :\n", true_solution)
print("PrÃ©diction modÃ¨le :\n", pred_grid)
print(f"Accuracy par pixel sur ce test : {np.mean(true_solution == pred_grid)*100:.2f}%")






def annotate_grid(ax, grid):
    rows, cols = grid.shape
    for i in range(rows):
        for j in range(cols):
            ax.text(j+0.5, i+0.5, str(grid[i, j]), ha='center', va='center',
                    color='white' if grid[i, j] < 5 else 'black', fontsize=12)

# GÃ©nÃ¨re un challenge avec plusieurs exemples train/test
challenge = generate_symmetric_task(num_train=2, num_test=1, size=(5, 5))



# Affiche les exemples d'entraÃ®nement (input/output)
for i, pair in enumerate(challenge["train"]):
    print(f"\nExemple d'entraÃ®nement {i+1} :")
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(pair["input"], cmap="tab20", vmin=0, vmax=9)
    annotate_grid(axs[0], np.array(pair["input"]))
    axs[0].set_title("Input (exemple)")
    axs[0].axis("off")
    axs[1].imshow(pair["output"], cmap="tab20", vmin=0, vmax=9)
    annotate_grid(axs[1], np.array(pair["output"]))
    axs[1].set_title("Output (solution)")
    axs[1].axis("off")
    plt.tight_layout()
    plt.show()

# Affiche l'exemple test : input, vraie solution, prÃ©diction modÃ¨le
test_input = np.array(challenge["test"][0]["input"])
true_solution = np.fliplr(test_input)
h, w = test_input.shape
X_test_pixels = np.array([[test_input[i, j], i / h, j / w] for i in range(h) for j in range(w)])
pred_pixels = clf_aug.predict(X_test_pixels)
pred_grid = pred_pixels.reshape((h, w))

print("\n=== Exemple TEST ===")
fig, axs = plt.subplots(1, 3, figsize=(12, 4))
for ax, grid, title in zip(
    axs,
    [test_input, true_solution, pred_grid],
    ["Test Input", "Vraie solution", "PrÃ©diction modÃ¨le"]
):
    ax.imshow(grid, cmap="tab20", vmin=0, vmax=9)
    annotate_grid(ax, grid)
    ax.set_title(title)
    ax.axis("off")
plt.tight_layout()
plt.show()

# Affiche aussi les matrices pour vÃ©rification numÃ©rique
print("Test Input :\n", test_input)
print("Vraie solution :\n", true_solution)
print("PrÃ©diction modÃ¨le :\n", pred_grid)
print(f"Accuracy par pixel sur ce test : {np.mean(true_solution == pred_grid)*100:.2f}%")


def predict_attempts(input_grid, clf):
    """Retourne deux prÃ©dictions pour une grille d'entrÃ©e avec le modÃ¨le donnÃ©."""
    h, w = len(input_grid), len(input_grid[0])
    arr = np.array(input_grid)
    # VÃ©rifie le nombre de features attendu par le modÃ¨le
    n_features = getattr(clf, "n_features_in_", 3)
    if n_features == 4:
        X_test_pixels = np.array([
            [arr[i, j], i / h, j / w, arr[i, w-j-1]]
            for i in range(h) for j in range(w)
        ])
    else:
        X_test_pixels = np.array([
            [arr[i, j], i / h, j / w]
            for i in range(h) for j in range(w)
        ])
    pred_pixels = clf.predict(X_test_pixels)
    pred_grid = pred_pixels.reshape((h, w)).tolist()
    return pred_grid, pred_grid
# GÃ©nÃ©ration du dictionnaire de soumission
submission = {}
for task_id, task in eval_challenges.items():
    submission[task_id] = []
    for i, test_case in enumerate(task["test"]):
        input_grid = test_case["input"]
        attempt_1, attempt_2 = predict_attempts(input_grid, clf_aug)
        # Affichage couleur pour vÃ©rification
        print(f"TÃ¢che {task_id} - Test {i}")
        show_grid(attempt_1, title=f"{task_id} - Attempt 1", mode="arc")
        show_grid(attempt_2, title=f"{task_id} - Attempt 2", mode="arc")
        submission[task_id].append({
            "attempt_1": attempt_1,
            "attempt_2": attempt_2
        })

# Sauvegarde au format JSON
with open("/kaggle/working/submission.json", "w") as f:
    json.dump(submission, f)
print("âœ… Fichier submission.json gÃ©nÃ©rÃ© et sauvegardÃ©.")


print(f"Nombre de challenges dans la soumission : {len(submission_data)}")




# 1. Charger le fichier submission.json
submission_path = "/kaggle/working/submission.json"
with open(submission_path, "r") as f:
    submission = json.load(f)

# 2. Afficher 3 exemples (format colonne, rÃ©sumÃ©)
for i, (task_id, preds) in enumerate(submission.items()):
    print(f"\n=== Task ID : {task_id} ===")
    for j, pred in enumerate(preds):
        print(f"  Test {j+1}:")
        print("    attempt_1 :")
        for row in pred["attempt_1"]:
            print("      ", row)
        print("    attempt_2 :")
        for row in pred["attempt_2"]:
            print("      ", row)
    if i >= 2:  # Affiche seulement 3 tasks
        break



# 3. VÃ©rifier les doublons dans submission.json (task_id)
task_ids = list(submission.keys())
if len(task_ids) != len(set(task_ids)):
    print("â�Œ Doublons de task_id trouvÃ©s dans submission.json")
else:
    print("âœ… Aucun doublon de task_id dans submission.json")

# 4. VÃ©rifier les doublons de prÃ©diction pour chaque test d'une mÃªme tÃ¢che
for task_id, preds in submission.items():
    seen = set()
    for pred in preds:
        key = (str(pred["attempt_1"]), str(pred["attempt_2"]))
        if key in seen:
            print(f"â�Œ Doublon de prÃ©diction dans la tÃ¢che {task_id}")
        seen.add(key)

# 5. Charger les autres fichiers ARC
with open(data_path / "arc-agi_training_challenges.json") as f:
    train_challenges = json.load(f)
with open(data_path / "arc-agi_test_challenges.json") as f:
    test_challenges = json.load(f)
with open(data_path / "arc-agi_evaluation_challenges.json") as f:
    eval_challenges = json.load(f)


# Collecte tous les hash des grilles existantes
def collect_all_hashes(*datasets):
    hashes = set()
    for dataset in datasets:
        for task in dataset.values():
            for pair in task.get("train", []):
                hashes.add(hash_grid(pair["input"]))
                hashes.add(hash_grid(pair["output"]))
            for pair in task.get("test", []):
                hashes.add(hash_grid(pair["input"]))
    return hashes

hashes_existants = collect_all_hashes(train_challenges, test_challenges, eval_challenges)

# VÃ©rifie si une prÃ©diction du fichier submission.json existe dÃ©jÃ  ailleurs
for task_id, preds in submission.items():
    for pred in preds:
        for attempt in ["attempt_1", "attempt_2"]:
            grid = pred[attempt]
            if hash_grid(grid) in hashes_existants:
                print(f"âš ï¸� PrÃ©diction dÃ©jÃ  prÃ©sente dans les datasets pour {task_id} ({attempt})")

print("VÃ©rification terminÃ©e.")

