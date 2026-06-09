import json
import random
import numpy as np
from collections import Counter
import matplotlib.pyplot as plt


with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
    train_data = json.load(f)

print(f"Total training tasks: {len(train_data)}")
sample_task_key = list(train_data.keys())[0]
print(json.dumps(train_data[sample_task_key], indent=2))


def plot_grid(grid, ax, title=""):
    # Function for visualizing a grid
    ax.imshow(grid, cmap='tab10', vmin=0, vmax=9)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

def visualize_task(task):
    # Visualizing only the train input/output pairs
    fig, axes = plt.subplots(2, len(task["train"]), figsize=(10, 5))

    # Visual depiction of train pairs
    for i, pair in enumerate(task["train"]):
        plot_grid(np.array(pair["input"]), axes[0, i], title=f"Train Input {i+1}")
        plot_grid(np.array(pair["output"]), axes[1, i], title=f"Train Output {i+1}")

    plt.tight_layout()
    plt.show()

sample_task = train_data[random.choice(list(train_data.keys()))]
visualize_task(sample_task)


def detect_shape_change(input_grid, output_grid):
    in_shape = np.array(input_grid).shape
    out_shape = np.array(output_grid).shape
    return in_shape, out_shape, in_shape != out_shape

sample_train = sample_task["train"]
for i, pair in enumerate(sample_train):
    in_shape, out_shape, changed = detect_shape_change(pair["input"], pair["output"])
    print(f"Train Pair {i+1}: Input Shape {in_shape} → Output Shape {out_shape}, Changed: {changed}")


def detect_color_mapping(input_grid, output_grid):
    input_colors = Counter(np.array(input_grid).flatten())
    output_colors = Counter(np.array(output_grid).flatten())
    return input_colors, output_colors

for i, pair in enumerate(sample_train):
    in_colors, out_colors = detect_color_mapping(pair["input"], pair["output"])
    print(f"Train Pair {i+1}: Input Colors {in_colors} → Output Colors {out_colors}")


def check_transformations(input_grid, output_grid):
    input_array = np.array(input_grid)
    output_array = np.array(output_grid)

    # Check flipping
    if np.array_equal(output_array, np.flipud(input_array)):
        return "Vertical Flip"
    elif np.array_equal(output_array, np.fliplr(input_array)):
        return "Horizontal Flip"
    
    # Check rotation
    if np.array_equal(output_array, np.rot90(input_array, 1)):
        return "Rotated 90°"
    elif np.array_equal(output_array, np.rot90(input_array, 2)):
        return "Rotated 180°"
    elif np.array_equal(output_array, np.rot90(input_array, 3)):
        return "Rotated 270°"

    return "No simple transformation detected"

for i, pair in enumerate(sample_train):
    transformation = check_transformations(pair["input"], pair["output"])
    print(f"Train Pair {i+1}: Transformation → {transformation}")


def detect_object_movement(input_grid, output_grid):
    input_positions = np.argwhere(np.array(input_grid) > 0)
    output_positions = np.argwhere(np.array(output_grid) > 0)
    movement = output_positions - input_positions if input_positions.shape == output_positions.shape else "Varied movement"
    return movement
    
for i, pair in enumerate(sample_train):
    movement = detect_object_movement(pair["input"], pair["output"])
    print(f"Train Pair {i+1}: Object Movement → {movement}")


def detect_transformation(input_grid, output_grid):
    input_array = np.array(input_grid)
    output_array = np.array(output_grid)
    
    if input_array.shape != output_array.shape:
        return "Resized"
    
    # Color Mapping
    input_colors = Counter(input_array.flatten())
    output_colors = Counter(output_array.flatten())
    if input_colors.keys() != output_colors.keys():
        return "Color Change"
    
    # Flipping
    if np.array_equal(output_array, np.flipud(input_array)):
        return "Vertical Flip"
    elif np.array_equal(output_array, np.fliplr(input_array)):
        return "Horizontal Flip"
    
    # Rotation
    if np.array_equal(output_array, np.rot90(input_array, 1)):
        return "Rotated 90°"
    elif np.array_equal(output_array, np.rot90(input_array, 2)):
        return "Rotated 180°"
    elif np.array_equal(output_array, np.rot90(input_array, 3)):
        return "Rotated 270°"
    
    return "Unknown Transformation"

def apply_transformation(input_grid, transformation):
    input_array = np.array(input_grid)
    
    if transformation == "Vertical Flip":
        return np.flipud(input_array).tolist()
    elif transformation == "Horizontal Flip":
        return np.fliplr(input_array).tolist()
    elif transformation == "Rotated 90°":
        return np.rot90(input_array, 1).tolist()
    elif transformation == "Rotated 180°":
        return np.rot90(input_array, 2).tolist()
    elif transformation == "Rotated 270°":
        return np.rot90(input_array, 3).tolist()
    elif transformation == "Resized":
        return input_grid
    elif transformation == "Color Change":
        return input_grid 
    
    return input_grid 


def solve_task(task):
    transformations = []
    
    for pair in task["train"]:
        transformation = detect_transformation(pair["input"], pair["output"])
        transformations.append(transformation)
    
    test_predictions = []
    for test_input in task["test"]:
        predicted_output = apply_transformation(test_input["input"], transformations[0]) 
        test_predictions.append(predicted_output)
    
    return test_predictions

sample_task = train_data[random.choice(list(train_data.keys()))]
predicted_outputs = solve_task(sample_task)

# Visualize results
for i, test in enumerate(sample_task["test"]):
    print(f"\nTest Input {i+1}:")
    print(np.array(test["input"]))
    print(f"Predicted Output {i+1}:")
    print(np.array(predicted_outputs[i]))


with open('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
    solutions_data = json.load(f)

random_task_key = next(iter(solutions_data.keys()))  
print(f"Random key: {random_task_key}")
print(f"Value type: {type(solutions_data[random_task_key])}")  
print(f"Sample content: {solutions_data[random_task_key]}") 


first_keys = list(solutions_data.keys())[:2] 
for key in first_keys:
    print(f"Key: {key}")
    print(f"Value: {solutions_data[key]}\n")  


correct = 0
total = 0

for task_key in train_data.keys():
    task = train_data[task_key]
    true_outputs = solutions_data[task_key] 

    predicted_outputs = solve_task(task) 

    for pred, true in zip(predicted_outputs, true_outputs):
        if np.array_equal(np.array(pred), np.array(true)):
            correct += 1
        total += 1

print(f"Accuracy: {correct}/{total} ({(correct/total)*100:.2f}%)")


for task_key in list(train_data.keys())[:5]: 
    task = train_data[task_key]
    true_outputs = solutions_data[task_key] 
    predicted_outputs = solve_task(task)  

    print(f"Task: {task_key}")
    print("Predicted Output:")
    print(predicted_outputs)
    print("True Output:")
    print(true_outputs)
    print("=" * 50)  

