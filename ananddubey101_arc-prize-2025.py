import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import OneHotEncoder
import tensorflow as tf
from tensorflow.keras import layers, models
import itertools
from collections import Counter
import copy
import os
from tqdm import tqdm

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)


class ARCSolver:
    def __init__(self):
        self.heuristic_methods = [
            self.identity_transform,
            self.color_replacement,
            self.flip_horizontal,
            self.flip_vertical,
            self.rotate_90,
            self.rotate_180,
            self.rotate_270,
            self.shift_up,
            self.shift_down,
            self.shift_left,
            self.shift_right,
            self.object_completion
        ]
        self.cnn_model = None
    
    def load_data(self, challenge_file, solution_file=None):
        """Load challenge data and solutions if available"""
        with open(challenge_file, 'r') as f:
            challenges = json.load(f)
        
        solutions = None
        if solution_file:
            with open(solution_file, 'r') as f:
                solutions = json.load(f)
        
        return challenges, solutions
    
    def visualize_grid(self, grid, title=None):
        """Visualize a single grid"""
        plt.figure(figsize=(5, 5))
        plt.imshow(grid, cmap='tab10', vmin=0, vmax=9)
        plt.colorbar(ticks=range(10))
        plt.grid(True, color='black', linewidth=0.5)
        if title:
            plt.title(title)
        plt.show()
    
    def visualize_task(self, task):
        """Visualize a task with train and test examples"""
        n_train = len(task['train'])
        n_test = len(task['test'])
        
        fig, axes = plt.subplots(n_train, 2, figsize=(10, 5*n_train))
        if n_train == 1:
            axes = [axes]
        
        for i, example in enumerate(task['train']):
            axes[i][0].imshow(example['input'], cmap='tab10', vmin=0, vmax=9)
            axes[i][0].set_title(f"Train Input {i+1}")
            axes[i][0].grid(True, color='black', linewidth=0.5)
            
            axes[i][1].imshow(example['output'], cmap='tab10', vmin=0, vmax=9)
            axes[i][1].set_title(f"Train Output {i+1}")
            axes[i][1].grid(True, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
        
        # Display test inputs
        fig, axes = plt.subplots(1, n_test, figsize=(5*n_test, 5))
        if n_test == 1:
            axes = [axes]
        
        for i, test in enumerate(task['test']):
            axes[i].imshow(test['input'], cmap='tab10', vmin=0, vmax=9)
            axes[i].set_title(f"Test Input {i+1}")
            axes[i].grid(True, color='black', linewidth=0.5)
        
        plt.tight_layout()
        plt.show()
    
    # ========== BASELINE HEURISTIC TRANSFORMS ==========
    
    def identity_transform(self, grid):
        """Return the grid unchanged"""
        return copy.deepcopy(grid)
    
    def color_replacement(self, input_grid, examples):
        """Map colors from input to output based on examples"""
        # Create color mapping from examples
        color_map = {}
        for example in examples:
            in_grid = example['input']
            out_grid = example['output']
            
            # Check if grids have the same shape
            if np.array(in_grid).shape != np.array(out_grid).shape:
                return None
            
            for i in range(len(in_grid)):
                for j in range(len(in_grid[0])):
                    src_color = in_grid[i][j]
                    target_color = out_grid[i][j]
                    
                    if src_color in color_map and color_map[src_color] != target_color:
                        # Inconsistent mapping
                        return None
                    
                    color_map[src_color] = target_color
        
        # Apply mapping to input grid
        output = copy.deepcopy(input_grid)
        for i in range(len(input_grid)):
            for j in range(len(input_grid[0])):
                if input_grid[i][j] in color_map:
                    output[i][j] = color_map[input_grid[i][j]]
        
        return output
    
    def flip_horizontal(self, grid):
        """Flip grid horizontally"""
        return [row[::-1] for row in grid]
    
    def flip_vertical(self, grid):
        """Flip grid vertically"""
        return grid[::-1]
    
    def rotate_90(self, grid):
        """Rotate grid 90 degrees clockwise"""
        return [list(row) for row in zip(*grid[::-1])]
    
    def rotate_180(self, grid):
        """Rotate grid 180 degrees"""
        return [row[::-1] for row in grid[::-1]]
    
    def rotate_270(self, grid):
        """Rotate grid 270 degrees clockwise (90 counter-clockwise)"""
        return [list(row) for row in zip(*grid)][::-1]
    
    def shift_up(self, grid, steps=1):
        """Shift grid up by steps"""
        result = copy.deepcopy(grid)
        h, w = len(grid), len(grid[0])
        for i in range(h):
            for j in range(w):
                if i + steps < h:
                    result[i][j] = grid[i + steps][j]
                else:
                    result[i][j] = 0
        return result
    
    def shift_down(self, grid, steps=1):
        """Shift grid down by steps"""
        result = copy.deepcopy(grid)
        h, w = len(grid), len(grid[0])
        for i in range(h-1, -1, -1):
            for j in range(w):
                if i - steps >= 0:
                    result[i][j] = grid[i - steps][j]
                else:
                    result[i][j] = 0
        return result
    
    def shift_left(self, grid, steps=1):
        """Shift grid left by steps"""
        result = copy.deepcopy(grid)
        h, w = len(grid), len(grid[0])
        for i in range(h):
            for j in range(w):
                if j + steps < w:
                    result[i][j] = grid[i][j + steps]
                else:
                    result[i][j] = 0
        return result
    
    def shift_right(self, grid, steps=1):
        """Shift grid right by steps"""
        result = copy.deepcopy(grid)
        h, w = len(grid), len(grid[0])
        for i in range(h):
            for j in range(w-1, -1, -1):
                if j - steps >= 0:
                    result[i][j] = grid[i][j - steps]
                else:
                    result[i][j] = 0
        return result
    
    def get_connected_components(self, grid, color=None):
        """Find connected components in the grid
        Returns a list of components, where each component is a list of (i, j) coordinates"""
        if not grid:
            return []
        
        h, w = len(grid), len(grid[0])
        visited = set()
        components = []
        
        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and (color is None or grid[i][j] == color):
                    # Start BFS from this cell
                    component = []
                    queue = [(i, j)]
                    cell_color = grid[i][j]
                    
                    while queue:
                        r, c = queue.pop(0)
                        if (r, c) in visited or grid[r][c] != cell_color:
                            continue
                        
                        visited.add((r, c))
                        component.append((r, c))
                        
                        # Check neighbors
                        for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited:
                                if grid[nr][nc] == cell_color:
                                    queue.append((nr, nc))
                    
                    components.append(component)
        
        return components
    
    def object_completion(self, grid):
        """Complete missing parts of objects based on patterns"""
        # This is a simplified version - in reality, this would be much more complex
        # Find connected components
        components = self.get_connected_components(grid)
        
        # For each component, try to identify if it's a partial shape
        # This is a placeholder - actual implementation would be more complex
        return grid
    
    # ========== ADVANCED ALGORITHMIC APPROACHES ==========
    
    def flood_fill(self, grid, i, j, target_color, replacement_color):
        """Flood fill algorithm to replace connected components"""
        if target_color == replacement_color:
            return grid
        
        h, w = len(grid), len(grid[0])
        result = copy.deepcopy(grid)
        
        if not (0 <= i < h and 0 <= j < w):
            return result
        
        if result[i][j] != target_color:
            return result
        
        stack = [(i, j)]
        while stack:
            r, c = stack.pop()
            if result[r][c] == target_color:
                result[r][c] = replacement_color
                
                # Add neighbors
                for dr, dc in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and result[nr][nc] == target_color:
                        stack.append((nr, nc))
        
        return result
    
    def extract_objects(self, grid):
        """Extract objects from the grid as separate features"""
        objects = []
        components = self.get_connected_components(grid)
        
        for component in components:
            # Find bounding box
            min_i = min(i for i, j in component)
            max_i = max(i for i, j in component)
            min_j = min(j for i, j in component)
            max_j = max(j for i, j in component)
            
            # Create a new grid for this object
            obj_height = max_i - min_i + 1
            obj_width = max_j - min_j + 1
            obj_grid = [[0 for _ in range(obj_width)] for _ in range(obj_height)]
            
            # Fill the object grid
            for i, j in component:
                obj_grid[i - min_i][j - min_j] = grid[i][j]
            
            objects.append({
                'grid': obj_grid,
                'position': (min_i, min_j),
                'size': (obj_height, obj_width),
                'color': grid[component[0][0]][component[0][1]]
            })
        
        return objects
    
    def detect_symmetry(self, grid):
        """Detect if grid has horizontal, vertical, or diagonal symmetry"""
        h, w = len(grid), len(grid[0])
        
        # Check horizontal symmetry
        h_symmetry = True
        for i in range(h):
            for j in range(w // 2):
                if grid[i][j] != grid[i][w-j-1]:
                    h_symmetry = False
                    break
            if not h_symmetry:
                break
        
        # Check vertical symmetry
        v_symmetry = True
        for i in range(h // 2):
            for j in range(w):
                if grid[i][j] != grid[h-i-1][j]:
                    v_symmetry = False
                    break
            if not v_symmetry:
                break
        
        # Check diagonal symmetry (top-left to bottom-right)
        if h == w:  # Only for square grids
            d1_symmetry = True
            for i in range(h):
                for j in range(i):
                    if grid[i][j] != grid[j][i]:
                        d1_symmetry = False
                        break
                if not d1_symmetry:
                    break
            
            # Check diagonal symmetry (top-right to bottom-left)
            d2_symmetry = True
            for i in range(h):
                for j in range(w):
                    if i + j == h - 1:
                        continue
                    if i + j < h - 1:
                        if grid[i][j] != grid[h-1-j][w-1-i]:
                            d2_symmetry = False
                            break
                if not d2_symmetry:
                    break
            
            return h_symmetry, v_symmetry, d1_symmetry, d2_symmetry
        
        return h_symmetry, v_symmetry, False, False
    
    def pattern_completion(self, grid, examples):
        """Try to complete a pattern based on training examples"""
        # This is a placeholder for a more complex implementation
        # In a real solution, you would analyze the patterns in examples
        # and try to extend them to complete the test grid
        return grid
    
    # ========== MACHINE LEARNING APPROACHES ==========
    
    def build_cnn_model(self, input_shape, max_output_shape):
        """Build a CNN model for transformation learning"""
        # Encode input grid
        input_layer = layers.Input(shape=input_shape + (1,))
        
        # Encoder
        x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_layer)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
        x = layers.MaxPooling2D((2, 2), padding='same')(x)
        
        # Latent representation
        x = layers.Flatten()(x)
        x = layers.Dense(256, activation='relu')(x)
        
        # Decoder
        x = layers.Dense(max_output_shape[0] * max_output_shape[1] * 64, activation='relu')(x)
        x = layers.Reshape((max_output_shape[0] // 4, max_output_shape[1] // 4, 64))(x)
        x = layers.Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
        x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
        
        # Output layer with 10 channels (one for each color 0-9)
        output_layer = layers.Conv2D(10, (3, 3), activation='softmax', padding='same')(x)
        
        model = models.Model(input_layer, output_layer)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        
        return model
    
    def preprocess_for_cnn(self, examples):
        """Preprocess examples for CNN training"""
        inputs = []
        outputs = []
        
        max_input_height = max(len(ex['input']) for ex in examples)
        max_input_width = max(len(ex['input'][0]) for ex in examples)
        max_output_height = max(len(ex['output']) for ex in examples)
        max_output_width = max(len(ex['output'][0]) for ex in examples)
        
        max_input_shape = (max_input_height, max_input_width)
        max_output_shape = (max_output_height, max_output_width)
        
        for example in examples:
            input_grid = example['input']
            output_grid = example['output']
            
            # Pad inputs and outputs to the maximum size
            padded_input = np.zeros(max_input_shape)
            padded_output = np.zeros(max_output_shape)
            
            for i in range(len(input_grid)):
                for j in range(len(input_grid[0])):
                    padded_input[i, j] = input_grid[i][j]
            
            for i in range(len(output_grid)):
                for j in range(len(output_grid[0])):
                    padded_output[i, j] = output_grid[i][j]
            
            inputs.append(padded_input)
            outputs.append(padded_output)
        
        return np.array(inputs), np.array(outputs), max_input_shape, max_output_shape
    
    def train_cnn_model(self, examples, epochs=100, batch_size=32):
        """Train a CNN model on the given examples"""
        inputs, outputs, input_shape, output_shape = self.preprocess_for_cnn(examples)
        
        # Reshape for CNN
        inputs = inputs.reshape(-1, input_shape[0], input_shape[1], 1)
        outputs = outputs.reshape(-1, output_shape[0], output_shape[1], 1)
        
        # Build and train the model
        self.cnn_model = self.build_cnn_model(input_shape, output_shape)
        self.cnn_model.fit(inputs, outputs, epochs=epochs, batch_size=batch_size, verbose=0)
        
        return self.cnn_model
    
    def predict_with_cnn(self, input_grid, input_shape, output_shape):
        """Predict the output grid using the trained CNN model"""
        if self.cnn_model is None:
            return None
        
        # Pad input to match the expected shape
        padded_input = np.zeros(input_shape)
        for i in range(min(len(input_grid), input_shape[0])):
            for j in range(min(len(input_grid[0]), input_shape[1])):
                padded_input[i, j] = input_grid[i][j]
        
        # Reshape for CNN
        input_tensor = padded_input.reshape(1, input_shape[0], input_shape[1], 1)
        
        # Get prediction
        prediction = self.cnn_model.predict(input_tensor)[0]
        
        # Convert prediction to grid
        predicted_grid = np.argmax(prediction, axis=-1)
        
        # Crop to the original output size (if needed)
        h, w = len(input_grid), len(input_grid[0])
        cropped_output = predicted_grid[:h, :w].tolist()
        
        return cropped_output
    
    def try_all_heuristics(self, task):
        """Try all heuristic methods and return the best matching one"""
        train_examples = task['train']
        test_examples = task['test']
        
        best_method = None
        best_score = -1
        
        # Try each heuristic method
        for method in self.heuristic_methods:
            correct_predictions = 0
            
            # Test the method on training examples
            for i in range(len(train_examples)):
                # Use all but the current example for training
                training_examples = train_examples[:i] + train_examples[i+1:]
                
                # Apply the method
                if method.__name__ == 'color_replacement' or method.__name__ == 'pattern_completion':
                    prediction = method(train_examples[i]['input'], training_examples)
                else:
                    prediction = method(train_examples[i]['input'])
                
                if prediction is None:
                    continue
                
                # Check if prediction matches the expected output
                expected = train_examples[i]['output']
                
                if self.grids_match(prediction, expected):
                    correct_predictions += 1
            
            # Calculate score
            score = correct_predictions / len(train_examples)
            
            if score > best_score:
                best_score = score
                best_method = method
        
        # If no heuristic worked well, return None
        if best_score < 0.5:  # Threshold can be adjusted
            return None
        
        # Apply the best method to test examples
        predictions = []
        for test in test_examples:
            if best_method.__name__ == 'color_replacement' or best_method.__name__ == 'pattern_completion':
                prediction = best_method(test['input'], train_examples)
            else:
                prediction = best_method(test['input'])
            
            predictions.append(prediction)
        
        return predictions
    
    def grids_match(self, grid1, grid2):
        """Check if two grids match exactly"""
        if len(grid1) != len(grid2) or len(grid1[0]) != len(grid2[0]):
            return False
        
        for i in range(len(grid1)):
            for j in range(len(grid1[0])):
                if grid1[i][j] != grid2[i][j]:
                    return False
        
        return True
    
    def solve_task(self, task):
        """Solve a single task using the best approach"""
        # First try heuristic methods
        heuristic_predictions = self.try_all_heuristics(task)
        
        if heuristic_predictions:
            return heuristic_predictions
        
        # If heuristics fail, try CNN
        train_examples = task['train']
        test_examples = task['test']
        
        # Train CNN model
        self.train_cnn_model(train_examples, epochs=50)
        
        # Make predictions
        inputs, _, input_shape, output_shape = self.preprocess_for_cnn(train_examples)
        
        predictions = []
        for test in test_examples:
            prediction = self.predict_with_cnn(test['input'], input_shape, output_shape)
            predictions.append(prediction)
        
        return predictions
    
    def solve_all_tasks(self, challenges):
        """Solve all tasks and return predictions"""
        predictions = {}
        
        for task_id, task in tqdm(challenges.items(), desc="Solving tasks"):
            try:
                task_predictions = self.solve_task(task)
                predictions[task_id] = [task_predictions]
            except Exception as e:
                print(f"Error solving task {task_id}: {e}")
                # Fallback to identity transform
                predictions[task_id] = [[self.identity_transform(test['input']) for test in task['test']]]
        
        return predictions
    
    def evaluate_predictions(self, predictions, solutions):
        """Evaluate predictions against ground truth solutions"""
        correct = 0
        total = 0
        
        for task_id, task_predictions in predictions.items():
            if task_id not in solutions:
                continue
            
            ground_truth = solutions[task_id]
            
            # Check if predictions match ground truth
            if len(task_predictions[0]) != len(ground_truth[0]):
                continue
            
            match = True
            for i in range(len(task_predictions[0])):
                if not self.grids_match(task_predictions[0][i], ground_truth[0][i]):
                    match = False
                    break
            
            if match:
                correct += 1
            
            total += 1
        
        accuracy = correct / total if total > 0 else 0
        print(f"Accuracy: {accuracy:.2f} ({correct}/{total})")
        
        return accuracy
    
    def save_predictions(self, predictions, output_file):
        """Save predictions to a JSON file"""
        with open(output_file, 'w') as f:
            json.dump(predictions, f)


# Main execution
if __name__ == "__main__":
    # Initialize solver
    solver = ARCSolver()
    
    # Set the paths for the ARC dataset
    base_path = "/kaggle/input/arc-prize-2025"
    
    # Load training data
    training_challenges, training_solutions = solver.load_data(
        os.path.join(base_path, 'arc-agi_training_challenges.json'), 
        os.path.join(base_path, 'arc-agi_training_solutions.json')
    )
    
    # Load evaluation data
    eval_challenges, eval_solutions = solver.load_data(
        os.path.join(base_path, 'arc-agi_evaluation_challenges.json'), 
        os.path.join(base_path, 'arc-agi_evaluation_solutions.json')
    )
    
    # Load test data
    test_challenges, _ = solver.load_data(
        os.path.join(base_path, 'arc-agi_test_challenges.json')
    )
    
    # Example: Visualize a specific task
    # First task ID
    first_task_id = list(training_challenges.keys())[0]
    solver.visualize_task(training_challenges[first_task_id])
    
    # Solve all evaluation tasks
    print("Solving evaluation tasks...")
    eval_predictions = solver.solve_all_tasks(eval_challenges)
    
    # Evaluate predictions
    print("Evaluating predictions...")
    solver.evaluate_predictions(eval_predictions, eval_solutions)
    
    # Solve test tasks
    print("Solving test tasks...")
    test_predictions = solver.solve_all_tasks(test_challenges)
    
    # Save predictions
    output_path = "submission.json"
    solver.save_predictions(test_predictions, output_path)
    
    print(f"Done! Predictions saved to {output_path}.")

