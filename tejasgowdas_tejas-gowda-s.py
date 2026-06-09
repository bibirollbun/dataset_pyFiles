import numpy as np
import json
import os
from pathlib import Path
import time
from typing import List, Dict, Any, Tuple
import multiprocessing as mp
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# Configuration
USE_GPU = torch.cuda.is_available()
DEVICE = torch.device("cuda" if USE_GPU else "cpu")
NUM_PROCESSES = mp.cpu_count() - 1 if mp.cpu_count() > 1 else 1
MAX_GRID_SIZE = 30
NUM_ATTEMPTS = 2
PATTERNS_TO_CHECK = 50

class ARCModel(nn.Module):
    def __init__(self, hidden_size=512):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(MAX_GRID_SIZE * MAX_GRID_SIZE * 10, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
        )
        
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=hidden_size, nhead=8, batch_first=True),
            num_layers=4
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, MAX_GRID_SIZE * MAX_GRID_SIZE * 10)
        )
        
    def forward(self, x):
        x = self.encoder(x)
        x = x.unsqueeze(1)
        x = self.transformer(x)
        x = x.squeeze(1)
        x = self.decoder(x)
        return x

class ARCSolver:
    def __init__(self):
        self.model = ARCModel().to(DEVICE) if USE_GPU else ARCModel()
        self.pattern_library = {}
        self.transformation_rules = []
        
    def load_task(self, task_data: Dict) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        """Load and parse a task into training and test examples"""
        train_inputs, train_outputs = [], []
        test_inputs = []
        
        for train_example in task_data["train"]:
            train_inputs.append(np.array(train_example["input"]))
            train_outputs.append(np.array(train_example["output"]))
            
        for test_example in task_data["test"]:
            test_inputs.append(np.array(test_example["input"]))
            
        return train_inputs, train_outputs, test_inputs
    
    def preprocess_grid(self, grid: np.ndarray) -> np.ndarray:
        """Preprocess a grid for model input"""
        # Pad or crop grid to MAX_GRID_SIZE
        h, w = grid.shape
        padded = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE), dtype=int)
        h_to_use, w_to_use = min(h, MAX_GRID_SIZE), min(w, MAX_GRID_SIZE)
        padded[:h_to_use, :w_to_use] = grid[:h_to_use, :w_to_use]
        
        # One-hot encode the grid
        one_hot = np.zeros((MAX_GRID_SIZE, MAX_GRID_SIZE, 10), dtype=np.float32)
        for i in range(10):
            one_hot[..., i] = (padded == i)
        
        return one_hot.flatten()
    
    def extract_patterns(self, grid: np.ndarray) -> Dict:
        """Extract notable patterns from a grid"""
        patterns = {}
        h, w = grid.shape
        
        # Extract shape outlines (non-zero elements)
        patterns["shapes"] = np.where(grid > 0)
        
        # Extract color distribution
        unique, counts = np.unique(grid, return_counts=True)
        patterns["colors"] = dict(zip(unique.tolist(), counts.tolist()))
        
        # Extract symmetry patterns
        patterns["h_symmetry"] = np.allclose(grid, np.fliplr(grid))
        patterns["v_symmetry"] = np.allclose(grid, np.flipud(grid))
        
        # Extract connectivity information
        connected_components = []
        visited = set()
        
        def dfs(i, j, color):
            component = []
            stack = [(i, j)]
            while stack:
                x, y = stack.pop()
                if (x, y) in visited or x < 0 or y < 0 or x >= h or y >= w or grid[x, y] != color:
                    continue
                visited.add((x, y))
                component.append((x, y))
                for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    stack.append((x + dx, y + dy))
            return component
        
        for i in range(h):
            for j in range(w):
                if (i, j) not in visited and grid[i, j] > 0:
                    component = dfs(i, j, grid[i, j])
                    connected_components.append((grid[i, j], component))
        
        patterns["components"] = connected_components
        return patterns
    
    def analyze_transformation(self, input_grid: np.ndarray, output_grid: np.ndarray) -> Dict:
        """Analyze the transformation from input to output"""
        transformation = {}
        
        # Check basic transformations
        transformation["rotation_90"] = np.allclose(output_grid, np.rot90(input_grid))
        transformation["rotation_180"] = np.allclose(output_grid, np.rot90(input_grid, 2))
        transformation["rotation_270"] = np.allclose(output_grid, np.rot90(input_grid, 3))
        transformation["h_flip"] = np.allclose(output_grid, np.fliplr(input_grid))
        transformation["v_flip"] = np.allclose(output_grid, np.flipud(input_grid))
        
        # Check color transformations
        input_colors = set(np.unique(input_grid))
        output_colors = set(np.unique(output_grid))
        
        transformation["color_mapping"] = {}
        if len(input_colors) == len(output_colors):
            for i_color in input_colors:
                mask = (input_grid == i_color)
                if np.sum(mask) > 0:
                    values, counts = np.unique(output_grid[mask], return_counts=True)
                    if len(values) == 1:
                        transformation["color_mapping"][int(i_color)] = int(values[0])
        
        # Check size transformations
        transformation["size_change"] = (output_grid.shape != input_grid.shape)
        
        # Check pattern-based transformations
        input_patterns = self.extract_patterns(input_grid)
        output_patterns = self.extract_patterns(output_grid)
        
        transformation["pattern_changes"] = {
            "shapes": len(input_patterns["components"]) != len(output_patterns["components"]),
            "colors": input_patterns["colors"] != output_patterns["colors"],
            "symmetry": (input_patterns["h_symmetry"] != output_patterns["h_symmetry"] or 
                         input_patterns["v_symmetry"] != output_patterns["v_symmetry"])
        }
        
        return transformation
    
    def apply_transformation(self, input_grid: np.ndarray, transformation: Dict) -> np.ndarray:
        """Apply a detected transformation to an input grid"""
        result = input_grid.copy()
        
        # Apply basic transformations
        if transformation.get("rotation_90", False):
            result = np.rot90(result)
        elif transformation.get("rotation_180", False):
            result = np.rot90(result, 2)
        elif transformation.get("rotation_270", False):
            result = np.rot90(result, 3)
        elif transformation.get("h_flip", False):
            result = np.fliplr(result)
        elif transformation.get("v_flip", False):
            result = np.flipud(result)
        
        # Apply color transformations
        if transformation.get("color_mapping"):
            for src, dst in transformation["color_mapping"].items():
                result[result == src] = dst
        
        return result
    
    def solve_by_pattern_matching(self, train_inputs: List[np.ndarray], 
                                 train_outputs: List[np.ndarray], 
                                 test_input: np.ndarray) -> List[np.ndarray]:
        """Solve a task by pattern matching approach"""
        predictions = []
        
        # Analyze transformations in training examples
        transformations = []
        for i_ex in range(len(train_inputs)):
            transformation = self.analyze_transformation(train_inputs[i_ex], train_outputs[i_ex])
            transformations.append(transformation)
        
        # Try applying each transformation to test input
        for transformation in transformations:
            pred = self.apply_transformation(test_input, transformation)
            predictions.append(pred)
        
        # If no transformation worked well, try some common operations
        if not predictions:
            # Try identity
            predictions.append(test_input.copy())
            
            # Try flipping
            predictions.append(np.fliplr(test_input))
            predictions.append(np.flipud(test_input))
            
            # Try rotations
            predictions.append(np.rot90(test_input))
            predictions.append(np.rot90(test_input, 2))
            
            # Try color inversion (assuming color values range from 0-9)
            max_color = np.max(test_input)
            if max_color > 0:
                inverted = max_color - test_input
                predictions.append(inverted)
        
        # Return top 2 unique predictions
        unique_preds = []
        for pred in predictions:
            if not any(np.array_equal(pred, existing) for existing in unique_preds):
                unique_preds.append(pred)
                if len(unique_preds) >= NUM_ATTEMPTS:
                    break
        
        # If we still need more predictions, add some variations
        while len(unique_preds) < NUM_ATTEMPTS:
            # Add a default prediction (either original input or slightly modified)
            if len(unique_preds) == 0:
                unique_preds.append(test_input.copy())
            else:
                # Apply a random transformation to the first prediction
                modified = unique_preds[0].copy()
                if np.random.random() > 0.5:
                    modified = np.rot90(modified)
                unique_preds.append(modified)
        
        return unique_preds[:NUM_ATTEMPTS]
    
    def solve_by_neural_model(self, train_inputs: List[np.ndarray], 
                             train_outputs: List[np.ndarray], 
                             test_input: np.ndarray) -> List[np.ndarray]:
        """Solve a task using the neural network model"""
        # Convert training examples to model format
        X_train = torch.stack([torch.tensor(self.preprocess_grid(grid), dtype=torch.float32) 
                              for grid in train_inputs]).to(DEVICE)
        y_train = torch.stack([torch.tensor(self.preprocess_grid(grid), dtype=torch.float32) 
                              for grid in train_outputs]).to(DEVICE)
        
        # Quick fine-tuning on this specific task
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        
        for epoch in range(50):  # Quick training
            optimizer.zero_grad()
            outputs = self.model(X_train)
            loss = F.mse_loss(outputs, y_train)
            loss.backward()
            optimizer.step()
        
        # Now predict on test input
        self.model.eval()
        with torch.no_grad():
            X_test = torch.tensor(self.preprocess_grid(test_input), dtype=torch.float32).unsqueeze(0).to(DEVICE)
            prediction = self.model(X_test).squeeze(0).cpu().numpy()
        
        # Convert model output back to grid format
        pred_reshaped = prediction.reshape(MAX_GRID_SIZE, MAX_GRID_SIZE, 10)
        grid_pred = np.argmax(pred_reshaped, axis=2)
        
        # Determine output shape (use the maximum of input and output training shapes)
        max_h = max([max(inp.shape[0], out.shape[0]) for inp, out in zip(train_inputs, train_outputs)])
        max_w = max([max(inp.shape[1], out.shape[1]) for inp, out in zip(train_inputs, train_outputs)])
        
        # Clip to detected shape
        pred1 = grid_pred[:max_h, :max_w].copy()
        
        # Generate a second prediction with slight variation
        pred2 = pred1.copy()
        if np.random.random() > 0.5:
            pred2 = np.fliplr(pred2)
        else:
            pred2 = np.rot90(pred2)
        
        return [pred1, pred2]
    
    def solve_task(self, task_data: Dict) -> List[List[np.ndarray]]:
        """Solve a single task and return predictions for all test examples"""
        train_inputs, train_outputs, test_inputs = self.load_task(task_data)
        all_predictions = []
        
        for test_input in test_inputs:
            # Try pattern matching approach first
            pattern_predictions = self.solve_by_pattern_matching(train_inputs, train_outputs, test_input)
            
            # If the task seems complex, also try neural approach
            if len(train_inputs) > 2 or any(grid.size > 25 for grid in train_inputs + train_outputs):
                neural_predictions = self.solve_by_neural_model(train_inputs, train_outputs, test_input)
                
                # Choose the best predictions from both approaches
                predictions = [pattern_predictions[0], neural_predictions[0]]
            else:
                predictions = pattern_predictions
            
            all_predictions.append(predictions)
        
        return all_predictions

def process_task(task_id: str, task_data: Dict, solver: ARCSolver) -> Tuple[str, List[Dict]]:
    """Process a single task and return formatted predictions"""
    predictions = solver.solve_task(task_data)
    formatted_preds = []
    
    for i, pred_pair in enumerate(predictions):
        pred_dict = {
            "attempt_1": pred_pair[0].tolist(),
            "attempt_2": pred_pair[1].tolist()
        }
        formatted_preds.append(pred_dict)
    
    return task_id, formatted_preds

def main():
    # Initialize solver
    solver = ARCSolver()
    
    # Load the evaluation tasks
    input_path = Path("../input/arc-prize-2025/evaluation")
    if not input_path.exists():
        input_path = Path("evaluation")  # Fallback path
    
    tasks = {}
    for file_path in input_path.glob("*.json"):
        with open(file_path, 'r') as f:
            task_data = json.load(f)
            tasks[file_path.stem] = task_data
    
    print(f"Loaded {len(tasks)} tasks for evaluation")
    
    # Process tasks in parallel
    results = {}
    with mp.Pool(processes=NUM_PROCESSES) as pool:
        process_func = partial(process_task, solver=solver)
        for task_id, preds in tqdm(pool.starmap(process_func, tasks.items()), total=len(tasks)):
            results[task_id] = preds
    
    # Save predictions to submission file
    with open('submission.json', 'w') as f:
        json.dump(results, f)
    
    print("Submission file created successfully!")

if __name__ == "__main__":
    start_time = time.time()
    main()
    print(f"Total execution time: {time.time() - start_time:.2f} seconds")

