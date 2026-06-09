#!/usr/bin/env python3

import json
import numpy as np
import pandas as pd
from pathlib import Path
from collections import Counter
from scipy.ndimage import label
import time

def apply_strategy_dominant_color(input_grid, train_examples):
    try:
        flat = input_grid.flatten()
        non_zero = flat[flat != 0]
        if len(non_zero) > 0:
            dominant = Counter(non_zero).most_common(1)[0][0]
            output = np.full_like(input_grid, dominant)
            return output
    except:
        pass
    return None

def apply_strategy_extract_pattern(input_grid, train_examples):
    try:
        non_zero_mask = input_grid != 0
        if np.any(non_zero_mask):
            rows, cols = np.where(non_zero_mask)
            min_row, max_row = rows.min(), rows.max()
            min_col, max_col = cols.min(), cols.max()
            
            pattern = input_grid[min_row:max_row+1, min_col:max_col+1].copy()
            output = np.zeros_like(input_grid)
            
            # Centra il pattern
            start_row = (output.shape[0] - pattern.shape[0]) // 2
            start_col = (output.shape[1] - pattern.shape[1]) // 2
            
            if start_row >= 0 and start_col >= 0:
                end_row = start_row + pattern.shape[0]
                end_col = start_col + pattern.shape[1]
                if end_row <= output.shape[0] and end_col <= output.shape[1]:
                    output[start_row:end_row, start_col:end_col] = pattern
                    return output
    except:
        pass
    return None

def apply_strategy_scale_pattern(input_grid, train_examples):
    try:
        if len(train_examples) == 0:
            return None
            
        # Calcola il fattore di scala medio dai training examples
        scale_factors = []
        for example in train_examples:
            inp = np.array(example['input'])
            out = np.array(example['output'])
            if inp.shape[0] > 0 and inp.shape[1] > 0:
                scale_h = out.shape[0] / inp.shape[0]
                scale_w = out.shape[1] / inp.shape[1]
                if scale_h == scale_w and scale_h > 0:
                    scale_factors.append(int(scale_h))
        
        if not scale_factors:
            return None
            
        scale = Counter(scale_factors).most_common(1)[0][0]
        
        if scale == 1:
            return input_grid.copy()
        elif scale > 1:
            # Upscaling
            output = np.repeat(np.repeat(input_grid, scale, axis=0), scale, axis=1)
            return output
        else:
            # Downscaling
            factor = int(1 / scale)
            output = input_grid[::factor, ::factor]
            return output
    except:
        pass
    return None

def apply_strategy_flip_horizontal(input_grid, train_examples):
    try:
        return np.fliplr(input_grid)
    except:
        pass
    return None

def apply_strategy_flip_vertical(input_grid, train_examples):
    try:
        return np.flipud(input_grid)
    except:
        pass
    return None

def apply_strategy_rotate_90(input_grid, train_examples):
    try:
        return np.rot90(input_grid)
    except:
        pass
    return None

def apply_strategy_color_swap(input_grid, train_examples):
    try:
        flat = input_grid.flatten()
        unique, counts = np.unique(flat, return_counts=True)
        
        if len(unique) >= 2:
            sorted_indices = np.argsort(-counts)
            color1 = unique[sorted_indices[0]]
            color2 = unique[sorted_indices[1]]
            
            output = input_grid.copy()
            mask1 = output == color1
            mask2 = output == color2
            output[mask1] = color2
            output[mask2] = color1
            return output
    except:
        pass
    return None

def apply_strategy_copy_to_output_size(input_grid, train_examples):
    try:
        if len(train_examples) == 0:
            return None
            
        # Trova le dimensioni di output pi첫 comuni
        output_shapes = [np.array(ex['output']).shape for ex in train_examples]
        if output_shapes:
            most_common_shape = Counter(output_shapes).most_common(1)[0][0]
            output = np.zeros(most_common_shape, dtype=input_grid.dtype)
            
            # Copia quanto possibile dell'input nell'output
            h = min(input_grid.shape[0], output.shape[0])
            w = min(input_grid.shape[1], output.shape[1])
            output[:h, :w] = input_grid[:h, :w]
            return output
    except:
        pass
    return None

def apply_strategy_remove_background(input_grid, train_examples):
    try:
        flat = input_grid.flatten()
        background = Counter(flat).most_common(1)[0][0]
        output = input_grid.copy()
        output[output == background] = 0
        return output
    except:
        pass
    return None

def apply_strategy_connected_components(input_grid, train_examples):
    try:
        # Maschera dei pixel non-zero
        mask = input_grid != 0
        labeled, num_features = label(mask)
        
        if num_features > 0:
            # Trova la componente pi첫 grande
            sizes = [(labeled == i).sum() for i in range(1, num_features + 1)]
            largest = np.argmax(sizes) + 1
            
            output = np.where(labeled == largest, input_grid, 0)
            return output
    except:
        pass
    return None

def solve_task_with_strategies(task_data):
    train_examples = task_data['train']
    test_input = np.array(task_data['test'][0]['input'])
    
    strategies = [
        apply_strategy_scale_pattern,
        apply_strategy_extract_pattern,
        apply_strategy_copy_to_output_size,
        apply_strategy_dominant_color,
        apply_strategy_connected_components,
        apply_strategy_flip_horizontal,
        apply_strategy_flip_vertical,
        apply_strategy_rotate_90,
        apply_strategy_color_swap,
        apply_strategy_remove_background,
    ]
    
    attempts = []
    
    for strategy in strategies:
        try:
            result = strategy(test_input, train_examples)
            if result is not None and result.size > 0:
                if not np.array_equal(result, test_input): 
                    attempts.append(result.tolist())
                    if len(attempts) >= 2:
                        break
        except Exception as e:
            continue
    
    while len(attempts) < 2:
        attempts.append(test_input.tolist())
    
    return {
        'attempt_1': attempts[0],
        'attempt_2': '[[0,0],[0,0]]'
    }

def solve_task_with_multiple_outputs(task_data):
    train_examples = task_data['train']
    test_inputs = task_data['test']
    
    results = []
    
    for test_pair in test_inputs:
        test_input = np.array(test_pair['input'])
        
        strategies = [
            apply_strategy_scale_pattern,
            apply_strategy_extract_pattern,
            apply_strategy_copy_to_output_size,
            apply_strategy_dominant_color,
            apply_strategy_connected_components,
        ]
        
        attempts = []
        
        for strategy in strategies:
            try:
                result = strategy(test_input, train_examples)
                if result is not None and result.size > 0:
                    if not np.array_equal(result, test_input):
                        attempts.append(result.tolist())
                        if len(attempts) >= 2:
                            break
            except:
                continue
        
        while len(attempts) < 2:
            attempts.append(test_input.tolist())
        
        results.append({
            'attempt_1': attempts[0],
            'attempt_2': '[[0,0],[0,0]]'
        })
    
    return results

def generate_submission(test_challenges_path, output_path='/kaggle/working/submission.json'):
    
    with open(test_challenges_path, 'r') as f:
        test_challenges = json.load(f)
    
    submission = {}
    total_tasks = len(test_challenges)
    
    for idx, (task_id, task_data) in enumerate(test_challenges.items(), 1):
        try:
            num_test_outputs = len(task_data['test'])
            
            if num_test_outputs == 1:
                result = solve_task_with_strategies(task_data)
                submission[task_id] = [result]
            else:
                results = solve_task_with_multiple_outputs(task_data)
                submission[task_id] = results
        
        except Exception as e:
            test_input = task_data['test'][0]['input']
            submission[task_id] = [{
                'attempt_1': test_input,
                'attempt_2': '[[0,0],[0,0]]'
            }]

    with open(output_path, 'w') as f:
        json.dump(submission, f)
       
    return submission

if __name__ == "__main__":
    """
    /kaggle/input/arc-prize-2025/arc-agi_training_solutions.json
    /kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json
    /kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json
    /kaggle/input/arc-prize-2025/sample_submission.json
    /kaggle/input/arc-prize-2025/arc-agi_training_challenges.json
    /kaggle/input/arc-prize-2025/arc-agi_test_challenges.json
    """
    
    test_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'
    output_path = '/kaggle/working/submission.json'

    submission = generate_submission(test_path, output_path)
    


