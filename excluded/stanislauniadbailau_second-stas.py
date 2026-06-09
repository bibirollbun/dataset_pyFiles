import json
import numpy as np
from collections import Counter
from typing import List, Dict, Any, Tuple
from scipy import ndimage

with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
    test_challenges = json.load(f)

try:
    with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
        eval_challenges = json.load(f)
    with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json', 'r') as f:
        eval_solutions = json.load(f)
    print(f"✅ Loaded {len(eval_challenges)} evaluation tasks for testing")
except:
    eval_challenges = None
    eval_solutions = None
    print("⚠️  Evaluation data not available")

print(f"✅ Loaded {len(test_challenges)} test challenges")


class GridTilingSolver:
    def solve(self, task):
        train_pairs = task['train']
        test_inputs = [pair['input'] for pair in task['test']]
        
        for train_pair in train_pairs:
            input_grid = np.array(train_pair['input'])
            output_grid = np.array(train_pair['output'])
            
            ih, iw = input_grid.shape
            oh, ow = output_grid.shape
            
            if oh % ih == 0 and ow % iw == 0:
                v_factor = oh // ih
                h_factor = ow // iw
                
                tiled = np.tile(input_grid, (v_factor, h_factor))
                if np.array_equal(tiled, output_grid):
                    results = []
                    for test_input in test_inputs:
                        test_input = np.array(test_input)
                        tiled_output = np.tile(test_input, (v_factor, h_factor))
                        results.append(tiled_output)
                    return results
        
        return [np.array(test_inputs[i]) for i in range(len(test_inputs))]

solver_a1 = GridTilingSolver()
print("✅ GridTilingSolver (A1) loaded")


class ColorSubstitutionSolver:
    def solve(self, task):
        train_pairs = task['train']
        test_inputs = [pair['input'] for pair in task['test']]
        
        color_maps = []
        for train_pair in train_pairs:
            input_grid = np.array(train_pair['input'])
            output_grid = np.array(train_pair['output'])
            
            if input_grid.shape == output_grid.shape:
                color_map = {}
                for i in range(input_grid.shape[0]):
                    for j in range(input_grid.shape[1]):
                        in_color = input_grid[i, j]
                        out_color = output_grid[i, j]
                        if in_color not in color_map:
                            color_map[in_color] = out_color
                        elif color_map[in_color] != out_color:
                            color_map = None
                            break
                    if color_map is None:
                        break
                
                if color_map:
                    color_maps.append(color_map)
        
        if not color_maps:
            return [np.array(test_inputs[i]) for i in range(len(test_inputs))]
        
        consensus_map = color_maps[0] if len(color_maps) == 1 else {}
        if len(color_maps) > 1:
            for color in color_maps[0]:
                if all(cm.get(color) == color_maps[0][color] for cm in color_maps):
                    consensus_map[color] = color_maps[0][color]
        
        results = []
        for test_input in test_inputs:
            test_input = np.array(test_input)
            output = test_input.copy()
            for i in range(output.shape[0]):
                for j in range(output.shape[1]):
                    color = output[i, j]
                    if color in consensus_map:
                        output[i, j] = consensus_map[color]
            results.append(output)
        
        return results

solver_b1 = ColorSubstitutionSolver()
print("✅ ColorSubstitutionSolver (B1) loaded")


def simple_dsl_solver(train_pairs, test_inputs):
    return [np.array(inp['input']) for inp in test_inputs]

def connected_components_solver(train_pairs, test_inputs):
    return [np.array(inp['input']) for inp in test_inputs]

def object_relationships_solver(train_pairs, test_inputs):
    return [np.array(inp['input']) for inp in test_inputs]

print("✅ Placeholder solvers (C1, D1, C4) loaded")


def extract_constraints(train_pairs):
    constraints = {
        'size': extract_size_constraints(train_pairs),
        'color': extract_color_constraints(train_pairs),
        'spatial': extract_spatial_constraints(train_pairs)
    }
    return constraints

def extract_size_constraints(train_pairs):
    size_relations = []
    for pair in train_pairs:
        in_shape = np.array(pair['input']).shape
        out_shape = np.array(pair['output']).shape
        if in_shape[0] > 0 and in_shape[1] > 0:
            size_relations.append({
                'in': in_shape,
                'out': out_shape,
                'ratio': (out_shape[0] / in_shape[0], out_shape[1] / in_shape[1])
            })
    return size_relations

def extract_color_constraints(train_pairs):
    color_info = []
    for pair in train_pairs:
        in_colors = set(np.array(pair['input']).flatten())
        out_colors = set(np.array(pair['output']).flatten())
        color_info.append({
            'in': in_colors,
            'out': out_colors,
            'new': out_colors - in_colors,
            'removed': in_colors - out_colors
        })
    return color_info

def extract_spatial_constraints(train_pairs):
    spatial_info = []
    for pair in train_pairs:
        in_grid = np.array(pair['input'])
        out_grid = np.array(pair['output'])
        spatial_info.append({
            'is_symmetric_h': np.array_equal(out_grid, np.fliplr(out_grid)),
            'is_symmetric_v': np.array_equal(out_grid, np.flipud(out_grid)),
            'is_rotated': in_grid.shape == out_grid.shape[::-1]
        })
    return spatial_info

print("✅ Constraint extraction functions loaded")


class SymmetrySolver:
    def solve(self, task):
        train_pairs = task['train']
        test_inputs = [pair['input'] for pair in task['test']]
        
        transformations = []
        for train_pair in train_pairs:
            input_grid = np.array(train_pair['input'])
            output_grid = np.array(train_pair['output'])
            
            if np.array_equal(output_grid, np.fliplr(input_grid)):
                transformations.append('flip_h')
            elif np.array_equal(output_grid, np.flipud(input_grid)):
                transformations.append('flip_v')
            elif np.array_equal(output_grid, np.rot90(input_grid, k=1)):
                transformations.append('rot90')
            elif np.array_equal(output_grid, np.rot90(input_grid, k=2)):
                transformations.append('rot180')
            elif np.array_equal(output_grid, np.rot90(input_grid, k=3)):
                transformations.append('rot270')
            elif np.array_equal(output_grid, input_grid.T):
                transformations.append('transpose')
        
        if not transformations:
            return [np.array(test_inputs[i]) for i in range(len(test_inputs))]
        
        most_common = Counter(transformations).most_common(1)[0][0]
        
        results = []
        for test_input in test_inputs:
            test_input = np.array(test_input)
            if most_common == 'flip_h':
                output = np.fliplr(test_input)
            elif most_common == 'flip_v':
                output = np.flipud(test_input)
            elif most_common == 'rot90':
                output = np.rot90(test_input, k=1)
            elif most_common == 'rot180':
                output = np.rot90(test_input, k=2)
            elif most_common == 'rot270':
                output = np.rot90(test_input, k=3)
            elif most_common == 'transpose':
                output = test_input.T
            else:
                output = test_input
            results.append(output)
        
        return results

solver_e1 = SymmetrySolver()
print("✅ SymmetrySolver (E1) loaded")


class PatternCompletionSolver:
    def solve(self, task):
        train_pairs = task['train']
        test_inputs = [pair['input'] for pair in task['test']]
        
        for train_pair in train_pairs:
            input_grid = np.array(train_pair['input'])
            output_grid = np.array(train_pair['output'])
            
            if output_grid.shape[0] == input_grid.shape[0] * 2:
                if np.array_equal(output_grid[:input_grid.shape[0], :], input_grid):
                    results = []
                    for test_input in test_inputs:
                        test_input = np.array(test_input)
                        doubled = np.vstack([test_input, test_input])
                        results.append(doubled)
                    return results
            
            if output_grid.shape[1] == input_grid.shape[1] * 2:
                if np.array_equal(output_grid[:, :input_grid.shape[1]], input_grid):
                    results = []
                    for test_input in test_inputs:
                        test_input = np.array(test_input)
                        doubled = np.hstack([test_input, test_input])
                        results.append(doubled)
                    return results
        
        return [np.array(test_inputs[i]) for i in range(len(test_inputs))]

solver_f1 = PatternCompletionSolver()
print("✅ PatternCompletionSolver (F1) loaded")


class SynthesisSolver:
    def solve(self, task):
        train_pairs = task['train']
        test_inputs = [pair['input'] for pair in task['test']]
        
        operations = [
            ('flip_h', lambda g: np.fliplr(g)),
            ('flip_v', lambda g: np.flipud(g)),
            ('rot90', lambda g: np.rot90(g, k=1)),
            ('transpose', lambda g: g.T),
        ]
        
        for op1_name, op1_func in operations:
            for op2_name, op2_func in operations:
                valid = True
                for train_pair in train_pairs:
                    input_grid = np.array(train_pair['input'])
                    output_grid = np.array(train_pair['output'])
                    
                    try:
                        result = op2_func(op1_func(input_grid))
                        if not np.array_equal(result, output_grid):
                            valid = False
                            break
                    except:
                        valid = False
                        break
                
                if valid:
                    results = []
                    for test_input in test_inputs:
                        test_input = np.array(test_input)
                        try:
                            result = op2_func(op1_func(test_input))
                            results.append(result)
                        except:
                            results.append(test_input)
                    return results
        
        return [np.array(test_inputs[i]) for i in range(len(test_inputs))]

solver_d2 = SynthesisSolver()
print("✅ SynthesisSolver (D2) loaded")


def portfolio_ensemble_solver_v6(task):
    task_train = task['train']
    task_test = task['test']
    
    constraints = extract_constraints(task_train)
    
    all_predictions = {}
    
    try:
        preds_d2 = solver_d2.solve(task)
        all_predictions['D2'] = {'preds': preds_d2, 'weight': 0.75}
    except Exception as e:
        all_predictions['D2'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_d1 = simple_dsl_solver(task_train, task_test)
        all_predictions['D1'] = {'preds': preds_d1, 'weight': 0.77}
    except Exception as e:
        all_predictions['D1'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_c1 = connected_components_solver(task_train, task_test)
        all_predictions['C1'] = {'preds': preds_c1, 'weight': 0.74}
    except Exception as e:
        all_predictions['C1'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_e1 = solver_e1.solve(task)
        all_predictions['E1'] = {'preds': preds_e1, 'weight': 0.72}
    except Exception as e:
        all_predictions['E1'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_b1 = solver_b1.solve(task)
        all_predictions['B1'] = {'preds': preds_b1, 'weight': 0.70}
    except Exception as e:
        all_predictions['B1'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_f1 = solver_f1.solve(task)
        all_predictions['F1'] = {'preds': preds_f1, 'weight': 0.69}
    except Exception as e:
        all_predictions['F1'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_a1 = solver_a1.solve(task)
        all_predictions['A1'] = {'preds': preds_a1, 'weight': 0.68}
    except Exception as e:
        all_predictions['A1'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    try:
        preds_c4 = object_relationships_solver(task_train, task_test)
        all_predictions['C4'] = {'preds': preds_c4, 'weight': 0.64}
    except Exception as e:
        all_predictions['C4'] = {'preds': [None] * len(task_test), 'weight': 0.0}
    
    final_outputs = []
    
    for i in range(len(task_test)):
        candidates = []
        for solver_name, data in all_predictions.items():
            if data['preds'] and i < len(data['preds']) and data['preds'][i] is not None:
                candidates.append({
                    'prediction': data['preds'][i],
                    'weight': data['weight'],
                    'solver': solver_name
                })
        
        candidates.sort(key=lambda x: x['weight'], reverse=True)
        
        attempt_1 = None
        attempt_2 = None
        attempt_1_solver = None
        
        if candidates:
            test_input = task_test[i]['input']
            
            for cand in candidates:
                pred_array = np.array(cand['prediction']) if isinstance(cand['prediction'], list) else cand['prediction']
                input_array = np.array(test_input)
                
                if not np.array_equal(pred_array, input_array):
                    attempt_1 = cand['prediction']
                    attempt_1_solver = cand['solver']
                    break
            
            for cand in candidates:
                if cand['solver'] != attempt_1_solver:
                    pred_array = np.array(cand['prediction']) if isinstance(cand['prediction'], list) else cand['prediction']
                    input_array = np.array(test_input)
                    
                    if not np.array_equal(pred_array, input_array):
                        attempt_2 = cand['prediction']
                        break
            
            if attempt_2 is None and len(candidates) > 1:
                attempt_2 = candidates[1]['prediction']
            
            if attempt_2 is None:
                attempt_2 = attempt_1
        
        final_outputs.append({
            'attempt_1': attempt_1 if attempt_1 is not None else task_test[i]['input'],
            'attempt_2': attempt_2 if attempt_2 is not None else task_test[i]['input']
        })
    
    return final_outputs

print("✅ Portfolio ensemble v6 with 8 solvers loaded")


if eval_challenges and eval_solutions:
    print("\n" + "="*70)
    print("EVALUATION SET TESTING")
    print("="*70)
    
    eval_submission = {}
    for task_id, task in eval_challenges.items():
        outputs = portfolio_ensemble_solver_v6(task)
        eval_submission[task_id] = outputs
    
    total_pairs = 0
    solved_pairs = 0
    
    for task_id, task_outputs in eval_submission.items():
        if task_id in eval_solutions:
            expected = eval_solutions[task_id]
            for i, pred in enumerate(task_outputs):
                if i < len(expected):
                    attempt_1 = np.array(pred['attempt_1'])
                    attempt_2 = np.array(pred['attempt_2'])
                    expected_output = np.array(expected[i])
                    
                    if np.array_equal(attempt_1, expected_output) or np.array_equal(attempt_2, expected_output):
                        solved_pairs += 1
                    total_pairs += 1
    
    accuracy = solved_pairs / total_pairs if total_pairs > 0 else 0
    print(f"\n✅ Evaluation Accuracy: {accuracy:.1%} ({solved_pairs}/{total_pairs})")
    print(f"   Target: >35-40% (v5 baseline)")
else:
    print("⚠️  Skipping evaluation (data not available)")


submission = {}

for task_id, task in test_challenges.items():
    output_dicts = portfolio_ensemble_solver_v6(task)
    
    def to_list(obj):
        if obj is None:
            return [[0]]
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if isinstance(obj, list):
            return [[int(cell) if hasattr(cell, 'item') else cell for cell in row] for row in obj]
        return obj
    
    task_outputs = []
    for out_dict in output_dicts:
        task_outputs.append({
            'attempt_1': to_list(out_dict['attempt_1']),
            'attempt_2': to_list(out_dict['attempt_2'])
        })
    
    submission[task_id] = task_outputs

with open('/kaggle/working/submission.json', 'w') as f:
    json.dump(submission, f)

print(f"\n✅ Submission generated: {len(submission)} tasks")

output_counts = Counter(len(outputs) for outputs in submission.values())
print(f"\nOutput distribution:")
for count in sorted(output_counts.keys()):
    print(f"  {count} output(s): {output_counts[count]} tasks")

