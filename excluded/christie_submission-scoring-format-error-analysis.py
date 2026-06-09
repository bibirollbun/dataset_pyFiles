import numpy as np

import json

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


submissionA = load_json("/kaggle/input/submission-check/submissionA.json") # Succeeded
submissionB = load_json("/kaggle/input/submission-check/submissionB.json") # Failed


if len(submissionA) != len(submissionB):
    print("The lengths are not the same.")
else:
    print("The lengths are the same: ", len(submissionA))


if set(submissionA.keys()) == set(submissionB.keys()):
    print("Dictionaries have the SAME keys.")
else:
    print("Dictionaries have DIFFERENT keys.")


def check_values_integer(data):
    """
    Check if all values in nested lists within a dictionary are integers.
    Keys are ignored in the checking process.
    
    Args:
        data (dict): Dictionary with nested structure containing lists
        
    Returns:
        bool: True if all values are int, False otherwise
    """

    def is_integer(value):
        return isinstance(value, (int))
        
    def check_values_recursive(obj):
        if isinstance(obj, dict):
            # Only check values, ignore keys
            return all(check_values_recursive(value) for value in obj.values())
        elif isinstance(obj, list):
            return all(check_values_recursive(item) for item in obj)
        else:
            # This is an actual value - check if it's integer
            return is_integer(obj)
    
    return check_values_recursive(data)


# Example usage:
data = {
    '00576224': [{'attempt_1': [[9, 3], [9, 9]], 'attempt_2': [[9, 3], [9, 9]]}],
    '007bbfb7': [{'attempt_1': [[9, 9, 3], [9, 2, 3], [9, 9, 9]],
                  'attempt_2': [[9, 9, 3], [9, 2, 3], [9, 9, 9]]}]}

# Check if all values are integer
print(f"All values are integer: {check_values_integer(data)}")

# Test with mixed data (non-int values)
mixed_data = {
    'key1': [{'attempt_1': [[9, 3.5], [9, 'hello']], 'attempt_2': [[9, 3.0], [9, None]]}]
}

print(f"Mixed data - All values numeric: {check_values_integer(mixed_data)}")


# Check if all values are integers
print(f"All values for submissionA are integers: {check_values_integer(submissionA)}")
print(f"All values for submissionA are integers: {check_values_integer(submissionB)}")


def get_shape(nested_list):
    """
    Get the shape of a nested list structure.
    
    Args:
        nested_list: A nested list structure
        
    Returns:
        tuple: Shape of the nested structure
    """
    if not isinstance(nested_list, list):
        return ()
    
    if not nested_list:
        return (0,)
    
    # Get the length of the outer list
    outer_length = len(nested_list)
    
    # Check if all elements are lists and get their lengths
    inner_lengths = []
    for item in nested_list:
        if isinstance(item, list):
            inner_lengths.append(len(item))
        else:
            # If item is not a list, it's just a single element
            return (outer_length,)
    
    # Check if all inner lists have the same length
    if inner_lengths and all(length == inner_lengths[0] for length in inner_lengths):
        return (outer_length, inner_lengths[0])
    else:
        # Irregular shape - return detailed structure
        return (outer_length, tuple(inner_lengths))

def shapes_are_compatible(shape1, shape2):
    """
    Check if two shapes are compatible (identical or transposed).
    
    Args:
        shape1, shape2: Tuples representing shapes
        
    Returns:
        bool: True if shapes are identical or transposed
    """
    if shape1 == shape2:
        return True
    
    # Check if they are 2D shapes that can be transposed
    if (len(shape1) == 2 and len(shape2) == 2 and 
        isinstance(shape1[0], int) and isinstance(shape1[1], int) and
        isinstance(shape2[0], int) and isinstance(shape2[1], int)):
        # Check if one is the transpose of the other
        return shape1 == (shape2[1], shape2[0])
    
    return False

def check_attempt_shapes(data):
    """
    Check if attempt_1 and attempt_2 have compatible shapes for each key.
    Compatible means identical shapes or transposed shapes.
    
    Args:
        data (dict): Dictionary with nested structure
        
    Returns:
        dict: Results for each key showing shape comparison
    """
    results = {}
    
    for key, value in data.items():
        key_result = {
            'shapes_compatible': False,
            'shapes_identical': False,
            'shapes_transposed': False,
            'attempt_1_shape': None,
            'attempt_2_shape': None,
            'valid': False,
            'issues': []
        }
        
        # Check if value is a list
        if not isinstance(value, list):
            key_result['issues'].append('Value is not a list')
        else:
            attempt_1_data = None
            attempt_2_data = None
            
            # Find attempt_1 and attempt_2 data
            for item in value:
                if isinstance(item, dict):
                    if 'attempt_1' in item:
                        attempt_1_data = item['attempt_1']
                    if 'attempt_2' in item:
                        attempt_2_data = item['attempt_2']
                else:
                    key_result['issues'].append('List contains non-dictionary items')
            
            # Check if both attempts exist
            if attempt_1_data is None:
                key_result['issues'].append('attempt_1 not found')
            if attempt_2_data is None:
                key_result['issues'].append('attempt_2 not found')
            
            # If both exist, compare shapes
            if attempt_1_data is not None and attempt_2_data is not None:
                key_result['attempt_1_shape'] = get_shape(attempt_1_data)
                key_result['attempt_2_shape'] = get_shape(attempt_2_data)
                
                # Check if shapes are identical
                key_result['shapes_identical'] = key_result['attempt_1_shape'] == key_result['attempt_2_shape']
                
                # Check if shapes are transposed
                if not key_result['shapes_identical']:
                    shape1 = key_result['attempt_1_shape']
                    shape2 = key_result['attempt_2_shape']
                    if (len(shape1) == 2 and len(shape2) == 2 and 
                        isinstance(shape1[0], int) and isinstance(shape1[1], int) and
                        isinstance(shape2[0], int) and isinstance(shape2[1], int)):
                        key_result['shapes_transposed'] = shape1 == (shape2[1], shape2[0])
                
                # Overall compatibility
                key_result['shapes_compatible'] = key_result['shapes_identical'] or key_result['shapes_transposed']
                
                if not key_result['shapes_compatible']:
                    key_result['issues'].append(
                        f"Shape incompatible: attempt_1{key_result['attempt_1_shape']} vs attempt_2{key_result['attempt_2_shape']}"
                    )
        
        # Overall validity
        key_result['valid'] = key_result['shapes_compatible'] and len([issue for issue in key_result['issues'] if 'Shape incompatible' in issue or 'not found' in issue]) == 0
        
        results[key] = key_result
    
    return results

def all_attempts_compatible_shapes(data):
    """
    Check if ALL keys have compatible shapes between attempt_1 and attempt_2.
    Compatible means identical or transposed shapes.
    
    Returns:
        bool: True if all attempts have compatible shapes, False otherwise
    """
    for key, value in data.items():
        if not isinstance(value, list):
            return False
        
        attempt_1_data = None
        attempt_2_data = None
        
        for item in value:
            if isinstance(item, dict):
                if 'attempt_1' in item:
                    attempt_1_data = item['attempt_1']
                if 'attempt_2' in item:
                    attempt_2_data = item['attempt_2']
        
        if attempt_1_data is None or attempt_2_data is None:
            return False
        
        if not shapes_are_compatible(get_shape(attempt_1_data), get_shape(attempt_2_data)):
            return False
    
    return True



print("\nTesting mismatched shapes:")
mismatched_results = check_attempt_shapes(submissionA)
for key, info in mismatched_results.items():
    status = "✓" if info['valid'] else "✗"

    if status == "✗":
        print(f"Key '{key}': {status}")
        print(f"  attempt_1 shape: {info['attempt_1_shape']}")
        print(f"  attempt_2 shape: {info['attempt_2_shape']}")
        if info['issues']:
            print(f"  Issues: {', '.join(info['issues'])}")
        print()


print("\nTesting mismatched shapes:")
mismatched_results = check_attempt_shapes(submissionB)
for key, info in mismatched_results.items():
    status = "✓" if info['valid'] else "✗"

    if status == "✗":
        print(f"Key '{key}': {status}")
        print(f"  attempt_1 shape: {info['attempt_1_shape']}")
        print(f"  attempt_2 shape: {info['attempt_2_shape']}")
        if info['issues']:
            print(f"  Issues: {', '.join(info['issues'])}")
        print()


def check_attempt_structure(data):
    """
    Check if each key has exactly one attempt_1 and one attempt_2.
    
    Args:
        data (dict): Dictionary with nested structure
        
    Returns:
        dict: Results for each key showing validation status
    """
    results = {}
    
    for key, value in data.items():
        # Initialize result for this key
        key_result = {
            'valid': True,
            'has_attempt_1': False,
            'has_attempt_2': False,
            'attempt_1_count': 0,
            'attempt_2_count': 0,
            'issues': []
        }
        
        # Check if value is a list
        if not isinstance(value, list):
            key_result['valid'] = False
            key_result['issues'].append('Value is not a list')
        else:
            # Count attempts across all dictionaries in the list
            for item in value:
                if isinstance(item, dict):
                    if 'attempt_1' in item:
                        key_result['attempt_1_count'] += 1
                    if 'attempt_2' in item:
                        key_result['attempt_2_count'] += 1
                else:
                    key_result['issues'].append('List contains non-dictionary items')
        
        # Check if we have exactly one of each
        key_result['has_attempt_1'] = key_result['attempt_1_count'] == 1
        key_result['has_attempt_2'] = key_result['attempt_2_count'] == 1
        
        # Add specific issues
        if key_result['attempt_1_count'] == 0:
            key_result['issues'].append('Missing attempt_1')
        elif key_result['attempt_1_count'] > 1:
            key_result['issues'].append(f'Multiple attempt_1 found ({key_result["attempt_1_count"]})')
            
        if key_result['attempt_2_count'] == 0:
            key_result['issues'].append('Missing attempt_2')
        elif key_result['attempt_2_count'] > 1:
            key_result['issues'].append(f'Multiple attempt_2 found ({key_result["attempt_2_count"]})')
        
        # Overall validity
        key_result['valid'] = (key_result['has_attempt_1'] and 
                              key_result['has_attempt_2'] and 
                              len(key_result['issues']) == 0)
        
        results[key] = key_result
    
    return results

# Simpler version that just returns True/False
def has_valid_attempt_structure(data):
    """
    Check if ALL keys have exactly one attempt_1 and one attempt_2.
    
    Returns:
        bool: True if all keys have valid structure, False otherwise
    """
    for key, value in data.items():
        if not isinstance(value, list):
            return False
        
        attempt_1_count = 0
        attempt_2_count = 0
        
        for item in value:
            if isinstance(item, dict):
                if 'attempt_1' in item:
                    attempt_1_count += 1
                if 'attempt_2' in item:
                    attempt_2_count += 1
            else:
                return False
        
        if attempt_1_count != 1 or attempt_2_count != 1:
            return False
    
    return True

# Example usage:
# data = {
#     '00576224': [{'attempt_1': [[9, 3], [9, 9]], 'attempt_2': [[9, 3], [9, 9]]}],
#     '007bbfb7': [{'attempt_1': [[9, 9, 3], [9, 2, 3], [9, 9, 9]],
#                   'attempt_2': [[9, 9, 3], [9, 2, 3], [9, 9, 9]]}]}

# # Detailed check
# results = check_attempt_structure(data)
# for key, info in results.items():
#     if info['valid']:
#         print(f"Key '{key}': Valid structure ✓")
#     else:
#         print(f"Key '{key}': Invalid - {', '.join(info['issues'])}")

# # Simple boolean check
# print(f"\nAll keys have valid attempt structure: {has_valid_attempt_structure(data)}")

# # Test with invalid data
# invalid_data = {
#     'key1': [{'attempt_1': [[9, 3]], 'attempt_2': [[9, 9]]}],  # Valid
#     'key2': [{'attempt_1': [[1, 2]]}],  # Missing attempt_2
#     'key3': [{'attempt_1': [[1, 2]], 'attempt_2': [[3, 4]]}, 
#              {'attempt_1': [[5, 6]]}]  # Duplicate attempt_1
# }

# print("\nTesting invalid data:")
# invalid_results = check_attempt_structure(invalid_data)
# for key, info in invalid_results.items():
#     status = "✓" if info['valid'] else "✗"
#     print(f"Key '{key}': {status} - attempt_1: {info['attempt_1_count']}, attempt_2: {info['attempt_2_count']}")
#     if info['issues']:
#         print(f"  Issues: {', '.join(info['issues'])}")


# Detailed check
# results = check_attempt_structure(submissionA)
# for key, info in results.items():
#     if info['valid']:
#         print(f"Key '{key}': Valid structure ✓")
#     else:
#         print(f"Key '{key}': Invalid - {', '.join(info['issues'])}")


# Detailed check
# results = check_attempt_structure(submissionB)
# for key, info in results.items():
#     if info['valid']:
#         print(f"Key '{key}': Valid structure ✓")
#     else:
#         print(f"Key '{key}': Invalid - {', '.join(info['issues'])}")


import matplotlib.pyplot as plt
from   matplotlib import colors
import seaborn as sns


base_path='/kaggle/input/arc-prize-2025/'

test_challenges   = load_json(base_path +'arc-agi_test_challenges.json')
    
# 0:black, 1:blue, 2:red, 3:green, 4:yellow, # 5:gray, 6:magenta, 7:orange, 8:sky, 9:brown
cmap = colors.ListedColormap(
    ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
     '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
norm = colors.Normalize(vmin=0, vmax=9)

plt.figure(figsize=(3, 1), dpi=150)
plt.imshow([list(range(10))], cmap=cmap, norm=norm)
plt.xticks(list(range(10)))
plt.yticks([])
plt.tick_params(axis='x', color='r', length=0, grid_color='none')
    
plt.show()


def plot_task(task, task_solutions, i, t, size=3, w1=0.5):
    """
    Plots the task and solutions
    """
    # Get dimensions
    n_train = len(task['train'])
    n_test = len(task['test'])
    
    # Create figure and axes
    # Layout: [train inputs] [test inputs] [attempt_1 outputs] [attempt_2 outputs]
    fig, axs = plt.subplots(2, n_train + n_test * 2, figsize=(size * (n_train + n_test * 2), size * 2))
        
    # If only one column, make it 2D array
    if n_train + n_test * 2 == 1:
        axs = axs.reshape(2, 1)
    
    # Plot training examples
    for j in range(n_train):
        plot_one(axs[0, j], j, task, 'train', 'input', w=w1)
        plot_one(axs[1, j], j, task, 'train', 'output', w=w1)
    
    # Plot test examples
    for k in range(n_test):
        # Extract the solution grid
        solution_dict = task_solutions[k]  # This is {'attempt_1': grid, 'attempt_2': grid}

        #attempt_1
        plot_one(axs[0, n_train + k], k, task, 'test', 'input', w=w1)
        solution_grid_1 = solution_dict['attempt_1']  # Extract the actual grid
        
        # Create temporary task structure for plotting
        temp_task = {'test': [{'output': solution_grid_1}]}
        plot_one(axs[1, n_train + k], k, temp_task, 'test', 'output', w=w1)
        axs[1, n_train + k].set_title(f'test output {k} - attempt 1')
    
        # attempt_2
        plot_one(axs[0, n_train + n_test + k], k, task, 'test', 'input', w=w1)
        solution_grid_2 = solution_dict['attempt_2']  # Extract the actual grid for attemp_2
        
        # Set the output for plotting
        temp_task = {'test': [{'output': solution_grid_2}]}
        plot_one(axs[1, n_train + n_test + k], k, temp_task, 'test', 'output', w=w1)
        axs[1, n_train + n_test + k].set_title(f'test output {k} - attempt 2')
    
    plt.tight_layout()
    plt.show()

def plot_one(ax, i, task, train_or_test, input_or_output, w=0.5):
    """
    Plot a single grid
    """
    
    input_matrix = task[train_or_test][i][input_or_output]
    
    # Convert to numpy array to ensure proper dtype
    input_matrix = np.array(input_matrix, dtype=int)
    
    ax.imshow(input_matrix, cmap=cmap, norm=norm)
    ax.grid(True, which='both', color='lightgrey', linewidth=w)
    ax.set_xticks(np.arange(-0.5, len(input_matrix[0]), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(input_matrix), 1), minor=True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(f'{train_or_test} {input_or_output} {i}')

for i, t in enumerate(list(test_challenges.keys())[:30]):
    task = test_challenges[t]
    task_solution = submissionB[t]  # This gets the list for this specific task
    plot_task(task, task_solution, i, t)

