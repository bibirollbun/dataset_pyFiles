"""
ARC-AGI Grid Editor - Python Implementation

This module provides functions to:
1. Convert action dictionaries to/from vectors
2. Perform grid editing operations (copy, rotate, fill, etc.)
3. Visualize grids with color mapping

Based on the TypeScript implementation in the ARC-AGI puzzle solver project.
"""
import random
from matplotlib.colors import ListedColormap
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from typing import Dict, List, Tuple, Optional, Union
from copy import deepcopy
from PIL import Image, ImageDraw
import json

# ============================================================================
# Constants and Color Mapping
# ============================================================================

COLOR_MAP = {
    -1: "#ffffff",  # White
    0: "#000000",   # Black
    1: "#1E93FF",   # Blue
    2: "#F93C31",   # Red
    3: "#4FCC30",   # Green
    4: "#FFDC00",   # Yellow
    5: "#999999",   # Gray
    6: "#E53AA3",   # Magenta
    7: "#FF851B",   # Orange
    8: "#87D8F1",   # Light Blue
    9: "#921231",   # Dark Red
}

color_map = {
    -1: "#ffffff",
    0: "#000000",
    1: "#1E93FF",
    2: "#F93C31",
    3: "#4FCC30",
    4: "#FFDC00",
    5: "#999999",
    6: "#E53AA3",
    7: "#FF851B",
    8: "#87D8F1",
    9: "#921231",
}

def create_custom_colormap():
    """Create a custom colormap from the color_map dictionary."""
    colors = [color_map[i] for i in range(-1, 10)]
    return ListedColormap(colors)

# Action types (9 possible actions, 'default' represented as all zeros)
ACTION_TYPES = [
    'resize', 'rotate', 'flip', 'project', 'copy', 
    'match', 're-scale', 'fill', 'fill-boundary'
]

# Direction types (10 possible directions, 'default' represented as all zeros)
DIRECTION_TYPES = [
    'up', 'down', 'left', 'right', 'north', 'south', 
    'east', 'west', 'horizontal', 'vertical'
]

# Source types (2 possible sources - based on TypeScript Position type)
SOURCE_TYPES = ['output', 'input']


# ============================================================================
# Action Vector Conversion Functions
# ============================================================================

def action_to_vector(action: Dict) -> np.ndarray:
    """
    Convert an action dictionary to a numpy vector representation.
    
    Args:
        action: Dictionary with keys:
            - action (str): Action type
            - position (dict): {x, y, source}
            - fromPosition (dict): {x, y, source}
            - size (dict): {width, height}
            - color (int): DIGIT value (-1 to 9)
            - targetColor (int): DIGIT value (-1 to 9)
            - isFillAll (bool)
            - scaleFactor (int) (1, 2 and 3)
            - direction (str): Direction type
    
    Returns:
        np.ndarray: Vector of length 31 representing the action
            [0-8]: action_type one-hot (9 values, all zeros = 'default')
            [9-10]: position (x, y)
            [11]: position source (0=output, 1=input)
            [12-13]: fromPosition (x, y)
            [14]: fromPosition source (0=output, 1=input)
            [15-16]: size (width, height)
            [17]: color (-1 to 9, stored as is)
            [18]: targetColor (-1 to 9, stored as is)
            [19]: isFillAll (0 or 1)
            [20]: scaleFactor (1, 2 and 3)
            [21-30]: direction one-hot (10 values, all zeros = 'default')
    """
    vector = np.zeros(31, dtype=np.float32)
    
    # Action type (one-hot encoding, 'default' = all zeros)
    action_type = action.get('action', 'default')
    if action_type != 'default' and action_type in ACTION_TYPES:
        vector[ACTION_TYPES.index(action_type)] = 1.0
    
    # Action type (one-hot encoding, 'default' = all zeros)
    action_type = action.get('action', 'default')
    if action_type != 'default' and action_type in ACTION_TYPES:
        vector[ACTION_TYPES.index(action_type)] = 1.0
    
    # Position
    pos = action.get('position', {'x': 0, 'y': 0, 'source': 'output'})
    vector[9] = pos.get('x', 0)
    vector[10] = pos.get('y', 0)
    vector[11] = SOURCE_TYPES.index(pos.get('source', 'output'))
    
    # From Position
    from_pos = action.get('fromPosition', {'x': 0, 'y': 0, 'source': 'output'})
    vector[12] = from_pos.get('x', 0)
    vector[13] = from_pos.get('y', 0)
    vector[14] = SOURCE_TYPES.index(from_pos.get('source', 'output'))
    
    # Size
    size = action.get('size', {'width': 1, 'height': 1})
    vector[15] = size.get('width', 1)
    vector[16] = size.get('height', 1)
    
    # Color values
    vector[17] = action.get('color', -1)
    vector[18] = action.get('targetColor', -1)
    
    # Fill all flag
    vector[19] = 1.0 if action.get('isFillAll', False) else 0.0
    
    # Scale factor
    vector[20] = action.get('scaleFactor', 1)
    
    # Direction (one-hot encoding, 'default' = all zeros)
    direction = action.get('direction', 'default')
    if direction != 'default' and direction in DIRECTION_TYPES:
        vector[21 + DIRECTION_TYPES.index(direction)] = 1.0
    
    return vector


def vector_to_action(vector: np.ndarray) -> Dict:
    """
    Convert a numpy vector back to an action dictionary.
    
    Args:
        vector: np.ndarray of length 31 representing the action
    
    Returns:
        Dictionary with action parameters
    """
    action = {}
    
    # Action type (from one-hot, all zeros = 'default')
    if np.any(vector[0:9]):
        action_idx = np.argmax(vector[0:9])
        action['action'] = ACTION_TYPES[action_idx]
    else:
        action['action'] = 'default'
    
    # Position
    action['position'] = {
        'x': int(vector[9]),
        'y': int(vector[10]),
        'source': SOURCE_TYPES[int(vector[11])]
    }
    
    # From Position
    action['fromPosition'] = {
        'x': int(vector[12]),
        'y': int(vector[13]),
        'source': SOURCE_TYPES[int(vector[14])]
    }
    
    # Size
    action['size'] = {
        'width': int(vector[15]),
        'height': int(vector[16])
    }
    
    # Color values
    action['color'] = int(vector[17])
    action['targetColor'] = int(vector[18])
    
    # Fill all flag
    action['isFillAll'] = bool(vector[19] > 0.5)
    
    # Scale factor
    action['scaleFactor'] = int(vector[20])
    
    # Direction (from one-hot, all zeros = 'default')
    if np.any(vector[21:31]):
        direction_idx = np.argmax(vector[21:31])
        action['direction'] = DIRECTION_TYPES[direction_idx]
    else:
        action['direction'] = 'default'
    
    return action

def parse_action_tensor_to_dict(action_tensor, input_rows: int = 30, input_cols: int = 30, output_rows: int = 30, output_cols: int = 30) -> Dict:
    '''
    Convert a PyTorch tensor representing an action vector into an action dictionary.
    [0-8]: action_type one-hot (9 values, all zeros = 'default')
    [9-10]: position (x, y)
    [11]: position source (0=output, 1=input)
    [12-13]: fromPosition (x, y)
    [14]: fromPosition source (0=output, 1=input)
    [15-16]: size (width, height)
    [17]: color (-1 to 9, stored as is)
    [18]: targetColor (-1 to 9, stored as is)
    [19]: isFillAll (0 or 1)
    [20]: scaleFactor (1, 2 and 3)
    [21-30]: direction one-hot (10 values, all zeros = 'default')
    '''
    action_vector_raw = action_tensor.numpy()
    # make sure the values are within valid ranges
    action_type_idx = np.argmax(action_vector_raw[0:9]) if np.any(action_vector_raw[0:9]) else -1
    action_type = ACTION_TYPES[action_type_idx] if action_type_idx != -1 else 'default'
    position = {
        'x': int(action_vector_raw[9]) if 0 <= int(action_vector_raw[9]) < output_cols else 0,
        'y': int(action_vector_raw[10]) if 0 <= int(action_vector_raw[10]) < output_rows else 0,
        'source': 'output'
    }
    from_position = {
        'x': int(action_vector_raw[12]) if (0 <= int(action_vector_raw[12]) < output_cols and SOURCE_TYPES[int(action_vector_raw[14])] == 'output') or (0 <= int(action_vector_raw[12]) < input_cols and SOURCE_TYPES[int(action_vector_raw[14])] == 'input') else 0,
        'y': int(action_vector_raw[13]) if (0 <= int(action_vector_raw[13]) < output_rows and SOURCE_TYPES[int(action_vector_raw[14])] == 'output') or (0 <= int(action_vector_raw[13]) < input_rows and SOURCE_TYPES[int(action_vector_raw[14])] == 'input') else 0,
        'source': SOURCE_TYPES[int(action_vector_raw[14]) if int(action_vector_raw[14]) in [0,1] else 0]
    }
    max_width = output_cols - position['x']
    max_height = output_rows - position['y']
    if from_position['source'] == 'input':
        max_width = min(max_width, input_cols - from_position['x'])
        max_height = min(max_height, input_rows - from_position['y'])
    direction = DIRECTION_TYPES[np.argmax(action_vector_raw[21:31])] if np.any(action_vector_raw[21:31]) else 'default'        
    min_width = 1
    min_height = 1
    if action_type == 'resize':
        max_width = 30
        max_height = 30
    if action_type == 'project':
        if direction in ['up', 'down']:
            min_height = 2
        elif direction in ['left', 'right']:
            min_width = 2
        elif direction in ['north', 'west', 'east', 'south']:
            if int(action_vector_raw[15]) < int(action_vector_raw[16]):
                max_width = 1
            else:
                max_height = 1
    if action_type == 'rotate':
        min_width = min_height = 2
    if min_width <= int(action_vector_raw[15]) <= max_width:
        width = int(action_vector_raw[15])
    else:
        if abs(int(action_vector_raw[15]) - min_width) < abs(int(action_vector_raw[15]) - max_width):
            width = min_width
        else:
            width = max_width
    if min_height <= int(action_vector_raw[16]) <= max_height:
        height = int(action_vector_raw[16])
    else:
        if abs(int(action_vector_raw[16]) - min_height) < abs(int(action_vector_raw[16]) - max_height):
            height = min_height
        else:
            height = max_height
    if action_type == 'rotate':
        width = height
    size = {
        'width': width,
        'height': height
    }
        
    color = int(action_vector_raw[17]) if -1 <= int(action_vector_raw[17]) <= 9 else -1
    target_color = int(action_vector_raw[18]) if -1 <= int(action_vector_raw[18]) <= 9 else -1
    is_fill_all = bool(action_vector_raw[19] > 0.5)
    scale_factor = int(action_vector_raw[20]) if 1 <= int(action_vector_raw[20]) <= 3 else 1

    return {
        'action': action_type,
        'position': position,
        'fromPosition': from_position,
        'size': size,
        'color': color,
        'targetColor': target_color,
        'isFillAll': is_fill_all,
        'scaleFactor': scale_factor,
        'direction': direction
    }


def generate_random_action(rows: int = 30, cols: int = 30, action: str = None,
                          input_rows: int = 30, input_cols: int = 30) -> Dict:
    """
    Generate a random valid action vector with reasonable attributes for each action type.
    
    Args:
        rows: Number of rows in the grid
        cols: Number of columns in the grid
        input_rows: Number of rows in the input grid
        input_cols: Number of columns in the input grid
    
    Returns:
        np.ndarray: Random action vector of length 31
    """
    # Randomly choose an action type (or 'default')
    if action:
        action_type = action
    else:
        action_type = np.random.choice(ACTION_TYPES)
    
    # Initialize action dictionary with defaults
    action = {
        'action': action_type,
        'position': {'x': 0, 'y': 0, 'source': 'output'},
        'fromPosition': {'x': 0, 'y': 0, 'source': 'output'},
        'size': {'width': 1, 'height': 1},
        'color': -1,
        'targetColor': -1,
        'isFillAll': False,
        'scaleFactor': 1,
        'direction': 'default'
    }
    
    # Generate random position and size
    x = np.random.randint(0, cols)
    y = np.random.randint(0, rows)
    width = np.random.randint(1, cols - x + 1)
    height = np.random.randint(1, rows - y + 1)
    
    # Set reasonable attributes based on action type
    if action_type == 'resize':
        # Resize needs size, optionally new position
        action['size'] = {
            'width': np.random.randint(1, 30),
            'height': np.random.randint(1, 30)
        }
    
    elif action_type == 'rotate':
        x = np.random.randint(0, cols - 1)
        y = np.random.randint(0, rows - 1)
        # Rotate needs position and square size
        size = np.random.randint(2, min(cols - x + 1, rows - y + 1))
        action['size'] = {'width': size, 'height': size}
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        action['fromPosition'] = action['position'].copy()
    
    elif action_type == 'flip':
        # Flip needs position, size, and direction (horizontal/vertical)
        action['direction'] = np.random.choice(['horizontal', 'vertical'])
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        action['size'] = {'width': width, 'height': height}
        action['fromPosition'] = action['position'].copy()
    
    elif action_type == 'project':
        # Project needs position, size, and direction
        # Can be rect (up/down/left/right) or line (north/south/east/west)
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        action['isFillAll'] = np.random.choice([True, False])
        if np.random.rand() < 0.5:
            # Rectangle projection
            action['direction'] = np.random.choice(['up', 'down', 'left', 'right'])
            if action['direction'] in ['up', 'down']:
                y = np.random.randint(0, rows - 1)
                height = np.random.randint(2, rows - x + 1)
            else:
                x = np.random.randint(0, cols - 1)
                width = np.random.randint(2, cols - x + 1)
        else:
            # Line projection (make it a line: width=1 or height=1)
            action['direction'] = np.random.choice(['north', 'south', 'east', 'west'])
            if np.random.rand() < 0.5:
                width = 1
            else:
                height = 1
        action['size'] = {'width': width, 'height': height}
        action['fromPosition'] = action['position'].copy()
    
    elif action_type == 'copy' or action_type == 'match':
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        # Copy/match can copy from input to output or within output
        if input_rows and input_cols and np.random.rand() < 0.5:
            # Copy from input to output
            from_x = np.random.randint(0, input_cols)
            from_y = np.random.randint(0, input_rows)
            from_width = np.random.randint(1, input_cols - from_x + 1)
            from_height = np.random.randint(1, input_rows - from_y + 1)
            width = min(width, from_width)
            height = min(height, from_height)
            action['fromPosition'] = {'x': from_x, 'y': from_y, 'source': 'input'}
        else:
            # Copy within output
            from_x = np.random.randint(0, cols)
            from_y = np.random.randint(0, rows)
            from_width = np.random.randint(1, cols - from_x + 1)
            from_height = np.random.randint(1, rows - from_y + 1)
            width = min(width, from_width)
            height = min(height, from_height)
            action['fromPosition'] = {'x': from_x, 'y': from_y, 'source': 'output'}
        action['size'] = {'width': width, 'height': height}
    
    elif action_type == 're-scale':
        # Re-scale needs position, size, and scale factor
        action['scaleFactor'] = np.random.choice([1, 2, 3])
        if np.random.rand() < 0.7:
            # The higher the scale factor, the smaller the size and the further top-left the position
            max_width = max(1, cols // action['scaleFactor'])
            max_height = max(1, rows // action['scaleFactor'])
            x = np.random.randint(0, cols - max_width + 1)
            y = np.random.randint(0, rows - max_height + 1)
            width = np.random.randint(1, max_width + 1)
            height = np.random.randint(1, max_height + 1)
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        action['size'] = {'width': width, 'height': height}
        action['fromPosition'] = action['position'].copy()
    
    elif action_type == 'fill':
        # Fill needs position and color
        action['color'] = np.random.randint(-1, 10)
        action['targetColor'] = np.random.randint(-1, 10)
        action['isFillAll'] = np.random.choice([True, False])
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        action['size'] = {'width': width, 'height': height}
        if np.random.rand() < 0.4:
            action['size'] = {'width': 1, 'height': 1}
        action['fromPosition'] = action['position'].copy()
    
    elif action_type == 'fill-boundary':
        # Fill-boundary needs position, color, and target color
        action['color'] = np.random.randint(-1, 10)
        action['targetColor'] = np.random.randint(-1, 10)
        while action['targetColor'] == action['color']:
            action['targetColor'] = np.random.randint(-1, 10)
        action['position'] = {'x': x, 'y': y, 'source': 'output'}
        action['size'] = {'width': 1, 'height': 1}
        action['fromPosition'] = action['position'].copy()
    
    elif action_type == 'default':
        # Default action - all values stay at defaults
        pass
    
    # Convert to vector
    return action

def generate_random_action_vector(rows: int = 30, cols: int = 30, action: str = None, input_rows: int = 30, input_cols: int = 30) -> np.ndarray:
    try:
        action = generate_random_action(rows, cols, action, input_rows, input_cols)
        return action_to_vector(action)
    except Exception as e:
        return None

# ============================================================================
# Grid Editing Functions
# ============================================================================

def boundary_fill(matrix: np.ndarray, x: int, y: int, 
                  target_color: int, replacement_color: int) -> None:
    """
    Perform boundary fill (flood fill) on a matrix in-place.
    
    Args:
        matrix: 2D numpy array to modify
        x: Starting x coordinate (column)
        y: Starting y coordinate (row)
        target_color: Color to replace
        replacement_color: Color to fill with
    """
    if target_color == replacement_color:
        return
    
    rows, cols = matrix.shape
    stack = [(x, y)]
    
    while stack:
        curr_x, curr_y = stack.pop()
        
        if curr_x < 0 or curr_x >= cols or curr_y < 0 or curr_y >= rows:
            continue
        if matrix[curr_y, curr_x] != target_color:
            continue
        
        matrix[curr_y, curr_x] = replacement_color
        
        stack.extend([
            (curr_x + 1, curr_y),
            (curr_x - 1, curr_y),
            (curr_x, curr_y + 1),
            (curr_x, curr_y - 1)
        ])


def project_rect(rect: np.ndarray, direction: str, isFillAll: bool) -> np.ndarray:
    """
    Project patterns in a rectangle in the specified direction.
    
    Args:
        rect: 2D numpy array
        direction: One of 'up', 'down', 'left', 'right'
    
    Returns:
        Modified rectangle
    """
    rows, cols = rect.shape
    
    def project(x: int, y: int, target_color: int, replacement_color: int):
        if x < 0 or x >= cols or y < 0 or y >= rows:
            return
        if rect[y, x] != target_color and not isFillAll:
            return
        rect[y, x] = replacement_color
        
        if direction == 'up':
            project(x, y - 1, target_color, replacement_color)
        elif direction == 'down':
            project(x, y + 1, target_color, replacement_color)
        elif direction == 'left':
            project(x - 1, y, target_color, replacement_color)
        elif direction == 'right':
            project(x + 1, y, target_color, replacement_color)
    
    if direction == 'up':
        for j in range(cols):
            project(j, rows - 2, rect[rows - 2, j], rect[rows - 1, j])
    elif direction == 'down':
        for j in range(cols):
            project(j, 1, rect[1, j], rect[0, j])
    elif direction == 'left':
        for i in range(rows):
            project(cols - 2, i, rect[i, cols - 2], rect[i, cols - 1])
    elif direction == 'right':
        for i in range(rows):
            project(1, i, rect[i, 1], rect[i, 0])
    
    return rect


def project_line(matrix: np.ndarray, position: Tuple[int, int], 
                 size: Tuple[int, int], direction: str, isFillAll: bool) -> None:
    """
    Project a line diagonally in the specified direction.
    
    Args:
        matrix: 2D numpy array to modify in-place
        position: (x, y) starting position
        size: (width, height) of the line
        direction: One of 'north', 'south', 'east', 'west'
    """
    rows, cols = matrix.shape
    x, y = position
    width, height = size
    
    def project(px: int, py: int, direction: str, 
                target_color: int, replacement_color: int):
        if px < 0 or px >= cols or py < 0 or py >= rows:
            return
        if matrix[py, px] != target_color and not isFillAll:
            return
        matrix[py, px] = replacement_color
        
        if direction == 'north':
            project(px + 1, py - 1, direction, target_color, replacement_color)
        elif direction == 'south':
            project(px - 1, py + 1, direction, target_color, replacement_color)
        elif direction == 'east':
            project(px + 1, py + 1, direction, target_color, replacement_color)
        elif direction == 'west':
            project(px - 1, py - 1, direction, target_color, replacement_color)
    
    if direction == 'north':
        for i in range(height):
            for j in range(width):
                nx, ny = x + j + 1, y + i - 1
                if 0 <= nx < cols and 0 <= ny < rows and ny + 1 < rows and nx - 1 >= 0:
                    project(nx, ny, direction, matrix[ny, nx], matrix[y + i, x + j])
    elif direction == 'south':
        for i in range(height):
            for j in range(width):
                nx, ny = x + j - 1, y + i + 1
                if 0 <= nx < cols and 0 <= ny < rows and ny - 1 >= 0 and nx + 1 < cols:
                    project(nx, ny, direction, matrix[ny, nx], matrix[y + i, x + j])
    elif direction == 'east':
        for i in range(height):
            for j in range(width):
                nx, ny = x + j + 1, y + i + 1
                if 0 <= nx < cols and 0 <= ny < rows and ny - 1 >= 0 and nx - 1 >= 0:
                    project(nx, ny, direction, matrix[ny, nx], matrix[y + i, x + j])
    elif direction == 'west':
        for i in range(height):
            for j in range(width):
                nx, ny = x + j - 1, y + i - 1
                if 0 <= nx < cols and 0 <= ny < rows and ny + 1 < rows and nx + 1 < cols:
                    project(nx, ny, direction, matrix[ny, nx], matrix[y + i, x + j])


def edit_grid(input_matrix: np.ndarray, output_matrix: np.ndarray, 
              action_vector: np.ndarray) -> np.ndarray:
    """
    Apply an edit operation to the output matrix based on the action vector.
    
    Note: This function only handles interactions between input and output matrices,
    not between example/test inputs and output (simplified compared to TS version).
    
    Args:
        input_matrix: Input grid (numpy array)
        output_matrix: Current output grid (numpy array)
        action_vector: Action encoded as vector
    
    Returns:
        New output matrix after applying the action
    """
    action = vector_to_action(action_vector)
    new_output = output_matrix.copy()
    
    action_type = action['action']
    pos = action['position']
    from_pos = action['fromPosition']
    size = action['size']
    color = action['color']
    target_color = action['targetColor']
    isFillAll = action['isFillAll']
    direction = action['direction']
    scale_factor = action['scaleFactor']
    
    x, y = pos['x'], pos['y']
    from_x, from_y = from_pos['x'], from_pos['y']
    width, height = size['width'], size['height']
    rows, cols = new_output.shape
    
    # RESIZE
    if action_type == 'resize':
        new_output = np.zeros((height, width), dtype=np.int32)
        min_rows = min(output_matrix.shape[0], height)
        min_cols = min(output_matrix.shape[1], width)
        new_output[:min_rows, :min_cols] = output_matrix[:min_rows, :min_cols]
        return new_output
    
    # FILL (an area fill)
    elif action_type == 'fill':
        for i in range(height):
            for j in range(width):
                if 0 <= y + i < rows and 0 <= x + j < cols:
                    if isFillAll or new_output[y + i, x + j] == target_color:
                        new_output[y + i, x + j] = color
        return new_output
    
    # FILL-BOUNDARY (flood fill)
    elif action_type == 'fill-boundary':
        if 0 <= y < rows and 0 <= x < cols:
            boundary_fill(new_output, x, y, target_color, color)
    
    # COPY
    elif action_type == 'copy':
        # Determine source matrix based on from_pos source
        if from_pos['source'] == 'input' or from_pos['source'] == 'test':
            source_matrix = input_matrix
        else:
            source_matrix = output_matrix
        
        # Ensure source_matrix is a 2D numpy array
        if hasattr(source_matrix, 'cpu'):  # Check if it's a torch tensor
            source_matrix = source_matrix.cpu().detach().numpy()
        if hasattr(source_matrix, 'squeeze'):  # Squeeze to 2D if needed
            while source_matrix.ndim > 2:
                source_matrix = source_matrix.squeeze(0)
        
        # Copy region from source to destination
        src_rows, src_cols = source_matrix.shape
        min_delta_rows = min(height, rows - y, src_rows - from_y)
        min_delta_cols = min(width, cols - x, src_cols - from_x)
        
        for i in range(min_delta_rows):
            for j in range(min_delta_cols):
                new_output[y + i, x + j] = source_matrix[from_y + i, from_x + j]
    
    # MATCH (compare and mark differences)
    elif action_type == 'match':
        # Similar to copy but marks differences with -1
        if from_pos['source'] == 'input' or from_pos['source'] == 'test':
            source_matrix = input_matrix
        else:
            source_matrix = output_matrix
        
        src_rows, src_cols = source_matrix.shape
        min_delta_rows = min(height, rows - y, src_rows - from_y)
        min_delta_cols = min(width, cols - x, src_cols - from_x)
        
        for i in range(min_delta_rows):
            for j in range(min_delta_cols):
                if new_output[y + i, x + j] == source_matrix[from_y + i, from_x + j]:
                    new_output[y + i, x + j] = source_matrix[from_y + i, from_x + j]
                else:
                    new_output[y + i, x + j] = -1
    
    # ROTATE (90 degrees clockwise, square only)
    elif action_type == 'rotate':
        if width == height and y + height <= rows and x + width <= cols:
            temp = new_output[y:y+height, x:x+width].copy()
            # Rotate 90 degrees clockwise: new[i,j] = old[n-1-j, i]
            for i in range(height):
                for j in range(width):
                    new_output[y + i, x + j] = temp[height - 1 - j, i]
    
    # FLIP
    elif action_type == 'flip':
        if y + height <= rows and x + width <= cols:
            temp = new_output[y:y+height, x:x+width].copy()
            if direction == 'horizontal':
                new_output[y:y+height, x:x+width] = np.fliplr(temp)
            elif direction == 'vertical':
                new_output[y:y+height, x:x+width] = np.flipud(temp)
    
    # PROJECT (rectangle)
    elif action_type == 'project' and direction in ['up', 'down', 'left', 'right']:
        if y + height <= rows and x + width <= cols:
            temp = new_output[y:y+height, x:x+width].copy()
            project_rect(temp, direction, isFillAll)
            new_output[y:y+height, x:x+width] = temp
    
    # PROJECT (line, diagonal)
    elif action_type == 'project' and direction in ['north', 'south', 'east', 'west']:
        if y + height <= rows and x + width <= cols and (width == 1 or height == 1):
            project_line(new_output, (x, y), (width, height), direction, isFillAll)
    
    # RE-SCALE
    elif action_type == 're-scale':
        new_rows = height * scale_factor
        new_cols = width * scale_factor
        min_delta_rows = min(new_rows, rows - y)
        min_delta_cols = min(new_cols, cols - x)
        
        for i in range(min_delta_rows):
            for j in range(min_delta_cols):
                new_output[y + i, x + j] = output_matrix[y + i // scale_factor, x + j // scale_factor]
    
    return new_output

def matrix_to_color_array(matrix: np.ndarray) -> np.ndarray:
    """
    Convert a matrix to a color array using the ARC color map.
    
    Args:
        matrix: 2D numpy array with values from -1 to 9
    
    Returns:
        3D numpy array (rows, cols, 3) with RGB values
    """
    rows, cols = matrix.shape
    color_array = np.zeros((rows, cols, 3))
    
    for i in range(rows):
        for j in range(cols):
            digit = matrix[i, j]
            if digit in COLOR_MAP:
                hex_color = COLOR_MAP[digit]
                # Convert hex to RGB
                rgb = tuple(int(hex_color[k:k+2], 16) / 255.0 for k in (1, 3, 5))
                color_array[i, j] = rgb
    
    return color_array

def matrix_to_image(matrix: np.ndarray) -> 'Image':
    """
    Convert a matrix to a PIL Image using the ARC color map.
    """
    color_array = matrix_to_color_array(matrix)
    return Image.fromarray(color_array, 'RGB')

# ============================================================================
# Visualization Function
# ============================================================================

def visualize_matrix(matrix: np.ndarray, title: str = "Grid Visualization", is_input: bool = False, 
                     figsize: Tuple[int, int] = (8, 8), pad_to_30: bool = True,
                     action: Optional[Dict] = None, ax: Optional[plt.Axes] = None) -> plt.Axes:
    """
    Visualize a matrix using the ARC color map.
    If the matrix is not 30x30 and pad_to_30 is True, it will be padded with -1 (white).
    
    Args:
        matrix: 2D numpy array with values from -1 to 9
        title: Title for the plot
        figsize: Figure size (width, height) - only used if ax is None
        pad_to_30: Whether to pad to 30x30 if smaller
        action: Optional action dictionary to visualize action region and details
        ax: Optional matplotlib axes to plot on. If None, creates new figure.
    
    Returns:
        The axes object used for plotting
    """
    # Convert to numpy array if not already
    if not isinstance(matrix, np.ndarray):
        matrix = np.array(matrix)
    
    # Pad to 30x30 if requested and necessary
    if pad_to_30 and (matrix.shape[0] != 30 or matrix.shape[1] != 30):
        padded = np.full((30, 30), -1, dtype=np.int32)
        rows, cols = matrix.shape
        padded[:rows, :cols] = matrix
        matrix = padded
    
    rows, cols = matrix.shape
    
    # Create color array
    color_array = matrix_to_color_array(matrix)
    
    # Create figure if ax not provided
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    
    # Display the image
    ax.imshow(color_array, interpolation='nearest')
    
    # Add grid lines
    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)
    
    # Remove major ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set title
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add action outline and label if action is provided and source is 'output'
    if action is not None:
        if not is_input:
            pos = action.get('position', {})
            x = pos.get('x', 0)
            y = pos.get('y', 0)
            size = action.get('size', {'width': 1, 'height': 1})
            width = size.get('width', 1)
            height = size.get('height', 1)
            
            # Draw rectangle outline (x, y are in grid coordinates, need to adjust for matplotlib)
            rect = mpatches.Rectangle((x - 0.5, y - 0.5), width, height,
                                        linewidth=3, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            
            # Build label from action details
            action_type = action.get('action', 'default')
            direction = action.get('direction', 'default')
            color_val = action.get('color', -1)
            target_color_val = action.get('targetColor', -1)
            is_fill_all = action.get('isFillAll', False)
            scale_factor = action.get('scaleFactor', 1)
            
            label_parts = []
            if action_type != 'default':
                label_parts.append(f"action:{action_type}")
            if direction != 'default':
                label_parts.append(f"dir:{direction}")
            if color_val != -1:
                label_parts.append(f"color:{color_val}")
            if target_color_val != -1:
                label_parts.append(f"target:{target_color_val}")
            if is_fill_all:
                label_parts.append("fillAll:True")
            if scale_factor != 1:
                label_parts.append(f"scale:{scale_factor}")
            
            action_label = ", ".join(label_parts) if label_parts else "action"
            
            # Add text label near the rectangle
            label_x = x + width / 2
            label_y = y - 0.7  # Position above the rectangle
            ax.text(label_x, label_y, action_label, 
                    color='lime', fontsize=9, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            if action.get('fromPosition', {}).get('source', 'output') == 'output':
                from_pos = action.get('fromPosition', {})
                from_x = from_pos.get('x', 0)
                from_y = from_pos.get('y', 0)
                if from_x != x or from_y != y:
                    rect = mpatches.Rectangle((from_x - 0.5, from_y - 0.5), width, height,
                                            linewidth=3, edgecolor='orange', facecolor='none')
                    ax.add_patch(rect)
                    label_x = from_x + width / 2
                    label_y = from_y - 0.7  # Position above the rectangle
                    ax.text(label_x, label_y, "From", 
                        color='orange', fontsize=9, fontweight='bold',
                        ha='center', va='bottom',
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
        elif is_input and action.get('fromPosition', {}).get('source', 'output') == 'input':
            pos = action.get('fromPosition', {})
            x = pos.get('x', 0)
            y = pos.get('y', 0)
            size = action.get('size', {})
            width = size.get('width', 1)
            height = size.get('height', 1)
            
            # Draw rectangle outline (x, y are in grid coordinates, need to adjust for matplotlib)
            rect = mpatches.Rectangle((x - 0.5, y - 0.5), width, height,
                                        linewidth=3, edgecolor='orange', facecolor='none')
            ax.add_patch(rect)

            # Build label from action details
            action_type = action.get('action', 'default')
            direction = action.get('direction', 'default')
            color_val = action.get('color', -1)
            target_color_val = action.get('targetColor', -1)
            is_fill_all = action.get('isFillAll', False)
            scale_factor = action.get('scaleFactor', 1)
            label_parts = []
            if action_type != 'default':
                label_parts.append(f"action:{action_type}")
            if direction != 'default':
                label_parts.append(f"dir:{direction}")
            if color_val != -1:
                label_parts.append(f"color:{color_val}")
            if target_color_val != -1:
                label_parts.append(f"target:{target_color_val}")
            if is_fill_all:
                label_parts.append("fillAll:True")
            if scale_factor != 1:
                label_parts.append(f"scale:{scale_factor}")
            action_label = ", ".join(label_parts) if label_parts else "action"
            # Add text label near the rectangle
            label_x = x
            label_y = y - 0.7  # Position above the rectangle
            ax.text(label_x, label_y, action_label, 
                    color='orange', fontsize=9, fontweight='bold',
                    ha='center', va='bottom',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))
            
    # Create legend
    legend_elements = []
    for digit in sorted(COLOR_MAP.keys()):
        if digit in matrix:  # Only show colors that appear in the matrix
            label = f"{digit}" if digit != -1 else "Empty"
            legend_elements.append(
                mpatches.Patch(color=COLOR_MAP[digit], label=label)
            )
    
    if legend_elements:
        ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), 
                 loc='upper left', fontsize=10)
    
    # Only call tight_layout and show if we created the figure
    if ax is None:
        plt.tight_layout()
        plt.show()
    
    return ax


def generate_random_action_sequence(
    input_matrix: np.ndarray = None,
    output_matrix: np.ndarray = None,
    sequence_length: int = 10,
    changes_required: bool = False,
    max_attempts_per_action: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a random sequence of actions and apply them step-by-step.
    
    Args:
        input_matrix: Input grid (numpy array). If None, uses a 3x3 zeros matrix.
        output_matrix: Initial output grid (numpy array). If None, uses a 3x3 zeros matrix.
        sequence_length: Number of actions in the sequence.
        changes_required: If True, each action must change the output matrix.
        max_attempts_per_action: Maximum attempts to find an action that changes the output.
    
    Returns:
        Tuple of:
            - action_vectors: np.ndarray of shape (sequence_length, 31) with action vectors
            - output_sequence: np.ndarray of shape (sequence_length + 1, rows, cols) with output matrices
                             Index 0 is the initial output, indices 1+ are outputs after each action
    """
    # Initialize input and output matrices
    if input_matrix is None:
        input_matrix = np.zeros((3, 3), dtype=np.int32)
    
    if output_matrix is None:
        output_matrix = np.zeros((3, 3), dtype=np.int32)
    
    rows, cols = output_matrix.shape
    input_rows, input_cols = input_matrix.shape

    # Initialize storage
    action_vectors = np.zeros((sequence_length, 31), dtype=np.float32)
    output_sequence = [output_matrix.copy()]
    
    current_output = output_matrix.copy()

    # 80% chance that the first action is resize
    count = 0
    from_input = False
    if np.random.rand() < 0.8:
        # Initialize action dictionary with defaults
        action = {
            'action': 'resize',
            'position': {'x': 0, 'y': 0, 'source': 'output'},
            'fromPosition': {'x': 0, 'y': 0, 'source': 'output'},
            'size': {'width': 1, 'height': 1},
            'color': -1,
            'targetColor': -1,
            'isFillAll': False,
            'scaleFactor': 1,
            'direction': 'default'
        }
        if np.random.rand() < 0.1:
            # 10% chance to use a random resize action
            action_vector = generate_random_action_vector(rows, cols, action='resize')
        else:
            # Resize to match input size
            from_input = True
            action['size'] = {'width': input_cols, 'height': input_rows}
            action_vector = action_to_vector(action)
        count += 1
        # Apply the action
        new_output = edit_grid(input_matrix, current_output, action_vector)
        rows, cols = new_output.shape
        action_vectors[0] = action_vector
        current_output = new_output
        output_sequence.append(new_output.copy())

        # 90% chance that the second action is copy from input to output
        if sequence_length > 1 and np.random.rand() < 0.9:
            if from_input:
                action['size'] = {'width': input_cols, 'height': input_rows}
                action['fromPosition'] = {'x': 0, 'y': 0, 'source': 'input'}
                action['action'] = 'copy'
                action_vector = action_to_vector(action)
            else:
                action_vector = generate_random_action_vector(rows, cols, action='copy', input_rows=input_rows, input_cols=input_cols)
            count += 1
            # Apply the action
            new_output = edit_grid(input_matrix, current_output, action_vector)
            action_vectors[1] = action_vector
            current_output = new_output
            output_sequence.append(new_output.copy())

    for step in range(count, sequence_length):
        action_found = False
        attempts = 0
        
        while attempts < max_attempts_per_action:
            # Generate a random action
            rows, cols = current_output.shape
            action_vector = generate_random_action_vector(rows, cols, input_rows=input_rows, input_cols=input_cols)
            if action_vector is None:
                attempts += 1
                continue
            
            # Apply the action
            action = vector_to_action(action_vector)
            action_type = action['action']
            if action_type == 'resize' and np.random.rand() < 0.5:
                action_vector = generate_random_action_vector(rows, cols, input_rows=input_rows, input_cols=input_cols)
                if action_vector is None:
                    continue
                action = vector_to_action(action_vector)
                action_type = action['action']
                
            if action_type == 'fill-boundary':
                x, y = action['position']['x'], action['position']['y']
                action['targetColor'] = current_output[y, x]
                while action['targetColor'] == action['color']:
                    action['color'] = np.random.randint(-1, 10)
                action_vector = action_to_vector(action)
            new_output = edit_grid(input_matrix, current_output, action_vector)
            rows, cols = new_output.shape
            
            # Check if changes are required
            if changes_required:
                # Check if the output actually changed
                if not np.array_equal(new_output, current_output):
                    action_found = True
                    action_vectors[step] = action_vector
                    current_output = new_output
                    output_sequence.append(new_output.copy())
                    break
            else:
                # No requirement for changes, accept any action
                action_found = True
                action_vectors[step] = action_vector
                current_output = new_output
                output_sequence.append(new_output.copy())
                break
            
            attempts += 1
        
        # If we couldn't find a valid action after max attempts
        if not action_found and changes_required:
            print(f"Warning: Could not find action that changes output at step {step} after {max_attempts_per_action} attempts.")
            # Use a 'fill' action with a random color as fallback
            fallback_action = {
                'action': 'fill',
                'position': {'x': np.random.randint(0, cols), 'y': np.random.randint(0, rows), 'source': 'output'},
                'fromPosition': {'x': 0, 'y': 0, 'source': 'output'},
                'size': {'width': 1, 'height': 1},
                'color': np.random.randint(0, 10),
                'targetColor': -1,
                'isFillAll': True,
                'scaleFactor': 1,
                'direction': 'default'
            }
            action_vector = action_to_vector(fallback_action)
            new_output = edit_grid(input_matrix, current_output, action_vector)
            rows, cols = new_output.shape
            
            action_vectors[step] = action_vector
            current_output = new_output
            output_sequence.append(new_output.copy())
    # output_sequence = np.array(output_sequence, dtype=np.int32)
    return action_vectors, output_sequence

def generate_output_sequence(
    input_matrix: np.ndarray,
    action_vectors: np.ndarray
) -> np.ndarray:
    """
    Generate the output sequence by applying the action vectors step-by-step.
    
    Args:
        input_matrix: Input grid (numpy array)
        action_vectors: np.ndarray of shape (sequence_length, 31) with action vectors
    Returns:
        output_sequence: np.ndarray of shape (sequence_length + 1, rows, cols) with output matrices
    """
    sequence_length = action_vectors.shape[0]
    output_sequence = []
    
    # Start with an initial output matrix of zeros
    current_output = np.zeros((3, 3), dtype=np.int32)
    output_sequence.append(current_output.copy())
    
    for step in range(sequence_length):
        action_vector = action_vectors[step]
        new_output = edit_grid(input_matrix, current_output, action_vector)
        output_sequence.append(new_output.copy())
        current_output = new_output
    
    return output_sequence

def get_matrix_color_order(matrix: np.ndarray) -> List[int]:
    """
    Get the order of colors as they appear in the matrix (row-wise).
    
    Args:
        matrix: 2D numpy array with values from -1 to 9
    
    Returns:
        List of unique colors in the order they appear
    """
    rows, cols = matrix.shape
    color_order = []
    seen_colors = set()
    
    for i in range(rows):
        for j in range(cols):
            color = matrix[i, j]
            if color not in seen_colors:
                seen_colors.add(color)
                color_order.append(color)
    
    return color_order

def swap_colors_in_matrix(matrix: np.ndarray) -> np.ndarray:
    """
    Swap colors in the matrix based on their order of appearance.
    
    Args:
        matrix: 2D numpy array with values from -1 to 9
    Returns:
        New matrix with colors swapped
    """
    color_order = get_matrix_color_order(matrix)
    all_colors = set(range(0, 10))
    non_existing_colors = list(all_colors - set(color_order))
    # either swap an existing color with a random color
    if np.random.rand() < 0.5 and len(non_existing_colors) > 0:
        color_to_swap = np.random.choice(color_order)
        new_color = np.random.choice(non_existing_colors)
        swapped_matrix = matrix.copy()
        swapped_matrix[matrix == color_to_swap] = new_color
        new_color_order = get_matrix_color_order(swapped_matrix)
        return swapped_matrix, color_order, new_color_order
    else:
        # or swap two existing colors
        if len(color_order) >= 2:
            color1, color2 = np.random.choice(color_order, size=2, replace=False)
            swapped_matrix = matrix.copy()
            swapped_matrix[matrix == color1] = -1  # Temporary marker
            swapped_matrix[matrix == color2] = color1
            swapped_matrix[swapped_matrix == -1] = color2
            new_color_order = get_matrix_color_order(swapped_matrix)
            return swapped_matrix, color_order, new_color_order
        
def swap_color_actions_in_sequence(
    action_vectors: np.ndarray,
    original_color_order: List[int],
    new_color_order: List[int]
) -> np.ndarray:
    """
    Swap color references in the action vectors based on the color order mapping.
    
    Args:
        action_vectors: np.ndarray of shape (sequence_length, 31) with action vectors
        original_color_order: List of original colors in order
        new_color_order: List of new colors in order
    Returns:
        New action_vectors with colors swapped
    """
    color_mapping = {orig: new for orig, new in zip(original_color_order, new_color_order)}
    swapped_action_vectors = action_vectors.copy()
    sequence_length = len(action_vectors)
    
    for step in range(sequence_length):
        action = vector_to_action(action_vectors[step])
        color = action.get('color', -1)
        target_color = action.get('targetColor', -1)
        
        if color in color_mapping:
            action['color'] = color_mapping[color]
        if target_color in color_mapping:
            action['targetColor'] = color_mapping[target_color]
        
        swapped_action_vectors[step] = action_to_vector(action)
    
    return swapped_action_vectors

def visualize_action_sequence(
    action_vectors: np.ndarray,
    output_sequence: np.ndarray,
    input_matrix: np.ndarray = None,
    max_steps_to_show: int = None,
    figsize_per_step: Tuple[int, int] = (6, 6)
) -> None:
    """
    Visualize a sequence of actions and their effects on the output matrix.
    
    Args:
        action_vectors: np.ndarray of shape (sequence_length, 31) with action vectors
        output_sequence: np.ndarray of shape (sequence_length + 1, rows, cols) with output matrices
        input_matrix: Optional input matrix to display alongside
        max_steps_to_show: Maximum number of steps to visualize (None = show all)
        figsize_per_step: Size of each subplot
    """
    sequence_length = action_vectors.shape[0]
    
    if max_steps_to_show is not None:
        steps_to_show = min(max_steps_to_show, sequence_length)
    else:
        steps_to_show = sequence_length
    
    # Determine number of columns: 1 for output, optionally 1 for input
    n_cols = 2 if input_matrix is not None else 1
    n_rows = steps_to_show + 1  # +1 for initial state
    
    fig, axes = plt.subplots(n_rows, n_cols, 
                            figsize=(figsize_per_step[0] * n_cols, 
                                   figsize_per_step[1] * n_rows))
    
    # Handle case where axes might not be 2D
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(-1, 1)
    
    # Show initial state
    visualize_matrix(output_sequence[0], title="Initial Output (Step 0)", 
                    pad_to_30=True, ax=axes[0, 0])
    if input_matrix is not None and n_cols > 1:
        visualize_matrix(input_matrix, title="Input (Reference)", is_input=True,
                        pad_to_30=True, ax=axes[0, 1])
        axes[0, 1].set_title("Input (Reference)", fontsize=12, fontweight='bold')
    
    # Show each step
    for step in range(steps_to_show):
        action = vector_to_action(action_vectors[step])
        title = f"Step {step + 1}: {action['action']}"
        
        visualize_matrix(output_sequence[step + 1], title=title,
                        pad_to_30=True, action=action, ax=axes[step + 1, 0])
        
        if input_matrix is not None and n_cols > 1:
            # Show input again for reference
            visualize_matrix(input_matrix, title="Input (Reference)", is_input=True,
                           pad_to_30=True, action=action, ax=axes[step + 1, 1])
    
    plt.tight_layout()
    plt.show()

def generate_random_matrix(rows: int = None, cols: int = None, density: float = None) -> np.ndarray:
    """
    Generate a random matrix with given dimensions and density of colored cells.
    
    Args:
        rows: Number of rows (default random between 1 and 30)
        cols: Number of columns (default random between 1 and 30)
        density: Fraction of cells that are colored (0 to 1)
    
    Returns:
        2D numpy array with values from -1 to 9
    """
    if rows is None:
        rows = np.random.randint(1, 31)
    if cols is None:
        cols = np.random.randint(1, 31)
    if density is None:
        density = np.random.uniform(0.1, 0.5)
    
    matrix = np.full((rows, cols), -1, dtype=np.int32)
    
    for i in range(rows):
        for j in range(cols):
            if np.random.rand() < density:
                matrix[i, j] = np.random.randint(0, 10)
    
    return matrix

def pad_matrix(np_array, target_w: int = 30, target_h: int = 30, pad_value= -1):
        """Pad a 2D numpy array to target width and height"""
        h, w = np_array.shape
        padded_array = np.full((target_h, target_w), pad_value, dtype=np_array.dtype)
        padded_array[0:h, 0:w] = np_array
        return padded_array


class ShapeGenerator:
    """Generate synthetic shapes for object detection training"""
    
    def __init__(self, max_size=30, values_range=(0, 9), min_size=2):
        self.max_size = max_size
        self.min_size = min_size
        self.values_range = values_range
        self.padding_value = -1

    def generate_circle(self, draw, bbox, value, filled=True):
        """Generate a circle shape"""
        x1, y1, x2, y2 = bbox
        points = [x1, y1, x2 - 1, y2 - 1]
        if filled:
            draw.ellipse(points, fill=value)
        else:
            draw.ellipse(points, outline=value, width=1)

    def generate_square(self, draw, bbox, value, filled=True):
        """Generate a square shape"""
        x1, y1, x2, y2 = bbox
        points = [x1, y1, x2 - 1, y2 - 1]
        if filled:
            draw.rectangle(points, fill=value)
        else:
            draw.rectangle(points, outline=value, width=1)

    def generate_rectangle(self, draw, bbox, value, filled=True):
        """Generate a rectangle shape"""
        x1, y1, x2, y2 = bbox
        points = [x1, y1, x2 - 1, y2 - 1]
        if filled:
            draw.rectangle(points, fill=value)
        else:
            draw.rectangle(points, outline=value, width=1)

    def generate_line(self, draw, bbox, direction, value):
        """Generate a line shape"""
        x1, y1, x2, y2 = bbox
        if direction < 3:
            points = [x1, y1, x2 - 1, y2 - 1]
            draw.line(points, fill=value, width=1)
        else:
            points = [x1, y2 - 1, x2 - 1, y1]
            draw.line(points, fill=value, width=1)

    def generate_triangle(self, draw, bbox, direction, value, filled=True):
        """Generate a triangle shape"""
        points = [
            (bbox[0], bbox[1]),
            (bbox[2] - 1, bbox[1]),
            (bbox[2] - 1, bbox[3] - 1),
            (bbox[0], bbox[3] - 1)
        ]
        middle_points = []
        for i in range(len(points)):
            next_index = (i + 1) % len(points)
            mid_x = (points[i][0] + points[next_index][0]) // 2
            mid_y = (points[i][1] + points[next_index][1]) // 2
            middle_points.append((mid_x, mid_y))
        def slice_except_index(lst, index):
            return [item for i, item in enumerate(lst) if i != index]
        if direction%2 == 0:
            points = slice_except_index(points, direction//2)
        else:
            middle_point = middle_points[direction//2]
            p1 = points[direction//2 + 2 if direction//2 + 2 < len(points) else direction//2 + 2 - len(points)]
            p2 = points[direction//2 + 3 if direction//2 + 3 < len(points) else direction//2 + 3 - len(points)]
            points = [p1, middle_point, p2]
        if filled:
            draw.polygon(points, fill=value)
        else:
            draw.polygon(points, outline=value, width=1)

    def generate_ellipse(self, draw, bbox, value, filled=True):
        """Generate an ellipse shape"""
        x1, y1, x2, y2 = bbox
        points = [x1, y1, x2 - 1, y2 - 1]
        if filled:
            draw.ellipse(points, fill=value)
        else:
            draw.ellipse(points, outline=value, width=1)
    
    def generate_cross(self, draw, bbox, value):
        """Generate a cross/plus shape"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w%2 == 0 or h%2 == 0:
            return  # Cross requires odd width and height
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        points = [
            (x1, center_y),
            (x2 - 1, center_y),
            (center_x, y1),
            (center_x, y2 - 1)
        ]
        draw.line([points[0], points[1]], fill=value, width=1)
        draw.line([points[2], points[3]], fill=value, width=1)

    def generate_diamond(self, draw, bbox, value, filled=True):
        """Generate a diamond/rhombus shape"""
        x1, y1, x2, y2 = bbox
        middle_points = [
            ((x1 + x2) // 2, y1),
            (x2 - 1, (y1 + y2) // 2),
            ((x1 + x2) // 2, y2 - 1),
            (x1, (y1 + y2) // 2)
        ]
        if filled:
            draw.polygon(middle_points, fill=value)
        else:
            draw.polygon(middle_points, outline=value, width=1)

    def generate_polygon(self, draw, points, value, filled=True):
        """Generate a polygon shape"""
        if filled:
            draw.polygon(points, fill=value)
        else:
            draw.polygon(points, outline=value, width=1)

    def generate_X(self, draw, bbox, value):
        """Generate a symetric x"""
        x1, y1, x2, y2 = bbox
        w, h = x2 - x1, y2 - y1
        if w != h:
            return  # X requires square bbox
        points = [
            (x1, y1),
            (x2 - 1, y2 - 1),
            (x1, y2 - 1),
            (x2 - 1, y1)
        ]
        draw.line([points[0], points[1]], fill=value, width=1)
        draw.line([points[2], points[3]], fill=value, width=1)

    def generate_L(self, draw, points, line_width, value):
        """Generate an L shape"""
        p1, p2, p3 = points
        l1 = [p1, p2]
        l2 = [p2, p3]
        draw.line(l1, fill=value, width=line_width)
        draw.line(l2, fill=value, width=line_width)

    def _generate_random_bbox(self, w, h, squared = False, min_w = 2, min_h = 2, max_w = 30, max_h = 30, existing_bboxes=[], impair=False):
        """Generate a random bbox within image dimensions"""
        box_w = random.randint(min_w, min(max_w, w))
        box_h = random.randint(min_h, min(max_h, h))
        x1 = random.randint(0, w - box_w)
        y1 = random.randint(0, h - box_h)
        bbox = self._compute_actual_bbox([x1, y1, x1 + box_w, y1 + box_h], w, h)
        if self._has_acceptable_overlap(bbox, existing_bboxes, 0.22) and (not squared or (bbox[2] - bbox[0]) == (bbox[3] - bbox[1])) and (bbox[2] - bbox[0] >= min_w) and (bbox[3] - bbox[1] >= min_h) or (impair and (bbox[2] - bbox[0]) % 2 == 1 and (bbox[3] - bbox[1]) % 2 == 1):
            z_index = self._compute_z_index(bbox, existing_bboxes)
            existing_bboxes.append(bbox)
            return bbox, existing_bboxes, z_index
        return self._generate_random_bbox(w, h, squared, min_w, min_h, max_w, max_h, existing_bboxes)

    def _compute_actual_bbox(self, bbox, w, h):
        """Ensure bbox coordinates are within bounds"""
        bbox = self._sort_bbox_coordinates(bbox)
        return [max(0, int(bbox[0])), max(0, int(bbox[1])),
                min(w, int(bbox[2])), min(h, int(bbox[3]))]
    
    def _generate_random_noise_matrix(self, w, h, bg_color, same_color=False):        
        if same_color:
            color = random.randint(0, 9)
            while color == bg_color:
                color = random.randint(0, 9)
            value_list = np.array([color])
        else:
            value_list = np.array(list(range(0, 9)))

        nb_position = random.randint(1, max(1, (w + h) // 4))
        positions = np.random.choice(range(w * h), nb_position, replace=False)
        rows = positions // w
        cols = positions % w
        random_values = np.random.choice(value_list, nb_position, replace=True)
        noise = np.zeros((h, w), dtype=np.uint8)
        noise[rows, cols] = random_values
        mask = np.ones((h, w), dtype=np.uint8)
        mask[rows, cols] = 0
        return noise, mask

    def _sort_bbox_coordinates(self, bbox):
        # Ensure bbox coordinates are in correct order
        if not (bbox[2] > bbox[0] and bbox[3] > bbox[1]):
            bbox_temp = [bbox[2], bbox[3], bbox[0], bbox[1]]
            bbox = bbox_temp
        return bbox

    def _is_aligned(self, point1, point2):
        """Check if two points are aligned along one axis"""
        return point1[0] == point2[0] or point1[1] == point2[1]
    
    def _same_point(self, point1, point2):
        """Check if two points are the same"""
        return point1[0] == point2[0] and point1[1] == point2[1]

    def _combine_bboxes(self, bbox1, bbox2):
        """Combine two bounding boxes into one that encompasses both"""
        x1 = min(bbox1[0], bbox2[0])
        y1 = min(bbox1[1], bbox2[1])
        x2 = max(bbox1[2], bbox2[2])
        y2 = max(bbox1[3], bbox2[3])
        return [x1, y1, x2, y2]

    def _compute_overlap(self, bbox1, bbox2):
        """Compute the overlap ratio between two bounding boxes"""
        # Calculate intersection
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 <= x1 or y2 <= y1:
            return 0.0
        
        intersection_area = (x2 - x1) * (y2 - y1)
        
        # Calculate areas
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Return the maximum overlap ratio relative to either bbox
        overlap_ratio1 = intersection_area / area1 if area1 > 0 else 0
        overlap_ratio2 = intersection_area / area2 if area2 > 0 else 0
        
        return max(overlap_ratio1, overlap_ratio2)
    
    def _compute_z_index(self, bbox, existing_bboxes):
        """Compute the z-index of a bbox based on overlaps with existing bboxes"""
        z_index = 0
        for existing_bbox in existing_bboxes:
            overlap = self._compute_overlap(bbox, existing_bbox)
            if overlap > 0:
                z_index += 1
        return z_index
    
    def _has_acceptable_overlap(self, new_bbox, existing_bboxes, max_overlap=0.3):
        """Check if a new bbox has acceptable overlap with existing bboxes"""
        for existing_bbox in existing_bboxes:
            overlap = self._compute_overlap(new_bbox, existing_bbox)
            if overlap > max_overlap:
                return False
        return True
    
    def add_random_shape(self, image_array, draw, w, h, bg_color=0, existing_bboxes=[], existing_colors=[]):
        """Add a random shape to the image and return its bounding box"""
        shape_types = [
            'circle_filled', 'circle_outline',
            'square_filled', 'square_outline',
            'rectangle_filled', 'rectangle_outline',
            'line', 'line_discrete', 'triangle_filled', 'triangle_outline',
            'ellipse_filled', 'ellipse_outline',
            'cross', 'diamond_filled', 'diamond_outline',
            'polygon_3', "polygon_4", "polygon_5", 
            # "polygon_6", "polygon_7", "polygon_8",
            'X', 'L', 'L_rect'
        ]
        
        shape_type = random.choice(shape_types)
        value = random.randint(*self.values_range)
        while value in existing_colors:
            value = random.randint(*self.values_range)
            # if random.random() < 0.01: # break potential infinite loop
            #     if value != bg_color:
            #         break
        try:
            if 'circle' in shape_type:
                if min(w, h) < 8:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=True, min_w=4, min_h=4, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                self.generate_circle(draw, bbox, value, filled='filled' in shape_type)
                if 'filled' in shape_type:
                    return [(bbox, shape_type.split('_')[0], value, z_index)]
                inner_bbox = [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1]
                return [(bbox, shape_type.split('_')[0], value, z_index), (inner_bbox, 'circle', bg_color, z_index)]

            elif 'square' in shape_type:
                if min(w, h) < 3:
                    return []
                if 'filled' in shape_type:
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=True, min_w=2, min_h=2, max_w=w , max_h=h , existing_bboxes=existing_bboxes)
                else:
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=True, min_w=3, min_h=3, max_w=w , max_h=h , existing_bboxes=existing_bboxes)
                self.generate_square(draw, bbox, value, filled='filled' in shape_type)
                if 'filled' in shape_type:
                    return [(bbox, shape_type.split('_')[0], value, z_index)]
                inner_bbox = [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1]
                return [(bbox, shape_type.split('_')[0], value, z_index), (inner_bbox, 'square', bg_color, z_index)]
            
            elif 'rectangle' in shape_type:
                if min(w, h) < 3:
                    return []
                if 'filled' in shape_type:
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=2, min_h=2, max_w=w , max_h=h , existing_bboxes=existing_bboxes)
                else:
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=3, min_h=3, max_w=w , max_h=h , existing_bboxes=existing_bboxes)
                r_w, r_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if r_w == r_h:
                    return []
                self.generate_rectangle(draw, bbox, value, filled='filled' in shape_type)
                if 'filled' in shape_type:
                    return [(bbox, shape_type.split('_')[0], value, z_index)]
                inner_bbox = [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1]
                return [(bbox, shape_type.split('_')[0], value, z_index), (inner_bbox, 'rectangle', bg_color, z_index)]


            elif shape_type == 'line':
                if min(w, h) < 5:
                    return []
                direction = random.randint(0, 3)  # 0: horizontal, 1: diagonal \, 2: vertical, 3: diagonal /
                if direction%2 == 0:
                    if direction == 0:
                        # horizontal line
                        bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=4, min_h=1, max_w=w, max_h=1, existing_bboxes=existing_bboxes)
                    else:
                        # vertical line
                        bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=1, min_h=4, max_w=1, max_h=h, existing_bboxes=existing_bboxes)
                else:
                    # diagonal line (need square bbox)
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=True, min_w=4, min_h=4, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                if bbox:
                    self.generate_line(draw, bbox, direction, value)
                    return [(bbox, shape_type.split('_')[0], value, z_index)]
                return []

            elif shape_type == 'line_discrete':
                if min(w, h) < 5:
                    return []
                direction = random.randint(0, 3)  # 0: horizontal, 1: diagonal \, 2: vertical, 3: diagonal /
                n_attempts = 20
                bbox = None
                while n_attempts > 0:
                    n_attempts -= 1
                    dash_length = random.randint(1, 3)
                    gap_length = random.randint(1, 3)
                    if direction%2 == 0:
                        if direction == 0:
                            # horizontal line
                            bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=4, min_h=1, max_w=w, max_h=1, existing_bboxes=existing_bboxes)
                            length = bbox[2] - bbox[0]
                        else:
                            # vertical line
                            bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=1, min_h=4, max_w=1, max_h=h, existing_bboxes=existing_bboxes)
                            length = bbox[3] - bbox[1]
                    else:
                        # diagonal line (need square bbox)
                        bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=True, min_w=4, min_h=4, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                        length = bbox[2] - bbox[0]
                    if length < dash_length + gap_length:
                        continue
                    # make sure the end of a line is not a gap
                    if length % (dash_length + gap_length) < dash_length and length % (dash_length + gap_length) != 0:
                        break
                if not bbox:
                    return []
                # Draw discrete line
                if direction%2 == 0:
                    if direction == 0:
                        # horizontal line
                        for x_start in range(bbox[0], bbox[2], dash_length + gap_length):
                            x_end = min(x_start + dash_length, bbox[2])
                            draw.line([x_start, bbox[1], x_end - 1, bbox[1]], fill=value, width=1)
                    else:
                        # vertical line
                        for y_start in range(bbox[1], bbox[3], dash_length + gap_length):
                            y_end = min(y_start + dash_length, bbox[3])
                            draw.line([bbox[0], y_start, bbox[0], y_end - 1], fill=value, width=1)
                    return [(bbox, 'line_discrete', value, z_index)]
                else:
                    x_start, y_start = bbox[0], bbox[3] - 1 if direction == 3 else bbox[1]
                    x_end, y_end = bbox[2] - 1, bbox[1] if direction == 3 else bbox[3] - 1
                    total_length = x_end - x_start + 1
                    for offset in range(0, total_length, dash_length + gap_length):
                        seg_start_x = x_start + offset
                        seg_start_y = y_start - offset if direction == 3 else y_start + offset
                        seg_end_x = min(seg_start_x + dash_length - 1, x_end)
                        seg_end_y = y_start - (seg_end_x - x_start) if direction == 3 else y_start + (seg_end_x - x_start)
                        draw.line([seg_start_x, seg_start_y, seg_end_x, seg_end_y], fill=value, width=1)
                    return [(bbox, 'line_discrete', value, z_index)]
            
            elif 'ellipse' in shape_type:
                if max(w, h) < 7 or min(w, h) < 4:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=6, min_h=4, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                self.generate_ellipse(draw, bbox, value, filled='filled' in shape_type)
                if 'filled' in shape_type:
                    return [(bbox, shape_type.split('_')[0], value, z_index)]
                inner_bbox = [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1]
                return [(bbox, shape_type.split('_')[0], value, z_index), (inner_bbox, 'ellipse', bg_color, z_index)]
            

            elif shape_type == 'cross':
                if min(w, h) < 3:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=3, min_h=3, max_w=w, max_h=h, existing_bboxes=existing_bboxes, impair=True)
                # Ensure odd width and height for cross
                w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                while w % 2 == 0 or h % 2 == 0:
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=3, min_h=3, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                self.generate_cross(draw, bbox, value)
                bbox_lines = [
                    [bbox[0], (bbox[1] + bbox[3]) // 2, bbox[2], (bbox[1] + bbox[3]) // 2 + 1],
                    [(bbox[0] + bbox[2]) // 2, bbox[1], (bbox[0] + bbox[2]) // 2 + 1, bbox[3]]
                ]
                return [(bbox, shape_type.split('_')[0], value, z_index), (bbox_lines[0], 'line', value, z_index), (bbox_lines[1], 'line', value, z_index)]
            
            elif 'diamond' in shape_type:
                if min(w, h) < 5:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=5, min_h=5, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                d_w, d_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                while d_w % 2 == 0 or d_h % 2 == 0:
                    bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=5, min_h=5, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                    d_w, d_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                self.generate_diamond(draw, bbox, value, filled='filled' in shape_type)
                if 'filled' in shape_type:
                    return [(bbox, shape_type.split('_')[0], value, z_index)]
                inner_bbox = [bbox[0] + 1, bbox[1] + 1, bbox[2] - 1, bbox[3] - 1]
                return [(bbox, shape_type.split('_')[0], value, z_index), (inner_bbox, 'diamond', bg_color, z_index)]

            elif 'polygon' in shape_type:
                if min(w, h) < 2 or w + h < 6 or h - w > max(w, h) // 1.4 or w - h > max(w, h) // 1.4:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=2, min_h=2, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                # The number of points is determined by the size of the bbox
                sqarea = math.sqrt((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
                # nb_points = random.randint(3, min(8, max(3, sqarea))) # 6 shapes max
                nb_points = random.randint(3, min(5, max(3, int(sqarea)))) # 3 shapes max
                x1, y1, x2, y2 = bbox
                points = []
                def existed_point(px, py, points):
                    for p in points:
                        if p[0] == px and p[1] == py:
                            return True
                    return False
                for i in range(nb_points):
                    px = random.randint(x1, x2 - 1)
                    py = random.randint(y1, y2 - 1)
                    while existed_point(px, py, points):
                        px = random.randint(x1, x2 - 1)
                        py = random.randint(y1, y2 - 1)
                    points.append((px, py))
                def sort_points_clockwise(pts):
                    center_x = sum(p[0] for p in pts) / len(pts)
                    center_y = sum(p[1] for p in pts) / len(pts)
                    return sorted(pts, key=lambda p: np.arctan2(p[1] - center_y, p[0] - center_x))
                points = sort_points_clockwise(points)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
                bbox = self._compute_actual_bbox(bbox, w, h)
                p_w, p_h = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if p_w + p_h < 6 or p_h - p_w > max(p_w, p_h) // 1.4 or p_w - p_h > max(p_w, p_h) // 1.4:
                    return []
                existing_bboxes.append(bbox)
                self.generate_polygon(draw, points, value, filled=True)
                map_point_to_name = {3: 'triangle', 4: 'quadrilateral', 5: 'pentagon', 6: 'hexagon', 7: 'heptagon', 8: 'octagon'}
                name = map_point_to_name[nb_points]
                return [(bbox, name, value, z_index)]
            
            elif 'X' in shape_type:
                if min(w, h) < 4:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=True, min_w=3, min_h=3, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                self.generate_X(draw, bbox, value)
                return [(bbox, shape_type.split('_')[0], value, z_index), (bbox, 'line', value, z_index)]

            elif shape_type == 'L':
                # directions: 0 to 7 (left, up-left, up, up-right, right, down-right, down, down-left)
                direction1 = random.randint(0, 7)
                direction2 = (direction1 + random.choice([-2, 2])) % 8
                if direction1 % 4 == 0:
                    line1_size = random.randint(2, max(2, w))
                    line2_size = random.randint(2, max(2, h))
                    while line1_size - line2_size > max(line1_size, line2_size) // 2 or line2_size - line1_size > max(line1_size, line2_size) // 2:
                        line1_size = random.randint(2, max(2, w))
                        line2_size = random.randint(2, max(2, h))
                elif direction1 % 4 == 2:
                    line1_size = random.randint(2, max(2, h))
                    line2_size = random.randint(2, max(2, w))
                    while line1_size - line2_size > max(line1_size, line2_size) // 2 or line2_size - line1_size > max(line1_size, line2_size) // 2:
                        line1_size = random.randint(2, max(2, h))
                        line2_size = random.randint(2, max(2, w))
                else:
                    line1_size = random.randint(2, min(w, h))
                    line2_size = random.randint(2, min(w, h))
                    while line1_size - line2_size > max(line1_size, line2_size) // 2 or line2_size - line1_size > max(line1_size, line2_size) // 2:
                        line1_size = random.randint(2, min(w, h))
                        line2_size = random.randint(2, min(w, h))
                x_start = random.randint(0, w - line1_size if direction1 in [0, 7, 1] else line1_size if direction1 in [3, 4, 5] else w - 1 - line2_size)
                y_start = random.randint(0, h - line2_size if direction1 in [1, 2, 3] else line2_size if direction1 in [5, 6, 7] else h - 1 - line1_size)
                x_corner = x_start + int(line1_size * np.cos(np.pi / 4 * direction1))
                y_corner = y_start - int(line1_size * np.sin(np.pi / 4 * direction1))
                x_end = x_corner + int(line2_size * np.cos(np.pi / 4 * direction2))
                y_end = y_corner - int(line2_size * np.sin(np.pi / 4 * direction2))
                if not (0 <= x_end < w and 0 <= y_end < h):
                    return []
                points = [(x_start, y_start), (x_corner, y_corner), (x_end, y_end)]
                line_width = random.randint(1, 2 if min(w, h) >= 4 else 1)
                xs = [p[0] for p in points]
                ys = [p[1] for p in points]
                bbox = [min(xs), min(ys), max(xs) + 1, max(ys) + 1]
                real_bbox = self._compute_actual_bbox(bbox, w, h)
                if bbox[2] - bbox[0] < 2 or bbox[3] - bbox[1] < 2 and self._compute_overlap(bbox, real_bbox) < 0.8:
                    return []
                bbox = real_bbox
                z_index = self._compute_z_index(bbox, existing_bboxes)
                existing_bboxes.append(bbox)
                self.generate_L(draw, points, 1, value)
                bbox_lines = [
                    self._compute_actual_bbox([min(x_start, x_corner), min(y_start, y_corner), max(x_start, x_corner) + 1, max(y_start, y_corner) + 1], w, h),
                    self._compute_actual_bbox([min(x_corner, x_end), min(y_corner, y_end), max(x_corner, x_end) + 1, max(y_corner, y_end) + 1], w, h)
                ]
                return [(bbox, shape_type.split('_')[0], value, z_index), (bbox_lines[0], 'line', value, z_index), (bbox_lines[1], 'line', value, z_index)]


            elif 'L_rect' in shape_type:
                if min(w, h) < 4:
                    return []
                bbox, existing_bboxes, z_index = self._generate_random_bbox(w, h, squared=False, min_w=2, min_h=2, max_w=w, max_h=h, existing_bboxes=existing_bboxes)
                w1, h1 = bbox[2] - bbox[0], bbox[3] - bbox[1]
                if w1 < h1:
                    h2 = w1
                    h4 = h1 - w1
                    w4 = w1
                    h3 = w1
                    x4 = bbox[0]    
                    right = random.choice([True, False])
                    if right:
                        w2 = random.randint(1, w - bbox[2])
                        x2 = bbox[2]
                        x3 = bbox[0]
                    else:
                        w2 = random.randint(1, bbox[0])
                        x2 = bbox[0] - w2
                        x3 = bbox[0] - w2
                    w3 = w2 + w1
                    top = random.choice([True, False])
                    if top:
                        y2 = bbox[1]
                        y3 = bbox[1]
                        y4 = bbox[1] + w1
                    else:
                        y2 = bbox[3] - h2
                        y3 = bbox[3] - w1
                        y4 = bbox[1]
                elif w1 > h1:
                    w2 = h1
                    w4 = w1 - h1
                    h4 = h1
                    w3 = h1
                    y4 = bbox[1]
                    top = random.choice([True, False])
                    if top:
                        h2 = random.randint(1, bbox[1])
                        y2 = bbox[1] - h2
                        y3 = bbox[1] - h2
                    else:
                        h2 = random.randint(1, h - bbox[3])
                        y2 = bbox[3]
                        y3 = bbox[1]
                    h3 = h2 + h1
                    right = random.choice([True, False])
                    if right:
                        x2 = bbox[2] - w2
                        x3 = bbox[2] - w2
                        x4 = bbox[0]
                    else:
                        x2 = bbox[0]
                        x3 = bbox[0]
                        x4 = bbox[0] + h1
                else:
                    return []  # skip square L_rect
                bbox2 = [x2, y2, x2 + w2, y2 + h2]
                bbox2 = self._compute_actual_bbox(bbox2, w, h)
                # Check overlaps
                if not self._has_acceptable_overlap(bbox2, existing_bboxes, 0.22):
                    return []
                z_index = self._compute_z_index(bbox, existing_bboxes)
                bbox3 = [x3, y3, x3 + w3, y3 + h3]
                bbox3 = self._compute_actual_bbox(bbox3, w, h)
                bbox4 = [x4, y4, x4 + w4, y4 + h4]
                bbox4 = self._compute_actual_bbox(bbox4, w, h)
                existing_bboxes.append(bbox2)
                L_rect_bbox = self._combine_bboxes(bbox, bbox2)
                # existing_bboxes.append(L_rect_bbox)
                self.generate_rectangle(draw, bbox, value, filled=True)
                self.generate_rectangle(draw, bbox2, value, filled=True)
                return [(L_rect_bbox, shape_type.split('_')[0], value, z_index), (bbox, 'rectangle', value, z_index), (bbox2, 'rectangle', value, z_index), (bbox3, 'rectangle', bg_color, z_index), (bbox4, 'rectangle', bg_color, z_index)]

        except Exception as e:
            # If shape generation fails, return empty list
            return []

        return []
    
    def generate_sample(self):
        """Generate a single training sample"""
        # Random dimensions (with minimum size constraint)
        # w = random.randint(self.min_size, self.max_size)
        # h = random.randint(self.min_size, self.max_size)
        # less probability for smaller sizes
        size_probs = [i for i in range(2, self.max_size + 1)]
        size_probs = [p / sum(size_probs) for p in size_probs]
        w = np.random.choice(range(2, self.max_size + 1), p=size_probs)
        h = np.random.choice(range(2, self.max_size + 1), p=size_probs)
        w = int(w)
        h = int(h)
        # Background color (mostly 0, sometimes random)
        bg_color = 0 if random.random() > 0.08 else random.randint(*self.values_range)
        existing_colors = [bg_color]
        # Create image with PIL for easy drawing
        img = Image.new('L', (w, h), bg_color)
        draw = ImageDraw.Draw(img)
        
        # Convert to numpy for noise addition
        img_array = np.array(img, dtype=np.float32)

        # Add background noise 25% chance (subtle)
        if random.random() < 0.25 and w * h > 5:
            same_color = bool(random.random() < 0.5)
            noise, mask = self._generate_random_noise_matrix(w, h, bg_color, same_color)
            img_array = np.clip(img_array * mask + noise, 0, 9)
            # Round to nearest integer
            img_array = np.round(img_array)
        
        # Convert back to PIL for shape drawing
        img = Image.fromarray(img_array.astype(np.uint8))
        draw = ImageDraw.Draw(img)
        
        # Add random number of shapes (1-5) with overlap checking
        num_shapes = random.randint(1, max(1, min(5, (w * h) // 20)))
        shapes_data = []
        max_attempts = 3  # Maximum attempts to place each shape
        existing_bboxes = []
        for shape_idx in range(num_shapes):
            placed = False
            for _ in range(max_attempts):
                value_tuple_list = self.add_random_shape(img_array, draw, w, h, bg_color, existing_bboxes, existing_colors)
                for value_tuple in value_tuple_list:
                    if not value_tuple:
                        continue
                    bbox, shape_class, value, z_index = value_tuple
                    shapes_data.append({
                        'bbox': bbox,
                        'class': shape_class,
                        'color': value,
                        'z_index': z_index
                    })
                    placed = True
            if placed:
                break
            # If we couldn't place the shape after max_attempts, just skip it
        
        # Convert back to numpy
        img_array = np.array(img, dtype=np.float32)

        # Add final noise layer (20% chance)
        if random.random() < 0.20 and w * h > 5:
            if w * h <= 7:
                pass  # skip noise for very small images
            same_color = bool(random.random() < 0.5)
            noise, mask = self._generate_random_noise_matrix(w, h, bg_color, same_color)
            img_array = np.clip(img_array * mask + noise, 0, 9)
            # Round to nearest integer
            img_array = np.round(img_array)

        return img_array, shapes_data


# ============================================================================
# Dataset Generation for Action Prediction Model
# ============================================================================

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import os
from tqdm import tqdm

class ActionPredictionDatasetGenerator:
    """
    Generate training examples for predicting next editing actions.
    
    Each example consists of:
    - Input: (input_matrix, current_output_matrix, final_output_matrix)
    - Output: next_action_vector and progression (step number)
    """
    
    def __init__(self, shape_generator=None, max_matrix_size=30, color_swap_ratio=0.2):
        """
        Initialize the dataset generator.
        
        Args:
            shape_generator: ShapeGenerator instance for creating input matrices
            max_matrix_size: Maximum size for matrices (default 30x30)
            color_swap_ratio: Ratio of sequences to apply color swapping (default 0.2)
        """
        self.shape_generator = shape_generator if shape_generator else ShapeGenerator(max_size=max_matrix_size)
        self.max_matrix_size = max_matrix_size
        self.color_swap_ratio = color_swap_ratio
        
    def generate_training_example(self, sequence_length=None, use_random_input=True):
        """
        Generate a single training example.
        
        Args:
            sequence_length: Length of action sequence (random 3-15 if None)
            use_random_input: If True, use ShapeGenerator for input, else use random matrix
            
        Returns:
            List of dictionaries, each containing:
                - 'input_matrix': Input grid (padded to 30x30)
                - 'current_output': Current output state (padded to 30x30)
                - 'final_output': Final output state (padded to 30x30)
                - 'next_action': Action vector (31-dim) for next step
                - 'step': Step number in sequence
                - 'progression': Normalized step (step / total_steps)
        """
        if sequence_length is None:
            sequence_length = np.random.randint(3, 16)
        
        # Generate input matrix
        if use_random_input and np.random.rand() < 0.7:
            # Use ShapeGenerator 70% of the time
            input_matrix, _ = self.shape_generator.generate_sample()
        else:
            # Use random matrix 30% of the time
            input_matrix = generate_random_matrix()
        
        # Generate action sequence and output sequence
        initial_output = np.zeros((3, 3), dtype=np.int32)
        action_vectors, output_sequence = generate_random_action_sequence(
            input_matrix=input_matrix,
            output_matrix=initial_output,
            sequence_length=sequence_length,
            changes_required=True,
            max_attempts_per_action=200
        )
        
        # Apply color swapping for enrichment (20% of the time)
        apply_color_swap = np.random.rand() < self.color_swap_ratio
        if apply_color_swap:
            # Get color order from input matrix
            original_color_order = get_matrix_color_order(input_matrix)
            
            # Only apply color swap if there are colors to swap
            if len(original_color_order) > 0:
                # Swap colors in input matrix
                swap_result = swap_colors_in_matrix(input_matrix)
                if swap_result is not None:
                    swapped_input, old_color_order, new_color_order = swap_result
                    
                    # Swap colors in action vectors
                    swapped_action_vectors = swap_color_actions_in_sequence(
                        action_vectors, old_color_order, new_color_order
                    )
                    
                    # Regenerate output sequence with swapped actions
                    swapped_output_sequence = generate_output_sequence(
                        swapped_input, swapped_action_vectors
                    )
                    
                    # Use swapped versions
                    input_matrix = swapped_input
                    action_vectors = swapped_action_vectors
                    output_sequence = swapped_output_sequence
        
        # Pad final output to 30x30
        final_output = output_sequence[-1]
        final_output_padded = pad_matrix(final_output, self.max_matrix_size, self.max_matrix_size)
        
        # Pad input matrix to 30x30
        input_padded = pad_matrix(input_matrix, self.max_matrix_size, self.max_matrix_size)
        
        # Create training examples for each step
        training_examples = []
        for step in range(sequence_length):
            current_output = output_sequence[step]
            current_output_padded = pad_matrix(current_output, self.max_matrix_size, self.max_matrix_size)
            next_action = action_vectors[step]
            
            # Calculate progression (normalized step)
            progression = (step + 1) / sequence_length
            
            example = {
                'input_matrix': input_padded.astype(np.int32),
                'current_output': current_output_padded.astype(np.int32),
                'final_output': final_output_padded.astype(np.int32),
                'next_action': next_action.astype(np.float32),
                'step': step,
                'total_steps': sequence_length,
                'progression': progression
            }
            training_examples.append(example)
        
        return training_examples
    
    def generate_dataset(self, num_sequences=1000, min_seq_length=3, max_seq_length=15, 
                        use_random_input=True, verbose=True):
        """
        Generate a complete dataset of training examples.
        
        Args:
            num_sequences: Number of action sequences to generate
            min_seq_length: Minimum sequence length
            max_seq_length: Maximum sequence length
            use_random_input: Whether to use random input matrices
            verbose: Whether to show progress bar
            
        Returns:
            List of training examples
        """
        all_examples = []
        
        iterator = tqdm(range(num_sequences), desc="Generating sequences") if verbose else range(num_sequences)
        
        for _ in iterator:
            seq_length = np.random.randint(min_seq_length, max_seq_length + 1)
            try:
                examples = self.generate_training_example(
                    sequence_length=seq_length,
                    use_random_input=use_random_input
                )
                all_examples.extend(examples)
            except Exception as e:
                if verbose:
                    print(f"\nWarning: Failed to generate sequence: {e}")
                continue
        
        if verbose:
            print(f"\nGenerated {len(all_examples)} training examples from {num_sequences} sequences")
        
        return all_examples


class ActionPredictionDataset(Dataset):
    """
    PyTorch Dataset for action prediction training.
    """
    
    def __init__(self, examples=None, file_path=None):
        """
        Initialize dataset from examples or load from file.
        
        Args:
            examples: List of training examples
            file_path: Path to load dataset from (if examples is None)
        """
        if examples is not None:
            self.examples = examples
        elif file_path is not None:
            self.load(file_path)
        else:
            raise ValueError("Either examples or file_path must be provided")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx):
        """
        Get a single training example.
        
        Returns:
            Dictionary with:
                - 'input_matrix': Tensor [30, 30]
                - 'current_output': Tensor [30, 30]
                - 'final_output': Tensor [30, 30]
                - 'next_action': Tensor [31]
                - 'step': int
                - 'total_steps': int
                - 'progression': float (normalized step)
        """
        example = self.examples[idx]
        
        return {
            'input_matrix': torch.from_numpy(example['input_matrix']).long(),
            'current_output': torch.from_numpy(example['current_output']).long(),
            'final_output': torch.from_numpy(example['final_output']).long(),
            'next_action': torch.from_numpy(example['next_action']).float(),
            'step': example['step'],
            'total_steps': example['total_steps'],
            'progression': torch.tensor(example['progression'], dtype=torch.float32)
        }
    
    def save(self, file_path):
        """
        Save dataset to file.
        
        Args:
            file_path: Path to save dataset
        """
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.examples, f)
        print(f"Dataset saved to {file_path}")
    
    def load(self, file_path):
        """
        Load dataset from file.
        
        Args:
            file_path: Path to load dataset from
        """
        with open(file_path, 'rb') as f:
            self.examples = pickle.load(f)
        print(f"Dataset loaded from {file_path} ({len(self.examples)} examples)")
    
    def get_statistics(self):
        """
        Get statistics about the dataset.
        
        Returns:
            Dictionary with dataset statistics
        """
        action_types = []
        sequence_lengths = []
        
        for example in self.examples:
            # Decode action type from vector
            action_vector = example['next_action']
            action_type_idx = np.argmax(action_vector[0:9])
            if np.any(action_vector[0:9]):
                action_types.append(ACTION_TYPES[action_type_idx])
            else:
                action_types.append('default')
            sequence_lengths.append(example['total_steps'])
        
        from collections import Counter
        action_counts = Counter(action_types)
        
        stats = {
            'total_examples': len(self.examples),
            'action_distribution': dict(action_counts),
            'avg_sequence_length': np.mean(sequence_lengths),
            'min_sequence_length': np.min(sequence_lengths),
            'max_sequence_length': np.max(sequence_lengths)
        }
        
        return stats
    
    def print_statistics(self):
        """Print dataset statistics in a readable format."""
        stats = self.get_statistics()
        
        print("=" * 60)
        print("Dataset Statistics")
        print("=" * 60)
        print(f"Total examples: {stats['total_examples']}")
        print(f"Average sequence length: {stats['avg_sequence_length']:.2f}")
        print(f"Min sequence length: {stats['min_sequence_length']}")
        print(f"Max sequence length: {stats['max_sequence_length']}")
        print("\nAction Distribution:")
        for action, count in sorted(stats['action_distribution'].items(), key=lambda x: x[1], reverse=True):
            percentage = (count / stats['total_examples']) * 100
            print(f"  {action:15s}: {count:6d} ({percentage:5.2f}%)")
        print("=" * 60)


def create_dataloader(dataset, batch_size=32, shuffle=True, num_workers=0):
    """
    Create a DataLoader for the dataset.
    
    Args:
        dataset: ActionPredictionDataset instance
        batch_size: Batch size
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        DataLoader instance
    """
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )


# ============================================================================
# Helper Functions for Dataset Visualization
# ============================================================================

def visualize_training_example(example, show_action_details=True):
    """
    Visualize a single training example.
    
    Args:
        example: Dictionary with training example data
        show_action_details: Whether to print action details
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Convert tensors to numpy if necessary
    if isinstance(example['input_matrix'], torch.Tensor):
        input_matrix = example['input_matrix'].numpy()
        current_output = example['current_output'].numpy()
        final_output = example['final_output'].numpy()
        next_action = example['next_action'].numpy()
    else:
        input_matrix = example['input_matrix']
        current_output = example['current_output']
        final_output = example['final_output']
        next_action = example['next_action']
    
    # Decode action
    action_dict = vector_to_action(next_action)
    
    # Visualize matrices
    visualize_matrix(input_matrix, title=f"Input Matrix", ax=axes[0], pad_to_30=False)
    visualize_matrix(current_output, title=f"Current Output (Step {example['step']})", 
                    ax=axes[1], pad_to_30=False, action=action_dict)
    visualize_matrix(final_output, title=f"Final Output (Goal)", ax=axes[2], pad_to_30=False)
    
    plt.tight_layout()
    
    if show_action_details:
        print(f"\nStep {example['step'] + 1}/{example['total_steps']}")
        print(f"Progression: {example.get('progression', (example['step'] + 1) / example['total_steps']):.2%}")
        print(f"Next Action: {action_dict['action']}")
        if action_dict['action'] != 'default':
            print(f"  Position: ({action_dict['position']['x']}, {action_dict['position']['y']}) "
                  f"from {action_dict['position']['source']}")
            print(f"  Size: {action_dict['size']['width']}x{action_dict['size']['height']}")
            if action_dict['direction'] != 'default':
                print(f"  Direction: {action_dict['direction']}")
            if action_dict['color'] != -1:
                print(f"  Color: {action_dict['color']}")
            if action_dict['targetColor'] != -1:
                print(f"  Target Color: {action_dict['targetColor']}")
    
    plt.show()


# ============================================================================
# Example Usage
# ============================================================================

def generate_and_save_dataset_example(num_sequences=100, save_path='./datasets/action_prediction_dataset.pkl'):
    """
    Example function to generate and save a dataset.
    
    Args:
        num_sequences: Number of sequences to generate
        save_path: Path to save the dataset
    """
    print("Initializing dataset generator...")
    generator = ActionPredictionDatasetGenerator()
    
    print(f"Generating {num_sequences} sequences...")
    examples = generator.generate_dataset(
        num_sequences=num_sequences,
        min_seq_length=3,
        max_seq_length=15,
        use_random_input=True,
        verbose=True
    )
    
    print("\nCreating dataset...")
    dataset = ActionPredictionDataset(examples=examples)
    
    print("\nDataset statistics:")
    dataset.print_statistics()
    
    print(f"\nSaving dataset to {save_path}...")
    dataset.save(save_path)
    
    return dataset


def load_and_create_dataloader_example(load_path='./datasets/action_prediction_dataset.pkl', batch_size=32):
    """
    Example function to load a dataset and create a dataloader.
    
    Args:
        load_path: Path to load the dataset from
        batch_size: Batch size for dataloader
        
    Returns:
        DataLoader instance
    """
    print(f"Loading dataset from {load_path}...")
    dataset = ActionPredictionDataset(file_path=load_path)
    
    print("\nDataset statistics:")
    dataset.print_statistics()
    
    print(f"\nCreating dataloader with batch_size={batch_size}...")
    dataloader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
    
    print(f"Dataloader created with {len(dataloader)} batches")
    
    return dataloader, dataset


print("Dataset generation utilities loaded successfully!")
print("\nQuick start:")
print("1. Generate dataset: dataset = generate_and_save_dataset_example(num_sequences=100)")
print("2. Load dataset: dataloader, dataset = load_and_create_dataloader_example()")
print("3. Visualize example: visualize_training_example(dataset[0])")
print("\nNOTE: Dataset enrichment is enabled with 20% color swapping for data augmentation!")


# ============================================================================
# Advanced Usage: Generate Large Dataset and Create DataLoader
# ============================================================================

def generate_large_dataset(num_sequences=1000, batch_size=32, save_path='./datasets/large_dataset.pkl'):
    """
    Generate a large dataset for actual training.
    
    Args:
        num_sequences: Number of action sequences to generate
        batch_size: Batch size for the dataloader
        save_path: Where to save the dataset
    """
    print(f"Generating large dataset with {num_sequences} sequences...")
    print("This may take a few minutes...\n")
    
    # Initialize generator
    generator = ActionPredictionDatasetGenerator(max_matrix_size=30, color_swap_ratio=0.2)
    
    # Generate dataset
    examples = generator.generate_dataset(
        num_sequences=num_sequences,
        min_seq_length=3,
        max_seq_length=15,
        use_random_input=True,
        verbose=True
    )
    
    # Create dataset
    dataset = ActionPredictionDataset(examples=examples)
    
    # Print statistics
    print("\n")
    dataset.print_statistics()
    
    # Save dataset
    dataset.save(save_path)
    
    # Create dataloader
    print(f"\nCreating DataLoader with batch_size={batch_size}...")
    train_loader = create_dataloader(dataset, batch_size=batch_size, shuffle=True)
    
    # Test the dataloader
    print(f"\nDataLoader created successfully!")
    print(f"Number of batches: {len(train_loader)}")
    
    # Get a sample batch
    sample_batch = next(iter(train_loader))
    print(f"\nSample batch shapes:")
    print(f"  input_matrix: {sample_batch['input_matrix'].shape}")
    print(f"  current_output: {sample_batch['current_output'].shape}")
    print(f"  final_output: {sample_batch['final_output'].shape}")
    print(f"  next_action: {sample_batch['next_action'].shape}")
    
    return dataset, train_loader


def split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """
    Split dataset into train/val/test sets.
    
    Args:
        dataset: ActionPredictionDataset instance
        train_ratio: Ratio for training set
        val_ratio: Ratio for validation set
        test_ratio: Ratio for test set
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    total = len(dataset.examples)
    train_size = int(total * train_ratio)
    val_size = int(total * val_ratio)
    
    train_examples = dataset.examples[:train_size]
    val_examples = dataset.examples[train_size:train_size + val_size]
    test_examples = dataset.examples[train_size + val_size:]
    
    train_dataset = ActionPredictionDataset(examples=train_examples)
    val_dataset = ActionPredictionDataset(examples=val_examples)
    test_dataset = ActionPredictionDataset(examples=test_examples)
    
    print(f"Dataset split:")
    print(f"  Train: {len(train_dataset)} examples ({train_ratio*100:.1f}%)")
    print(f"  Val:   {len(val_dataset)} examples ({val_ratio*100:.1f}%)")
    print(f"  Test:  {len(test_dataset)} examples ({test_ratio*100:.1f}%)")
    
    return train_dataset, val_dataset, test_dataset


# Example: Uncomment to generate a large dataset
# dataset, train_loader = generate_large_dataset(num_sequences=1000, batch_size=32)

# Example: Split an existing dataset
# train_ds, val_ds, test_ds = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

print("Advanced dataset utilities loaded!")
print("\nTo generate a large dataset, run:")
print("  dataset, train_loader = generate_large_dataset(num_sequences=1000, batch_size=32)")
print("\nTo split a dataset, run:")
print("  train_ds, val_ds, test_ds = split_dataset(dataset)")


# ============================================================================
# Model Architecture for Action Prediction
# ============================================================================

import torch.nn as nn
import torch.nn.functional as F

class ActionPredictionModel(nn.Module):
    """
    Neural network for predicting next editing actions and progression.
    
    Takes three 30x30 matrices as input:
    - input_matrix: The original input
    - current_output: Current state of output
    - final_output: Target/goal state
    
    Outputs:
    - action_vector: 31-dimensional action vector
    - progression: Normalized step position in sequence (0 to 1)
    """
    
    def __init__(self, embedding_dim=64, hidden_dim=512, num_colors=11):
        """
        Initialize the model.
        
        Args:
            embedding_dim: Dimension for color embeddings
            hidden_dim: Hidden dimension for fully connected layers
            num_colors: Number of possible colors (11: -1 to 9)
        """
        super(ActionPredictionModel, self).__init__()
        
        # Color embedding layer (maps -1 to 9 -> 11 possible values)
        self.color_embedding = nn.Embedding(num_colors, embedding_dim)
        
        # Convolutional layers for each matrix
        self.conv1 = nn.Conv2d(embedding_dim, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.3)
        
        # Calculate flattened size after convolutions
        # After 3 pools: 30 -> 15 -> 7 -> 3, so final size is 3x3x256
        flattened_size = 3 * 3 * 256 * 3  # *3 because we have 3 input matrices
        
        # Shared fully connected layers
        self.fc1 = nn.Linear(flattened_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # Action prediction head
        self.fc_action = nn.Linear(hidden_dim // 2, 31)  # Output: 31-dim action vector
        
        # Progression prediction head
        self.fc_progression = nn.Linear(hidden_dim // 2, 1)  # Output: scalar progression value
        
        self.batch_norm1 = nn.BatchNorm1d(hidden_dim)
        self.batch_norm2 = nn.BatchNorm1d(hidden_dim // 2)
    
    def process_matrix(self, matrix):
        """
        Process a single matrix through conv layers.
        
        Args:
            matrix: [batch_size, 30, 30] with values -1 to 9
            
        Returns:
            Processed features [batch_size, 256, 3, 3]
        """
        # Shift values from [-1, 9] to [0, 10] for embedding
        matrix = matrix + 1 # Now in range [0, 10]
        
        # Embed colors: [batch_size, 30, 30] -> [batch_size, 30, 30, embedding_dim]
        embedded = self.color_embedding(matrix)
        
        # Transpose for conv2d: [batch_size, embedding_dim, 30, 30]
        embedded = embedded.permute(0, 3, 1, 2)
        
        # Apply convolutions with pooling
        x = F.relu(self.conv1(embedded))
        x = self.pool(x)  # [batch_size, 128, 15, 15]
        
        x = F.relu(self.conv2(x))
        x = self.pool(x)  # [batch_size, 256, 7, 7]
        
        x = F.relu(self.conv3(x))
        x = self.pool(x)  # [batch_size, 256, 3, 3]
        
        return x.contiguous()
    
    def forward(self, input_matrix, current_output, final_output):
        """
        Forward pass.
        
        Args:
            input_matrix: [batch_size, 30, 30]
            current_output: [batch_size, 30, 30]
            final_output: [batch_size, 30, 30]
            
        Returns:
            Tuple of:
                - action_vector: [batch_size, 31]
                - progression: [batch_size, 1] (normalized step position 0-1)
        """
        # Process each matrix
        input_features = self.process_matrix(input_matrix)
        current_features = self.process_matrix(current_output)
        final_features = self.process_matrix(final_output)
        
        # Flatten and concatenate
        input_flat = input_features.view(input_features.size(0), -1)
        current_flat = current_features.view(current_features.size(0), -1)
        final_flat = final_features.view(final_features.size(0), -1)
        
        combined = torch.cat([input_flat, current_flat, final_flat], dim=1)
        
        # Shared fully connected layers
        x = F.relu(self.batch_norm1(self.fc1(combined)))
        x = self.dropout(x)
        
        x = F.relu(self.batch_norm2(self.fc2(x)))
        x = self.dropout(x)
        
        # Action prediction head
        action_vector = self.fc_action(x)
        
        # Progression prediction head
        progression = torch.sigmoid(self.fc_progression(x))  # Sigmoid to bound [0, 1]
        
        return action_vector, progression


def custom_action_loss(predicted_action, target_action, predicted_progression=None, target_progression=None):
    """
    Custom loss function for action and progression prediction.
    
    Combines:
    - BCE loss for action type (one-hot, indices 0-8)
    - MSE loss for continuous values (positions, sizes, colors)
    - BCE loss for direction (one-hot, indices 21-30)
    - MSE loss for progression (if provided)
    
    Args:
        predicted_action: [batch_size, 31]
        target_action: [batch_size, 31]
        predicted_progression: [batch_size, 1] (optional)
        target_progression: [batch_size, 1] or [batch_size] (optional)
        
    Returns:
        Total loss (scalar)
    """
    # Action type loss (indices 0-8) - BCE with logits
    action_type_pred = predicted_action[:, 0:9]
    action_type_target = target_action[:, 0:9]
    action_type_loss = F.binary_cross_entropy_with_logits(action_type_pred, action_type_target)
    
    # Continuous values loss (indices 9-20) - MSE
    continuous_pred = predicted_action[:, 9:21]
    continuous_target = target_action[:, 9:21]
    continuous_loss = F.mse_loss(continuous_pred, continuous_target)
    
    # Direction loss (indices 21-30) - BCE with logits
    direction_pred = predicted_action[:, 21:31]
    direction_target = target_action[:, 21:31]
    direction_loss = F.binary_cross_entropy_with_logits(direction_pred, direction_target)
    
    # Weighted combination
    total_loss = 2.0 * action_type_loss + continuous_loss + 1.5 * direction_loss
    
    # Add progression loss if provided
    if predicted_progression is not None and target_progression is not None:
        # Ensure target_progression has correct shape
        if target_progression.dim() == 1:
            target_progression = target_progression.unsqueeze(1)
        
        progression_loss = F.mse_loss(predicted_progression, target_progression)
        total_loss = total_loss + 0.5 * progression_loss  # Weight for progression loss
    
    return total_loss


# Example training loop
def train_epoch(model, dataloader, optimizer, device='cpu'):
    """
    Train for one epoch.
    
    Args:
        model: ActionPredictionModel instance
        dataloader: DataLoader with training data
        optimizer: PyTorch optimizer
        device: 'cpu' or 'cuda'
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    total_loss = 0.0
    
    for batch in dataloader:
        # Move data to device
        input_matrix = batch['input_matrix'].to(device)
        current_output = batch['current_output'].to(device)
        final_output = batch['final_output'].to(device)
        next_action = batch['next_action'].to(device)
        progression = batch['progression'].to(device)
        
        # Forward pass
        predicted_action, predicted_progression = model(input_matrix, current_output, final_output)
        
        # Compute loss
        loss = custom_action_loss(predicted_action, next_action, 
                                 predicted_progression, progression)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


# Example usage
print("Example Model Architecture Loaded!")
print("\nModel now predicts both ACTION and PROGRESSION!")
print("\nTo use the model:")
print("1. Create model: model = ActionPredictionModel()")
print("2. Create optimizer: optimizer = torch.optim.Adam(model.parameters(), lr=0.001)")
print("3. Train: loss = train_epoch(model, train_loader, optimizer)")
print("4. Predict: action_vector, progression = model(input_m, current_m, final_m)")
print("\nModel architecture:")
model = ActionPredictionModel()
print(model)
print(f"\nTotal parameters: {sum(p.numel() for p in model.parameters()):,}")


# Step 1: Generate dataset with color swapping augmentation
dataset, train_loader = generate_large_dataset(
    num_sequences=25000,
    batch_size=256,
    save_path='./datasets/my_dataset.pkl'
)

# Step 2: Split dataset
train_ds, val_ds, test_ds = split_dataset(dataset, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)

# Step 3: Create dataloaders
train_loader = create_dataloader(train_ds, batch_size=512, shuffle=True)
val_loader = create_dataloader(val_ds, batch_size=512, shuffle=False)

# Step 4: Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ActionPredictionModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print(f"Training on device: {device}")
print(f"Model predicts: ACTION (31-dim) + PROGRESSION (1-dim)")
print(f"Dataset includes 20% color-swapped sequences for augmentation")

# Step 5: Training loop
num_epochs = 200
for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {train_loss:.4f}")

# Step 6: Visualize predictions
model.eval()
with torch.no_grad():
    sample = dataset[0]
    input_m = sample['input_matrix'].unsqueeze(0).to(device)
    current_m = sample['current_output'].unsqueeze(0).to(device)
    final_m = sample['final_output'].unsqueeze(0).to(device)
    
    predicted_action, predicted_progression = model(input_m, current_m, final_m)
    predicted_action_dict = vector_to_action(predicted_action.cpu().numpy()[0])
    true_action_dict = vector_to_action(sample['next_action'].numpy())
    
    print("\n" + "="*60)
    print("PREDICTION COMPARISON")
    print("="*60)
    print(f"Predicted progression: {predicted_progression.item():.4f}")
    print(f"True progression: {sample['progression'].item():.4f}")
    print(f"Progression error: {abs(predicted_progression.item() - sample['progression'].item()):.4f}")
    print(f"\nPredicted action: {predicted_action_dict['action']}")
    print(f"True action: {true_action_dict['action']}")
    print("="*60)


def validate(model, val_loader, device='cpu'):
    """
    Validate the model on the validation set.

    Args:
        model: ActionPredictionModel instance
        val_loader: DataLoader with validation data
        device: 'cpu' or 'cuda'

    Returns:
        Average loss on the validation set
    """
    model.eval()
    correct_actions = 0
    total_actions = 0
    total_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            # Move data to device
            input_matrix = batch['input_matrix'].to(device)
            current_output = batch['current_output'].to(device)
            final_output = batch['final_output'].to(device)
            next_action = batch['next_action'].to(device)
            progression = batch['progression'].to(device)
            # Forward pass
            predicted_action, predicted_progression = model(input_matrix, current_output, final_output)
            predicted_action_dicts = [parse_action_tensor_to_dict(vec) for vec in predicted_action.to('cpu')]
            target_action_dicts = [vector_to_action(vec) for vec in next_action.to('cpu').numpy()]
            # accumulate accuracy metrics
            for pred_dict, target_dict in zip(predicted_action_dicts, target_action_dicts):
                if pred_dict['action'] == target_dict['action']:
                    correct_actions += 1
                total_actions += 1
            # Compute loss
            loss = custom_action_loss(predicted_action, next_action, 
                                     predicted_progression, progression)
            total_loss += loss.item()
    accuracy = correct_actions / total_actions if total_actions > 0 else 0.0
    avg_loss = total_loss / len(val_loader)
    print(f"Validation Accuracy: {accuracy*100:.2f}%, Average Loss: {avg_loss:.4f}")

validate(model, val_loader, device)


def compare_np_arrays(arr1, arr2) -> bool:
    """
    Compare two numpy arrays and print differences.

    Args:
        arr1: First numpy array
        arr2: Second numpy array
    """
    if arr1.shape != arr2.shape:
        return False
    differences = np.where(arr1 != arr2)
    if len(differences[0]) == 0:
        return True
    else:
        return False
    
def compare_tensors(tensor1, tensor2) -> bool:
    """
    Compare two PyTorch tensors and print differences.

    Args:
        tensor1: First tensor
        tensor2: Second tensor
    """
    if tensor1.size() != tensor2.size():
        return False
    differences = (tensor1 != tensor2).nonzero(as_tuple=False)
    if differences.size(0) == 0:
        return True
    else:
        return False


base_path='/kaggle/input/arc-prize-2025/'
def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data

test_challenges = load_json(base_path +'arc-agi_test_challenges.json')
test_challenges = list(test_challenges.items())
submission = {}
for key, task in test_challenges:
    examples = task['train']
    tests = task['test']
    action_sequences_scores = []
    all_action_sequences = []
    color_order_list = []
    for example in examples:
        input_matrix = np.array(example['input'])
        color_order_list.append(get_matrix_color_order(input_matrix))
        output_matrix = np.array(example['output'])
        current_output_matrix = np.zeros((3, 3), dtype=np.int32)
        padded_input = pad_matrix(input_matrix)
        padded_output = pad_matrix(output_matrix)
        padded_initial_output = pad_matrix(current_output_matrix)
        input = torch.tensor(padded_input).unsqueeze(0).to(device)
        final_output = torch.tensor(padded_output).unsqueeze(0).to(device)
        current_output = torch.tensor(padded_initial_output).unsqueeze(0).to(device)
        action_sequence = []
        for i in range(16):
            predicted_action, predicted_progression = model(input, current_output, final_output)
            predicted_action = predicted_action.squeeze(0).detach().cpu()
            predicted_progression = predicted_progression.squeeze(0).detach().cpu()
            action = parse_action_tensor_to_dict(predicted_action)
            action_vector = action_to_vector(action)
            action_sequence.append(action_vector)
            current_output_matrix = edit_grid(input_matrix, current_output_matrix, action_vector)
            current_output_padded = pad_matrix(current_output_matrix)
            current_output = torch.tensor(current_output_padded).unsqueeze(0).to(device)
            if compare_np_arrays(current_output_matrix, output_matrix) or abs(predicted_progression.item() - 1.0) < 0.01 or i == 15:
                action_sequences_scores.append(1 if compare_np_arrays(current_output_matrix, output_matrix) else predicted_progression.item())
                break
        all_action_sequences.append(action_sequence)
    solutions = []
    for test in tests:
        input_matrix = np.array(test['input'])
        test_color_order = get_matrix_color_order(input_matrix)
        attempts = {}
        n_attempts = 2
        sorted_action_sequence_scores = np.argsort(action_sequences_scores)[::-1]
        sorted_action_sequences = [all_action_sequences[idx] for idx in sorted_action_sequence_scores][:n_attempts]
        sorted_color_orders = [color_order_list[idx] for idx in sorted_action_sequence_scores][:n_attempts]
        for i in range(n_attempts):
            action_sequence = sorted_action_sequences[i]
            example_color_order = sorted_color_orders[i]
            current_output_matrix = np.zeros((3, 3), dtype=np.int32)
            action_sequence = swap_color_actions_in_sequence(action_sequence, example_color_order, test_color_order)
            for action_vector in action_sequence:
                current_output_matrix = edit_grid(input_matrix, current_output_matrix, action_vector)
            attempt_key = 'attempt_' + str(i+1)
            attempts[attempt_key] = current_output_matrix.tolist()
        solutions.append(attempts)
    submission[key] = solutions
    print(f"Processed task {key}")
json_output_path = './submission.json'
with open(json_output_path, 'w') as f:
    json.dump(submission, f)

