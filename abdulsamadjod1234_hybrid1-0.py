# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


import numpy as np
import json
import logging
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
import os
import time
from scipy.ndimage import zoom, label, find_objects
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Input, UpSampling2D, Concatenate, Softmax, MaxPooling2D, Conv2DTranspose
import networkx as nx
from scipy.spatial import ConvexHull
from sklearn.metrics import jaccard_score  # For IoU

# Configuration
DEBUG_MODE = True
MAX_GRID_SIZE = 30
TRAIN_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
TEST_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
SUBMISSION_PATH = "/kaggle/working/submission.json"

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

# Logging setup
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

# Utility Functions
def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def force_same_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr.copy()
    if arr.size == 0:
        return np.ones(target_shape, dtype=int) * 1
    H, W = target_shape
    h, w = arr.shape
    result = np.ones(target_shape, dtype=int) * 1
    h_min, w_min = min(H, h), min(W, w)
    y0, x0 = (H - h_min) // 2, (W - w_min) // 2
    ys, xs = (h - h_min) // 2, (w - w_min) // 2
    result[y0:y0+h_min, x0:x0+w_min] = arr[ys:ys+h_min, xs:xs+w_min]
    return result

def force_grid_cap(arr: np.ndarray) -> np.ndarray:
    H, W = arr.shape
    if H <= MAX_GRID_SIZE and W <= MAX_GRID_SIZE:
        return arr
    return force_same_shape(arr, (min(H, MAX_GRID_SIZE), min(W, MAX_GRID_SIZE)))

def detect_background(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    return int(np.argmax(np.bincount(grid.flatten(), minlength=10)))

def load_json_file(json_path: str) -> Dict:
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return {}

# Advanced Rule Solver with Improvements
class AdvancedRuleSolver:
    def __init__(self):
        self.task_patterns = defaultdict(dict)
        self.global_cnn = self.build_unet_model()  # Global pretrained CNN
        self.transformations = [
            lambda x: x, lambda x: np.rot90(x, 1), lambda x: np.rot90(x, 2), lambda x: np.rot90(x, 3), np.fliplr, np.flipud
        ]
        self.stats = {'accuracy': [], 'feature_stats': [], 'iou_scores': []}  # For diagnostics

    # Improved CNN: Deeper UNet with downsampling and residual blocks
    def build_unet_model(self):
        def residual_block(x, filters):
            shortcut = Conv2D(filters, (1, 1), padding='same')(x)
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = Conv2D(filters, (3, 3), padding='same')(x)
            x = tf.keras.layers.BatchNormalization()(x)
            x = tf.keras.layers.ReLU()(x)
            x = tf.keras.layers.Add()([x, shortcut])
            return x

        input_layer = Input(shape=(30, 30, 1))
        # Encoder with downsampling
        e1 = residual_block(input_layer, 32)
        p1 = MaxPooling2D((2, 2))(e1)
        e2 = residual_block(p1, 64)
        p2 = MaxPooling2D((2, 2))(e2)
        e3 = residual_block(p2, 128)
        # Decoder with upsampling
        d1 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(e3)
        d1 = Concatenate()([d1, e2])
        d1 = residual_block(d1, 64)
        d2 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(d1)
        d2 = Concatenate()([d2, e1])
        d2 = residual_block(d2, 32)
        output = Conv2D(10, (1, 1), activation='softmax', padding='same')(d2)  # Per-pixel softmax over 10 colors
        model = Model(inputs=input_layer, outputs=output)
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')
        return model

    # Better Feature Extraction
    def extract_features(self, grid: np.ndarray) -> dict:
        # Graph features
        G = nx.Graph()
        rows, cols = grid.shape
        for i in range(rows):
            for j in range(cols):
                G.add_node((i, j), color=grid[i, j])
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    if 0 <= ni < rows and 0 <= nj < cols and grid[i, j] == grid[ni, nj]:
                        G.add_edge((i, j), (ni, nj))
        graph_features = {'num_components': nx.number_connected_components(G), 'avg_degree': np.mean([d for n, d in G.degree()]) if G.number_of_nodes() > 0 else 0}

        # Symmetry detectors
        symmetry = {
            'horizontal': np.array_equal(grid, np.flipud(grid)),
            'vertical': np.array_equal(grid, np.fliplr(grid)),
            'rot90': np.array_equal(grid, np.rot90(grid)),
            'rot180': np.array_equal(grid, np.rot90(grid, 2)),
            'rot270': np.array_equal(grid, np.rot90(grid, 3)),
            'diagonal': np.array_equal(grid, grid.T)
        }

        # Shape descriptors
        bg = detect_background(grid)
        mask = grid != bg
        coords = np.argwhere(mask)
        if coords.size == 0:
            shape_features = {'bbox': (0,0,0,0), 'centroid': (0,0), 'hull_area': 0}
        else:
            min_r, min_c = coords.min(0)
            max_r, max_c = coords.max(0)
            centroid = tuple(coords.mean(0).astype(int))
            hull = ConvexHull(coords)
            hull_points = coords[hull.vertices]
            x = hull_points[:, 0]
            y = hull_points[:, 1]
            hull_area = 0.5 * np.abs(np.dot(x, np.roll(y,1)) - np.dot(y, np.roll(x,1)))
            shape_features = {'bbox': (min_r, min_c, max_r, max_c), 'centroid': centroid, 'hull_area': hull_area}
        
        features = {**graph_features, **symmetry, **shape_features}
        self.stats['feature_stats'].append(features)
        return features

    # Improved Rule Mining
    def apply_rules(self, input_grid: np.ndarray, task_id: str) -> np.ndarray:
        if task_id not in self.task_patterns:
            return input_grid
        patterns = self.task_patterns[task_id]

        # Improved Color mapping: Object-wise
        result = input_grid.copy()
        bg = detect_background(result)
        labeled, num_objects = label(result != bg)
        for obj_id in range(1, num_objects + 1):
            obj_mask = labeled == obj_id
            obj_colors = result[obj_mask]
            obj_old_color = Counter(obj_colors).most_common(1)[0][0]
            obj_new_color = patterns['color_mappings'].get(obj_old_color, Counter()).most_common(1)
            if obj_new_color:
                result[obj_mask] = obj_new_color[0][0]

        # Object copy/translation
        features = self.extract_features(input_grid)
        if features['num_components'] > 1:
            bbox = features['bbox']
            shift = (bbox[2] - bbox[0], bbox[3] - bbox[1])
            result = np.roll(result, shift, axis=(0, 1))

        # Improved Repetition / tiling: Autocorrelation
        autocorrelation = np.abs(np.fft.ifft2(np.fft.fft2(result) * np.conj(np.fft.fft2(result))))
        autocorrelation = autocorrelation / np.max(autocorrelation)
        repetition_period = np.unravel_index(np.argmax(autocorrelation[1:, 1:]), autocorrelation.shape)
        if repetition_period[0] > 0 and repetition_period[1] > 0:
            tile_size = repetition_period
            result = np.tile(result[:tile_size[0], :tile_size[1]], (result.shape[0] // tile_size[0] + 1, result.shape[1] // tile_size[1] + 1))[:result.shape[0], :result.shape[1]]

        return result

    def learn_task(self, task_id: str, train_examples: List[Dict]):
        patterns = {
            'input_shapes': Counter(),
            'output_shapes': Counter(),
            'color_mappings': defaultdict(Counter),
            'transformations': Counter(),
            'patterns': [],
            'repetition': False, 'tile_size': (1,1),
            'background_color': 0
        }
        X_train, y_train = [], []
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            patterns['input_shapes'][input_grid.shape] += 1
            patterns['output_shapes'][output_grid.shape] += 1
            # Object-wise color mapping
            bg_in = detect_background(input_grid)
            bg_out = detect_background(output_grid)
            patterns['background_color'] = bg_out
            labeled_in, num_in = label(input_grid != bg_in)
            labeled_out, labeled_out_num = label(output_grid != bg_out)
            if num_in == labeled_out_num:
                for obj_id in range(1, num_in + 1):
                    obj_mask_in = labeled_in == obj_id
                    obj_mask_out = labeled_out == obj_id
                    obj_old_color = Counter(input_grid[obj_mask_in]).most_common(1)[0][0]
                    obj_new_color = Counter(output_grid[obj_mask_out]).most_common(1)[0][0]
                    patterns['color_mappings'][obj_old_color][obj_new_color] += 1
            for idx, transform in enumerate(self.transformations):
                transformed = transform(input_grid)
                if transformed.shape == output_grid.shape and np.array_equal(transformed, output_grid):
                    patterns['transformations'][f'transform_{idx}'] += 1
            patterns['patterns'].append(self.extract_features(input_grid))
            # Check repetition with autocorrelation
            autocorrelation = np.abs(np.fft.ifft2(np.fft.fft2(input_grid) * np.conj(np.fft.fft2(input_grid))))
            autocorrelation = autocorrelation / np.max(autocorrelation)
            repetition_period = np.unravel_index(np.argmax(autocorrelation[1:, 1:]), autocorrelation.shape)
            if repetition_period[0] > 0 and repetition_period[1] > 0:
                patterns['repetition'] = True
                patterns['tile_size'] = repetition_period

            padded_input = np.pad(input_grid, ((0, max(0, 30-input_grid.shape[0])), (0, max(0, 30-input_grid.shape[1]))), mode='constant')[:30, :30]
            padded_output = np.pad(output_grid, ((0, max(0, 30-output_grid.shape[0])), (0, max(0, 30-output_grid.shape[1]))), mode='constant')[:30, :30]
            X_train.append(padded_input[..., np.newaxis])
            y_train.append(tf.one_hot(padded_output, depth=10))  # One-hot for softmax

        self.task_patterns[task_id] = patterns
        if X_train:
            X_train = np.array(X_train)
            y_train = np.array(y_train)
            # Finetune global CNN per task for meta-learning
            self.unet_model.fit(X_train, y_train, epochs=10, verbose=0)  # More epochs for better generalization

    def fix_zero_predictions(self, grid: np.ndarray, input_grid: np.ndarray, task_id: str) -> np.ndarray:
        if grid.size == 0:
            return np.array([[1]])
        zero_count = np.sum(grid == 0)
        if zero_count / grid.size > 0.5:
            result = grid.copy()
            bg_color = self.task_patterns[task_id].get('background_color', 1)
            result[result == 0] = bg_color
            return result
        return grid

    def score_prediction(self, candidate: np.ndarray, input_grid: np.ndarray, task_id: str) -> float:
        score = 0.5
        if np.all(candidate == candidate[0, 0]) and not np.all(input_grid == input_grid[0, 0]):
            score -= 0.3
        if task_id in self.task_patterns:
            colors = set(np.unique(candidate))
            train_colors = set(self.task_patterns[task_id]['color_mappings'].keys())
            score += 0.1 * len(colors.intersection(train_colors))
        if not np.array_equal(candidate, input_grid):
            score += 0.2
        # Improved scoring
        candidate_features = self.extract_features(candidate)
        input_features = self.extract_features(input_grid)
        # IoU of objects
        bg_c = detect_background(candidate)
        bg_i = detect_background(input_grid)
        mask_c = candidate != bg_c
        mask_i = input_grid != bg_i
        iou = jaccard_score(mask_c.flatten(), mask_i.flatten())
        score += 0.2 * iou
        # Symmetry match
        sym_match = sum(1 for k in ['horizontal', 'vertical', 'rot90', 'rot180', 'rot270', 'diagonal'] if candidate_features[k] == input_features[k]) / 6
        score += 0.1 * sym_match
        # Component count match
        comp_match = 1 if candidate_features['num_components'] == input_features['num_components'] else 0
        score += 0.1 * comp_match
        # Diagnostic: Log IoU
        self.stats['iou_scores'].append(iou)
        logger.debug(f"IoU score: {iou}")
        return max(0.1, min(1.0, score))

    def predict(self, input_grid: np.ndarray, task_id: str) -> Tuple[np.ndarray, np.ndarray]:
        if input_grid.size == 0:
            return np.array([[1]]), np.array([[2]])
        target_shape = self.task_patterns[task_id]['output_shapes'].most_common(1)[0][0] if task_id in self.task_patterns else input_grid.shape
        candidates = []

        # Attempt 1: Rule-based prediction
        attempt1 = self.apply_rules(input_grid, task_id)
        attempt1 = force_same_shape(attempt1, target_shape)
        attempt1 = self.fix_zero_predictions(attempt1, input_grid, task_id)
        candidates.append((attempt1, self.score_prediction(attempt1, input_grid, task_id)))

        # Attempt 2: UNet prediction
        padded_input = np.pad(input_grid, ((0, max(0, 30-input_grid.shape[0])), (0, max(0, 30-input_grid.shape[1]))), mode='constant')[:30, :30][..., np.newaxis]
        unet_pred = self.unet_model.predict(padded_input[np.newaxis, ...], verbose=0)
        unet_pred = np.argmax(unet_pred, axis=-1)[0]
        attempt2 = force_same_shape(unet_pred.astype(int), target_shape)
        attempt2 = self.fix_zero_predictions(attempt2, input_grid, task_id)
        candidates.append((attempt2, self.score_prediction(attempt2, input_grid, task_id)))

        # Improved Ensemble: Generate diverse candidates
        for transform in self.transformations:
            diverse = transform(input_grid)
            diverse = self.apply_rules(diverse, task_id)
            diverse = force_same_shape(diverse, target_shape)
            diverse = self.fix_zero_predictions(diverse, input_grid, task_id)
            score = self.score_prediction(diverse, input_grid, task_id)
            candidates.append((diverse, score))

        # Select best and diverse attempts
        candidates.sort(key=lambda x: x[1], reverse=True)
        attempt1 = candidates[0][0]
        attempt2 = None
        for candidate, _ in candidates[1:]:
            if not np.array_equal(candidate, attempt1):
                attempt2 = candidate
                break
        if attempt2 is None:
            attempt2 = np.ones_like(input_grid) * 2

        # Diagnostics: Log stats
        self.stats['accuracy'].append(jaccard_score(input_grid.flatten() > 0, attempt1.flatten() > 0))
        logger.info(f"Average accuracy so far: {np.mean(self.stats['accuracy']):.2f}")
        logger.info(f"Average IoU so far: {np.mean(self.stats['iou_scores']):.2f}")

        if DEBUG_MODE:
            logger.debug(f"Task {task_id}: Input shape={input_grid.shape}, Attempt1 shape={attempt1.shape}, Attempt2 shape={attempt2.shape}")
            logger.debug(f"Attempt1:\n{attempt1}\nAttempt2:\n{attempt2}")

        return force_grid_cap(attempt1), force_grid_cap(attempt2)

def train_on_dataset(solver: AdvancedRuleSolver, train_json_path: str):
    train_data = load_json_file(train_json_path)
    if not train_data:
        logger.error("No training data found!")
        return
    for task_id, task_info in train_data.items():
        train_examples = task_info.get("train", [])
        if train_examples:
            solver.learn_task(task_id, train_examples)
            logger.info(f"Trained on task {task_id} with {len(train_examples)} examples")

def submission_writer(predictions_dict: Dict, filename: str = SUBMISSION_PATH) -> bool:
    try:
        formatted_predictions = {}
        for task_id, predictions_list in predictions_dict.items():
            formatted_predictions[task_id] = []
            for attempt1, attempt2 in predictions_list:
                formatted_predictions[task_id].append({
                    "attempt_1": to_serializable(attempt1),
                    "attempt_2": to_serializable(attempt2)
                })
        with open(filename, "w") as f:
            json.dump(formatted_predictions, f, indent=2)
        logger.info(f"âœ… Submission saved: {filename}")
        return True
    except Exception as e:
        logger.error(f"â�Œ Submission write error: {e}")
        return False

def process_test_file(test_json_path: str, submission_path: str) -> bool:
    start_time = time.time()
    try:
        test_data = load_json_file(test_json_path)
        if not test_data:
            logger.error("No test data found!")
            return False
        solver = AdvancedRuleSolver()
        train_on_dataset(solver, TRAIN_JSON_PATH)
        all_predictions = {}
        for task_id, task_info in test_data.items():
            train_examples = task_info.get("train", [])
            test_examples = task_info.get("test", [])
            logger.info(f"Processing task {task_id}: {len(train_examples)} train, {len(test_examples)} test")
            if train_examples:
                solver.learn_task(task_id, train_examples)
            task_predictions = []
            for test_ex in test_examples:
                input_grid = np.array(test_ex["input"])
                pred1, pred2 = solver.predict(input_grid, task_id)
                task_predictions.append((pred1, pred2))
            all_predictions[task_id] = task_predictions
        success = submission_writer(all_predictions, submission_path)
        logger.info(f"Runtime: {time.time() - start_time:.2f}s")
        return success
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting ARC Solver...")
    if not os.path.exists(TEST_JSON_PATH):
        logger.error(f"Test file missing: {TEST_JSON_PATH}")
        os.exit(1)
    success = process_test_file(TEST_JSON_PATH, SUBMISSION_PATH)
    if success:
        logger.info("ğŸ�‰ Submission created!")
    else:
        logger.error("â�Œ Submission failed")
        #cell1


import numpy as np
import json
import logging
from typing import List, Dict, Tuple, Any
from collections import defaultdict, Counter
import os
from scipy.ndimage import label, find_objects
from sklearn.metrics import jaccard_score

# Configuration
DEBUG_MODE = True
MAX_GRID_SIZE = 30
TRAIN_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
TEST_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
SUBMISSION_PATH = "/kaggle/working/submission.json"

# Set random seeds for reproducibility
np.random.seed(42)

# Logging setup
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

# Utility Functions
def to_serializable(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    return obj

def force_same_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Resize array to target shape by center-cropping/padding"""
    if arr.shape == target_shape:
        return arr.copy()
    if arr.size == 0:
        return np.ones(target_shape, dtype=int) * 1
    
    H, W = target_shape
    h, w = arr.shape
    result = np.ones(target_shape, dtype=int) * 1
    h_min, w_min = min(H, h), min(W, w)
    y0, x0 = (H - h_min) // 2, (W - w_min) // 2
    ys, xs = (h - h_min) // 2, (w - w_min) // 2
    result[y0:y0+h_min, x0:x0+w_min] = arr[ys:ys+h_min, xs:xs+w_min]
    return result

def force_grid_cap(arr: np.ndarray) -> np.ndarray:
    """Ensure grid doesn't exceed maximum size"""
    H, W = arr.shape
    if H <= MAX_GRID_SIZE and W <= MAX_GRID_SIZE:
        return arr
    return force_same_shape(arr, (min(H, MAX_GRID_SIZE), min(W, MAX_GRID_SIZE)))

def detect_background(grid: np.ndarray) -> int:
    """Detect the most common color (background)"""
    if grid.size == 0:
        return 0
    return int(np.argmax(np.bincount(grid.flatten(), minlength=10)))

def load_json_file(json_path: str) -> Dict:
    """Load JSON data from file"""
    try:
        with open(json_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Error loading JSON: {e}")
        return {}

# Advanced Rule Solver
class AdvancedRuleSolver:
    def __init__(self):
        self.task_patterns = defaultdict(dict)
        self.transformations = [
            lambda x: x, 
            lambda x: np.rot90(x, 1), 
            lambda x: np.rot90(x, 2), 
            lambda x: np.rot90(x, 3), 
            np.fliplr, 
            np.flipud
        ]
        self.stats = {'accuracy': [], 'iou_scores': []}

    def extract_features(self, grid: np.ndarray) -> dict:
        """Extract features from grid for pattern recognition"""
        features = {}
        
        # Basic shape features
        features['shape'] = grid.shape
        features['size'] = grid.size
        
        # Color features
        unique, counts = np.unique(grid, return_counts=True)
        features['color_distribution'] = dict(zip(unique.tolist(), counts.tolist()))
        features['background_color'] = detect_background(grid)
        
        # Symmetry features
        features['horizontal_symmetry'] = np.array_equal(grid, np.flipud(grid))
        features['vertical_symmetry'] = np.array_equal(grid, np.fliplr(grid))
        features['rotational_symmetry'] = np.array_equal(grid, np.rot90(grid, 2))
        
        # Object detection
        bg = detect_background(grid)
        mask = grid != bg
        labeled, num_objects = label(mask)
        features['num_objects'] = num_objects
        
        # Object features
        if num_objects > 0:
            object_sizes = []
            for i in range(1, num_objects + 1):
                object_mask = labeled == i
                object_sizes.append(np.sum(object_mask))
            features['object_sizes'] = object_sizes
            features['avg_object_size'] = np.mean(object_sizes) if object_sizes else 0
        
        return features

    def apply_rules(self, input_grid: np.ndarray, task_id: str) -> np.ndarray:
        """Apply learned rules to transform input grid"""
        if task_id not in self.task_patterns:
            return input_grid.copy()
            
        patterns = self.task_patterns[task_id]
        result = input_grid.copy()
        
        # Apply color mapping if learned
        if 'color_mappings' in patterns:
            for old_color, new_color in patterns['color_mappings'].items():
                result[input_grid == old_color] = new_color
        
        # Apply shape transformation if output shape is different
        if 'output_shape' in patterns and input_grid.shape != patterns['output_shape']:
            result = force_same_shape(result, patterns['output_shape'])
        
        # Apply common transformations
        if 'common_transformation' in patterns:
            transform_idx = patterns['common_transformation']
            if transform_idx < len(self.transformations):
                result = self.transformations[transform_idx](result)
        
        return result

    def learn_task(self, task_id: str, train_examples: List[Dict]):
        """Learn patterns from training examples"""
        patterns = {
            'input_shapes': Counter(),
            'output_shapes': Counter(),
            'color_mappings': defaultdict(Counter),
            'transformations': Counter(),
        }
        
        for ex in train_examples:
            input_grid = np.array(ex['input'])
            output_grid = np.array(ex['output'])
            
            # Record shapes
            patterns['input_shapes'][input_grid.shape] += 1
            patterns['output_shapes'][output_grid.shape] += 1
            
            # Learn color mappings
            input_unique = np.unique(input_grid)
            output_unique = np.unique(output_grid)
            
            # Simple 1:1 color mapping
            if len(input_unique) == len(output_unique):
                for in_color, out_color in zip(input_unique, output_unique):
                    patterns['color_mappings'][in_color][out_color] += 1
            
            # Learn transformations
            for idx, transform in enumerate(self.transformations):
                transformed = transform(input_grid)
                if np.array_equal(transformed, output_grid):
                    patterns['transformations'][idx] += 1
        
        # Determine most common patterns
        if patterns['output_shapes']:
            patterns['output_shape'] = patterns['output_shapes'].most_common(1)[0][0]
        
        if patterns['color_mappings']:
            patterns['color_mappings'] = {
                color: counter.most_common(1)[0][0] 
                for color, counter in patterns['color_mappings'].items()
            }
        
        if patterns['transformations']:
            patterns['common_transformation'] = patterns['transformations'].most_common(1)[0][0]
        
        self.task_patterns[task_id] = patterns

    def fix_zero_predictions(self, grid: np.ndarray, input_grid: np.ndarray, task_id: str) -> np.ndarray:
        """Fix predictions that are mostly zeros"""
        if grid.size == 0:
            return np.array([[1]])
            
        zero_count = np.sum(grid == 0)
        if zero_count / grid.size > 0.5:
            result = grid.copy()
            bg_color = self.task_patterns[task_id].get('background_color', 1) if task_id in self.task_patterns else 1
            result[result == 0] = bg_color
            return result
            
        return grid

    def score_prediction(self, candidate: np.ndarray, input_grid: np.ndarray, task_id: str) -> float:
        """Score a prediction based on various heuristics"""
        score = 0.5
        
        # Penalize constant predictions when input is not constant
        if np.all(candidate == candidate[0, 0]) and not np.all(input_grid == input_grid[0, 0]):
            score -= 0.3
            
        # Reward predictions that use colors seen in training
        if task_id in self.task_patterns:
            colors = set(np.unique(candidate))
            train_colors = set(self.task_patterns[task_id].get('color_mappings', {}).keys())
            score += 0.1 * len(colors.intersection(train_colors))
            
        # Reward predictions that are different from input
        if not np.array_equal(candidate, input_grid):
            score += 0.2
            
        # Score based on object IoU
        bg_c = detect_background(candidate)
        bg_i = detect_background(input_grid)
        mask_c = candidate != bg_c
        mask_i = input_grid != bg_i
        
        if mask_i.any() and mask_c.any():  # Only calculate if there are objects
            iou = jaccard_score(mask_i.flatten(), mask_c.flatten())
            score += 0.2 * iou
            self.stats['iou_scores'].append(iou)
        
        return max(0.1, min(1.0, score))

    def predict(self, input_grid: np.ndarray, task_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two prediction attempts for the input grid"""
        if input_grid.size == 0:
            return np.array([[1]]), np.array([[2]])
            
        # Determine target shape
        if task_id in self.task_patterns and self.task_patterns[task_id].get('output_shape'):
            target_shape = self.task_patterns[task_id]['output_shape']
        else:
            target_shape = input_grid.shape
            
        candidates = []
        
        # Attempt 1: Rule-based prediction
        attempt1 = self.apply_rules(input_grid, task_id)
        attempt1 = force_same_shape(attempt1, target_shape)
        attempt1 = self.fix_zero_predictions(attempt1, input_grid, task_id)
        candidates.append((attempt1, self.score_prediction(attempt1, input_grid, task_id)))
        
        # Attempt 2: Apply transformations to rule-based prediction
        for transform in self.transformations:
            transformed = transform(attempt1)
            transformed = force_same_shape(transformed, target_shape)
            transformed = self.fix_zero_predictions(transformed, input_grid, task_id)
            score = self.score_prediction(transformed, input_grid, task_id)
            candidates.append((transformed, score))
        
        # Select best and diverse attempts
        candidates.sort(key=lambda x: x[1], reverse=True)
        attempt1 = candidates[0][0]
        
        # Find a different attempt for attempt2
        attempt2 = None
        for candidate, _ in candidates[1:]:
            if not np.array_equal(candidate, attempt1):
                attempt2 = candidate
                break
                
        if attempt2 is None:
            # Fallback: create a different pattern
            attempt2 = np.ones_like(attempt1) * 2
            attempt2 = self.fix_zero_predictions(attempt2, input_grid, task_id)
        
        if DEBUG_MODE:
            logger.debug(f"Task {task_id}: Input shape={input_grid.shape}")
            logger.debug(f"Attempt1 shape={attempt1.shape}, Attempt2 shape={attempt2.shape}")
        
        return force_grid_cap(attempt1), force_grid_cap(attempt2)

def train_on_dataset(solver: AdvancedRuleSolver, train_json_path: str):
    """Train solver on the training dataset"""
    train_data = load_json_file(train_json_path)
    if not train_data:
        logger.error("No training data found!")
        return
        
    for task_id, task_info in train_data.items():
        train_examples = task_info.get("train", [])
        if train_examples:
            solver.learn_task(task_id, train_examples)
            logger.info(f"Trained on task {task_id} with {len(train_examples)} examples")

def submission_writer(predictions_dict: Dict, filename: str = SUBMISSION_PATH) -> bool:
    """Write predictions to submission file"""
    try:
        formatted_predictions = {}
        for task_id, predictions_list in predictions_dict.items():
            formatted_predictions[task_id] = []
            for attempt1, attempt2 in predictions_list:
                formatted_predictions[task_id].append({
                    "attempt_1": to_serializable(attempt1),
                    "attempt_2": to_serializable(attempt2)
                })
                
        with open(filename, "w") as f:
            json.dump(formatted_predictions, f, indent=2)
            
        logger.info(f"âœ… Submission saved: {filename}")
        return True
    except Exception as e:
        logger.error(f"â�Œ Submission write error: {e}")
        return False

def process_test_file(test_json_path: str, submission_path: str) -> bool:
    """Process test file and generate submission"""
    try:
        test_data = load_json_file(test_json_path)
        if not test_data:
            logger.error("No test data found!")
            return False
            
        solver = AdvancedRuleSolver()
        train_on_dataset(solver, TRAIN_JSON_PATH)
        
        all_predictions = {}
        for task_id, task_info in test_data.items():
            train_examples = task_info.get("train", [])
            test_examples = task_info.get("test", [])
            
            logger.info(f"Processing task {task_id}: {len(train_examples)} train, {len(test_examples)} test")
            
            if train_examples:
                solver.learn_task(task_id, train_examples)
                
            task_predictions = []
            for test_ex in test_examples:
                input_grid = np.array(test_ex["input"])
                pred1, pred2 = solver.predict(input_grid, task_id)
                task_predictions.append((pred1, pred2))
                
            all_predictions[task_id] = task_predictions
            
        success = submission_writer(all_predictions, submission_path)
        return success
    except Exception as e:
        logger.error(f"Processing error: {e}")
        return False

if __name__ == "__main__":
    logger.info("Starting ARC Solver...")
    
    if not os.path.exists(TEST_JSON_PATH):
        logger.error(f"Test file missing: {TEST_JSON_PATH}")
        exit(1)
        
    success = process_test_file(TEST_JSON_PATH, SUBMISSION_PATH)
    
    if success:
        logger.info("ğŸ�‰ Submission created successfully!")
    else:
        logger.error("â�Œ Submission failed")
        exit(1)
        #cell2


import numpy as np
import json
import logging
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict, Counter
from copy import deepcopy
import itertools
from scipy import ndimage

# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ARCSolver:
    def __init__(self):
        self.pattern_db = defaultdict(dict)
        
    def learn_from_task(self, task_id: str, examples: List[Dict]):
        """Learn patterns from training examples of a task"""
        inputs = [np.array(ex["input"]) for ex in examples]
        outputs = [np.array(ex["output"]) for ex in examples]
        
        # Store input-output pairs
        self.pattern_db[task_id]["examples"] = list(zip(inputs, outputs))
        
        # Analyze common transformations
        self._analyze_transformations(task_id, inputs, outputs)
        
    def _analyze_transformations(self, task_id: str, inputs: List[np.ndarray], outputs: List[np.ndarray]):
        """Analyze what transformations are applied from input to output"""
        transformations = []
        
        for inp, out in zip(inputs, outputs):
            # Check if it's a simple copy
            if np.array_equal(inp, out):
                transformations.append("copy")
                continue
                
            # Check for resizing
            if inp.shape != out.shape:
                transformations.append("resize")
                continue
                
            # Check for color mapping
            if not np.array_equal(inp, out) and inp.shape == out.shape:
                unique_pairs = set(zip(inp.flatten(), out.flatten()))
                if len(unique_pairs) < 20:  # Reasonable threshold for color mapping
                    transformations.append("color_map")
                    continue
                    
            # Check for rotations/flips
            for rotation in [0, 1, 2, 3]:
                rotated = np.rot90(inp, rotation)
                if np.array_equal(rotated, out):
                    transformations.append(f"rotate_{rotation*90}")
                    break
                    
                # Check flip combinations
                flipped_v = np.flipud(rotated)
                if np.array_equal(flipped_v, out):
                    transformations.append(f"rotate_{rotation*90}_flipv")
                    break
                    
                flipped_h = np.fliplr(rotated)
                if np.array_equal(flipped_h, out):
                    transformations.append(f"rotate_{rotation*90}_fliph")
                    break
            else:
                transformations.append("complex")
                
        # Store the most common transformation
        if transformations:
            counter = Counter(transformations)
            self.pattern_db[task_id]["common_transform"] = counter.most_common(1)[0][0]
    
    def solve(self, input_grid: np.ndarray, task_id: str) -> List[np.ndarray]:
        """Generate solutions for an input grid based on learned patterns"""
        attempts = []
        
        # If we have examples for this task, try to apply learned patterns
        if task_id in self.pattern_db and "examples" in self.pattern_db[task_id]:
            attempts.extend(self._apply_learned_patterns(input_grid, task_id))
        
        # Always include some general heuristic approaches
        attempts.extend(self._general_heuristics(input_grid))
        
        # Ensure we have at least 2 attempts
        while len(attempts) < 2:
            attempts.append(input_grid.copy())  # Fallback: just copy input
            
        # Return exactly 2 attempts
        return attempts[:2]
    
    def _apply_learned_patterns(self, input_grid: np.ndarray, task_id: str) -> List[np.ndarray]:
        """Apply patterns learned from training examples"""
        attempts = []
        examples = self.pattern_db[task_id]["examples"]
        
        # Try to find the most similar example input
        best_match_idx = -1
        best_similarity = -1
        
        for i, (example_in, example_out) in enumerate(examples):
            # Resize both grids to the same shape for comparison
            if input_grid.shape != example_in.shape:
                # Resize the smaller grid to match the larger one
                max_shape = (max(input_grid.shape[0], example_in.shape[0]), 
                            max(input_grid.shape[1], example_in.shape[1]))
                
                input_resized = np.zeros(max_shape, dtype=input_grid.dtype)
                example_resized = np.zeros(max_shape, dtype=example_in.dtype)
                
                # Copy original content to resized arrays
                input_resized[:input_grid.shape[0], :input_grid.shape[1]] = input_grid
                example_resized[:example_in.shape[0], :example_in.shape[1]] = example_in
                
                # Calculate similarity with resized arrays
                color_similarity = min(1.0, np.sum(input_resized == example_resized) / max(input_resized.size, 1))
                shape_similarity = 0.2  # Different shapes get lower similarity
            else:
                # Same shape, calculate direct similarity
                color_similarity = min(1.0, np.sum(input_grid == example_in) / max(input_grid.size, 1))
                shape_similarity = 1.0
            
            similarity = shape_similarity * 0.4 + color_similarity * 0.6
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match_idx = i
        
        if best_match_idx >= 0:
            _, example_out = examples[best_match_idx]
            
            # If shapes match, try direct application
            if input_grid.shape == example_out.shape:
                attempts.append(example_out.copy())
            
            # Try to apply the common transformation if identified
            if "common_transform" in self.pattern_db[task_id]:
                transform = self.pattern_db[task_id]["common_transform"]
                attempts.append(self._apply_transformation(input_grid, transform))
        
        return attempts
    
    def _apply_transformation(self, grid: np.ndarray, transform: str) -> np.ndarray:
        """Apply a specific transformation to a grid"""
        if transform == "copy":
            return grid.copy()
        
        elif transform.startswith("rotate_"):
            # Extract rotation angle
            angle = int(transform.split("_")[1])
            rotations = angle // 90
            return np.rot90(grid, rotations)
        
        elif "flip" in transform:
            if "flipv" in transform:
                result = np.flipud(grid)
            if "fliph" in transform:
                result = np.fliplr(grid)
            return result
        
        elif transform == "resize":
            # For resize, we need to determine the target size
            # This is a simplified approach - in practice, we'd need to know the target size
            if grid.shape[0] > 1 and grid.shape[1] > 1:
                # Simple scaling - reduce by half
                new_shape = (max(1, grid.shape[0] // 2), max(1, grid.shape[1] // 2))
                return self._resize_grid(grid, new_shape)
        
        # Default: return the original grid
        return grid.copy()
    
    def _resize_grid(self, grid: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Resize grid to target shape using nearest neighbor interpolation"""
        if grid.shape == target_shape:
            return grid.copy()
        
        # Calculate scaling factors
        y_scale = target_shape[0] / grid.shape[0]
        x_scale = target_shape[1] / grid.shape[1]
        
        # Create coordinate arrays
        y_coords = np.arange(target_shape[0]) / y_scale
        x_coords = np.arange(target_shape[1]) / x_scale
        
        # Use nearest neighbor interpolation
        y_indices = np.floor(y_coords).astype(int)
        x_indices = np.floor(x_coords).astype(int)
        
        # Ensure indices are within bounds
        y_indices = np.clip(y_indices, 0, grid.shape[0] - 1)
        x_indices = np.clip(x_indices, 0, grid.shape[1] - 1)
        
        # Create resized grid
        resized = grid[y_indices[:, None], x_indices]
        return resized
    
    def _general_heuristics(self, input_grid: np.ndarray) -> List[np.ndarray]:
        """Apply general heuristic approaches to solve ARC tasks"""
        attempts = []
        
        # Heuristic 1: Try to find and extend patterns
        pattern_attempt = self._find_patterns(input_grid)
        if pattern_attempt is not None:
            attempts.append(pattern_attempt)
        
        # Heuristic 2: Try symmetry detection
        symmetry_attempt = self._detect_symmetry(input_grid)
        if symmetry_attempt is not None:
            attempts.append(symmetry_attempt)
        
        # Heuristic 3: Try object detection and manipulation
        object_attempt = self._detect_objects(input_grid)
        if object_attempt is not None:
            attempts.append(object_attempt)
            
        return attempts
    
    def _find_patterns(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Try to find and extend patterns in the grid"""
        # Check for row patterns
        if len(grid) > 1:
            row_pattern = self._extend_pattern([grid[i] for i in range(min(3, len(grid)))])
            if row_pattern is not None and len(row_pattern) == len(grid[0]):
                result = np.tile(row_pattern, (len(grid), 1))
                if not np.array_equal(result, grid):
                    return result
        
        # Check for column patterns
        if len(grid[0]) > 1:
            col_pattern = self._extend_pattern([grid[:, i] for i in range(min(3, len(grid[0])))])
            if col_pattern is not None and len(col_pattern) == len(grid):
                result = np.tile(col_pattern, (len(grid[0]), 1)).T
                if not np.array_equal(result, grid):
                    return result
                    
        return None
    
    def _extend_pattern(self, sequences: List[np.ndarray]) -> Optional[np.ndarray]:
        """Try to extend a pattern from sample sequences"""
        if len(sequences) < 2:
            return None
            
        # Check for arithmetic sequences
        diffs = [sequences[i+1] - sequences[i] for i in range(len(sequences)-1)]
        if all(np.array_equal(diffs[0], diff) for diff in diffs):
            next_seq = sequences[-1] + diffs[0]
            return next_seq
            
        # Check for repeating patterns
        for pattern_len in range(1, min(8, len(sequences[0]) // 2 + 1)):
            pattern = sequences[0][:pattern_len]
            if all(np.array_equal(seq[:pattern_len], pattern) for seq in sequences):
                if all(np.array_equal(seq, np.tile(pattern, len(seq) // pattern_len + 1)[:len(seq)]) 
                       for seq in sequences):
                    return np.tile(pattern, len(sequences[0]) // pattern_len + 1)[:len(sequences[0])]
        
        return None
    
    def _detect_symmetry(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Detect and complete symmetry in the grid"""
        h, w = grid.shape
        
        # Check for horizontal symmetry
        if h > 2 and h % 2 == 1:
            mid = h // 2
            if np.array_equal(grid[:mid], np.flipud(grid[mid+1:])):
                return grid  # Already symmetric
        
        # Check for vertical symmetry
        if w > 2 and w % 2 == 1:
            mid = w // 2
            if np.array_equal(grid[:, :mid], np.fliplr(grid[:, mid+1:])):
                return grid  # Already symmetric
                
        # Try to make it symmetric by mirroring
        if h >= 2:
            mirrored = np.vstack([grid, np.flipud(grid)])
            if not np.array_equal(mirrored, grid):
                return mirrored
                
        if w >= 2:
            mirrored = np.hstack([grid, np.fliplr(grid)])
            if not np.array_equal(mirrored, grid):
                return mirrored
                
        return None
    
    def _detect_objects(self, grid: np.ndarray) -> Optional[np.ndarray]:
        """Detect and manipulate objects in the grid"""
        # Simple object detection by connected components
        labeled, num_features = ndimage.label(grid > 0)
        
        if num_features > 0:
            # Find the largest object
            sizes = ndimage.sum(grid > 0, labeled, range(1, num_features + 1))
            if len(sizes) > 0:
                largest_obj = np.argmax(sizes) + 1
                obj_mask = labeled == largest_obj
                
                # Try some manipulations
                attempts = []
                
                # Attempt 1: Move to center
                center_y, center_x = np.array(grid.shape) // 2
                obj_y, obj_x = np.where(obj_mask)
                if len(obj_y) > 0 and len(obj_x) > 0:
                    mean_y, mean_x = np.mean(obj_y), np.mean(obj_x)
                    
                    shift_y, shift_x = int(center_y - mean_y), int(center_x - mean_x)
                    shifted = ndimage.shift(obj_mask.astype(int), (shift_y, shift_x))
                    obj_color = grid[obj_y[0], obj_x[0]] if len(obj_y) > 0 else 1
                    attempts.append(np.where(shifted, obj_color, 0))
                
                # Attempt 2: Rotate 90 degrees
                rotated = ndimage.rotate(obj_mask.astype(int), 90, reshape=False)
                obj_color = grid[obj_y[0], obj_x[0]] if len(obj_y) > 0 else 1
                attempts.append(np.where(rotated, obj_color, 0))
                
                # Return the first attempt that's different from original
                for attempt in attempts:
                    if not np.array_equal(attempt, grid):
                        return attempt
                        
        return None

def load_task_data(file_path: str) -> Dict[str, Any]:
    """Load task data from a JSON file"""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return {}

def create_submission(solver: ARCSolver, test_data: Dict[str, Any]) -> Dict[str, Any]:
    """Create a submission in the required format"""
    submission = {}
    
    for task_id, task_info in test_data.items():
        test_examples = task_info.get("test", [])
        task_predictions = []
        
        # Learn from training examples if available
        if "train" in task_info:
            solver.learn_from_task(task_id, task_info["train"])
        
        for example in test_examples:
            input_grid = np.array(example["input"])
            solutions = solver.solve(input_grid, task_id)
            
            # Ensure we have exactly two attempts
            if len(solutions) < 2:
                solutions.extend([input_grid.copy()] * (2 - len(solutions)))
            
            # Convert to list format for JSON serialization
            attempt1 = solutions[0].tolist()
            attempt2 = solutions[1].tolist()
            
            task_predictions.append({
                "attempt_1": attempt1,
                "attempt_2": attempt2
            })
        
        submission[task_id] = task_predictions
    
    return submission

def save_submission(submission: Dict[str, Any], file_path: str):
    """Save submission to a JSON file"""
    try:
        with open(file_path, 'w') as f:
            json.dump(submission, f, indent=2)
        logger.info(f"Submission saved to {file_path}")
    except Exception as e:
        logger.error(f"Error saving submission to {file_path}: {e}")

def main():
    # Initialize solver
    solver = ARCSolver()
    
    # Load test data
    test_file = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
    test_data = load_task_data(test_file)
    
    if not test_data:
        logger.error("Failed to load test data")
        return
    
    # Load training data for additional learning
    train_file = "/kaggle/input/arc-agi-prize-2025/arc-agi_training_challenges.json"
    train_data = load_task_data(train_file)
    
    if train_data:
        for task_id, task_info in train_data.items():
            if "train" in task_info:
                solver.learn_from_task(task_id, task_info["train"])
    
    # Generate submission
    submission = create_submission(solver, test_data)
    
    # Save submission
    output_file = "/kaggle/working/submission.json"
    save_submission(submission, output_file)
    
    logger.info("Submission process completed successfully")

if __name__ == "__main__":
    main()
    #cell3


import numpy as np
import json
import logging
import os
import time
from typing import Dict, List, Tuple
from collections import Counter, defaultdict

# Configuration
DEBUG_MODE = True
MAX_GRID_SIZE = 30
TRAIN_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
TEST_JSON_PATH = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"
SUBMISSION_PATH = "/kaggle/working/submission.json"

# Set random seeds
np.random.seed(42)

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Utility Functions
def to_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(x) for x in obj]
    return obj

def load_json(path: str) -> Dict:
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"JSON load failed: {e}")
        return {}

def submission_writer(preds: Dict, filename: str) -> bool:
    try:
        with open(filename, "w") as f:
            json.dump(preds, f, indent=2)
        logger.info(f"âœ… Submission saved: {filename}")
        return True
    except Exception as e:
        logger.error(f"Submission write failed: {e}")
        return False

def force_same_shape(arr: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    if arr.shape == target_shape:
        return arr.copy()
    if arr.size == 0:
        return np.ones(target_shape, dtype=int)
    H, W = target_shape
    h, w = arr.shape
    result = np.ones(target_shape, dtype=int)
    h_min, w_min = min(H, h), min(W, w)
    y0, x0 = (H - h_min) // 2, (W - w_min) // 2
    ys, xs = (h - h_min) // 2, (w - w_min) // 2
    result[y0:y0+h_min, x0:x0+w_min] = arr[ys:ys+h_min, xs:xs+w_min]
    return result

def detect_background(grid: np.ndarray) -> int:
    if grid.size == 0:
        return 0
    return int(np.argmax(np.bincount(grid.flatten(), minlength=10)))

# Advanced Binary Pattern Encoder with Color Mapping
class BinaryPatternEncoder:
    def __init__(self):
        # Binary to color mapping with proper color numbers (0-9)
        self.binary_mapping = {
            # 2x2 patterns (4 bits)
            "0000": 0, "0001": 1, "0010": 2, "0011": 3, 
            "0100": 4, "0101": 5, "0110": 6, "0110": 7,
            "1000": 8, "1001": 9, "1010": 2, "1011": 3,
            "1100": 4, "1101": 5, "1110": 6, "1111": 7,
            
            # 3x3 patterns (9 bits)
            "000000000": 0, "000000001": 1, "000000010": 2, "000000011": 3,
            "000000100": 4, "000000101": 5, "000000110": 6, "000000111": 7,
            "000001000": 8, "000001001": 9, "000001010": 2, "000001011": 3,
            "000001100": 4, "000001101": 5, "000001110": 6, "000001111": 7,
            
            # 4x4 patterns (16 bits)
            "0000000000000000": 0, "1111111111111111": 9,
            "1010101010101010": 5, "0101010101010101": 3,
            "1100110011001100": 6, "0011001100110011": 3,
            "1111000011110000": 8, "0000111100001111": 4,
            
            # Diagonal patterns
            "1000010000100001": 5, "0001000100010001": 3,
            "1000100010001000": 8, "0100010001000100": 4,
            
            # Border patterns
            "1111011110111101": 8, "1011110111101111": 6,
            "1111111111111110": 9, "0111111111111111": 7,
        }
        
    def encode_pattern(self, binary_pattern: str) -> int:
        """Encode a binary pattern to a color value (0-9)"""
        return self.binary_mapping.get(binary_pattern, 1)  # Default to color 1 if pattern not found
    
    def extract_binary_patterns(self, grid: np.ndarray) -> np.ndarray:
        """Extract binary patterns from grid and encode them to colors"""
        if grid.size == 0:
            return np.array([[1]])
            
        bg_color = detect_background(grid)
        binary_grid = (grid != bg_color).astype(int)
        result = np.zeros_like(grid)
        
        H, W = grid.shape
        
        # Process 2x2 patterns
        for i in range(H - 1):
            for j in range(W - 1):
                pattern = binary_grid[i:i+2, j:j+2]
                pattern_str = ''.join(pattern.flatten().astype(str))
                result[i, j] = self.encode_pattern(pattern_str)
        
        # Process 3x3 patterns for remaining cells
        for i in range(H):
            for j in range(W):
                if result[i, j] == 0:  # Only process cells not covered by 2x2
                    min_i, max_i = max(0, i-1), min(H, i+2)
                    min_j, max_j = max(0, j-1), min(W, j+2)
                    pattern = binary_grid[min_i:max_i, min_j:max_j]
                    pattern_str = ''.join(pattern.flatten().astype(str))
                    result[i, j] = self.encode_pattern(pattern_str)
        
        return result

# Enhanced Pattern Detection
class PatternDetector:
    def __init__(self):
        self.patterns = {
            'checkerboard': lambda x: (x[0, 0] != x[0, 1]) and (x[0, 0] == x[1, 1]),
            'solid_color': lambda x: np.all(x == x[0, 0]),
            'border': lambda x: (x[0, :] == x[0, 0]).all() and (x[-1, :] == x[0, 0]).all() and 
                              (x[:, 0] == x[0, 0]).all() and (x[:, -1] == x[0, 0]).all(),
            'diagonal': lambda x: np.all(np.diag(x) == x[0, 0]) and np.all(np.diag(np.fliplr(x)) == x[0, -1])
        }

    def detect_pattern(self, grid: np.ndarray) -> str:
        if grid.size < 4:
            return "small_grid"
        
        for pattern_name, pattern_func in self.patterns.items():
            try:
                if pattern_func(grid):
                    return pattern_name
            except:
                continue
        return "unknown"

# Advanced Zero Fixer
class AdvancedZeroFixer:
    def __init__(self):
        self.zero_stats = {'fixed': 0, 'total': 0}
    
    def fix_zeros(self, grid: np.ndarray, original_grid: np.ndarray) -> np.ndarray:
        self.zero_stats['total'] += 1
        
        if grid.size == 0:
            return np.array([[1]])
            
        # Count zeros
        zero_mask = grid == 0
        zero_count = np.sum(zero_mask)
        
        # If no zeros, return as-is
        if zero_count == 0:
            return grid.copy()
        
        # Make a copy to work with
        result = grid.copy()
        
        # Strategy: Replace zeros with most common non-zero color from original
        non_zero_colors = original_grid[original_grid != 0]
        if len(non_zero_colors) > 0:
            most_common = Counter(non_zero_colors).most_common(1)[0][0]
            result[zero_mask] = most_common
            self.zero_stats['fixed'] += 1
        
        return result

# Hybrid Solver with Binary Pattern Encoder
class HybridSolver:
    def __init__(self):
        self.task_patterns = {}
        self.pattern_detector = PatternDetector()
        self.binary_encoder = BinaryPatternEncoder()
        self.zero_fixer = AdvancedZeroFixer()
        self.common_transforms = [
            lambda x: x,  # identity
            lambda x: np.rot90(x, 1),
            lambda x: np.rot90(x, 2), 
            lambda x: np.rot90(x, 3),
            lambda x: np.fliplr(x),
            lambda x: np.flipud(x)
        ]

    def learn_task(self, task_id: str, train_examples: List[Dict]):
        patterns = {
            'input_shapes': Counter(),
            'output_shapes': Counter(),
            'color_mappings': defaultdict(Counter),
            'common_patterns': Counter(),
            'transformations': Counter()
        }
        
        for ex in train_examples:
            inp = np.array(ex['input'])
            out = np.array(ex['output'])
            
            patterns['input_shapes'][inp.shape] += 1
            patterns['output_shapes'][out.shape] += 1
            
            # Color mapping analysis
            for old, new in zip(inp.flatten(), out.flatten()):
                if old != new:
                    patterns['color_mappings'][old][new] += 1
            
            # Pattern detection
            input_pattern = self.pattern_detector.detect_pattern(inp)
            output_pattern = self.pattern_detector.detect_pattern(out)
            patterns['common_patterns'][(input_pattern, output_pattern)] += 1
            
            # Transformation detection
            for idx, transform in enumerate(self.common_transforms):
                transformed = transform(inp)
                if transformed.shape == out.shape and np.array_equal(transformed, out):
                    patterns['transformations'][idx] += 1
        
        self.task_patterns[task_id] = patterns
        logger.debug(f"Learned patterns for task {task_id}")

    def apply_binary_patterns(self, grid: np.ndarray) -> np.ndarray:
        """Apply binary pattern encoding to input grid (Attempt 1)"""
        result = self.binary_encoder.extract_binary_patterns(grid)
        result = self.zero_fixer.fix_zeros(result, grid)
        return result

    def apply_learned_rules(self, grid: np.ndarray, task_id: str) -> np.ndarray:
        """Apply learned rules to transform input grid (Attempt 2)"""
        if task_id not in self.task_patterns:
            result = grid.copy()
        else:
            patterns = self.task_patterns[task_id]
            result = grid.copy()
            
            # Apply most common transformation
            if patterns['transformations']:
                best_transform_idx = patterns['transformations'].most_common(1)[0][0]
                result = self.common_transforms[best_transform_idx](result)
            
            # Apply color mappings
            if patterns['color_mappings']:
                color_map = {}
                for old_color, mappings in patterns['color_mappings'].items():
                    if mappings:
                        color_map[old_color] = mappings.most_common(1)[0][0]
                
                if color_map:
                    vec_func = np.vectorize(lambda x: color_map.get(x, x))
                    result = vec_func(result)
            
            # Resize to most common output shape
            if patterns['output_shapes']:
                target_shape = patterns['output_shapes'].most_common(1)[0][0]
                result = force_same_shape(result, target_shape)
        
        # Apply zero fix
        result = self.zero_fixer.fix_zeros(result, grid)
        return result

# Ensemble Solver with Multiple Strategies
class ARCEnsembleSolver:
    def __init__(self, hybrid_solver):
        self.hybrid_solver = hybrid_solver
        self.stats = {'fallbacks': 0, 'total': 0}

    def generate_diverse_attempts(self, grid: np.ndarray, task_id: str) -> Tuple[np.ndarray, np.ndarray]:
        """Generate two diverse attempts using different strategies"""
        
        # Attempt 1: Binary pattern encoding
        attempt1 = self.hybrid_solver.apply_binary_patterns(grid)
        
        # Attempt 2: Learned rules
        attempt2 = self.hybrid_solver.apply_learned_rules(grid, task_id)
        
        # Ensure attempts are different
        if np.array_equal(attempt1, attempt2):
            # Create a different pattern by adding 1 to all cells (mod 10, avoiding 0)
            attempt2 = (attempt2 + 1) % 10
            attempt2[attempt2 == 0] = 1
        
        self.stats['total'] += 1
        if np.array_equal(attempt1, grid) and np.array_equal(attempt2, grid):
            self.stats['fallbacks'] += 1
        
        return attempt1, attempt2

    def predict_task(self, task_data: Dict, task_id: str) -> List[Tuple[np.ndarray, np.ndarray]]:
        preds = []
        test_examples = task_data.get("test", [])
        
        for ex in test_examples:
            grid = np.array(ex["input"])
            preds.append(self.generate_diverse_attempts(grid, task_id))
            
        return preds

# MAIN EXECUTION
def main():
    logger.info("ğŸš€ Starting Enhanced ARC Solver Pipeline...")
    start_time = time.time()
    
    # Load data
    train_data = load_json(TRAIN_JSON_PATH)
    test_data = load_json(TEST_JSON_PATH)
    
    if not test_data:
        logger.error("â�Œ Test data not found")
        return False

    # Initialize solvers
    hybrid = HybridSolver()
    solver = ARCEnsembleSolver(hybrid)

    # Learn from training data
    if train_data:
        logger.info("ğŸ“š Learning from training data...")
        for task_id, task in train_data.items():
            if "train" in task:
                hybrid.learn_task(task_id, task["train"])
        logger.info(f"âœ… Learned patterns from {len(train_data)} tasks")

    # Generate predictions
    logger.info("ğŸ”� Generating predictions...")
    all_preds = {}
    for task_id, task in test_data.items():
        logger.info(f"Processing task {task_id}")
        
        # Learn from any training examples in test task
        if "train" in task:
            hybrid.learn_task(task_id, task["train"])
            
        all_preds[task_id] = solver.predict_task(task, task_id)

    # Format submission - ensure proper JSON structure
    formatted_preds = {}
    for tid, pred_list in all_preds.items():
        formatted_preds[tid] = []
        for attempt1, attempt2 in pred_list:
            formatted_preds[tid].append({
                "attempt_1": to_serializable(attempt1),
                "attempt_2": to_serializable(attempt2)
            })

    # Save submission
    success = submission_writer(formatted_preds, SUBMISSION_PATH)
    
    # Log statistics
    zero_fix_stats = hybrid.zero_fixer.zero_stats
    logger.info(f"ğŸ”§ Zero fix stats: {zero_fix_stats['fixed']}/{zero_fix_stats['total']} grids had zeros fixed")
    logger.info(f"ğŸ“Š Stats: {solver.stats['fallbacks']}/{solver.stats['total']} fallbacks used")
    logger.info(f"â�° Total runtime: {time.time() - start_time:.2f}s")
    logger.info("ğŸ�‰ Pipeline completed successfully!" if success else "â�Œ Pipeline failed")
    
    return success

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)

