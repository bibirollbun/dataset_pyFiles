import json
import numpy as np
from collections import Counter, defaultdict
import copy


class ARCPatternSolver:
    """Advanced pattern-based solver for ARC tasks"""

    def __init__(self):
        self.patterns = []

    def analyze_transformation(self, input_grid, output_grid):
        """Analyze transformation pattern between input and output"""
        patterns = {}

        # Pattern 1: Size transformation
        in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
        out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0

        if in_h > 0 and in_w > 0:
            patterns['size_ratio'] = (out_h / in_h, out_w / in_w)
        else:
            patterns['size_ratio'] = (1, 1)

        # Pattern 2: Color mapping
        patterns['color_map'] = {}
        if in_h == out_h and in_w == out_w:
            for i in range(min(in_h, out_h)):
                for j in range(min(in_w, out_w)):
                    in_c = input_grid[i][j]
                    out_c = output_grid[i][j]
                    if in_c not in patterns['color_map']:
                        patterns['color_map'][in_c] = []
                    patterns['color_map'][in_c].append(out_c)

        # Pattern 3: Identity check
        patterns['is_identity'] = (input_grid == output_grid)

        # Pattern 4: Rotation/Flip
        patterns['rotation'] = self.check_rotation(input_grid, output_grid)
        patterns['flip'] = self.check_flip(input_grid, output_grid)

        # Pattern 5: Repetition pattern
        patterns['repetition'] = self.check_repetition(input_grid, output_grid)

        return patterns

    def check_rotation(self, input_grid, output_grid):
        """Check if output is a rotation of input"""
        if not input_grid or not output_grid:
            return None

        in_arr = np.array(input_grid)
        out_arr = np.array(output_grid)

        for k in [1, 2, 3]:
            rotated = np.rot90(in_arr, k)
            if rotated.shape == out_arr.shape and np.array_equal(rotated, out_arr):
                return k * 90
        return None

    def check_flip(self, input_grid, output_grid):
        """Check if output is a flip of input"""
        if not input_grid or not output_grid:
            return None

        in_arr = np.array(input_grid)
        out_arr = np.array(output_grid)

        if in_arr.shape != out_arr.shape:
            return None

        if np.array_equal(np.fliplr(in_arr), out_arr):
            return 'horizontal'

        if np.array_equal(np.flipud(in_arr), out_arr):
            return 'vertical'

        return None

    def check_repetition(self, input_grid, output_grid):
        """Check if output is a repetition of input"""
        if not input_grid or not output_grid:
            return None

        in_h, in_w = len(input_grid), len(input_grid[0]) if input_grid else 0
        out_h, out_w = len(output_grid), len(output_grid[0]) if output_grid else 0

        if in_h == 0 or in_w == 0:
            return None

        repeat_h = out_h // in_h if in_h > 0 else 0
        repeat_w = out_w // in_w if in_w > 0 else 0

        if repeat_h > 1 or repeat_w > 1:
            is_repetition = True
            for i in range(out_h):
                for j in range(out_w):
                    expected = input_grid[i % in_h][j % in_w]
                    if output_grid[i][j] != expected:
                        is_repetition = False
                        break
                if not is_repetition:
                    break

            if is_repetition:
                return (repeat_h, repeat_w)

        return None

    def learn_from_examples(self, train_examples):
        """Learn patterns from training examples"""
        all_patterns = []

        for example in train_examples:
            patterns = self.analyze_transformation(example['input'], example['output'])
            all_patterns.append(patterns)

        consistent_patterns = {}

        # Check size ratio consistency
        size_ratios = [p['size_ratio'] for p in all_patterns]
        if len(set(size_ratios)) == 1:
            consistent_patterns['size_ratio'] = size_ratios[0]

        # Check rotation consistency
        rotations = [p['rotation'] for p in all_patterns if p['rotation'] is not None]
        if rotations and len(set(rotations)) == 1:
            consistent_patterns['rotation'] = rotations[0]

        # Check flip consistency
        flips = [p['flip'] for p in all_patterns if p['flip'] is not None]
        if flips and len(set(flips)) == 1:
            consistent_patterns['flip'] = flips[0]

        # Check repetition consistency
        repetitions = [p['repetition'] for p in all_patterns if p['repetition'] is not None]
        if repetitions and len(set(repetitions)) == 1:
            consistent_patterns['repetition'] = repetitions[0]

        # Check identity
        identities = [p['is_identity'] for p in all_patterns]
        if all(identities):
            consistent_patterns['is_identity'] = True

        # Color mapping
        color_mappings = defaultdict(list)
        for p in all_patterns:
            for in_c, out_cs in p['color_map'].items():
                if out_cs:
                    most_common = Counter(out_cs).most_common(1)[0][0]
                    color_mappings[in_c].append(most_common)

        consistent_color_map = {}
        for in_c, out_cs in color_mappings.items():
            if out_cs and len(set(out_cs)) == 1:
                consistent_color_map[in_c] = out_cs[0]

        if consistent_color_map:
            consistent_patterns['color_map'] = consistent_color_map

        return consistent_patterns

    def apply_patterns(self, test_input, patterns):
        """Apply learned patterns to test input"""
        result = copy.deepcopy(test_input)

        if not result:
            return [[0]]

        # Apply rotation
        if 'rotation' in patterns:
            result = np.rot90(np.array(result), patterns['rotation'] // 90).tolist()

        # Apply flip
        if 'flip' in patterns:
            result_arr = np.array(result)
            if patterns['flip'] == 'horizontal':
                result = np.fliplr(result_arr).tolist()
            elif patterns['flip'] == 'vertical':
                result = np.flipud(result_arr).tolist()

        # Apply repetition
        if 'repetition' in patterns:
            repeat_h, repeat_w = patterns['repetition']
            in_arr = np.array(result)
            result = np.tile(in_arr, (repeat_h, repeat_w)).tolist()

        # Apply size ratio
        if 'size_ratio' in patterns and patterns['size_ratio'] != (1, 1):
            scale_h, scale_w = patterns['size_ratio']
            in_h, in_w = len(result), len(result[0]) if result else 0

            if in_h > 0 and in_w > 0:
                out_h = int(in_h * scale_h)
                out_w = int(in_w * scale_w)

                scaled = []
                for i in range(out_h):
                    row = []
                    for j in range(out_w):
                        src_i = int(i / scale_h)
                        src_j = int(j / scale_w)
                        if src_i < in_h and src_j < in_w:
                            row.append(result[src_i][src_j])
                        else:
                            row.append(0)
                    scaled.append(row)
                result = scaled

        # Apply color mapping
        if 'color_map' in patterns:
            color_map = patterns['color_map']
            h, w = len(result), len(result[0]) if result else 0
            for i in range(h):
                for j in range(w):
                    if result[i][j] in color_map:
                        result[i][j] = color_map[result[i][j]]

        # Handle identity
        if patterns.get('is_identity', False):
            result = copy.deepcopy(test_input)

        return result

    def predict(self, train_examples, test_input):
        """Generate prediction for test input"""
        patterns = self.learn_from_examples(train_examples)
        prediction = self.apply_patterns(test_input, patterns)
        return prediction


# Load test challenges
test_challenges_path = '/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json'

print("Loading test challenges...")
with open(test_challenges_path, 'r') as f:
    test_challenges = json.load(f)

print(f"Loaded {len(test_challenges)} test tasks")


# Generate predictions
solver = ARCPatternSolver()
predictions = {}

print("\nGenerating predictions...")
for idx, (task_id, task_data) in enumerate(test_challenges.items(), 1):
    if idx % 50 == 0:
        print(f"Progress: {idx}/{len(test_challenges)}")

    train_examples = task_data['train']
    test_examples = task_data['test']

    task_predictions = []

    for test_case in test_examples:
        test_input = test_case['input']

        # Attempt 1: Advanced pattern-based
        try:
            attempt_1 = solver.predict(train_examples, test_input)
        except Exception:
            attempt_1 = test_input

        # Attempt 2: Simple heuristics
        try:
            if train_examples:
                first_train = train_examples[0]
                in_h = len(first_train['input'])
                in_w = len(first_train['input'][0]) if first_train['input'] else 0
                test_h = len(test_input)
                test_w = len(test_input[0]) if test_input else 0

                if in_h == test_h and in_w == test_w:
                    attempt_2 = first_train['output']
                else:
                    attempt_2 = test_input
            else:
                attempt_2 = test_input
        except Exception:
            attempt_2 = test_input

        task_predictions.append({
            'attempt_1': attempt_1,
            'attempt_2': attempt_2
        })

    predictions[task_id] = task_predictions

print(f"\nGenerated predictions for {len(predictions)} tasks")


# Save submission
with open('submission.json', 'w') as f:
    json.dump(predictions, f)

print("Submission saved to submission.json")

