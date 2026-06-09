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



import os,json,zipfile
import numpy as np
from functools import reduce
from itertools import product,combinations

class QuantumPatternEngine:
   
    
    def __init__(self):
       
        self.weights = np.ones(16) * 0.5
        
       
        self.quantum_ops = {
            'i': (lambda: lambda g: g, 1.0),
            'h': (lambda: lambda g: [r[::-1] for r in g], 0.95),
            'v': (lambda: lambda g: g[::-1], 0.95),
            '1': (lambda: lambda g: [list(r) for r in zip(*g[::-1])], 0.9),
            '2': (lambda: lambda g: [r[::-1] for r in g[::-1]], 0.9),
            '3': (lambda: lambda g: [list(r) for r in zip(*g)][::-1], 0.9),
            't': (lambda: lambda g: [list(r) for r in zip(*g)], 0.85),
            'c': (lambda: lambda g: [r[1:-1] for r in g[1:-1]] if len(g)>2 and len(g[0])>2 else g, 0.8),
            'd': (lambda: lambda g: [r[::2] for r in g[::2]], 0.75),
            'u': (lambda: lambda g: [[v for v in r for _ in range(2)] for r in g for _ in range(2)], 0.7),
            'x': (lambda: lambda g: [r*2 for r in (g*2)], 0.65),
            'H': (lambda: lambda g: [r+r[::-1] for r in g], 0.6),
            'V': (lambda: lambda g: g+g[::-1], 0.6),
        }
        

        self._cache = {}
        
    def _tensor_hash(self, tensor):
        """Generate unique hash for tensor-based pattern matching"""
        return hash(tuple(map(tuple, tensor))) if hasattr(tensor, '__iter__') else hash(tensor)
    
    def detect_quantum(self, examples):
        """
        Quantum-inspired pattern detection using superposition of states
        """
        if not examples:
            return 'i', {}, 0.0
        
    
        tensor_pairs = [(np.array(inp), np.array(out)) for inp, out in examples]
        
      
        state_vector = np.zeros(len(self.quantum_ops))
        
       
        results = []
        

        for idx, (op_name, (op_gen, weight)) in enumerate(self.quantum_ops.items()):
            op = op_gen()
            score = self._evaluate_transform(tensor_pairs, op, op_name)
            state_vector[idx] = score * weight * self.weights[idx]
            results.append((op_name, {}, state_vector[idx]))
        
       
        color_result = self._detect_color_mapping(tensor_pairs)
        if color_result[2] > 0:
            results.append(color_result)
        
      
        advanced_results = [
            self._detect_fill_pattern(tensor_pairs),
            self._detect_boundary_pattern(tensor_pairs),
            self._detect_staircase_pattern(tensor_pairs),
            self._detect_composite_pattern(tensor_pairs)
        ]
        
        results.extend([r for r in advanced_results if r[2] > 0])
        
     
        best = max(results, key=lambda x: x[2]) if results else ('i', {}, 0.0)
        
    
        if best[2] > 0.8:
            for idx, (op_name, _, _) in enumerate(results[:len(self.quantum_ops)]):
                if op_name == best[0]:
                    self.weights[idx] = min(1.0, self.weights[idx] * 1.1)
                else:
                    self.weights[idx] = max(0.1, self.weights[idx] * 0.95)
        
        return best
    
    def _evaluate_transform(self, pairs, op, op_name):
        """Evaluate transformation accuracy using tensor operations"""
        matches = 0
        for inp, out in pairs:
            try:
             
                if op_name in ['h', 'v', '1', '2', '3', 't']:
                    transformed = self._apply_numpy_transform(inp, op_name)
                else:
                    inp_list = inp.tolist()
                    transformed = np.array(op(inp_list))
                
                if np.array_equal(transformed, out):
                    matches += 1
            except:
                pass
        
        return matches / len(pairs) if pairs else 0
    
    def _apply_numpy_transform(self, arr, op_name):
        """Apply transformation using numpy operations"""
        transforms = {
            'h': lambda a: np.fliplr(a),
            'v': lambda a: np.flipud(a),
            '1': lambda a: np.rot90(a),
            '2': lambda a: np.rot90(a, 2),
            '3': lambda a: np.rot90(a, 3),
            't': lambda a: a.T if a.shape[0] == a.shape[1] else a
        }
        return transforms.get(op_name, lambda a: a)(arr)
    
    def _detect_color_mapping(self, pairs):
        """Detect color value transformations using statistical analysis"""
        all_mappings = []
        
        for inp, out in pairs:
            if inp.shape != out.shape:
                continue
            
            mapping = {}
            for value in np.unique(inp):
                mask = (inp == value)
                out_values = out[mask]
                if len(np.unique(out_values)) == 1:
                    mapping[int(value)] = int(out_values[0])
                else:
                    break
            else:
                all_mappings.append(mapping)
        
        if all_mappings and all(m == all_mappings[0] for m in all_mappings):
            if not all(k == v for k, v in all_mappings[0].items()):
                return 'M', {'m': all_mappings[0]}, 1.0
        
    
        if all(len(np.unique(out)) == 1 for _, out in pairs):
            fill_value = int(np.unique(pairs[0][1])[0])
            return 'F', {'c': fill_value}, 1.0
        
       
        mode_matches = 0
        for inp, out in pairs:
            if inp.shape == out.shape and inp.size > 0:
                flat = inp.ravel()
                if flat.min() >= 0:
                    mode = np.bincount(flat).argmax()
                    if np.all(out == mode):
                        mode_matches += 1
        
        if mode_matches == len(pairs):
            return 'f', {}, 1.0
        
        return 'i', {}, 0
    
    def _detect_fill_pattern(self, pairs):
        """Detect fill patterns using frequency analysis"""
        # Implementation of 'f' pattern
        matches = 0
        for inp, out in pairs:
            if inp.shape == out.shape and inp.size > 0:
                flat = inp.ravel()
                if flat.min() >= 0:
                    counts = np.bincount(flat)
                    if counts.size > 0:
                        mode = counts.argmax()
                        expected = np.full(inp.shape, mode)
                        if np.array_equal(out, expected):
                            matches += 1
        
        score = matches / len(pairs) if pairs else 0
        return ('f', {}, score) if score > 0 else ('i', {}, 0)
    
    def _detect_boundary_pattern(self, pairs):
        """Detect object boundary patterns"""
        def get_boundary(arr):
            h, w = arr.shape
            boundary = np.zeros_like(arr)
            for i in range(h):
                for j in range(w):
                    if arr[i, j] != 0:
                        # Check if on boundary
                        is_boundary = False
                        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]:
                            ni, nj = i + di, j + dj
                            if ni < 0 or ni >= h or nj < 0 or nj >= w or arr[ni, nj] == 0:
                                is_boundary = True
                                break
                        if is_boundary:
                            boundary[i, j] = arr[i, j]
            return boundary
        
        matches = 0
        for inp, out in pairs:
            if inp.shape == out.shape:
                boundary = get_boundary(inp)
                if np.array_equal(boundary, out):
                    matches += 1
        
        score = matches / len(pairs) if pairs else 0
        return ('o', {}, score) if score > 0 else ('i', {}, 0)
    
    def _detect_staircase_pattern(self, pairs):
        """Detect staircase elimination patterns"""
        def remove_staircase(arr):
            h, w = arr.shape
            result = arr.copy()
            for i in range(h - 1):
                for j in range(w - 1):
                    if arr[i, j] != arr[i+1, j+1] and arr[i+1, j+1] != 0:
                        result[i, j] = 0
            return result
        
        matches = 0
        for inp, out in pairs:
            if inp.shape == out.shape:
                expected = remove_staircase(inp)
                if np.array_equal(expected, out):
                    matches += 1
        
        score = matches / len(pairs) if pairs else 0
        return ('s', {}, score) if score > 0 else ('i', {}, 0)
    
    def _detect_composite_pattern(self, pairs):
        """Detect composite transformations using combinatorial search"""
        best = ('i', {}, 0)

        
        basic_ops = ['h', 'v', '1', '2', '3']
        for op in basic_ops:
            matches = 0
            for inp, out in pairs:
                transformed = self._apply_numpy_transform(inp, op)
                if np.array_equal(transformed, out):
                    matches += 1
            
            score = matches / len(pairs)
            if score > best[2]:
                best = (op, {}, score)
        
        return best

def generate_minimal_code(pattern, params=None):
    """Generate ultra-minimal code for pattern execution"""
    

    codes = {
        'i': 'p=lambda g:g',
        'h': 'p=lambda g:[r[::-1]for r in g]',
        'v': 'p=lambda g:g[::-1]',
        '1': 'p=lambda g:[list(r)for r in zip(*g[::-1])]',
        '2': 'p=lambda g:[r[::-1]for r in g[::-1]]',
        '3': 'p=lambda g:[list(r)for r in zip(*g)][::-1]',
        't': 'p=lambda g:[list(r)for r in zip(*g)]',
        'c': 'p=lambda g:[r[1:-1]for r in g[1:-1]]',
        'd': 'p=lambda g:[r[::2]for r in g[::2]]',
        'u': 'p=lambda g:[[v for v in r for _ in(0,1)]for r in g for _ in(0,1)]',
        'x': 'p=lambda g:[r+r for r in(g+g)]',
        'H': 'p=lambda g:[r+r[::-1]for r in g]',
        'V': 'p=lambda g:g+g[::-1]',
        'f': 'p=lambda g:[[max(set(sum(g,[])),key=lambda x:sum(r.count(x)for r in g))]*len(g[0])]*len(g)',
        's': 'p=lambda g:[[0if i<len(g)-1and j<len(g[0])-1and g[i][j]!=g[i+1][j+1]and g[i+1][j+1]else g[i][j]for j in range(len(g[0]))]for i in range(len(g))]',
        'o': 'p=lambda g:[[g[i][j]if g[i][j]and any(i+a<0or i+a>=len(g)or j+b<0or j+b>=len(g[0])or not g[i+a][j+b]for a,b in((0,1),(0,-1),(1,0),(-1,0)))else 0for j in range(len(g[0]))]for i in range(len(g))]'
    }
    
    if pattern == 'M' and params and 'm' in params:
        mapping = params['m']
        if len(mapping) == 1:
            k, v = list(mapping.items())[0]
            return f'p=lambda g:[[{v}if x=={k}else x for x in r]for r in g]'
        
       
        items = ','.join(f'{k}:{v}' for k, v in mapping.items())
        return f'p=lambda g:[[{{{items}}}.get(x,x)for x in r]for r in g]'
    
    if pattern == 'F' and params:
        c = params.get('c', 1)
        return f'p=lambda g:[[{c}]*len(g[0])]*len(g)'
    
    return codes.get(pattern, 'p=lambda g:g')

def apply_pattern(pattern, params, grid):
    """Apply detected pattern to grid"""
    
    g = grid
    
    if pattern == 'i': return g
    elif pattern == 'h': return [r[::-1] for r in g]
    elif pattern == 'v': return g[::-1]
    elif pattern == '1': return [list(r) for r in zip(*g[::-1])]
    elif pattern == '2': return [r[::-1] for r in g[::-1]]
    elif pattern == '3': return [list(r) for r in zip(*g)][::-1]
    elif pattern == 't': return [list(r) for r in zip(*g)]
    elif pattern == 'c': return [r[1:-1] for r in g[1:-1]] if len(g) > 2 and len(g[0]) > 2 else g
    elif pattern == 'd': return [r[::2] for r in g[::2]]
    elif pattern == 'u': return [[v for v in r for _ in (0,1)] for r in g for _ in (0,1)]
    elif pattern == 'x': return [r+r for r in (g+g)]
    elif pattern == 'H': return [r+r[::-1] for r in g]
    elif pattern == 'V': return g+g[::-1]
    elif pattern == 'f':
        if g and g[0]:
            flat = sum(g, [])
            if flat:
                mode = max(set(flat), key=lambda x: sum(r.count(x) for r in g))
                return [[mode]*len(g[0])]*len(g)
        return g
    elif pattern == 'F':
        c = params.get('c', 1) if params else 1
        return [[c]*len(g[0])]*len(g) if g and g[0] else g
    elif pattern == 'M':
        if params and 'm' in params:
            mapping = params['m']
            return [[mapping.get(x, x) for x in r] for r in g]
        return g
    elif pattern == 'o':
        h, w = len(g), len(g[0]) if g else 0
        result = [[0]*w for _ in range(h)]
        for i in range(h):
            for j in range(w):
                if g[i][j]:
                    is_boundary = any(
                        i+di < 0 or i+di >= h or j+dj < 0 or j+dj >= w or not g[i+di][j+dj]
                        for di, dj in [(0,1), (0,-1), (1,0), (-1,0)]
                    )
                    if is_boundary:
                        result[i][j] = g[i][j]
        return result
    elif pattern == 's':
        h, w = len(g), len(g[0]) if g else 0
        result = [row[:] for row in g]
        for i in range(h-1):
            for j in range(w-1):
                if g[i][j] != g[i+1][j+1] and g[i+1][j+1]:
                    result[i][j] = 0
        return result
    else:
        return g

def main():
    """Main execution with quantum pattern detection"""
    tasks_dir = os.environ.get('TASKS_DIR', '/kaggle/input/google-code-golf-2025')
    output_dir = os.environ.get('OUTPUT_DIR', 'solutions')
    os.makedirs(output_dir, exist_ok=True)
    
    engine = QuantumPatternEngine()
    total_score = 0
    statistics = []
    
    for task_id in range(400):
        task_name = f'task{task_id:03d}'
        task_path = os.path.join(tasks_dir, f'{task_name}.json')
        
        if not os.path.exists(task_path):
            continue
        
        try:
           
            with open(task_path, 'r') as f:
                task_data = json.load(f)
            
            
            examples = [(item['input'], item['output']) for item in task_data.get('train', [])]
            
           
            pattern, params, confidence = engine.detect_quantum(examples)
            
            
            test_cases = task_data.get('test', [])
            outputs = [apply_pattern(pattern, params, test['input']) for test in test_cases]

            
            code = generate_minimal_code(pattern, params)
            score = max(1, 2500 - len(code))
            
            
            with open(os.path.join(output_dir, f'{task_name}.py'), 'w') as f:
                f.write('#\n' + code + '\n')
            
            output_json = {'output': outputs[0]} if len(outputs) == 1 else {'outputs': outputs}
            with open(os.path.join(output_dir, f'{task_name}.out.json'), 'w') as f:
                json.dump(output_json, f)
            
            total_score += score
            statistics.append({
                'task': task_name,
                'pattern': pattern,
                'confidence': confidence,
                'score': score,
                'code_length': len(code)
            })
            
            print(f"{task_name}: {pattern} (confidence={confidence:.3f}, score={score}, len={len(code)})")
            
        except Exception as e:
            print(f"Error processing {task_name}: {e}")
    
    
    with zipfile.ZipFile('submission.zip', 'w') as zipf:
        for root, _, files in os.walk(output_dir):
            for filename in files:
                file_path = os.path.join(root, filename)
                arc_path = os.path.join(os.path.basename(root), filename)
                zipf.write(file_path, arc_path)
    
    
    if statistics:
        detected = sum(1 for s in statistics if s['pattern'] != 'i' or s['confidence'] == 1.0)
        total = len(statistics)
        avg_confidence = sum(s['confidence'] for s in statistics) / total
        avg_score = total_score / total
        
        print('=' * 60)
        print(f"TOTAL SCORE: {total_score}")
        print(f"PATTERNS DETECTED: {detected}/{total} ({100*detected/total:.1f}%)")
        print(f"AVERAGE CONFIDENCE: {avg_confidence:.3f}")
        print(f"AVERAGE SCORE: {avg_score:.1f}")
        print(f"NEURAL WEIGHTS: {engine.weights[:8].round(3)}")
    else:
        print("No tasks processed")

if __name__ == '__main__':
    main()










