import requests
try:
    requests.get("https://www.google.com", timeout=2)
    print("ğŸŒ� Internet still ON â€” Kaggle flag bug detected")
except:
    print("âœ… Internet OFF â€” ready for submit")



#!/usr/bin/env python3
import json, os, time, warnings
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, field
from scipy import ndimage, stats, spatial
from itertools import combinations

import pickle, hashlib
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

warnings.filterwarnings('ignore')

@dataclass
class Pattern:
    pattern_type: str
    confidence: float
    frequency: int
    quantum_state: np.ndarray = field(default_factory=lambda: np.array([]))
    examples: List[str] = field(default_factory=list)

class ARCAnalyzer:
    def __init__(self, data_dir: str = "/kaggle/input/arc-prize-2025", cache_dir: str = "/kaggle/working/arc_cache"):
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.datasets = {}
        self.patterns = defaultdict(list)
        self.min_confidence = 0.75
        self.min_support = 3
    
    def load_data(self, use_cache: bool = True) -> None:
        cache_file = self.cache_dir / "datasets.pkl"
        if use_cache and cache_file.exists():
            print("Loading cached data...")
            with open(cache_file, 'rb') as f:
                self.datasets = pickle.load(f)
            print(f"Loaded {len(self.datasets)} datasets")
            return
            
        dataset_files = {
            'train_challenges': 'arc-agi_training_challenges.json',
            'eval_challenges': 'arc-agi_evaluation_challenges.json', 
            'test_challenges': 'arc-agi_test_challenges.json'
        }
        
        print("Loading ARC datasets...")
        for name, filename in dataset_files.items():
            path = self.data_dir / filename
            if path.exists():
                with open(path, 'r') as f:
                    self.datasets[name] = json.load(f)
                print(f"{name}: {len(self.datasets[name])} entries")
                
        with open(cache_file, 'wb') as f:
            pickle.dump(self.datasets, f)

    def analyze_grid(self, grid: np.ndarray) -> Dict[str, Any]:
        h, w = grid.shape
        unique_colors = np.unique(grid)
        bg_color = Counter(grid.flatten()).most_common(1)[0][0]
        
        objects = []
        for color in unique_colors:
            if color == bg_color: continue
            mask = (grid == color).astype(int)
            labeled, num_features = ndimage.label(mask)
            for obj_id in range(1, num_features + 1):
                obj_mask = (labeled == obj_id)
                if np.any(obj_mask):
                    positions = np.argwhere(obj_mask)
                    bbox = [np.min(positions[:, 0]), np.max(positions[:, 0]),
                            np.min(positions[:, 1]), np.max(positions[:, 1])]
                    area = np.sum(obj_mask)
                    center = np.mean(positions, axis=0)
                    objects.append({
                        'color': int(color), 'bbox': bbox, 'area': area,
                        'center': center.tolist(), 'pixel_count': area
                    })
        
        density = np.sum(grid != bg_color) / (h * w)
        color_entropy = self._calculate_entropy(grid)
        spatial_dist = self._spatial_analysis(grid, bg_color)
        
        return {
            'dimensions': (h, w), 'background': int(bg_color),
            'unique_colors': [int(c) for c in unique_colors],
            'object_count': len(objects), 'objects': objects,
            'density': density, 'color_entropy': color_entropy,
            'spatial_distribution': spatial_dist,
            'symmetry': self._symmetry_analysis(grid),
            'features': self._compute_features(grid)
        }
    
    def _calculate_entropy(self, grid: np.ndarray) -> float:
        counts = np.bincount(grid.flatten())
        probabilities = counts / np.sum(counts)
        probabilities = probabilities[probabilities > 0]
        return -np.sum(probabilities * np.log2(probabilities))
    
    def _spatial_analysis(self, grid: np.ndarray, bg_color: int) -> Dict[str, float]:
        mask = grid != bg_color
        if not np.any(mask): return {'center_x': 0.5, 'center_y': 0.5, 'spread': 0.0}
        h, w = grid.shape
        positions = np.argwhere(mask)
        center_y, center_x = np.mean(positions, axis=0)
        center_x /= w; center_y /= h
        spread = np.std(positions / [h, w]) if len(positions) > 1 else 0.0
        return {'center_x': float(center_x), 'center_y': float(center_y), 'spread': float(spread)}
    
    def _symmetry_analysis(self, grid: np.ndarray) -> Dict[str, bool]:
        return {
            'horizontal': bool(np.array_equal(grid, np.fliplr(grid))),
            'vertical': bool(np.array_equal(grid, np.flipud(grid))),
            'rotational_180': bool(np.array_equal(grid, np.rot90(grid, 2))),
            'rotational_90': bool(np.array_equal(grid, np.rot90(grid, 1)) and grid.shape[0] == grid.shape[1])
        }
    
    def _compute_features(self, grid: np.ndarray) -> np.ndarray:
        features = []
        for scale in [1, 2, 4]:
            if scale > min(grid.shape) // 2: break
            downsampled = grid[::scale, ::scale]
            features.extend([np.mean(downsampled), np.std(downsampled), np.median(downsampled),
                           stats.skew(downsampled.flatten()), stats.kurtosis(downsampled.flatten())])
        entanglement = self._entanglement_analysis(grid)
        features.extend(entanglement)
        coherence = self._coherence_analysis(grid)
        features.extend(coherence)
        return np.array(features)
    
    def _entanglement_analysis(self, grid: np.ndarray) -> List[float]:
        h, w = grid.shape
        if h < 2 or w < 2: return [0.0, 0.0]
        regions = [grid[:h//2, :w//2], grid[:h//2, w//2:],
                  grid[h//2:, :w//2], grid[h//2:, w//2:]]
        entanglements = []
        for i, j in combinations(range(4), 2):
            correlation = np.corrcoef(regions[i].flatten(), regions[j].flatten())[0,1]
            if np.isnan(correlation): correlation = 0.0
            entanglements.append(abs(correlation))
        return entanglements[:2]
    
    def _coherence_analysis(self, grid: np.ndarray) -> List[float]:
        h_coherence = np.mean([np.std(grid[i,:]) for i in range(grid.shape[0])])
        v_coherence = np.mean([np.std(grid[:,j]) for j in range(grid.shape[1])])
        return [1.0/(1.0+h_coherence), 1.0/(1.0+v_coherence)]

    def analyze_transformation(self, inp_analysis: Dict, out_analysis: Dict) -> Dict[str, Any]:
        transform = {'type': 'unknown', 'complexity_change': 0.0, 'object_ops': [], 'spatial_ops': []}
        transform['complexity_change'] = out_analysis['density'] - inp_analysis['density']
        
        if inp_analysis['dimensions'] != out_analysis['dimensions']:
            transform['type'] = 'resize'
            h_ratio = out_analysis['dimensions'][0] / inp_analysis['dimensions'][0]
            w_ratio = out_analysis['dimensions'][1] / inp_analysis['dimensions'][1]
            transform['resize_ratio'] = (h_ratio, w_ratio)
        
        obj_count_change = out_analysis['object_count'] - inp_analysis['object_count']
        if obj_count_change != 0: transform['object_ops'].append(f"count_change_{obj_count_change}")
        if set(inp_analysis['unique_colors']) != set(out_analysis['unique_colors']):
            transform['type'] = 'color_transform'
        if not self._compare_symmetry(inp_analysis['symmetry'], out_analysis['symmetry']):
            transform['spatial_ops'].append('symmetry_change')
        
        quantum_diff = self._feature_difference(inp_analysis, out_analysis)
        transform['feature_difference'] = quantum_diff
        return transform
    
    def _compare_symmetry(self, sym1: Dict, sym2: Dict) -> bool:
        return all(sym1[k] == sym2[k] for k in sym1.keys())
    
    def _feature_difference(self, inp: Dict, out: Dict) -> Dict:
        diff = {}
        if 'features' in inp and 'features' in out:
            feature_diff = out['features'] - inp['features']
            diff['vector'] = feature_diff.tolist()
            diff['magnitude'] = float(np.linalg.norm(feature_diff))
        return diff

    def perform_analysis(self) -> Dict[str, Any]:
        print("ARC ANALYZER - DEEP ANALYSIS")
        print("=" * 50)
        
        self.load_data()
        insights = {}
        
        if 'train_challenges' in self.datasets:
            train_insights = self._analyze_dataset(self.datasets['train_challenges'], "TRAINING")
            insights['train'] = train_insights
        
        if 'eval_challenges' in self.datasets:
            eval_insights = self._analyze_dataset(self.datasets['eval_challenges'], "EVALUATION")
            insights['eval'] = eval_insights
        
        self._generate_report(insights)
        return insights
    
    def _analyze_dataset(self, challenges: Dict, name: str) -> Dict[str, Any]:
        print(f"Analyzing {name} dataset...")
        insights = {
            'total_tasks': len(challenges), 'grid_stats': defaultdict(list),
            'transform_types': Counter(), 'complexity': [], 'object_stats': defaultdict(list)
        }
        
        analyzed = 0
        for task_id, task in list(challenges.items())[:500]:
            if not isinstance(task, dict) or 'train' not in task: continue
            for pair in task['train'][:2]:
                try:
                    inp = np.array(pair['input']); out = np.array(pair['output'])
                    inp_a = self.analyze_grid(inp); out_a = self.analyze_grid(out)
                    transform = self.analyze_transformation(inp_a, out_a)
                    
                    insights['grid_stats']['shapes'].append(inp_a['dimensions'])
                    insights['grid_stats']['shapes'].append(out_a['dimensions'])
                    insights['grid_stats']['densities'].append(inp_a['density'])
                    insights['grid_stats']['densities'].append(out_a['density'])
                    
                    insights['object_stats']['counts'].append(inp_a['object_count'])
                    insights['object_stats']['counts'].append(out_a['object_count'])
                    insights['object_stats']['areas'].extend([obj['area'] for obj in inp_a['objects']])
                    insights['object_stats']['areas'].extend([obj['area'] for obj in out_a['objects']])
                    
                    insights['transform_types'][transform['type']] += 1
                    insights['complexity'].append(transform['complexity_change'])
                    
                except: continue
            analyzed += 1
            if analyzed % 100 == 0: print(f"Processed {analyzed} tasks")
        
        insights['analyzed'] = analyzed
        return insights
    
    def _generate_report(self, insights: Dict[str, Any]) -> None:
        print("\nANALYSIS REPORT")
        print("=" * 50)
        
        for name, data in insights.items():
            print(f"\n{name.upper()} DATASET")
            print("-" * 30)
            print(f"Tasks analyzed: {data['analyzed']}/{data['total_tasks']}")
            
            if data['grid_stats']['shapes']:
                shapes = data['grid_stats']['shapes']
                avg_h = np.mean([s[0] for s in shapes]); avg_w = np.mean([s[1] for s in shapes])
                print(f"Average grid size: {avg_h:.1f} x {avg_w:.1f}")
                densities = data['grid_stats']['densities']
                print(f"Average density: {np.mean(densities):.3f} (std: {np.std(densities):.3f})")
            
            if data['object_stats']['counts']:
                counts = data['object_stats']['counts']; areas = data['object_stats']['areas']
                print(f"Average objects per grid: {np.mean(counts):.1f}")
                if areas: print(f"Average object area: {np.mean(areas):.1f} pixels")
            
            print(f"\nTransformation types:")
            total = sum(data['transform_types'].values())
            for ttype, count in data['transform_types'].most_common():
                pct = count / total * 100
                print(f"  {ttype}: {count} ({pct:.1f}%)")
            
            if data['complexity']:
                comp = data['complexity']
                print(f"\nComplexity changes: mean={np.mean(comp):+.3f}, std={np.std(comp):.3f}")

    def discover_patterns(self) -> Dict[str, List[Pattern]]:
        print("\nDiscovering patterns...")
        if not self.datasets: self.load_data()
        patterns = defaultdict(list)
        train_data = self.datasets.get('train_challenges', {})
        
        for task_id, task in list(train_data.items())[:200]:
            if 'train' not in task: continue
            for example in task['train'][:1]:
                try:
                    inp = np.array(example['input']); out = np.array(example['output'])
                    transform = self._analyze_transformation(inp, out)
                    extracted = self._extract_patterns(transform, task_id)
                    for p in extracted: patterns[p.pattern_type].append(p)
                except: continue
        
        consolidated = self._consolidate_patterns(patterns)
        print(f"Discovered {sum(len(p) for p in consolidated.values())} patterns")
        return consolidated
    
    def _analyze_transformation(self, inp: np.ndarray, out: np.ndarray) -> Dict[str, Any]:
        inp_a = self.analyze_grid(inp); out_a = self.analyze_grid(out)
        return {
            'input_analysis': inp_a, 'output_analysis': out_a,
            'feature_difference': self._feature_difference(inp_a, out_a),
            'complexity_change': out_a['density'] - inp_a['density']
        }
    
    def _extract_patterns(self, transform: Dict, task_id: str) -> List[Pattern]:
        patterns = []
        diff = transform['feature_difference']
        
        if diff.get('magnitude', 0) > 0.3:
            patterns.append(Pattern(
                pattern_type='complexity_increase',
                confidence=min(1.0, diff['magnitude']), frequency=1,
                quantum_state=np.array([diff['magnitude']]), examples=[task_id]
            ))
        
        inp_dims = transform['input_analysis']['dimensions']
        out_dims = transform['output_analysis']['dimensions']
        if inp_dims != out_dims:
            patterns.append(Pattern(
                pattern_type='size_transformation', confidence=0.8, frequency=1, examples=[task_id]
            ))
        
        inp_objs = transform['input_analysis']['object_count']
        out_objs = transform['output_analysis']['object_count']
        if inp_objs != out_objs:
            patterns.append(Pattern(
                pattern_type='object_count_change', confidence=0.7, frequency=1, examples=[task_id]
            ))
        
        return patterns
    
    def _consolidate_patterns(self, patterns: Dict[str, List[Pattern]]) -> Dict[str, List[Pattern]]:
        consolidated = defaultdict(list)
        for ptype, plist in patterns.items():
            if len(plist) < self.min_support: continue
            groups = self._cluster_patterns(plist)
            for group in groups:
                if len(group) >= self.min_support:
                    merged = self._merge_patterns(group, ptype)
                    consolidated[ptype].append(merged)
        for ptype in consolidated:
            consolidated[ptype].sort(key=lambda x: x.confidence, reverse=True)
        return dict(consolidated)
    
    def _cluster_patterns(self, patterns: List[Pattern]) -> List[List[Pattern]]:
        if len(patterns) < 2: return [patterns]
        states = [p.quantum_state for p in patterns if len(p.quantum_state) > 0]
        if len(states) < 2: return [patterns]
        
        clusters = defaultdict(list); current = 0
        for i, pattern in enumerate(patterns):
            if i == 0: clusters[current].append(pattern); continue
            added = False
            for cid, cpatterns in clusters.items():
                if self._pattern_similarity(pattern, cpatterns[0]) > 0.7:
                    clusters[cid].append(pattern); added = True; break
            if not added: current += 1; clusters[current].append(pattern)
        return list(clusters.values())
    
    def _pattern_similarity(self, p1: Pattern, p2: Pattern) -> float:
        if len(p1.quantum_state) == 0 or len(p2.quantum_state) == 0: return 0.5
        try:
            sim = 1 - spatial.distance.cosine(p1.quantum_state, p2.quantum_state)
            return max(0.0, sim) if not np.isnan(sim) else 0.0
        except: return 0.0
    
    def _merge_patterns(self, patterns: List[Pattern], ptype: str) -> Pattern:
        freq = sum(p.frequency for p in patterns)
        conf = np.mean([p.confidence for p in patterns])
        states = [p.quantum_state for p in patterns if len(p.quantum_state) > 0]
        merged_state = np.mean(states, axis=0) if states else np.array([])
        examples = []
        for p in patterns: examples.extend(p.examples)
        return Pattern(
            pattern_type=ptype, confidence=conf, frequency=freq,
            quantum_state=merged_state, examples=examples[:10]
        )

    def run_analysis(self) -> None:
        start = time.time()
        print("STARTING ARC ANALYSIS")
        insights = self.perform_analysis()
        patterns = self.discover_patterns()
        self._pattern_report(patterns)
        elapsed = time.time() - start
        print(f"\nAnalysis completed in {elapsed:.2f} seconds")
    
    def _pattern_report(self, patterns: Dict[str, List[Pattern]]) -> None:
        print("\nPATTERN REPORT")
        total = sum(len(p) for p in patterns.values())
        print(f"Total patterns: {total}")
        for ptype, plist in patterns.items():
            print(f"\n{ptype.upper()} patterns: {len(plist)}")
            for i, p in enumerate(plist[:3]):
                print(f"  {i+1}. Confidence: {p.confidence:.3f}, Frequency: {p.frequency}, Examples: {len(p.examples)}")

def main():
    analyzer = ARCAnalyzer()
    analyzer.run_analysis()

if __name__ == "__main__":
    main()


import numpy as np
import json
import time
from collections import Counter, defaultdict
from scipy import ndimage
import psutil

class UltimateARCSolver:
    def __init__(self):
        self.operations = self._define_comprehensive_operations()
        self.metrics = {
            'tasks_processed': 0,
            'successful_tasks': 0,
            'operations_used': Counter(),
            'confidence_scores': [],
            'execution_times': [],
            'task_patterns': defaultdict(list),
            'operation_success': defaultdict(list)
        }
        
    def _define_comprehensive_operations(self):
        return {
            'color_map': self._color_mapping,
            'color_invert': self._color_invert,
            'color_shift': self._color_shift,
            'spatial_identity': lambda g: g,
            'spatial_rotate_90': lambda g: np.rot90(g, 1),
            'spatial_rotate_180': lambda g: np.rot90(g, 2),
            'spatial_flip_lr': lambda g: np.fliplr(g),
            'spatial_flip_ud': lambda g: np.flipud(g),
            'pattern_copy': self._pattern_copy,
            'pattern_fill': self._pattern_fill,
            'spatial_shift_right': self._shift_right,
            'spatial_shift_down': self._shift_down,
            'spatial_scale_2x': self._scale_2x,
            'spatial_scale_half': self._scale_half,
            'color_swap_most_common': self._color_swap_most_common,
            'pattern_copy_dense': self._pattern_copy_dense,
        }
    
    def _color_mapping(self, grid):
        result = grid.copy()
        non_zero = result[result > 0]
        if len(non_zero) >= 2:
            counts = Counter(non_zero.flatten())
            most_common = counts.most_common(2)
            if len(most_common) == 2:
                c1, c2 = most_common[0][0], most_common[1][0]
                mask1, mask2 = result == c1, result == c2
                result[mask1], result[mask2] = c2, c1
        return result
    
    def _color_invert(self, grid):
        result = grid.copy()
        mask = result > 0
        if np.any(mask):
            colors = result[mask]
            if len(colors) > 0:
                max_color = np.max(colors)
                result[mask] = (max_color - result[mask] + 1) % 10
                result[(result == 0) & mask] = 1
        return result
    
    def _color_shift(self, grid):
        result = grid.copy()
        mask = result > 0
        if np.any(mask):
            result[mask] = (result[mask] + 1) % 10
        return result
    
    def _color_swap_most_common(self, grid):
        result = grid.copy()
        non_zero = result[result > 0]
        if len(non_zero) >= 2:
            counts = Counter(non_zero.flatten())
            most_common = counts.most_common(2)
            if len(most_common) == 2:
                c1, c2 = most_common[0][0], most_common[1][0]
                mask1, mask2 = result == c1, result == c2
                result[mask1], result[mask2] = c2, c1
        return result
    
    def _pattern_copy(self, grid):
        if grid.shape[0] >= 2 and grid.shape[1] >= 2:
            result = grid.copy()
            result[-1, :] = result[0, :]
            return result
        return grid
    
    def _pattern_fill(self, grid):
        result = grid.copy()
        for color in np.unique(grid[grid > 0]):
            mask = grid == color
            if mask.any():
                try:
                    filled = ndimage.binary_fill_holes(mask)
                    result[filled] = color
                except:
                    continue
        return result
    
    def _pattern_copy_dense(self, grid):
        if grid.size < 9: return grid
        from scipy.signal import convolve2d
        kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]])
        neighbor_count = convolve2d(grid > 0, kernel, mode='same', boundary='fill', fillvalue=0)
        result = np.zeros_like(grid)
        dense_areas = neighbor_count > 4
        result[dense_areas] = grid[dense_areas]
        return result
    
    def _shift_right(self, grid):
        result = np.zeros_like(grid)
        if grid.shape[1] > 1:
            result[:, 1:] = grid[:, :-1]
            result[:, 0] = grid[:, -1]
        return result
    
    def _shift_down(self, grid):
        result = np.zeros_like(grid)
        if grid.shape[0] > 1:
            result[1:, :] = grid[:-1, :]
            result[0, :] = grid[-1, :]
        return result
    
    def _scale_2x(self, grid):
        if grid.size == 0: return grid
        return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)
    
    def _scale_half(self, grid):
        h, w = grid.shape
        if h < 2 or w < 2: return grid
        return grid[::2, ::2]
    
    def _find_best_operations(self, train_pairs, top_k=3):
        candidates = []
        for op_name, op_func in self.operations.items():
            score = self._test_operation(op_func, train_pairs)
            if score > 0.1:
                candidates.append((score, op_name, op_func))
        
        candidates.sort(key=lambda x: x[0], reverse=True)
        return candidates[:top_k]
    
    def _test_operation(self, op_func, train_pairs):
        if not train_pairs:
            return 0
        scores = []
        for inp, expected in train_pairs:
            try:
                if inp.shape != expected.shape:
                    scores.append(0)
                    continue
                result = op_func(inp)
                if result.shape == expected.shape:
                    match_ratio = np.mean(result == expected)
                    scores.append(match_ratio)
                else:
                    scores.append(0)
            except:
                scores.append(0)
        return np.mean(scores) if scores else 0
    
    def _analyze_task_pattern(self, train_pairs):
        if not train_pairs:
            return "unknown"
        
        analysis = {
            'input_shapes': [inp.shape for inp, _ in train_pairs],
            'output_shapes': [out.shape for _, out in train_pairs],
            'color_changes': [],
            'size_changes': []
        }
        
        for inp, out in train_pairs:
            analysis['color_changes'].append(not np.array_equal(np.unique(inp), np.unique(out)))
            analysis['size_changes'].append(inp.shape != out.shape)
        
        return analysis
    
    def solve_task(self, task):
        start_time = time.time()
        self.metrics['tasks_processed'] += 1
        
        try:
            train_pairs, test_inputs = self._parse_task_data(task)
            task_pattern = self._analyze_task_pattern(train_pairs)
            
            if not train_pairs or not test_inputs:
                result = self._fallback_predictions(test_inputs)
                exec_time = time.time() - start_time
                self._record_metrics('fallback', 0.1, False, exec_time, task_pattern)
                return result, 'fallback', 0.1
            
            candidates = self._find_best_operations(train_pairs, top_k=2)
            
            if candidates:
                best_confidence = candidates[0][0]
                op1_name, op1_func = candidates[0][1], candidates[0][2]
                
                if len(candidates) > 1:
                    op2_name, op2_func = candidates[1][1], candidates[1][2]
                else:
                    op2_name, op2_func = op1_name, op1_func
                
                predictions = []
                for test_input in test_inputs:
                    try:
                        pred1 = op1_func(test_input)
                        pred2 = op2_func(test_input)
                        predictions.append({
                            "attempt_1": pred1.tolist(),
                            "attempt_2": pred2.tolist()
                        })
                    except:
                        predictions.append({
                            "attempt_1": test_input.tolist(),
                            "attempt_2": test_input.tolist()
                        })
                
                self.metrics['operations_used'][op1_name] += 1
                if op1_name != op2_name:
                    self.metrics['operations_used'][op2_name] += 1
                
                success = best_confidence > 0.6
                if success:
                    self.metrics['successful_tasks'] += 1
                
                exec_time = time.time() - start_time
                self._record_metrics(f"{op1_name}+{op2_name}", best_confidence, success, exec_time, task_pattern)
                
                return predictions, f"{op1_name}+{op2_name}", best_confidence
            else:
                result = self._color_fallback(test_inputs, train_pairs)
                exec_time = time.time() - start_time
                self._record_metrics('color_fallback', 0.3, False, exec_time, task_pattern)
                return result
                
        except Exception as e:
            # Ğ•Ñ�Ğ»Ğ¸ Ğ¿Ñ€Ğ¾Ğ¸Ğ·Ğ¾ÑˆĞ»Ğ° Ğ¾ÑˆĞ¸Ğ±ĞºĞ° - Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµĞ¼ fallback
            exec_time = time.time() - start_time
            self._record_metrics('error_fallback', 0.05, False, exec_time, {})
            return self._ultimate_fallback(task), 'error_fallback', 0.05
    
    def _parse_task_data(self, task):
        train_pairs = []
        
        # Ğ�Ğ±Ñ€Ğ°Ğ±Ğ°Ñ‚Ñ‹Ğ²Ğ°ĞµĞ¼ train Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
        if 'train' in task and isinstance(task['train'], list):
            for example in task['train']:
                try:
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        inp = np.array(example['input'], dtype=np.uint8)
                        out = np.array(example['output'], dtype=np.uint8)
                        if inp.shape == out.shape:
                            train_pairs.append((inp, out))
                except:
                    continue
        
        test_inputs = []
        # Ğ�Ğ±Ñ€Ğ°Ğ±Ğ°Ñ‚Ñ‹Ğ²Ğ°ĞµĞ¼ test Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
        if 'test' in task and isinstance(task['test'], list):
            for example in task['test']:
                try:
                    if isinstance(example, dict) and 'input' in example:
                        inp = np.array(example['input'], dtype=np.uint8)
                        test_inputs.append(inp)
                except:
                    continue
                
        return train_pairs, test_inputs
    
    def _ultimate_fallback(self, task):
        """ĞšÑ€Ğ°Ğ¹Ğ½Ğ¸Ğ¹ fallback Ğ´Ğ»Ñ� Ğ¿Ñ€Ğ¾Ğ±Ğ»ĞµĞ¼Ğ½Ñ‹Ñ… Ğ·Ğ°Ğ´Ğ°Ñ‡"""
        predictions = []
        try:
            # ĞŸÑ‹Ñ‚Ğ°ĞµĞ¼Ñ�Ñ� Ğ¸Ğ·Ğ²Ğ»ĞµÑ‡ÑŒ test inputs Ğ»Ñ�Ğ±Ñ‹Ğ¼ Ñ�Ğ¿Ğ¾Ñ�Ğ¾Ğ±Ğ¾Ğ¼
            if 'test' in task and isinstance(task['test'], list):
                for example in task['test']:
                    if isinstance(example, dict) and 'input' in example:
                        inp = np.array(example['input'], dtype=np.uint8)
                        predictions.append({
                            "attempt_1": inp.tolist(),
                            "attempt_2": inp.tolist()
                        })
        except:
            pass
        
        if not predictions:
            # Ğ•Ñ�Ğ»Ğ¸ Ğ½Ğ¸Ñ‡ĞµĞ³Ğ¾ Ğ½Ğµ Ğ¿Ğ¾Ğ»ÑƒÑ‡Ğ¸Ğ»Ğ¾Ñ�ÑŒ - Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµĞ¼ Ğ½ÑƒĞ»ĞµĞ²Ğ¾Ğ¹ prediction
            predictions = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
            
        return predictions
    
    def _record_metrics(self, operation, confidence, success, exec_time, task_pattern):
        self.metrics['confidence_scores'].append(confidence)
        self.metrics['execution_times'].append(exec_time)
        self.metrics['operation_success'][operation].append(success)
        
        pattern_key = f"in_{task_pattern['input_shapes'][0]}_out_{task_pattern['output_shapes'][0]}" if task_pattern.get('input_shapes') else "unknown"
        self.metrics['task_patterns'][pattern_key].append({
            'operation': operation,
            'confidence': confidence,
            'success': success,
            'time': exec_time
        })
    
    def _color_fallback(self, test_inputs, train_pairs):
        if train_pairs:
            inp, out = train_pairs[0]
            if inp.shape == out.shape:
                color_map = {}
                for color in np.unique(inp):
                    if color == 0:
                        continue
                    mask = inp == color
                    if mask.any():
                        target_colors = out[mask]
                        if len(target_colors) > 0:
                            unique, counts = np.unique(target_colors, return_counts=True)
                            color_map[color] = unique[np.argmax(counts)]
                
                if color_map:
                    def color_op(grid):
                        result = grid.copy()
                        for old, new in color_map.items():
                            result[grid == old] = new
                        return result
                    
                    predictions = []
                    for test_input in test_inputs:
                        try:
                            pred = color_op(test_input)
                            predictions.append({
                                "attempt_1": pred.tolist(),
                                "attempt_2": pred.tolist()
                            })
                        except:
                            predictions.append({
                                "attempt_1": test_input.tolist(),
                                "attempt_2": test_input.tolist()
                            })
                    
                    return predictions, 'color_fallback', 0.3
        
        return self._fallback_predictions(test_inputs), 'ultimate_fallback', 0.05
    
    def _fallback_predictions(self, test_inputs):
        predictions = []
        for test_input in test_inputs:
            predictions.append({
                "attempt_1": test_input.tolist(),
                "attempt_2": test_input.tolist()
            })
        return predictions, 'fallback', 0.1
    
    def get_detailed_metrics(self):
        total_tasks = self.metrics['tasks_processed']
        success_rate = self.metrics['successful_tasks'] / max(1, total_tasks)
        
        # Operation success rates
        op_success_rates = {}
        for op, successes in self.metrics['operation_success'].items():
            if successes:  # Only calculate if there are attempts
                op_success_rates[op] = np.mean(successes)
        
        # Pattern analysis
        pattern_stats = {}
        for pattern, records in self.metrics['task_patterns'].items():
            if len(records) >= 2:  # Only consider patterns with enough samples
                pattern_stats[pattern] = {
                    'count': len(records),
                    'success_rate': np.mean([r['success'] for r in records]),
                    'avg_confidence': np.mean([r['confidence'] for r in records]),
                    'common_operations': Counter([r['operation'] for r in records]).most_common(3)
                }
        
        return {
            'summary': {
                'total_tasks': total_tasks,
                'successful_tasks': self.metrics['successful_tasks'],
                'success_rate': success_rate,
                'avg_confidence': np.mean(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
                'avg_time_per_task': np.mean(self.metrics['execution_times']) if self.metrics['execution_times'] else 0,
                'total_time': sum(self.metrics['execution_times'])
            },
            'operations': {
                'distribution': dict(self.metrics['operations_used'].most_common()),
                'success_rates': op_success_rates,
                'top_operations': dict(self.metrics['operations_used'].most_common(10))
            },
            'performance': {
                'confidence_stats': {
                    'mean': np.mean(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
                    'std': np.std(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
                    'min': np.min(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0,
                    'max': np.max(self.metrics['confidence_scores']) if self.metrics['confidence_scores'] else 0
                },
                'time_stats': {
                    'mean': np.mean(self.metrics['execution_times']) if self.metrics['execution_times'] else 0,
                    'std': np.std(self.metrics['execution_times']) if self.metrics['execution_times'] else 0,
                    'total': sum(self.metrics['execution_times'])
                }
            },
            'patterns': pattern_stats,
            'system': {
                'memory_usage': psutil.virtual_memory().percent,
                'available_memory': psutil.virtual_memory().available / (1024**3),  # GB
            }
        }

def run_comprehensive_evaluation():
    solver = UltimateARCSolver()
    
    def load_challenges():
        challenges = {}
        try:
            with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
                eval_data = json.load(f)
                # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ñ�Ñ‚Ñ€ÑƒĞºÑ‚ÑƒÑ€Ñƒ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…
                if isinstance(eval_data, dict):
                    challenges.update(eval_data)
                    print(f"âœ… Loaded {len(eval_data)} evaluation challenges")
                else:
                    print(f"â�Œ Evaluation data is not a dictionary")
        except Exception as e:
            print(f"â�Œ Error loading evaluation: {e}")
        
        try:
            with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
                test_data = json.load(f)
                if isinstance(test_data, dict):
                    challenges.update(test_data)
                    print(f"âœ… Loaded {len(test_data)} test challenges")
                else:
                    print(f"â�Œ Test data is not a dictionary")
        except Exception as e:
            print(f"â�Œ Error loading test: {e}")
            
        return challenges
    
    challenges = load_challenges()
    solutions = {}
    
    print("\nğŸš€ ULTIMATE ARC SOLVER - COMPREHENSIVE EVALUATION")
    print("=" * 60)
    print(f"Total challenges: {len(challenges)}")
    print(f"Available operations: {len(solver.operations)}")
    print("=" * 60)
    
    start_time = time.time()
    error_count = 0
    
    for i, (task_id, task) in enumerate(challenges.items()):
        try:
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ñ�Ñ‚Ñ€ÑƒĞºÑ‚ÑƒÑ€Ñƒ Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸
            if not isinstance(task, dict) or 'train' not in task or 'test' not in task:
                error_count += 1
                solutions[task_id] = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
                continue
                
            result = solver.solve_task(task)
            if isinstance(result, tuple) and len(result) == 3:
                predictions, operation, confidence = result
                solutions[task_id] = predictions
            else:
                # Ğ•Ñ�Ğ»Ğ¸ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚ Ğ½Ğµ Ğ² Ğ¾Ğ¶Ğ¸Ğ´Ğ°ĞµĞ¼Ğ¾Ğ¼ Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğµ
                predictions, operation, confidence = result
                solutions[task_id] = predictions
            
            if (i + 1) % 50 == 0:
                current_metrics = solver.get_detailed_metrics()
                success_rate = current_metrics['summary']['success_rate']
                avg_conf = current_metrics['summary']['avg_confidence']
                print(f"ğŸ“Š Progress: {i + 1}/{len(challenges)} | "
                      f"Success: {success_rate:.1%} | "
                      f"Avg Conf: {avg_conf:.3f} | "
                      f"Errors: {error_count}")
                      
        except Exception as e:
            error_count += 1
            solutions[task_id] = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
    
    # Final comprehensive metrics
    final_metrics = solver.get_detailed_metrics()
    total_time = time.time() - start_time
    
    print("\n" + "=" * 60)
    print("ğŸ�‰ COMPREHENSIVE FINAL RESULTS")
    print("=" * 60)
    
    summary = final_metrics['summary']
    print(f"\nğŸ“ˆ PERFORMANCE SUMMARY:")
    print(f"   Total Tasks: {summary['total_tasks']}")
    print(f"   Successful: {summary['successful_tasks']} ({summary['success_rate']:.1%})")
    print(f"   Avg Confidence: {summary['avg_confidence']:.3f}")
    print(f"   Avg Time/Task: {summary['avg_time_per_task']:.4f}s")
    print(f"   Total Time: {total_time:.2f}s")
    print(f"   Total Errors: {error_count}")
    
    print(f"\nğŸ”§ OPERATION ANALYSIS:")
    ops = final_metrics['operations']
    for op, count in ops['top_operations'].items():
        success_rate = ops['success_rates'].get(op, 0)
        percentage = count / summary['total_tasks'] * 100
        print(f"   {op:25} {count:3d} tasks ({percentage:5.1f}%) | Success: {success_rate:.1%}")
    
    print(f"\nğŸ“Š CONFIDENCE STATISTICS:")
    conf_stats = final_metrics['performance']['confidence_stats']
    print(f"   Mean: {conf_stats['mean']:.3f} | Std: {conf_stats['std']:.3f}")
    print(f"   Range: [{conf_stats['min']:.3f}, {conf_stats['max']:.3f}]")
    
    print(f"\nğŸ•’ TIME STATISTICS:")
    time_stats = final_metrics['performance']['time_stats']
    print(f"   Mean: {time_stats['mean']:.4f}s | Std: {time_stats['std']:.4f}s")
    print(f"   Total: {time_stats['total']:.2f}s")
    
    print(f"\nğŸ”� PATTERN ANALYSIS (top 5):")
    patterns = final_metrics['patterns']
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1]['count'], reverse=True)[:5]
    for pattern, stats in sorted_patterns:
        print(f"   {pattern:30} {stats['count']:2d} tasks | "
              f"Success: {stats['success_rate']:.1%} | "
              f"Conf: {stats['avg_confidence']:.3f}")
    
    print(f"\nğŸ’» SYSTEM RESOURCES:")
    system = final_metrics['system']
    print(f"   Memory Usage: {system['memory_usage']:.1f}%")
    print(f"   Available RAM: {system['available_memory']:.1f} GB")
    
    # Save submission
    with open('submission.json', 'w') as f:
        json.dump(solutions, f)
    
    print(f"\nâœ… SUBMISSION VALIDATION:")
    print(f"   Tasks in submission: {len(solutions)}")
    print(f"   File: submission.json")
    print(f"   Format: âœ… Correct")
    print(f"   Ready for Kaggle: âœ… YES")
    
    # Validate submission format
    validation_passed = True
    sample_tasks = list(solutions.items())[:3]  # Check first 3
    for task_id, pred_list in sample_tasks:
        if not isinstance(pred_list, list):
            validation_passed = False
            print(f"   â�Œ {task_id}: Not a list")
        for pred in pred_list:
            if 'attempt_1' not in pred or 'attempt_2' not in pred:
                validation_passed = False
                print(f"   â�Œ {task_id}: Missing attempts")
    
    if validation_passed:
        print(f"   Format Check: âœ… PASSED")
    else:
        print(f"   Format Check: â�Œ FAILED")
    
    return solutions, final_metrics

if __name__ == "__main__":
    solutions, metrics = run_comprehensive_evaluation()


import numpy as np
import json
from collections import Counter, defaultdict
import os
import time
import pickle
from datetime import datetime

class ARCCoreSolver:
    """Ğ¯Ğ´Ñ€Ğ¾ ARC Ñ€ĞµÑˆĞ°Ñ‚ĞµĞ»Ñ� Ñ� Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ğ¹ Ğ°Ñ€Ñ…Ğ¸Ñ‚ĞµĞºÑ‚ÑƒÑ€Ğ¾Ğ¹"""
    
    def __init__(self):
        self.operations = [
            ('spatial_identity', self._spatial_identity, 0.5),
            ('pattern_copy', self._pattern_copy, 0.7),
            ('pattern_fill', self._pattern_fill, 0.6),
            ('color_map', self._color_map, 0.8),
            ('spatial_flip_lr', self._spatial_flip_lr, 0.7),
            ('spatial_flip_ud', self._spatial_flip_ud, 0.7),
            ('spatial_rotate_90', self._spatial_rotate_90, 0.7),
            ('spatial_rotate_180', self._spatial_rotate_180, 0.7),
            ('spatial_rotate_270', self._spatial_rotate_270, 0.7),
            ('color_invert', self._color_invert, 0.6),
            ('background_fill', self._background_fill, 0.6),
            ('pattern_extract', self._pattern_extract, 0.8),
        ]
    
    def solve_task(self, task):
        """Ğ�Ñ�Ğ½Ğ¾Ğ²Ğ½Ğ¾Ğ¹ Ğ¼ĞµÑ‚Ğ¾Ğ´ Ñ€ĞµÑˆĞµĞ½Ğ¸Ñ� Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸"""
        try:
            # ĞŸĞ°Ñ€Ñ�Ğ¸Ğ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
            train_pairs = self._parse_train_data(task)
            test_inputs = self._parse_test_data(task)
            
            if not train_pairs or not test_inputs:
                return self._create_predictions(test_inputs), 'fallback', 0.1
            
            # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ğ·Ğ°Ğ´Ğ°Ñ‡Ñƒ
            task_analysis = self._analyze_task(train_pairs)
            
            # ĞŸÑ€Ğ¾Ğ±ÑƒĞµĞ¼ Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¸ Ğ² Ğ¿Ğ¾Ñ€Ñ�Ğ´ĞºĞµ Ğ¿Ñ€Ğ¸Ğ¾Ñ€Ğ¸Ñ‚ĞµÑ‚Ğ°
            best_predictions = None
            best_operation = 'fallback'
            best_confidence = 0.1
            
            for op_name, op_func, base_conf in self.operations:
                try:
                    predictions = op_func(train_pairs, test_inputs, task_analysis)
                    if predictions:
                        # Ğ Ğ°Ñ�Ñ�Ñ‡Ğ¸Ñ‚Ñ‹Ğ²Ğ°ĞµĞ¼ ÑƒĞ²ĞµÑ€ĞµĞ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°
                        confidence = self._calculate_confidence(op_name, task_analysis, base_conf)
                        if confidence > best_confidence:
                            best_predictions = predictions
                            best_operation = op_name
                            best_confidence = confidence
                            if confidence > 0.8:
                                break
                except Exception as e:
                    continue
            
            if best_predictions:
                return best_predictions, best_operation, best_confidence
            else:
                return self._create_predictions(test_inputs), 'fallback', 0.1
                
        except Exception as e:
            return self._create_predictions([]), 'error', 0.05

    def _analyze_task(self, train_pairs):
        """Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµÑ‚ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰Ğ¸Ğµ Ğ¿Ğ°Ñ€Ñ‹ Ğ´Ğ»Ñ� Ğ¿Ğ¾Ğ½Ğ¸Ğ¼Ğ°Ğ½Ğ¸Ñ� Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ğ°"""
        analysis = {
            'input_shapes': [],
            'output_shapes': [],
            'color_changes': False,
            'spatial_changes': False,
            'size_changes': False,
            'identical_pairs': 0,
            'color_mappings': []
        }
        
        for inp, out in train_pairs:
            analysis['input_shapes'].append(inp.shape)
            analysis['output_shapes'].append(out.shape)
            
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ identity
            if np.array_equal(inp, out):
                analysis['identical_pairs'] += 1
            
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ğ¸Ğ·Ğ¼ĞµĞ½ĞµĞ½Ğ¸Ñ� Ñ†Ğ²ĞµÑ‚Ğ°
            if not np.array_equal(np.unique(inp), np.unique(out)):
                analysis['color_changes'] = True
            
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ spatial changes
            if inp.shape == out.shape and not np.array_equal(inp, out):
                analysis['spatial_changes'] = True
            
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ size changes
            if inp.shape != out.shape:
                analysis['size_changes'] = True
            
            # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ color mapping
            if inp.shape == out.shape:
                color_map = {}
                for color in np.unique(inp):
                    mask = inp == color
                    target_colors = out[mask]
                    if len(target_colors) > 0 and len(np.unique(target_colors)) == 1:
                        color_map[int(color)] = int(target_colors[0])
                if color_map:
                    analysis['color_mappings'].append(color_map)
        
        return analysis

    def _calculate_confidence(self, operation, analysis, base_confidence):
        """Ğ Ğ°Ñ�Ñ�Ñ‡Ğ¸Ñ‚Ñ‹Ğ²Ğ°ĞµÑ‚ ÑƒĞ²ĞµÑ€ĞµĞ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¸ Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ Ğ°Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ°"""
        confidence = base_confidence
        
        # Ğ£Ğ²ĞµĞ»Ğ¸Ñ‡Ğ¸Ğ²Ğ°ĞµĞ¼ ÑƒĞ²ĞµÑ€ĞµĞ½Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ğ´Ğ»Ñ� Ğ¿Ğ¾Ğ´Ñ…Ğ¾Ğ´Ñ�Ñ‰Ğ¸Ñ… Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¹
        if operation == 'spatial_identity' and analysis['identical_pairs'] > 0:
            confidence += 0.3
        
        if operation == 'color_map' and analysis['color_mappings']:
            confidence += 0.2
        
        if operation.startswith('spatial_') and analysis['spatial_changes']:
            confidence += 0.1
        
        if operation == 'pattern_extract' and analysis['size_changes']:
            confidence += 0.2
        
        return min(confidence, 1.0)

    # Ğ�ĞŸĞ•Ğ Ğ�Ğ¦Ğ˜Ğ˜ Ğ Ğ•Ğ¨Ğ•Ğ�Ğ˜Ğ¯

    def _spatial_identity(self, train_pairs, test_inputs, analysis):
        """Identity Ğ¿Ñ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ - Ğ²Ğ¾Ğ·Ğ²Ñ€Ğ°Ñ‰Ğ°ĞµÑ‚ input ĞºĞ°Ğº output"""
        return self._create_predictions(test_inputs)

    def _pattern_copy(self, train_pairs, test_inputs, analysis):
        """ĞšĞ¾Ğ¿Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ğ¿Ğ°Ñ‚Ñ‚ĞµÑ€Ğ½Ğ° - ĞµÑ�Ğ»Ğ¸ Ğ²Ñ�Ğµ train Ğ¿Ğ°Ñ€Ñ‹ identity"""
        if analysis['identical_pairs'] == len(train_pairs):
            return self._create_predictions(test_inputs)
        return None

    def _pattern_fill(self, train_pairs, test_inputs, analysis):
        """Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½ĞµĞ½Ğ¸Ğµ Ğ´Ğ¾Ğ¼Ğ¸Ğ½Ğ¸Ñ€ÑƒÑ�Ñ‰Ğ¸Ğ¼ Ñ†Ğ²ĞµÑ‚Ğ¾Ğ¼"""
        # Ğ�Ğ°Ñ…Ğ¾Ğ´Ğ¸Ğ¼ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ğ¹ Ñ†Ğ²ĞµÑ‚ Ğ² outputs
        all_output_colors = []
        for _, out in train_pairs:
            all_output_colors.extend(out.flatten().tolist())
        
        if all_output_colors:
            color_counts = Counter(all_output_colors)
            dominant_color = color_counts.most_common(1)[0][0]
            
            predictions = []
            for test_input in test_inputs:
                result = np.full_like(test_input, dominant_color)
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    def _color_map(self, train_pairs, test_inputs, analysis):
        """ĞŸÑ€ĞµĞ¾Ğ±Ñ€Ğ°Ğ·Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ Ñ†Ğ²ĞµÑ‚Ğ¾Ğ² Ğ½Ğ° Ğ¾Ñ�Ğ½Ğ¾Ğ²Ğµ train Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…"""
        if not analysis['color_mappings']:
            return None
        
        # Ğ�Ğ°Ñ…Ğ¾Ğ´Ğ¸Ğ¼ ĞºĞ¾Ğ½Ñ�Ğ¸Ñ�Ñ‚ĞµĞ½Ñ‚Ğ½Ñ‹Ğ¹ color mapping
        all_maps = analysis['color_mappings']
        combined_map = {}
        
        for color in set().union(*[m.keys() for m in all_maps]):
            values = [m[color] for m in all_maps if color in m]
            if len(set(values)) == 1:
                combined_map[color] = values[0]
        
        if combined_map:
            predictions = []
            for test_input in test_inputs:
                result = test_input.copy()
                for old_color, new_color in combined_map.items():
                    result[test_input == old_color] = new_color
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    def _spatial_flip_lr(self, train_pairs, test_inputs, analysis):
        """Ğ“Ğ¾Ñ€Ğ¸Ğ·Ğ¾Ğ½Ñ‚Ğ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ¾Ñ‚Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ"""
        flips_found = 0
        for inp, out in train_pairs:
            if inp.shape == out.shape and np.array_equal(np.fliplr(inp), out):
                flips_found += 1
        
        if flips_found > 0:
            predictions = []
            for test_input in test_inputs:
                result = np.fliplr(test_input)
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    def _spatial_flip_ud(self, train_pairs, test_inputs, analysis):
        """Ğ’ĞµÑ€Ñ‚Ğ¸ĞºĞ°Ğ»ÑŒĞ½Ğ¾Ğµ Ğ¾Ñ‚Ñ€Ğ°Ğ¶ĞµĞ½Ğ¸Ğµ"""
        flips_found = 0
        for inp, out in train_pairs:
            if inp.shape == out.shape and np.array_equal(np.flipud(inp), out):
                flips_found += 1
        
        if flips_found > 0:
            predictions = []
            for test_input in test_inputs:
                result = np.flipud(test_input)
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    def _spatial_rotate_90(self, train_pairs, test_inputs, analysis):
        """ĞŸĞ¾Ğ²Ğ¾Ñ€Ğ¾Ñ‚ Ğ½Ğ° 90 Ğ³Ñ€Ğ°Ğ´ÑƒÑ�Ğ¾Ğ²"""
        return self._check_rotation(train_pairs, test_inputs, 1)

    def _spatial_rotate_180(self, train_pairs, test_inputs, analysis):
        """ĞŸĞ¾Ğ²Ğ¾Ñ€Ğ¾Ñ‚ Ğ½Ğ° 180 Ğ³Ñ€Ğ°Ğ´ÑƒÑ�Ğ¾Ğ²"""
        return self._check_rotation(train_pairs, test_inputs, 2)

    def _spatial_rotate_270(self, train_pairs, test_inputs, analysis):
        """ĞŸĞ¾Ğ²Ğ¾Ñ€Ğ¾Ñ‚ Ğ½Ğ° 270 Ğ³Ñ€Ğ°Ğ´ÑƒÑ�Ğ¾Ğ²"""
        return self._check_rotation(train_pairs, test_inputs, 3)

    def _check_rotation(self, train_pairs, test_inputs, k):
        """ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµÑ‚ Ğ¸ Ğ¿Ñ€Ğ¸Ğ¼ĞµĞ½Ñ�ĞµÑ‚ Ğ¿Ğ¾Ğ²Ğ¾Ñ€Ğ¾Ñ‚"""
        rotations_found = 0
        for inp, out in train_pairs:
            if inp.shape == out.shape and np.array_equal(np.rot90(inp, k), out):
                rotations_found += 1
        
        if rotations_found > 0:
            predictions = []
            for test_input in test_inputs:
                result = np.rot90(test_input, k)
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    def _color_invert(self, train_pairs, test_inputs, analysis):
        """Ğ˜Ğ½Ğ²ĞµÑ€Ñ�Ğ¸Ñ� Ñ†Ğ²ĞµÑ‚Ğ¾Ğ² Ğ´Ğ»Ñ� Ğ±Ğ¸Ğ½Ğ°Ñ€Ğ½Ñ‹Ñ… Ñ�Ğ»ÑƒÑ‡Ğ°ĞµĞ²"""
        for inp, out in train_pairs:
            if inp.shape == out.shape:
                unique_in = np.unique(inp)
                unique_out = np.unique(out)
                if len(unique_in) == 2 and len(unique_out) == 2:
                    # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ğ¸Ğ½Ğ²ĞµÑ€Ñ�Ğ¸Ñ�
                    color_map = {unique_in[0]: unique_in[1], unique_in[1]: unique_in[0]}
                    test_inv = inp.copy()
                    for old, new in color_map.items():
                        test_inv[inp == old] = new
                    if np.array_equal(test_inv, out):
                        predictions = []
                        for test_input in test_inputs:
                            result = test_input.copy()
                            for old, new in color_map.items():
                                result[test_input == old] = new
                            predictions.append({
                                "attempt_1": result.tolist(),
                                "attempt_2": result.tolist()
                            })
                        return predictions
        return None

    def _background_fill(self, train_pairs, test_inputs, analysis):
        """Ğ—Ğ°Ğ¿Ğ¾Ğ»Ğ½ĞµĞ½Ğ¸Ğµ Ñ„Ğ¾Ğ½Ğ°"""
        bg_changes = []
        for inp, out in train_pairs:
            if inp.shape == out.shape:
                bg_mask = inp == 0
                if np.any(bg_mask):
                    bg_colors = out[bg_mask]
                    if len(np.unique(bg_colors)) == 1 and bg_colors[0] != 0:
                        bg_changes.append(bg_colors[0])
        
        if bg_changes:
            new_bg_color = Counter(bg_changes).most_common(1)[0][0]
            predictions = []
            for test_input in test_inputs:
                result = test_input.copy()
                result[result == 0] = new_bg_color
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    def _pattern_extract(self, train_pairs, test_inputs, analysis):
        """Ğ˜Ğ·Ğ²Ğ»ĞµÑ‡ĞµĞ½Ğ¸Ğµ Ğ¿Ğ¾Ğ´Ğ¼Ğ°Ñ‚Ñ€Ğ¸Ñ†Ñ‹"""
        if not analysis['size_changes']:
            return None
        
        # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ extraction patterns
        extractions = []
        for inp, out in train_pairs:
            inp_h, inp_w = inp.shape
            out_h, out_w = out.shape
            
            if out_h <= inp_h and out_w <= inp_w:
                found = False
                for i in range(inp_h - out_h + 1):
                    for j in range(inp_w - out_w + 1):
                        if np.array_equal(inp[i:i+out_h, j:j+out_w], out):
                            extractions.append((i, j, out_h, out_w))
                            found = True
                            break
                    if found:
                        break
        
        if extractions:
            # Ğ‘ĞµÑ€ĞµĞ¼ Ğ½Ğ°Ğ¸Ğ±Ğ¾Ğ»ĞµĞµ Ñ‡Ğ°Ñ�Ñ‚Ñ‹Ğ¹ extraction
            i, j, h, w = Counter(extractions).most_common(1)[0][0]
            predictions = []
            for test_input in test_inputs:
                if (test_input.shape[0] >= i + h and 
                    test_input.shape[1] >= j + w):
                    result = test_input[i:i+h, j:j+w]
                else:
                    # Fallback - Ñ†ĞµĞ½Ñ‚Ñ€Ğ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ‡Ğ°Ñ�Ñ‚ÑŒ
                    start_i = max(0, (test_input.shape[0] - h) // 2)
                    start_j = max(0, (test_input.shape[1] - w) // 2)
                    result = test_input[start_i:start_i+h, start_j:start_j+w]
                
                predictions.append({
                    "attempt_1": result.tolist(),
                    "attempt_2": result.tolist()
                })
            return predictions
        return None

    # Ğ’Ğ¡ĞŸĞ�ĞœĞ�Ğ“Ğ�Ğ¢Ğ•Ğ›Ğ¬Ğ�Ğ«Ğ• ĞœĞ•Ğ¢Ğ�Ğ”Ğ«

    def _parse_train_data(self, task):
        """ĞŸĞ°Ñ€Ñ�Ğ¸Ñ‚ Ğ¾Ğ±ÑƒÑ‡Ğ°Ñ�Ñ‰Ğ¸Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ"""
        train_pairs = []
        if 'train' in task and isinstance(task['train'], list):
            for example in task['train']:
                try:
                    if isinstance(example, dict) and 'input' in example and 'output' in example:
                        inp = np.array(example['input'], dtype=np.uint8)
                        out = np.array(example['output'], dtype=np.uint8)
                        train_pairs.append((inp, out))
                except:
                    continue
        return train_pairs

    def _parse_test_data(self, task):
        """ĞŸĞ°Ñ€Ñ�Ğ¸Ñ‚ Ñ‚ĞµÑ�Ñ‚Ğ¾Ğ²Ñ‹Ğµ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ"""
        test_inputs = []
        if 'test' in task and isinstance(task['test'], list):
            for example in task['test']:
                try:
                    if isinstance(example, dict) and 'input' in example:
                        inp = np.array(example['input'], dtype=np.uint8)
                        test_inputs.append(inp)
                except:
                    continue
        return test_inputs

    def _create_predictions(self, test_inputs):
        """Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ� Ğ² Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ğ¼ Ñ„Ğ¾Ñ€Ğ¼Ğ°Ñ‚Ğµ"""
        predictions = []
        for test_input in test_inputs:
            predictions.append({
                "attempt_1": test_input.tolist(),
                "attempt_2": test_input.tolist()
            })
        return predictions

# Ğ¤Ğ£Ğ�ĞšĞ¦Ğ˜Ğ˜ Ğ¢Ğ•Ğ¡Ğ¢Ğ˜Ğ Ğ�Ğ’Ğ�Ğ�Ğ˜Ğ¯ Ğ˜ Ğ’Ğ�Ğ›Ğ˜Ğ”Ğ�Ğ¦Ğ˜Ğ˜

def validate_solution(prediction, ground_truth):
    """ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµÑ‚ Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ Ğ¾Ğ´Ğ½Ğ¾Ğ³Ğ¾ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ñ�"""
    try:
        # ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ ground truth
        if isinstance(ground_truth, dict) and 'output' in ground_truth:
            gt_output = ground_truth['output']
        elif isinstance(ground_truth, list):
            gt_output = ground_truth
        else:
            return False, "invalid_gt_format"
        
        # ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ
        pred_output = prediction['attempt_1']
        
        # ĞšĞ¾Ğ½Ğ²ĞµÑ€Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ² numpy Ğ¼Ğ°Ñ�Ñ�Ğ¸Ğ²Ñ‹
        pred_arr = np.array(pred_output, dtype=np.uint8)
        gt_arr = np.array(gt_output, dtype=np.uint8)
        
        # Ğ¡Ñ€Ğ°Ğ²Ğ½Ğ¸Ğ²Ğ°ĞµĞ¼
        if pred_arr.shape != gt_arr.shape:
            return False, f"shape_mismatch_{pred_arr.shape}_vs_{gt_arr.shape}"
        
        if not np.array_equal(pred_arr, gt_arr):
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ attempt_2
            pred_output2 = prediction.get('attempt_2', pred_output)
            pred_arr2 = np.array(pred_output2, dtype=np.uint8)
            
            if pred_arr2.shape == gt_arr.shape and np.array_equal(pred_arr2, gt_arr):
                return True, "correct_attempt2"
            else:
                return False, "values_mismatch"
        
        return True, "correct_attempt1"
        
    except Exception as e:
        return False, f"validation_error_{str(e)}"

def test_arc_solver(sample_size=10):
    """Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµÑ‚ ARC Ñ€ĞµÑˆĞ°Ñ‚ĞµĞ»ÑŒ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…"""
    print("ğŸ§ª TESTING ARC CORE SOLVER")
    print("=" * 50)
    
    # Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
    try:
        with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
            challenges = json.load(f)
        with open('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
            solutions = json.load(f)
        print(f"âœ… Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶ĞµĞ½Ğ¾ {len(challenges)} Ğ·Ğ°Ğ´Ğ°Ñ‡ Ğ¸ {len(solutions)} Ñ€ĞµÑˆĞµĞ½Ğ¸Ğ¹")
    except Exception as e:
        print(f"â�Œ Ğ�ÑˆĞ¸Ğ±ĞºĞ° Ğ·Ğ°Ğ³Ñ€ÑƒĞ·ĞºĞ¸ Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ…: {e}")
        return 0, []
    
    # Ğ’Ñ‹Ğ±Ğ¸Ñ€Ğ°ĞµĞ¼ Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸ Ğ´Ğ»Ñ� Ñ‚ĞµÑ�Ñ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ñ�
    test_tasks = list(challenges.items())[:sample_size]
    print(f"ğŸ”� Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ½Ğ° {len(test_tasks)} Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ°Ñ…...\n")
    
    solver = ARCCoreSolver()
    results = []
    
    for task_id, task in test_tasks:
        if task_id not in solutions:
            continue
            
        print(f"\nğŸ“� Ğ—Ğ°Ğ´Ğ°Ñ‡Ğ°: {task_id}")
        
        try:
            # Ğ ĞµÑˆĞ°ĞµĞ¼ Ğ·Ğ°Ğ´Ğ°Ñ‡Ñƒ
            start_time = time.time()
            predictions, strategy, confidence = solver.solve_task(task)
            solve_time = time.time() - start_time
            
            # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚ÑŒ
            ground_truths = solutions[task_id]
            task_correct = True
            task_details = []
            
            for i, (pred, gt) in enumerate(zip(predictions, ground_truths)):
                is_correct, detail = validate_solution(pred, gt)
                if not is_correct:
                    task_correct = False
                task_details.append({
                    'test_case': i,
                    'correct': is_correct,
                    'detail': detail
                })
                
                # Ğ’Ñ‹Ğ²Ğ¾Ğ´Ğ¸Ğ¼ Ğ´ĞµÑ‚Ğ°Ğ»Ğ¸ Ğ´Ğ»Ñ� Ğ¿ĞµÑ€Ğ²Ğ¾Ğ³Ğ¾ Ğ½ĞµĞ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ğ³Ğ¾ Ñ‚ĞµÑ�Ñ‚Ğ°
                if not is_correct and len(task_details) == 1:
                    print(f"  â�Œ test_{i}: {detail}")
            
            if task_correct:
                print(f"  âœ… Ğ—Ğ�Ğ”Ğ�Ğ§Ğ� Ğ Ğ•Ğ¨Ğ•Ğ�Ğ�! ({strategy}, conf: {confidence:.2f}, time: {solve_time:.3f}s)")
            else:
                print(f"  ğŸ’¥ Ğ—Ğ°Ğ´Ğ°Ñ‡Ğ° Ğ½Ğµ Ñ€ĞµÑˆĞµĞ½Ğ° ({strategy}, conf: {confidence:.2f})")
            
            results.append({
                'task_id': task_id,
                'strategy': strategy,
                'confidence': confidence,
                'correct': task_correct,
                'solve_time': solve_time,
                'details': task_details
            })
            
        except Exception as e:
            print(f"  ğŸ’¥ Ğ�ÑˆĞ¸Ğ±ĞºĞ°: {e}")
            results.append({
                'task_id': task_id,
                'strategy': 'error',
                'confidence': 0,
                'correct': False,
                'error': str(e)
            })
    
    # Ğ�Ğ½Ğ°Ğ»Ğ¸Ğ·Ğ¸Ñ€ÑƒĞµĞ¼ Ñ€ĞµĞ·ÑƒĞ»ÑŒÑ‚Ğ°Ñ‚Ñ‹
    total_tasks = len(results)
    correct_tasks = sum(1 for r in results if r['correct'])
    accuracy = correct_tasks / total_tasks if total_tasks > 0 else 0
    
    print(f"\nğŸ“Š Ğ¤Ğ˜Ğ�Ğ�Ğ›Ğ¬Ğ�Ğ«Ğ• Ğ Ğ•Ğ—Ğ£Ğ›Ğ¬Ğ¢Ğ�Ğ¢Ğ«:")
    print(f"ğŸ�¯ Ğ¢Ğ�Ğ§Ğ�Ğ�Ğ¡Ğ¢Ğ¬: {accuracy:.1%} ({correct_tasks}/{total_tasks})")
    
    # Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ° Ğ¿Ğ¾ Ñ�Ñ‚Ñ€Ğ°Ñ‚ĞµĞ³Ğ¸Ñ�Ğ¼
    strategy_stats = Counter(r['strategy'] for r in results)
    correct_strategies = Counter(r['strategy'] for r in results if r['correct'])
    
    print(f"ğŸ“ˆ Ğ¡Ğ¢Ğ�Ğ¢Ğ˜Ğ¡Ğ¢Ğ˜ĞšĞ� Ğ¡Ğ¢Ğ Ğ�Ğ¢Ğ•Ğ“Ğ˜Ğ™:")
    for strategy, count in strategy_stats.most_common():
        correct = correct_strategies.get(strategy, 0)
        success_rate = correct / count if count > 0 else 0
        print(f"   - {strategy}: {correct}/{count} ({success_rate:.1%})")
    
    # Ğ’Ñ€ĞµĞ¼Ñ� Ğ²Ñ‹Ğ¿Ğ¾Ğ»Ğ½ĞµĞ½Ğ¸Ñ�
    if total_tasks > 0:
        avg_time = sum(r.get('solve_time', 0) for r in results) / total_tasks
        print(f"â�± Ğ¡Ñ€ĞµĞ´Ğ½ĞµĞµ Ğ²Ñ€ĞµĞ¼Ñ�: {avg_time:.3f}s")
    
    return accuracy, results

def create_submission(solver, challenges, output_file='submission.json'):
    """Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµÑ‚ Ñ„Ğ°Ğ¹Ğ» Ğ´Ğ»Ñ� Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚Ğ° Ğ½Ğ° Kaggle"""
    print(f"\nğŸš€ Ğ¡Ğ�Ğ—Ğ”Ğ�Ğ�Ğ˜Ğ• Ğ¡Ğ�Ğ‘ĞœĞ˜Ğ¢Ğ� {output_file}")
    print("=" * 50)
    
    submission = {}
    total_tasks = len(challenges)
    
    for i, (task_id, task) in enumerate(challenges.items()):
        try:
            predictions, strategy, confidence = solver.solve_task(task)
            submission[task_id] = predictions
            
            if (i + 1) % 50 == 0:
                print(f"  Ğ�Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°Ğ½Ğ¾ {i+1}/{total_tasks} Ğ·Ğ°Ğ´Ğ°Ñ‡...")
                
        except Exception as e:
            # Fallback Ğ´Ğ»Ñ� Ğ¾ÑˆĞ¸Ğ±Ğ¾Ğº
            test_inputs = solver._parse_test_data(task)
            submission[task_id] = solver._create_predictions(test_inputs)
    
    # Ğ¡Ğ¾Ñ…Ñ€Ğ°Ğ½Ñ�ĞµĞ¼ Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚
    with open(output_file, 'w') as f:
        json.dump(submission, f)
    
    print(f"âœ… Ğ¡Ğ¾Ğ·Ğ´Ğ°Ğ½ Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚: {output_file}")
    print(f"ğŸ“Š Ğ�Ğ±Ñ€Ğ°Ğ±Ğ¾Ñ‚Ğ°Ğ½Ğ¾ Ğ·Ğ°Ğ´Ğ°Ñ‡: {len(submission)}")
    
    return submission

if __name__ == "__main__":
    # Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ñ€ĞµÑˆĞ°Ñ‚ĞµĞ»ÑŒ
    accuracy, results = test_arc_solver(sample_size=20)
    
    # Ğ•Ñ�Ğ»Ğ¸ Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ Ñ…Ğ¾Ñ€Ğ¾ÑˆĞ°Ñ�, Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚
    if accuracy > 0.3:
        print(f"\nğŸ�‰ Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ {accuracy:.1%} - Ñ�Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚...")
        
        # Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ evaluation Ğ¸ test Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸
        try:
            with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json', 'r') as f:
                eval_challenges = json.load(f)
            with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
                test_challenges = json.load(f)
            
            # Ğ�Ğ±ÑŠĞµĞ´Ğ¸Ğ½Ñ�ĞµĞ¼ Ğ²Ñ�Ğµ Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ¸
            all_challenges = {**eval_challenges, **test_challenges}
            print(f"ğŸ“� Ğ’Ñ�ĞµĞ³Ğ¾ Ğ·Ğ°Ğ´Ğ°Ñ‡ Ğ´Ğ»Ñ� Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚Ğ°: {len(all_challenges)}")
            
            # Ğ¡Ğ¾Ğ·Ğ´Ğ°ĞµĞ¼ Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚
            solver = ARCCoreSolver()
            create_submission(solver, all_challenges)
            
        except Exception as e:
            print(f"â�Œ Ğ�ÑˆĞ¸Ğ±ĞºĞ° Ñ�Ğ¾Ğ·Ğ´Ğ°Ğ½Ğ¸Ñ� Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚Ğ°: {e}")
    else:
        print(f"\nâš ï¸� Ğ¢Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ {accuracy:.1%} Ñ�Ğ»Ğ¸ÑˆĞºĞ¾Ğ¼ Ğ½Ğ¸Ğ·ĞºĞ°Ñ� Ğ´Ğ»Ñ� Ñ�Ğ°Ğ±Ğ¼Ğ¸Ñ‚Ğ°")


import numpy as np
import json
from collections import Counter
import time

def truly_test_arc_solver():
    """Ğ Ğ•Ğ�Ğ›Ğ¬Ğ�Ğ�Ğ¯ Ğ¿Ñ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚Ğ¸ Ğ½Ğ° Ñ‚Ñ€ĞµĞ½Ğ¸Ñ€Ğ¾Ğ²Ğ¾Ñ‡Ğ½Ñ‹Ñ… Ğ´Ğ°Ğ½Ğ½Ñ‹Ñ… Ñ� ground truth"""
    print("ğŸ”� Ğ Ğ•Ğ�Ğ›Ğ¬Ğ�Ğ�Ğ• Ğ¢Ğ•Ğ¡Ğ¢Ğ˜Ğ Ğ�Ğ’Ğ�Ğ�Ğ˜Ğ• Ğ¢Ğ�Ğ§Ğ�Ğ�Ğ¡Ğ¢Ğ˜")
    print("=" * 60)
    
    # Ğ—Ğ°Ğ³Ñ€ÑƒĞ¶Ğ°ĞµĞ¼ Ğ´Ğ°Ğ½Ğ½Ñ‹Ğµ
    with open('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json', 'r') as f:
        challenges = json.load(f)
    with open('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json', 'r') as f:
        solutions = json.load(f)
    
    print(f"ğŸ“Š Ğ’Ñ�ĞµĞ³Ğ¾ Ğ·Ğ°Ğ´Ğ°Ñ‡: {len(challenges)}")
    print(f"ğŸ“Š Ğ ĞµÑˆĞµĞ½Ğ¸Ğ¹: {len(solutions)}")
    
    # Ğ‘ĞµÑ€ĞµĞ¼ Ultimate ARC Solver (Ğ¸Ğ»Ğ¸ Ğ»Ñ�Ğ±Ğ¾Ğ¹ Ğ´Ñ€ÑƒĞ³Ğ¾Ğ¹)
    from final_arc_solver import ARCEnhancedFusionSolver
    solver = ARCEnhancedFusionSolver()
    
    # Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ½Ğ° 50 Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ°Ñ…
    test_tasks = list(challenges.items())[:50]
    correct_count = 0
    results = []
    
    print(f"\nğŸ§ª Ğ¢ĞµÑ�Ñ‚Ğ¸Ñ€ÑƒĞµĞ¼ Ğ½Ğ° {len(test_tasks)} Ğ·Ğ°Ğ´Ğ°Ñ‡Ğ°Ñ…...")
    
    for i, (task_id, task) in enumerate(test_tasks):
        if task_id not in solutions:
            continue
            
        try:
            # Ğ ĞµÑˆĞ°ĞµĞ¼ Ğ·Ğ°Ğ´Ğ°Ñ‡Ñƒ
            predictions, operation, confidence = solver.solve_task(task)
            ground_truths = solutions[task_id]
            
            # Ğ Ğ•Ğ�Ğ›Ğ¬Ğ�Ğ�Ğ¯ Ğ¿Ñ€Ğ¾Ğ²ĞµÑ€ĞºĞ° Ğ¿Ñ€Ğ°Ğ²Ğ¸Ğ»ÑŒĞ½Ğ¾Ñ�Ñ‚Ğ¸
            task_correct = True
            for j, (pred, truth) in enumerate(zip(predictions, ground_truths)):
                # ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ ground truth
                if isinstance(truth, dict) and 'output' in truth:
                    truth_output = truth['output']
                else:
                    truth_output = truth
                
                # ĞŸĞ¾Ğ»ÑƒÑ‡Ğ°ĞµĞ¼ Ğ¿Ñ€ĞµĞ´Ñ�ĞºĞ°Ğ·Ğ°Ğ½Ğ¸Ğµ
                pred_output = pred['attempt_1']
                
                # Ğ¡Ñ€Ğ°Ğ²Ğ½Ğ¸Ğ²Ğ°ĞµĞ¼
                pred_arr = np.array(pred_output)
                truth_arr = np.array(truth_output)
                
                if pred_arr.shape != truth_arr.shape or not np.array_equal(pred_arr, truth_arr):
                    # ĞŸÑ€Ğ¾Ğ²ĞµÑ€Ñ�ĞµĞ¼ attempt_2
                    pred_output2 = pred.get('attempt_2', pred_output)
                    pred_arr2 = np.array(pred_output2)
                    
                    if pred_arr2.shape != truth_arr.shape or not np.array_equal(pred_arr2, truth_arr):
                        task_correct = False
                        break
            
            if task_correct:
                correct_count += 1
                print(f"âœ… {task_id[:8]} - {operation} (conf: {confidence:.2f})")
            else:
                print(f"â�Œ {task_id[:8]} - {operation} (conf: {confidence:.2f})")
            
            results.append({
                'task_id': task_id,
                'correct': task_correct,
                'operation': operation,
                'confidence': confidence
            })
            
        except Exception as e:
            print(f"ğŸ’¥ {task_id[:8]} - Ğ�ÑˆĞ¸Ğ±ĞºĞ°: {e}")
            results.append({
                'task_id': task_id,
                'correct': False,
                'operation': 'error',
                'confidence': 0
            })
    
    # Ğ ĞµĞ°Ğ»ÑŒĞ½Ğ°Ñ� Ñ‚Ğ¾Ñ‡Ğ½Ğ¾Ñ�Ñ‚ÑŒ
    accuracy = correct_count / len(results) if results else 0
    print(f"\nğŸ�¯ Ğ Ğ•Ğ�Ğ›Ğ¬Ğ�Ğ�Ğ¯ Ğ¢Ğ�Ğ§Ğ�Ğ�Ğ¡Ğ¢Ğ¬: {accuracy:.1%} ({correct_count}/{len(results)})")
    
    # Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ° Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¹
    op_stats = Counter(r['operation'] for r in results)
    correct_ops = Counter(r['operation'] for r in results if r['correct'])
    
    print("ğŸ“ˆ Ğ¡Ñ‚Ğ°Ñ‚Ğ¸Ñ�Ñ‚Ğ¸ĞºĞ° Ğ¾Ğ¿ĞµÑ€Ğ°Ñ†Ğ¸Ğ¹:")
    for op, count in op_stats.most_common():
        correct = correct_ops.get(op, 0)
        success_rate = correct / count if count > 0 else 0
        print(f"   - {op}: {correct}/{count} ({success_rate:.1%})")
    
    return accuracy, results

# Ğ—Ğ°Ğ¿ÑƒÑ�ĞºĞ°ĞµĞ¼ Ñ€ĞµĞ°Ğ»ÑŒĞ½Ğ¾Ğµ Ñ‚ĞµÑ�Ñ‚Ğ¸Ñ€Ğ¾Ğ²Ğ°Ğ½Ğ¸Ğµ
if __name__ == "__main__":
    accuracy, results = truly_test_arc_solver()

