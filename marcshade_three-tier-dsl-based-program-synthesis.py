import numpy as np
import json


class DSLPrimitives:
    @staticmethod
    def rot90(grid): return np.rot90(grid, k=1)
    @staticmethod
    def rot180(grid): return np.rot90(grid, k=2)
    @staticmethod
    def rot270(grid): return np.rot90(grid, k=3)
    @staticmethod
    def hmirror(grid): return np.fliplr(grid)
    @staticmethod
    def vmirror(grid): return np.flipud(grid)

    @staticmethod
    def upscale(grid, scale=2):
        h, w = grid.shape
        result = np.zeros((h*scale, w*scale), dtype=grid.dtype)
        for i in range(h):
            for j in range(w):
                result[i*scale:(i+1)*scale, j*scale:(j+1)*scale] = grid[i,j]
        return result

    @staticmethod
    def downscale(grid, scale=2):
        h, w = grid.shape
        if h % scale or w % scale: return grid
        return grid[::scale, ::scale]

    @staticmethod
    def replace(grid, old_color, new_color):
        result = grid.copy()
        result[result == old_color] = new_color
        return result

    @staticmethod
    def tile(grid, times=2):
        return np.tile(grid, (times, times))

    @staticmethod
    def self_tile(grid):
        return np.tile(grid, (2, 2))

    @staticmethod
    def shift_right(grid, amount=1, background=0):
        result = np.full_like(grid, background)
        if 0 < amount < grid.shape[1]:
            result[:, amount:] = grid[:, :-amount]
        return result

    @staticmethod
    def shift_left(grid, amount=1, background=0):
        result = np.full_like(grid, background)
        if 0 < amount < grid.shape[1]:
            result[:, :-amount] = grid[:, amount:]
        return result

    @staticmethod
    def shift_up(grid, amount=1, background=0):
        result = np.full_like(grid, background)
        if 0 < amount < grid.shape[0]:
            result[:-amount, :] = grid[amount:, :]
        return result

    @staticmethod
    def shift_down(grid, amount=1, background=0):
        result = np.full_like(grid, background)
        if 0 < amount < grid.shape[0]:
            result[amount:, :] = grid[:-amount, :]
        return result

    @staticmethod
    def fill_holes(grid, fill_color=4):
        result = grid.copy()
        h, w = grid.shape
        visited = np.zeros((h,w), dtype=bool)

        def flood(r, c):
            if r<0 or r>=h or c<0 or c>=w or visited[r,c]: return
            visited[r,c] = True
            if result[r,c] == 0:
                for dr,dc in [(0,1),(1,0),(0,-1),(-1,0)]:
                    flood(r+dr, c+dc)

        for i in range(h):
            flood(i,0); flood(i,w-1)
        for j in range(w):
            flood(0,j); flood(h-1,j)

        result[~visited & (result==0)] = fill_color
        return result


class V15ImprovedSolver:
    def __init__(self):
        self.prims = {
            'rot90': DSLPrimitives.rot90,
            'rot180': DSLPrimitives.rot180,
            'rot270': DSLPrimitives.rot270,
            'hmirror': DSLPrimitives.hmirror,
            'vmirror': DSLPrimitives.vmirror,
            'upscale': DSLPrimitives.upscale,
            'downscale': DSLPrimitives.downscale,
            'replace': DSLPrimitives.replace,
            'tile': DSLPrimitives.tile,
            'shift_right': DSLPrimitives.shift_right,
            'shift_left': DSLPrimitives.shift_left,
            'shift_up': DSLPrimitives.shift_up,
            'shift_down': DSLPrimitives.shift_down,
            'fill_holes': DSLPrimitives.fill_holes,
            'self_tile': DSLPrimitives.self_tile,
        }

    def try_prim(self, grid, name, **params):
        try:
            result = self.prims[name](grid, **params) if params else self.prims[name](grid)
            return result if isinstance(result, np.ndarray) and result.size > 0 else None
        except:
            return None

    def score(self, pred, exp):
        return float(np.mean(pred == exp)) if pred.shape == exp.shape else 0.0
    
    def verify_all_train(self, prim, params, train_examples, threshold=0.80):
        """Verify primitive works on ALL training examples"""
        if not train_examples:
            return False, 0.0
        
        scores = []
        for inp, out in train_examples:
            result = self.try_prim(inp, prim, **params)
            if result is not None:
                sc = self.score(result, out)
                scores.append(sc)
            else:
                scores.append(0.0)
        
        avg_score = sum(scores) / len(scores) if scores else 0.0
        all_pass = all(s >= threshold for s in scores)
        return all_pass, avg_score

    def depth1(self, train_examples, test_input):
        """V15: Lower threshold to 80%, verify on all training"""
        best_score = 0.0
        best_prim = None
        best_params = None

        for prim in self.prims:
            params_list = [{}]
            
            if prim == 'downscale':
                params_list = [{'scale': s} for s in [2, 3, 4, 5, 6]]
            elif prim == 'upscale':
                params_list = [{'scale': s} for s in [2, 3, 4, 5]]
            elif prim in ['shift_right', 'shift_left', 'shift_up', 'shift_down']:
                params_list = [{'amount': a, 'background': 0} for a in [1, 2, 3, 4, 5]]
            elif prim == 'fill_holes':
                params_list = [{'fill_color': c} for c in range(10)]
            elif prim == 'tile':
                params_list = [{'times': t} for t in [2, 3, 4]]
            elif prim == 'replace':
                params_list = [{'old_color': o, 'new_color': n} 
                              for o in range(10) for n in range(10) if o != n][:20]  # Limit combinations

            for params in params_list:
                # Check average score across all training
                all_pass, avg_score = self.verify_all_train(prim, params, train_examples, threshold=0.80)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_prim = prim
                    best_params = params

        # Lower threshold: 80% instead of 90%
        if best_prim is not None and best_score >= 0.80:
            return self.try_prim(test_input, best_prim, **best_params)
        return None

    def depth2(self, train_examples, test_input):
        """V15: Lower threshold to 70%, try more combinations"""
        best_score = 0.0
        best_p1 = None
        best_p2 = None
        best_params1 = None
        best_params2 = None
        
        # Prioritize likely combinations
        priority_prims = ['rot90', 'rot180', 'rot270', 'hmirror', 'vmirror', 
                         'downscale', 'upscale', 'fill_holes', 'tile']
        prim_names = priority_prims + [p for p in self.prims.keys() if p not in priority_prims]

        for p1 in prim_names[:12]:  # Limit search
            params1_list = [{}]
            if p1 == 'downscale':
                params1_list = [{'scale': s} for s in [2, 3, 4]]
            elif p1 == 'fill_holes':
                params1_list = [{'fill_color': c} for c in [1, 2, 3, 4, 5, 6, 7, 8, 9]]
                
            for params1 in params1_list:
                for p2 in prim_names[:12]:
                    params2_list = [{}]
                    if p2 == 'downscale':
                        params2_list = [{'scale': s} for s in [2, 3]]
                    
                    for params2 in params2_list:
                        # Test on training
                        total_score = 0.0
                        count = 0
                        for inp, out in train_examples:
                            try:
                                mid = self.try_prim(inp, p1, **params1)
                                if mid is None: continue
                                result = self.try_prim(mid, p2, **params2)
                                if result is not None:
                                    sc = self.score(result, out)
                                    total_score += sc
                                    count += 1
                            except:
                                pass
                        
                        avg_score = total_score / count if count > 0 else 0.0
                        if avg_score > best_score:
                            best_score = avg_score
                            best_p1 = p1
                            best_p2 = p2
                            best_params1 = params1
                            best_params2 = params2

        # Lower threshold: 70% instead of 85%
        if best_p1 is not None and best_score >= 0.70:
            try:
                mid = self.try_prim(test_input, best_p1, **best_params1)
                if mid is not None:
                    return self.try_prim(mid, best_p2, **best_params2)
            except:
                pass
        return None

    def solve_task(self, task_data):
        try:
            train = [(np.array(ex['input']), np.array(ex['output']))
                     for ex in task_data.get('train', [])]
            if not train: return None

            predictions = []
            for test_case in task_data.get('test', []):
                test_input = np.array(test_case['input'])
                
                result = self.depth1(train, test_input)
                if result is None:
                    result = self.depth2(train, test_input)
                
                predictions.append(result.tolist() if result is not None else None)

            return predictions if predictions else None
        except:
            return None


# Load TEST challenges (240 tasks for competition submission)
with open('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json', 'r') as f:
    test_challenges = json.load(f)

print(f"Loaded {len(test_challenges)} test tasks")


solver = V15ImprovedSolver()
submission = {}
FALLBACK_GRID = [[0]]

for task_id, task_data in test_challenges.items():
    try:
        predictions = solver.solve_task(task_data)
        
        if predictions:
            task_predictions = []
            for pred in predictions:
                if pred and len(pred) > 0:
                    task_predictions.append({
                        'attempt_1': pred,
                        'attempt_2': pred
                    })
                else:
                    task_predictions.append({
                        'attempt_1': FALLBACK_GRID,
                        'attempt_2': FALLBACK_GRID
                    })
            submission[task_id] = task_predictions
        else:
            submission[task_id] = [{
                'attempt_1': FALLBACK_GRID,
                'attempt_2': FALLBACK_GRID
            }]
    except Exception as e:
        submission[task_id] = [{
            'attempt_1': FALLBACK_GRID,
            'attempt_2': FALLBACK_GRID
        }]

print(f"Generated predictions for {len(submission)} tasks")


with open('submission.json', 'w') as f:
    json.dump(submission, f)

print("Submission saved to submission.json")

