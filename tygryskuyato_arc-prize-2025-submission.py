import numpy as np
import pandas as pd
import os
import subprocess
import sys
import json
import copy
from collections import deque
from itertools import product, count
from typing import List, Dict, Any, Callable, Tuple, Set
import heapq # Thư viện cho Hàng đợi Ưu tiên

# Cài đặt 'scipy'
print("Installing/Verifying 'scipy'...")
try:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scipy"])
    print("'scipy' is installed.")
except Exception as e:
    print(f"Could not install scipy: {e}. Assuming it's pre-installed.")

import importlib
importlib.invalidate_caches()

try:
    from scipy.ndimage import label, find_objects, binary_fill_holes
    print("Successfully imported 'scipy.ndimage'.")
except ImportError:
    print("FATAL ERROR: Could not import 'scipy.ndimage'.")
    # Gán hàm giả để code không crash
    def label(x, structure=None): return x, 0
    def find_objects(x): return []
    def binary_fill_holes(x): return x

print("Imports complete.")

def load_json(file_path):
    """Tải và đọc file JSON."""
    with open(file_path, 'r') as f:
        return json.load(f)

# ==================================================================
# 2. Các Lớp và Hàm hỗ trợ DSL
# ==================================================================

class ArcObject:
    """Một class đơn giản để lưu thông tin về một đối tượng được phát hiện."""
    def __init__(self, grid_slice: np.array, color: int, bbox: Tuple[slice, slice], mask: np.array):
        self.grid = grid_slice
        self.color = color
        self.bbox = bbox
        self.mask = mask
        self.height = grid_slice.shape[0]
        self.width = grid_slice.shape[1]
        self.num_pixels = np.sum(mask)
        self.y_min = bbox[0].start
        self.x_min = bbox[1].start
        self.centroid = (self.y_min + self.height / 2, self.x_min + self.width / 2)
        
    def __repr__(self):
        return f"Obj(color={self.color}, shape={self.grid.shape}, pixels={self.num_pixels}, pos=({self.y_min},{self.x_min}))"

def detect_objects(grid: np.array, connectivity: int = 8) -> List[ArcObject]:
    """Phát hiện tất cả các đối tượng liền kề trong grid."""
    objects = []
    structure = np.ones((3,3)) if connectivity == 8 else np.array([[0,1,0],[1,1,1],[0,1,0]])
    
    for color in range(1, 10):
        color_mask = (grid == color)
        if not color_mask.any():
            continue
        
        labeled_array, num_features = label(color_mask, structure=structure)
        slices = find_objects(labeled_array)
        
        for i, slc in enumerate(slices):
            if slc is None: continue
            obj_mask_in_bbox = (labeled_array[slc] == i + 1)
            obj_grid_slice = grid[slc]
            obj_grid = obj_grid_slice * obj_mask_in_bbox
            full_mask = (labeled_array == i + 1)
            
            objects.append(ArcObject(
                grid_slice=obj_grid,
                color=color,
                bbox=slc,
                mask=full_mask
            ))
            
    objects.sort(key=lambda o: (o.bbox[0].start, o.bbox[1].start))
    return objects

def _paste_grid(main_grid, sub_grid, top_left_y, top_left_x, ignore_color_0=True):
    """Hàm hỗ trợ: Dán 'sub_grid' vào 'main_grid'."""
    g_h, g_w = main_grid.shape
    s_h, s_w = sub_grid.shape
    
    y_start, y_end = int(top_left_y), int(top_left_y + s_h)
    x_start, x_end = int(top_left_x), int(top_left_x + s_w)
    
    s_y_start, s_x_start = 0, 0
    if y_start < 0: s_y_start = -y_start; y_start = 0
    if x_start < 0: s_x_start = -x_start; x_start = 0

    s_y_end = s_h - max(0, y_end - g_h)
    s_x_end = s_w - max(0, x_end - g_w)
    
    y_end = min(y_end, g_h)
    x_end = min(x_end, g_w)
    
    if (y_start < y_end) and (x_start < x_end) and (s_y_start < s_y_end) and (s_x_start < s_x_end):
        sub_grid_to_paste = sub_grid[s_y_start:s_y_end, s_x_start:s_x_end]
        if sub_grid_to_paste.size == 0:
            return main_grid

        paste_region = main_grid[y_start:y_end, x_start:x_end]
        
        if paste_region.shape != sub_grid_to_paste.shape:
            h_min = min(paste_region.shape[0], sub_grid_to_paste.shape[0])
            w_min = min(paste_region.shape[1], sub_grid_to_paste.shape[1])
            paste_region = paste_region[:h_min, :w_min]
            sub_grid_to_paste = sub_grid_to_paste[:h_min, :w_min]
        
        if ignore_color_0:
            mask_to_paste = (sub_grid_to_paste != 0)
            paste_region[mask_to_paste] = sub_grid_to_paste[mask_to_paste]
        else:
            paste_region[:] = sub_grid_to_paste
            
    return main_grid

def _draw_line_simple(grid, y1, x1, y2, x2, color):
    """Vẽ 1 đường thẳng (chỉ ngang/dọc/chéo 45 độ)."""
    y1, x1, y2, x2 = int(y1), int(x1), int(y2), int(x2)
    dy, dx = (y2 - y1), (x2 - x1)
    
    if dx == 0: # Doc
        for y in range(min(y1, y2), max(y1, y2) + 1):
            if 0 <= y < grid.shape[0]: grid[y, x1] = color
    elif dy == 0: # Ngang
        for x in range(min(x1, x2), max(x1, x2) + 1):
            if 0 <= x < grid.shape[1]: grid[y1, x] = color
    elif abs(dx) == abs(dy): # Cheo
        step_y = 1 if dy > 0 else -1
        step_x = 1 if dx > 0 else -1
        for i in range(abs(dx) + 1):
            y, x = y1 + i * step_y, x1 + i * step_x
            if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
                grid[y, x] = color
    return grid

# --- [NÂNG CẤP] Các hàm hỗ trợ cho DSL mới ---
def _get_centroid(obj: ArcObject) -> Tuple[float, float]:
    """Tính tọa độ trung tâm (centroid) của một đối tượng."""
    # (Cách đơn giản, chỉ là trung tâm BBox)
    return obj.centroid

def _scale_grid_nn(grid: np.array, scale_factor: int) -> np.array:
    """Phóng to grid bằng Nearest Neighbor."""
    if scale_factor == 1: return grid
    if scale_factor <= 0: return np.array([[0]], dtype=int)
    h, w = grid.shape
    new_h, new_w = h * scale_factor, w * scale_factor
    
    # Tạo chỉ mục y và x đã phóng to
    y_indices = np.repeat(np.arange(h), scale_factor)
    x_indices = np.repeat(np.arange(w), scale_factor)
    
    # Dùng indexing để tạo grid mới
    scaled_grid = grid[y_indices[:, None], x_indices]
    return scaled_grid

def _find_symmetry_axis(grid: np.array) -> Tuple[int | None, int | None]:
    """Tìm trục đối xứng ngang (y) hoặc dọc (x) đầu tiên."""
    h, w = grid.shape
    
    # Kiem tra truc doc (x)
    for x in range(w - 1):
        # x là chỉ số *sau* đường kẻ
        left_width = x + 1
        right_width = w - left_width
        width_to_check = min(left_width, right_width)
        
        left_part = grid[:, left_width - width_to_check : left_width]
        right_part = grid[:, left_width : left_width + width_to_check]
        
        if np.array_equal(left_part, np.fliplr(right_part)):
            return (None, x) # Trả về chỉ số (x) của đường kẻ
            
    # Kiem tra truc ngang (y)
    for y in range(h - 1):
        top_height = y + 1
        bottom_height = h - top_height
        height_to_check = min(top_height, bottom_height)
        
        top_part = grid[top_height - height_to_check : top_height, :]
        bottom_part = grid[top_height : top_height + height_to_check, :]
        
        if np.array_equal(top_part, np.flipud(bottom_part)):
            return (y, None) # Trả về chỉ số (y) của đường kẻ

    return (None, None)

# ==================================================================
# 3. Định nghĩa Ngôn ngữ Chuyên biệt (DSL) Mở rộng
# ==================================================================

# --- 3.1 DSL Đơn giản (Không tham số) ---
def op_identity(grid: np.array) -> np.array: return grid
def op_rotate_90(grid: np.array) -> np.array: return np.rot90(grid, k=1)
def op_rotate_180(grid: np.array) -> np.array: return np.rot90(grid, k=2)
def op_rotate_270(grid: np.array) -> np.array: return np.rot90(grid, k=3)
def op_reflect_horizontal(grid: np.array) -> np.array: return np.fliplr(grid)
def op_reflect_vertical(grid: np.array) -> np.array: return np.flipud(grid)
def op_crop_to_content(grid: np.array) -> np.array:
    r_min, r_max, c_min, c_max = _get_bbox(grid)
    if r_min == 0 and r_max == 0 and c_min == 0 and c_max == 0:
         if grid.size > 0 and (grid.shape != (1,1) or grid[0,0] == 0):
             return np.array([[0]], dtype=int) 
         elif grid.size == 0:
             return np.array([[0]], dtype=int)
    return grid[r_min:r_max+1, c_min:c_max+1]

# --- [NÂNG CẤP] 3.2 DSL Mới: Đối xứng ---
def op_complete_symmetry_x(grid: np.array) -> np.array:
    """Tìm trục đối xứng dọc và hoàn thiện nó."""
    y_axis, x_axis = _find_symmetry_axis(grid)
    if x_axis is None: return grid # Không tìm thấy trục
    
    h, w = grid.shape
    x_line = x_axis + 1
    left_width = x_line
    right_width = w - left_width
    
    new_grid = grid.copy()
    if left_width > right_width: # Lấy bên trái, lật sang phải
        part_to_flip = grid[:, x_line - right_width - 1 : x_line]
        new_grid[:, x_line:] = np.fliplr(part_to_flip)[:, 1:]
    elif right_width > left_width: # Lấy bên phải, lật sang trái
        part_to_flip = grid[:, x_line : x_line + left_width + 1]
        new_grid[:, :x_line] = np.fliplr(part_to_flip)[:, :-1]
    return new_grid

def op_complete_symmetry_y(grid: np.array) -> np.array:
    """Tìm trục đối xứng ngang và hoàn thiện nó."""
    y_axis, x_axis = _find_symmetry_axis(grid)
    if y_axis is None: return grid # Không tìm thấy trục

    h, w = grid.shape
    y_line = y_axis + 1
    top_height = y_line
    bottom_height = h - top_height
    
    new_grid = grid.copy()
    if top_height > bottom_height: # Lấy bên trên, lật xuống dưới
        part_to_flip = grid[y_line - bottom_height - 1 : y_line, :]
        new_grid[y_line:, :] = np.flipud(part_to_flip)[1:, :]
    elif bottom_height > top_height: # Lấy bên dưới, lật lên trên
        part_to_flip = grid[y_line : y_line + top_height + 1, :]
        new_grid[:y_line, :] = np.flipud(part_to_flip)[:-1, :]
    return new_grid


SIMPLE_DSL = [
    op_identity, op_rotate_90, op_rotate_180, op_rotate_270,
    op_reflect_horizontal, op_reflect_vertical, op_crop_to_content,
    op_complete_symmetry_x, op_complete_symmetry_y # <<< Thêm DSL mới
]

# --- 3.3 DSL Phức tạp (Có tham số) ---
def op_recolor_by_color(grid: np.array, color_in: int, color_out: int) -> np.array:
    if color_in == color_out: return grid
    new_grid = grid.copy()
    new_grid[grid == color_in] = color_out
    return new_grid

def op_fill_background(grid: np.array, color: int) -> np.array:
    return op_recolor_by_color(grid, 0, color)

def op_copy_object_by_color(grid: np.array, color: int) -> np.array:
    objects = detect_objects(grid)
    color_objects = [o for o in objects if o.color == color]
    if len(color_objects) < 2: return grid
    pattern = color_objects[0]; destinations = color_objects[1:]
    new_grid = grid.copy()
    for obj in destinations:
        new_grid = _paste_grid(new_grid, pattern.grid, obj.y_min, obj.x_min)
    return new_grid

def op_connect_dots(grid: np.array, dot_color: int, line_color: int) -> np.array:
    coords = np.argwhere(grid == dot_color)
    if len(coords) < 2: return grid
    new_grid = grid.copy()
    coords = sorted(coords, key=lambda c: (c[1], c[0])) 
    for i in range(len(coords) - 1):
        y1, x1 = coords[i]; y2, x2 = coords[i+1]
        new_grid = _draw_line_simple(new_grid, y1, x1, y2, x2, line_color)
    return new_grid

# --- [NÂNG CẤP] 3.4 DSL Mới: Lồng nhau, Di chuyển, Co giãn ---
def op_fill_enclosed_bg(grid: np.array, new_color: int) -> np.array:
    """Tìm các vùng màu 0 bị "bao vây" và tô chúng bằng màu mới."""
    # binary_fill_holes hoạt động trên True/False
    # Nó sẽ lấp đầy các lỗ (False) bên trong các vùng True
    # Chúng ta muốn lấp đầy các lỗ (0) bên trong các vùng khác 0
    # -> Tạo mask "bất cứ thứ gì KHÔNG phải màu nền"
    walls = (grid != 0)
    holes = binary_fill_holes(walls)
    pixels_to_fill = holes & ~walls # Vùng được lấp đầy (True) VÀ không phải là tường (False)
    
    new_grid = grid.copy()
    new_grid[pixels_to_fill] = new_color
    return new_grid

def op_scale_object(grid: np.array, color: int, scale_factor: int) -> np.array:
    """Tìm đối tượng đầu tiên màu `color` và phóng to nó."""
    if scale_factor == 1: return grid
    objects = detect_objects(grid)
    obj_to_scale = next((o for o in objects if o.color == color), None)
    if obj_to_scale is None: return grid
    
    scaled_obj_grid = _scale_grid_nn(obj_to_scale.grid, scale_factor)
    
    # Xóa đối tượng cũ và dán đối tượng mới vào
    new_grid = grid.copy()
    new_grid[obj_to_scale.mask] = 0 # Xóa đối tượng cũ
    new_grid = _paste_grid(new_grid, scaled_obj_grid, obj_to_scale.y_min, obj_to_scale.x_min)
    return new_grid

def op_move_object_to_nearest(grid: np.array, move_color: int, target_color: int) -> np.array:
    """Di chuyển đối tượng A (move_color) đến đối tượng B (target_color) gần nhất."""
    objects = detect_objects(grid)
    move_objs = [o for o in objects if o.color == move_color]
    target_objs = [o for o in objects if o.color == target_color]
    if not move_objs or not target_objs: return grid
    
    move_obj = move_objs[0] # Chỉ di chuyển đối tượng đầu tiên
    move_centroid = _get_centroid(move_obj)
    
    # Tìm target gần nhất
    def dist(c1, c2):
        return (c1[0] - c2[0])**2 + (c1[1] - c2[1])**2
        
    nearest_target = min(target_objs, key=lambda o: dist(_get_centroid(o), move_centroid))
    target_centroid = _get_centroid(nearest_target)
    
    # Tính vector di chuyển (từ trung tâm A đến trung tâm B)
    dy = target_centroid[0] - move_centroid[0]
    dx = target_centroid[1] - move_centroid[1]
    
    # Vị trí dán mới (góc trên-trái)
    new_y_min = move_obj.y_min + dy
    new_x_min = move_obj.x_min + dx
    
    new_grid = grid.copy()
    new_grid[move_obj.mask] = 0 # Xóa đối tượng cũ
    new_grid = _paste_grid(new_grid, move_obj.grid, new_y_min, new_x_min)
    return new_grid

# ==================================================================
# 4. Thuật toán Tìm kiếm (Guided Search - Level 3.5)
# ==================================================================

PARAMETERIZED_DSL = [
    (op_recolor_by_color,  {"color_in": range(0, 10), "color_out": range(0, 10)}),
    (op_fill_background,   {"color": range(1, 10)}),
    (op_copy_object_by_color, {"color": range(1, 10)}),
    (op_connect_dots,      {"dot_color": range(1, 10), "line_color": range(1, 10)}),
    # <<< [NÂNG CẤP] Thêm DSL mới vào không gian tìm kiếm
    (op_fill_enclosed_bg,  {"new_color": range(1, 10)}),
    (op_scale_object,      {"color": range(1, 10), "scale_factor": [2, 3]}), # Thử scale x2, x3
    (op_move_object_to_nearest, {"move_color": range(1, 10), "target_color": range(1, 10)})
]

Program = Tuple[Callable[..., np.array], Dict[str, Any]]
ProgramTuple = Tuple[Program, ...] 
ProgramHash = Tuple[Tuple[str, Tuple[Tuple[str, Any], ...]], ...]

def serialize_program(program: ProgramTuple) -> ProgramHash:
    """Chuyển đổi một program (chứa dict) thành một tuple lồng hashable."""
    return tuple( 
        (op.__name__, tuple(sorted(params.items()))) 
        for op, params in program 
    )

def apply_program(program: ProgramTuple, input_grid: np.array) -> np.array:
    current_grid = copy.deepcopy(input_grid)
    for op_func, op_params in program:
        try:
            current_grid = op_func(current_grid, **op_params)
        except Exception:
            return np.array([[-1]]) # Grid lỗi
    return current_grid

def pad_to_match(g1: np.array, g2: np.array) -> Tuple[np.array, np.array]:
    if g1.size == 0: g1 = np.array([[0]], dtype=int)
    if g2.size == 0: g2 = np.array([[0]], dtype=int)
    h1, w1 = g1.shape; h2, w2 = g2.shape
    h_max, w_max = max(h1, h2), max(w1, w2)
    
    padded_g1 = np.full((h_max, w_max), 0, dtype=int)
    padded_g1[:h1, :w1] = g1
    padded_g2 = np.full((h_max, w_max), 0, dtype=int)
    padded_g2[:h2, :w2] = g2
    return padded_g1, padded_g2

# --- [NÂNG CẤP] Hàm Heuristic Cost "Thông minh" ---
def heuristic_cost_smart(program: ProgramTuple, train_pairs: list) -> int:
    """
    Hàm "chấm điểm" (Heuristic) thông minh hơn.
    Cost càng thấp, chương trình càng "gần đúng".
    """
    total_cost = 0
    for pair in train_pairs:
        train_in = np.array(pair['input'])
        train_out = np.array(pair['output'])
        
        try:
            predicted_out = apply_program(program, train_in)
            
            # 1. Phạt nặng nếu crash
            if predicted_out.shape == (1,1) and predicted_out[0,0] == -1:
                total_cost += 10000
                continue
            
            # 2. Phạt lỗi kích thước (Shape Penalty)
            h_pred, w_pred = predicted_out.shape
            h_true, w_true = train_out.shape
            shape_penalty = abs(h_pred - h_true) + abs(w_pred - w_true)
            total_cost += shape_penalty * 20 # Phạt nặng hơn cho sai kích thước
            
            # 3. Phạt lỗi số lượng đối tượng (Object Count Penalty)
            pred_objects = detect_objects(predicted_out)
            true_objects = detect_objects(train_out)
            object_count_penalty = abs(len(pred_objects) - len(true_objects))
            total_cost += object_count_penalty * 5 # Phạt cho mỗi đối tượng bị thiếu/thừa
            
            # 4. Phạt lỗi bộ màu sắc (Color Set Penalty)
            pred_colors = set(np.unique(predicted_out)) - {0}
            true_colors = set(np.unique(train_out)) - {0}
            color_penalty = len(pred_colors.symmetric_difference(true_colors))
            total_cost += color_penalty * 3 # Phạt cho mỗi màu bị sai
            
            # 5. Phạt lỗi pixel (Pixel Penalty)
            padded_pred, padded_true = pad_to_match(predicted_out, train_out)
            pixel_diff = np.sum(padded_pred != padded_true)
            total_cost += pixel_diff
            
        except Exception:
            total_cost += 10000 
            
    return total_cost
# --- [HẾT NÂNG CẤP] ---

# Cache từ vựng
base_vocabulary_cache: List[Program] | None = None

def get_base_vocabulary() -> List[Program]:
    """Tạo hoặc trả về 'Từ vựng' (Vocabulary) các lệnh cơ sở đã cache."""
    global base_vocabulary_cache
    if base_vocabulary_cache is not None:
        return base_vocabulary_cache
    
    base_vocabulary: List[Program] = []
    
    for func in SIMPLE_DSL:
        base_vocabulary.append( (func, {}) )
        
    for func, param_grid in PARAMETERIZED_DSL:
        param_names = param_grid.keys()
        param_values = param_grid.values()
        
        for param_combination in product(*param_values):
            params_dict = dict(zip(param_names, param_combination))
            if func == op_recolor_by_color and params_dict['color_in'] == params_dict['color_out']:
                continue
            if func == op_connect_dots and params_dict['dot_color'] == params_dict['line_color']:
                continue
            if func == op_move_object_to_nearest and params_dict['move_color'] == params_dict['target_color']:
                continue
            base_vocabulary.append( (func, params_dict) )
            
    print(f"  -> Kích thước Từ vựng (Vocabulary) DSL: {len(base_vocabulary)} lệnh cơ sở.")
    base_vocabulary_cache = base_vocabulary
    return base_vocabulary_cache

def solve_task_guided(task: dict, max_depth: int = 3, max_programs_to_check: int = 10000) -> list | None:
    """Tìm kiếm chương trình bằng Guided Best-First Search (sử dụng heapq)."""
    train_pairs = task['train']
    test_inputs_raw = [p['input'] for p in task['test']]
    base_vocab = get_base_vocabulary()
    
    # Hàng đợi ưu tiên: (cost, depth, tie_breaker, program_tuple)
    pq: List[Tuple[int, int, int, ProgramTuple]] = []
    visited: Set[ProgramHash] = set()
    tie_breaker = count() 
    
    # 1. Khởi tạo hàng đợi với các chương trình độ sâu 1
    for base_op in base_vocab:
        program = (base_op,) 
        cost = heuristic_cost_smart(program, train_pairs) # <<< [NÂNG CẤP] Dùng heuristic mới
        
        if cost == 0:
            program_name = " -> ".join(f"{op.__name__}({params})" for op, params in program)
            print(f"  -> Tìm thấy lời giải (depth=1): {program_name}")
            test_preds = [apply_program(program, np.array(test_in)).tolist() for test_in in test_inputs_raw]
            return test_preds
        
        program_hash = serialize_program(program)
        heapq.heappush(pq, (cost, 1, next(tie_breaker), program))
        visited.add(program_hash)
        
    programs_checked = 0
    
    # 2. Vòng lặp tìm kiếm chính
    while pq and programs_checked < max_programs_to_check:
        cost, depth, _, current_program = heapq.heappop(pq)
        programs_checked += 1
        
        if programs_checked % 1000 == 0:
             print(f"  -> Đã kiểm tra {programs_checked} chương trình. Cost thấp nhất hiện tại: {cost}")

        if depth < max_depth:
            for base_op in base_vocab:
                new_program = current_program + (base_op,)
                new_program_hash = serialize_program(new_program)
                
                if new_program_hash not in visited:
                    visited.add(new_program_hash)
                    new_cost = heuristic_cost_smart(new_program, train_pairs) # <<< [NÂNG CẤP] Dùng heuristic mới
                    
                    if new_cost == 0:
                        program_name = " -> ".join(f"{op.__name__}({params})" for op, params in new_program)
                        print(f"  -> Tìm thấy lời giải (depth={depth+1}): {program_name}")
                        test_preds = [apply_program(new_program, np.array(test_in)).tolist() for test_in in test_inputs_raw]
                        return test_preds
                    
                    heapq.heappush(pq, (new_cost, depth + 1, next(tie_breaker), new_program))

    print(f"  -> Không tìm thấy lời giải sau khi kiểm tra {programs_checked} chương trình.")
    return None

# ==================================================================
# 5. Vòng lặp chính và Tạo Submission
# ==================================================================

print("\nBắt đầu xử lý các task (với DSL Level 3.5 + Guided Search)...")

DATA_PATH = "/kaggle/input/arc-prize-2025"
TEST_CHALLENGES_PATH = os.path.join(DATA_PATH, "arc-agi_test_challenges.json")

try:
    test_challenges = load_json(TEST_CHALLENGES_PATH)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file test. Thử tải file evaluation làm dự phòng...")
    EVAL_CHALLENGES_PATH = os.path.join(DATA_PATH, "arc-agi_evaluation-challenges.json")
    try:
        test_challenges = load_json(EVAL_CHALLENGES_PATH)
        print("Đã tải file evaluation.")
    except FileNotFoundError:
        print("Lỗi: Không tìm thấy file evaluation. Thoát.")
        test_challenges = {} 

submission = {}
tasks_solved = 0
total_tasks = len(test_challenges)

print("Đang khởi tạo Từ vựng DSL (có thể mất vài giây)...")
_ = get_base_vocabulary() # Gọi lần đầu để cache
print("...Từ vựng đã sẵn sàng.")


for i, (task_id, task) in enumerate(test_challenges.items()):
    
    print(f"\n[{i+1}/{total_tasks}] Đang xử lý Task: {task_id}")
    
    # Tăng giới hạn kiểm tra lên 10,000 và độ sâu 2
    predicted_outputs_list = solve_task_guided(task, max_depth=2, max_programs_to_check=10000) 
    
    formatted_predictions = []
    
    if predicted_outputs_list is None:
        # print("  -> Không tìm thấy lời giải. Dùng dự đoán mặc định.")
        num_test_inputs = len(task['test'])
        default_grid = [[0]] 
        
        for _ in range(num_test_inputs):
            formatted_predictions.append({
                "attempt_1": default_grid,
                "attempt_2": default_grid 
            })
            
    else:
        tasks_solved += 1
        for pred_grid in predicted_outputs_list:
            formatted_predictions.append({
                "attempt_1": pred_grid,
                "attempt_2": pred_grid
            })

    submission[task_id] = formatted_predictions

print("\n...Hoàn tất xử lý.")
print(f"Tổng số task: {total_tasks}")
print(f"Số task giải được: {tasks_solved}")

# Ghi ra file submission.json
submission_path = "submission.json"
with open(submission_path, 'w') as f:
    json.dump(submission, f)

print(f"File submission đã được lưu tại: {submission_path}")

