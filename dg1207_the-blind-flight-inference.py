import tensorflow as tf
import numpy as np
import os

# Load your classifier
classifier = tf.keras.models.load_model("/kaggle/input/map-rec/keras/default/1/map_recognition.keras")

# Setup Data Pipeline (Crucial for 10k images)
test_ds = tf.keras.utils.image_dataset_from_directory(
    '/kaggle/input/the-blind-flight-synapse-drive-ps-1/SynapseDrive_Dataset/test/images/',
    labels=None, 
    image_size=(256, 256), # Match your CNN
    batch_size=8, 
    shuffle=False
)

# Get all file paths in the exact order the dataset will read them
file_paths = test_ds.file_paths

# Run batch prediction
preds = classifier.predict(test_ds)
class_indices = np.argmax(preds, axis=1)

# Route paths into three lists
desert_files = [file_paths[i] for i, cls in enumerate(class_indices) if cls == 0]
forest_files = [file_paths[i] for i, cls in enumerate(class_indices) if cls == 1]
lab_files = [file_paths[i] for i, cls in enumerate(class_indices) if cls == 2]


%pip install ultralytics


import torch, gc
gc.collect()
torch.cuda.empty_cache()

import os
import json
import heapq
import pandas as pd
import numpy as np
from ultralytics import YOLO
from itertools import islice
import gc

def chunked(iterable, size):
    it = iter(iterable)
    while True:
        batch = list(islice(it, size))
        if not batch:
            break
        yield batch

# 1. Helper Functions
def heuristic(a, b):
    # Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(grid, start, goal):
    rows, cols = 20, 20
    INF = float('inf')
    dist = [[INF]*cols for _ in range(rows)]
    parent = {}
    pq = []

    dist[start[0]][start[1]] = 0
    # Priority queue stores (f_score, g_score, (r, c))
    heapq.heappush(pq, (heuristic(start, goal), 0, start))

    while pq:
        _, cur_cost, (r, c) = heapq.heappop(pq)

        if (r, c) == goal:
            break

        if cur_cost > dist[r][c]:
            continue

        for dr, dc in [(1,0), (-1,0), (0,1), (0,-1)]:
            nr, nc = r + dr, c + dc

            if 0 <= nr < rows and 0 <= nc < cols:
                cell_val = grid[nr][nc]
                if cell_val <0: continue # Wall
                
                new_cost = cur_cost + cell_val
                if new_cost < dist[nr][nc]:
                    dist[nr][nc] = new_cost
                    parent[(nr, nc)] = (r, c)
                    priority = new_cost + heuristic((nr, nc), goal)
                    heapq.heappush(pq, (priority, new_cost, (nr, nc)))

    if goal not in parent:
        s=''
        x=goal[1]-start[1]
        y=goal[0]-start[0]
        if x<0:
            s=s+'l'*abs(x)
        else:
            s=s+'r'*abs(x)
        if y<0:
            s=s+'u'*abs(x)
        else:
            s=s+'d'*abs(x) 
        return s
    # 2. Path Reconstruction & Correct Direction Mapping
    path = []
    cur = goal
    while cur != start:
        path.append(cur)
        cur = parent[cur]
    path.append(start)
    path.reverse() # Start to Goal

    # Mapping matrix movements to characters
    # Row +1 = down, Row -1 = up, Col +1 = right, Col -1 = left
    DIR_MAP = {(1, 0): 'd', (-1, 0): 'u', (0, 1): 'r', (0, -1): 'l'}
    
    directions = ''
    for i in range(len(path) - 1):
        r1, c1 = path[i]
        r2, c2 = path[i+1]
        move = (r2 - r1, c2 - c1)
        directions += DIR_MAP.get(move, '')
        
    return directions

# 3. Processing Function
def process_yolo(model_path, image_list, output_file):
    if not image_list: return
    
    yolo_model = YOLO(model_path)
    # stream=True is efficient for large datasets
    
    D = {
        'rover': 1.2, 'sand': 1.2, 'quicksand': 3.7, 'cactus': -1, 
        'rocks': -1, 'dirt': 1.5, 'puddle': 2.8, 'startship': 1.5, 
        'floor': 1.0, 'glue': 3.0, 'drone': 1.0, 'wall': -1, 'tree':-1, 'plasma':-1
    }

    for result in yolo_model.predict(source=image_list, stream=False, max_det=400, cache=False, workers=0, batch=1):
        img_name = os.path.basename(result.path)
        all_data = []
        
        # Build detection list
        for box in result.boxes:
            all_data.append({
                "class": yolo_model.names[int(box.cls[0])],
                "xmax": float(box.xyxy[0][2]),
                "ymax": float(box.xyxy[0][3])
            })

        if not all_data: continue
        
        # Sort and Group into Grid (20x20 logic)
        df = pd.DataFrame(all_data).sort_values(by='ymax').reset_index(drop=True)
        
        img_h = 640 
        tolerance = (img_h / 20) * 0.5 
        
        rows_list = []
        current_row = [df.iloc[0].to_dict()]
        
        for i in range(1, len(df)):
            if abs(df.iloc[i]['ymax'] - current_row[-1]['ymax']) < tolerance:
                current_row.append(df.iloc[i].to_dict())
            else:
                rows_list.append(sorted(current_row, key=lambda x: x['xmax']))
                current_row = [df.iloc[i].to_dict()]
        rows_list.append(sorted(current_row, key=lambda x: x['xmax']))

        # Create Matrix
        matrix = np.full((20, 20), 1.0, dtype=float) # Default to 1.0 cost
        start, goal = (0, 0), (19, 19)
        start_found=False
        goal_found= False
        for r_idx, row_data in enumerate(rows_list):
            if r_idx >= 20: break
            for c_idx, obj in enumerate(row_data):
                if c_idx >= 20: break
                
                cls_name = obj['class']
                if cls_name == 'goal':
                    if goal_found==False:
                        goal = (r_idx, c_idx)
                        matrix[r_idx, c_idx] = 1.0
                        goal_found=True
                elif cls_name in ['rover', 'startship', 'drone']:
                    if start_found==False:
                        start = (r_idx, c_idx)
                        matrix[r_idx, c_idx] = D[cls_name]
                        start_found=True
                else:
                    matrix[r_idx, c_idx] = D[cls_name]

        # Apply Boost from JSON
        # Logic assumes a specific folder structure for velocity JSONs
        try:
            js_path = result.path.replace('/images/', '/velocities/').replace('.png', '.json')
            with open(js_path, 'r') as f:
                boost_data = json.load(f)
                boost_array = np.array(boost_data["boost"])
                matrix = matrix - boost_array
        except Exception:
            pass # Skip if JSON is missing

        # Calculate Path
        ans = astar(matrix, start, goal)
        
        # Write to results file
        output_file.write(f"{img_name.replace('.png', '')},{ans}\n")

# 4. Execution
# Open file once to handle all model runs
CHUNK_SIZE = 20 # SAFE: 50–100 recommended for YOLO-11x + 400 objs

with open('results.csv', 'w') as F:

    # ---- DESERT ----
    for i, chunk in enumerate(chunked(desert_files, CHUNK_SIZE)):
        print(f"[DESERT] Chunk {i+1}")
        process_yolo(
            "/kaggle/input/the-blind-flight/runs/detect/des/weights/best.pt",
            chunk,
            F
        )

    # ---- FOREST ----
    for i, chunk in enumerate(chunked(forest_files, CHUNK_SIZE)):
        print(f"[FOREST] Chunk {i+1}")
        process_yolo(
            "/kaggle/input/the-blind-flight/runs/detect/forest/weights/best.pt",
            chunk,
            F
        )

    # ---- LAB ----
    for i, chunk in enumerate(chunked(lab_files, CHUNK_SIZE)):
        print(f"[LAB] Chunk {i+1}")
        process_yolo(
            "/kaggle/input/the-blind-flight/runs/detect/lab/weights/best.pt",
            chunk,
            F
        )
# 1. Load the CSV as strings to prevent immediate stripping
df = pd.read_csv('results.csv', header=None, dtype={0: str})

# 2. Pad column 0 with zeros until it is 4 characters long
# '1' becomes '0001', '12' becomes '0012', etc.
df[0] = df[0].str.zfill(4)

# 3. Sort by the padded strings (now they sort correctly!)
df_sorted = df.sort_values(by=0)

# 4. Save back to CSV
df_sorted.to_csv('results.csv', index=False, header=False)

