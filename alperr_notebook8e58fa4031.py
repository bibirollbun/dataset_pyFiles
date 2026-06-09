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


import json
from pathlib import Path
from tqdm import tqdm

# 1. Test verisinin yolu
TEST_PATH = Path('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json')
print(TEST_PATH)

# 2. Test verisini yÃ¼kle
with open(TEST_PATH) as f:
    test_data = json.load(f)

# 3. JSON yapÄ±sÄ±nÄ± kontrol et
print("JSON anahtarlarÄ±:", list(test_data.keys()))

# 4. SonuÃ§larÄ± buraya ekleyeceÄŸiz
submission = {}

# 5. JSON yapÄ±sÄ±na gÃ¶re doÄŸru anahtarÄ± kullan
# EÄŸer "root" anahtarÄ± varsa onu kullan, yoksa doÄŸrudan test_data'yÄ± kullan
if "root" in test_data:
    tasks = test_data["root"]
else:
    tasks = test_data

print(f"Toplam gÃ¶rev sayÄ±sÄ±: {len(tasks)}")

# 6. Her gÃ¶rev (task_id) iÃ§in dÃ¶ngÃ¼
for task_id, task in tqdm(tasks.items()):
    task_predictions = []
    
    # 7. Her test input'u iÃ§in tahmin yapÄ±lacak
    for test_item in task["test"]:
        inp = test_item["input"]  # Grid biÃ§iminde 2D liste
        
        # ðŸ”§ Dummy tahmin (gerÃ§ek modelle deÄŸiÅŸtirilmeli)
        # GÃ¼venli bir dummy output oluÅŸtur
        try:
            # Input'un boyutlarÄ±nÄ± al
            height = len(inp)
            width = len(inp[0]) if height > 0 else 0
            
            # Basit bir kopyalama stratejisi
            if height >= 2 and width >= 2:
                # Ä°lk 2x2'yi kopyala
                dummy_output = [row[:2] for row in inp[:2]]
            elif height >= 1 and width >= 1:
                # Tek satÄ±r varsa onu kopyala
                dummy_output = [inp[0][:1]]
            else:
                # BoÅŸ input iÃ§in varsayÄ±lan
                dummy_output = [[0]]
                
        except (IndexError, TypeError):
            # Hata durumunda gÃ¼venli varsayÄ±lan
            dummy_output = [[0]]
        
        # 8. Bu formatta kaydediyoruz: {"attempt_1": [...], "attempt_2": [...]}
        task_predictions.append({
            "attempt_1": dummy_output,
            "attempt_2": dummy_output
        })
    
    # 9. GÃ¶rev ID'sine karÅŸÄ±lÄ±k sonucu ekle
    submission[task_id] = task_predictions

print(f"Submission hazÄ±rlandÄ±. Toplam gÃ¶rev: {len(submission)}")

# 10. Dosyaya yaz
with open("submission.json", "w") as f:
    json.dump(submission, f, indent=2)

print("submission.json dosyasÄ± oluÅŸturuldu!")


import json
from pathlib import Path

TRAIN_PATH = Path('/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json')

with open(TRAIN_PATH) as f:
    train_tasks = json.load(f)

# EÄŸer "root" anahtarÄ± varsa, onu kullan
if "root" in train_tasks:
    train_tasks = train_tasks["root"]

print(f"Toplam eÄŸitim gÃ¶revi sayÄ±sÄ±: {len(train_tasks)}")

# Ä°lk birkaÃ§ task_id'yi gÃ¶relim
task_ids = list(train_tasks.keys())
print("BazÄ± task_id Ã¶rnekleri:", task_ids[:5])



some_task_id = task_ids[0]  # Ä°lk task Ã¶rneÄŸi

task = train_tasks[some_task_id]

print(f"SeÃ§ilen task id: {some_task_id}")
print(f"Train Ã¶rnek sayÄ±sÄ±: {len(task['train'])}")

for i, pair in enumerate(task['train']):
    print(f"Ã–rnek {i+1}:")
    print("Input:")
    for row in pair['input']:
        print(row)
    print("Output:")
    for row in pair['output']:
        print(row)
    print("---------")



def rotate_90(grid):
    return [list(row) for row in zip(*grid[::-1])]

def rotate_180(grid):
    return rotate_90(rotate_90(grid))

def rotate_270(grid):
    return rotate_90(rotate_180(grid))

def flip_horizontal(grid):
    return [row[::-1] for row in grid]

def flip_vertical(grid):
    return grid[::-1]

def invert_colors(grid):
    return [[9 - cell for cell in row] for row in grid]

def grids_equal(grid1, grid2):
    if len(grid1) != len(grid2):
        return False
    for r1, r2 in zip(grid1, grid2):
        if r1 != r2:
            return False
    return True



def find_matching_transformations(input_grid, output_grid):
    transformations = {
        "original": lambda g: g,
        "rotate_90": rotate_90,
        "rotate_180": rotate_180,
        "rotate_270": rotate_270,
        "flip_horizontal": flip_horizontal,
        "flip_vertical": flip_vertical,
        "invert_colors": invert_colors,
        # Kombinasyonlar da ekleyebiliriz, Ã¶rneÄŸin flip + invert:
        "flip_horizontal + invert_colors": lambda g: invert_colors(flip_horizontal(g)),
        "flip_vertical + invert_colors": lambda g: invert_colors(flip_vertical(g)),
        "rotate_90 + invert_colors": lambda g: invert_colors(rotate_90(g)),
        # Daha fazla kombinasyon gerekirse eklenebilir
    }
    
    matches = []
    
    for name, func in transformations.items():
        try:
            transformed = func(input_grid)
            if grids_equal(transformed, output_grid):
                matches.append(name)
        except Exception as e:
            # HatalarÄ± gÃ¶z ardÄ± et
            pass
    
    return matches



summary = {}
for task_id, task in train_tasks.items():
    task_matches = []
    for idx, pair in enumerate(task['train']):
        input_grid = pair['input']
        output_grid = pair['output']
        matches = find_matching_transformations(input_grid, output_grid)
        task_matches.append({
            "pair_index": idx,
            "matches": matches
        })
    summary[task_id] = task_matches

# Ã–rnek Ã§Ä±ktÄ±: Ä°lk 5 gÃ¶revin ilk Ã§iftlerindeki dÃ¶nÃ¼ÅŸÃ¼mler
for i, (task_id, matches) in enumerate(summary.items()):
    
    print(f"Task ID: {task_id}")
    for match in matches:
        print(f"  Pair {match['pair_index']} iÃ§in eÅŸleÅŸen dÃ¶nÃ¼ÅŸÃ¼mler: {match['matches']}")
    print("---------")


import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleFCN(nn.Module):
    def __init__(self):
        super(SimpleFCN, self).__init__()
        # Basit 3 katmanlÄ± CNN
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # -> (32, H, W)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1) # -> (64, H, W)
        self.conv3 = nn.Conv2d(64, 10, kernel_size=3, padding=1) # -> (10, H, W) Ã§Ä±ktÄ± katmanÄ±

    def forward(self, x, target_size=None):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)  # logits (10 sÄ±nÄ±f)
        if target_size is not None:
            # Output'u hedef boyuta ayarla (Ã¶r: output grid boyutu)
            x = F.adaptive_avg_pool2d(x, output_size=target_size)
        return x



def grid_to_tensor(grid):
    # grid: List[List[int]] deÄŸerleri 0-9
    # Ã‡Ä±ktÄ±: (1, H, W) FloatTensor, 0-9 arasÄ± deÄŸerler normalize edilmeden bÄ±rakÄ±ldÄ±
    import numpy as np
    arr = np.array(grid, dtype=np.float32)
    tensor = torch.from_numpy(arr).unsqueeze(0)  # (1, H, W)
    return tensor

def output_grid_to_target(grid):
    # grid: List[List[int]], 0-9 sÄ±nÄ±f etiketleri
    # Ã‡Ä±ktÄ±: (H, W) LongTensor (CrossEntropyLoss iÃ§in)
    import numpy as np
    arr = np.array(grid, dtype=np.int64)
    tensor = torch.from_numpy(arr)  # (H, W)
    return tensor



import torch
import torch.nn as nn
import torch.optim as optim

# Model ve optimizasyon ayarlarÄ±
model = SimpleFCN()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

# EÄŸitim parametreleri
num_epochs = 10  # Daha fazla epoch iÃ§in deÄŸeri artÄ±rabilirsin

for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    print(f"\n=== Epoch {epoch + 1}/{num_epochs} ===")

    for task_index, (task_id, task) in enumerate(train_tasks.items()):
        print(f"\n>>> Task {task_index + 1}/{len(train_tasks)} | Task ID: {task_id}")

        for i, pair in enumerate(task['train']):
            input_grid = pair['input']
            output_grid = pair['output']

            x = grid_to_tensor(input_grid).unsqueeze(0)  # (1, 1, H, W)
            y = output_grid_to_target(output_grid).unsqueeze(0)  # (1, H, W)

            optimizer.zero_grad()

            logits = model(x, target_size=y.shape[-2:])  # (1, 10, H_out, W_out)

            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            # Her batch'te bilgilendirici Ã§Ä±ktÄ± ver
            print(f"[Task {task_index + 1} | Sample {i + 1}/{len(task['train'])}] Loss: {loss.item():.4f}")

    avg_loss = total_loss / num_batches
    print(f"\n[Epoch {epoch + 1}] Ortalama Loss: {avg_loss:.4f}")



import json
from pathlib import Path

TRAIN_SOL_PATH = Path('/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json')

with open(TRAIN_SOL_PATH) as f1:
    training_solutions = json.load(f1)

# EÄŸer "root" anahtarÄ± varsa, onun altÄ±ndaki gÃ¶revlere eriÅŸelim
if "root" in training_solutions:
    training_solutions = training_solutions["root"]

print(f"Toplam eÄŸitim gÃ¶revi sayÄ±sÄ±: {len(training_solutions)}")

# Ä°lk birkaÃ§ task_id'yi gÃ¶relim
task_ids = list(training_solutions.keys())
print("BazÄ± task_id Ã¶rnekleri:", task_ids[:5])

# Ã–rnek bir task_id iÃ§in kaÃ§ output var ve ilk outputun boyutlarÄ±
sample_task_id = task_ids[0]
outputs = training_solutions[sample_task_id]

print(f"\nTask ID: {sample_task_id} iÃ§in output sayÄ±sÄ±: {len(outputs)}")

# Ä°lk outputun satÄ±r ve sÃ¼tun sayÄ±sÄ± (2D liste)
first_output_grid = outputs[0]
height = len(first_output_grid)
width = len(first_output_grid[0]) if height > 0 else 0
print(f"Ä°lk output grid boyutu: {height} x {width}")



challenge_task_ids = set(train_tasks.keys())
solution_task_ids = set(training_solutions.keys())

print(f"Training challenges gÃ¶rev sayÄ±sÄ±: {len(challenge_task_ids)}")
print(f"Training solutions gÃ¶rev sayÄ±sÄ±: {len(solution_task_ids)}")

common_tasks = challenge_task_ids.intersection(solution_task_ids)
print(f"Her iki dosyada da bulunan gÃ¶rev sayÄ±sÄ±: {len(common_tasks)}")

# Eksik gÃ¶rev var mÄ± kontrol et
missing_in_solutions = challenge_task_ids - solution_task_ids
missing_in_challenges = solution_task_ids - challenge_task_ids

print(f"Solutions'da olmayan gÃ¶rev sayÄ±sÄ±: {len(missing_in_solutions)}")
print(f"Challenges'da olmayan gÃ¶rev sayÄ±sÄ±: {len(missing_in_challenges)}")



# train_tasks ve training_solutions zaten yukarÄ±da yÃ¼klenmiÅŸ ve "root" altÄ±na eriÅŸilmiÅŸ varsayÄ±yorum

challenge_task_ids = set(train_tasks.keys())
solution_task_ids = set(training_solutions.keys())

print(f"Training challenges gÃ¶rev sayÄ±sÄ±: {len(challenge_task_ids)}")
print(f"Training solutions gÃ¶rev sayÄ±sÄ±: {len(solution_task_ids)}")

common_tasks = challenge_task_ids.intersection(solution_task_ids)
print(f"Her iki dosyada da bulunan gÃ¶rev sayÄ±sÄ±: {len(common_tasks)}")

missing_in_solutions = challenge_task_ids - solution_task_ids
missing_in_challenges = solution_task_ids - challenge_task_ids

print(f"Solutions dosyasÄ±nda olmayan gÃ¶rev sayÄ±sÄ±: {len(missing_in_solutions)}")
print(f"Challenges dosyasÄ±nda olmayan gÃ¶rev sayÄ±sÄ±: {len(missing_in_challenges)}")

if len(missing_in_solutions) > 0:
    print(f"\nSolutions dosyasÄ±nda olmayan bazÄ± task_id'ler: {list(missing_in_solutions)[:5]}")
if len(missing_in_challenges) > 0:
    print(f"\nChallenges dosyasÄ±nda olmayan bazÄ± task_id'ler: {list(missing_in_challenges)[:5]}")



print(TRAIN_SOL_PATH)


import torch
import torch.nn as nn

model.eval()  # Modeli deÄŸerlendirme moduna al
criterion = nn.CrossEntropyLoss()

total_val_loss = 0.0
num_val_samples = 0

with torch.no_grad():  # Grad hesaplamayÄ± kapat
    for task_id, task in train_tasks.items():
        test_pairs = task['test']

        true_outputs = training_solutions.get(task_id, None)
        if true_outputs is None:
            print(f"Task {task_id} iÃ§in ground truth yok!")
            continue

        for i, test_item in enumerate(test_pairs):
            test_input = test_item['input']
            x = grid_to_tensor(test_input).unsqueeze(0)  # (1, 1, H, W)

            # true_outputs genelde liste; i ile doÄŸru outputu alÄ±yoruz
            if isinstance(true_outputs, list):
                true_output_grid = true_outputs[i]
            else:
                true_output_grid = true_outputs  # Nadir durumda tek output varsa

            y = output_grid_to_target(true_output_grid).unsqueeze(0)  # (1, H, W)

            logits = model(x, target_size=y.shape[-2:])  # (1, 10, H_out, W_out)
            val_loss = criterion(logits, y)

            total_val_loss += val_loss.item()
            num_val_samples += 1

val_loss_avg = total_val_loss / num_val_samples if num_val_samples > 0 else float('nan')
print(f"Validation Loss (training_challenges test kÄ±smÄ±): {val_loss_avg:.4f}")



# 1. Evaluation challenges dosyasÄ±nÄ± yÃ¼kle
EVAL_CHALLENGES_PATH = Path('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json')
with open(EVAL_CHALLENGES_PATH) as f:
    eval_tasks = json.load(f)
if "root" in eval_tasks:
    eval_tasks = eval_tasks["root"]
print(f"Evaluation gÃ¶rev sayÄ±sÄ±: {len(eval_tasks)}")

# 2. Evaluation solutions dosyasÄ±nÄ± yÃ¼kle
EVAL_SOL_PATH = Path('/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json')
with open(EVAL_SOL_PATH) as f:
    eval_solutions = json.load(f)
if "root" in eval_solutions:
    eval_solutions = eval_solutions["root"]
print(f"Evaluation Ã§Ã¶zÃ¼mleri sayÄ±sÄ±: {len(eval_solutions)}")


# 3. Model ve loss fonksiyonu (daha Ã¶nce tanÄ±mlÄ± model kullanÄ±lacak)
model.eval()
criterion = nn.CrossEntropyLoss()

total_val_loss = 0.0
num_val_samples = 0

with torch.no_grad():
    for task_id, task in eval_tasks.items():
        test_pairs = task['test']
        true_outputs = eval_solutions.get(task_id, None)
        if true_outputs is None:
            print(f"Task {task_id} iÃ§in ground truth yok!")
            continue

        for i, test_item in enumerate(test_pairs):
            test_input = test_item['input']
            x = grid_to_tensor(test_input).unsqueeze(0)

            # true_outputs formatÄ± Ã¶nce kontrol edilmeli, Ã¶rneÄŸin:
            if isinstance(true_outputs, list):
                true_output_grid = true_outputs[i] if isinstance(true_outputs[i], list) else true_outputs[i]['output']
            else:
                true_output_grid = true_outputs

            y = output_grid_to_target(true_output_grid).unsqueeze(0)

            logits = model(x, target_size=y.shape[-2:])
            val_loss = criterion(logits, y)

            total_val_loss += val_loss.item()
            num_val_samples += 1

val_loss_avg = total_val_loss / num_val_samples
print(f"Evaluation Set Validation Loss: {val_loss_avg:.4f}")


import json

TEST_CHALLENGES_PATH = Path('/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json')
with open(TEST_CHALLENGES_PATH) as f:
    test_tasks = json.load(f)
if "root" in test_tasks:
    test_tasks = test_tasks["root"]

submission = {}

model.eval()
with torch.no_grad():
    for task_id, task in test_tasks.items():
        predictions = []
        for test_item in task['test']:
            test_input = test_item['input']
            x = grid_to_tensor(test_input).unsqueeze(0)

            # Tahmin Ã¼ret, Ã§Ä±ktÄ± boyutu iÃ§in kendi stratejine gÃ¶re ayarla (Ã¶r: input boyutuna eÅŸitle)
            output_size = (len(test_input), len(test_input[0]))  # Ã–rnek: output input ile aynÄ± boyutta
            logits = model(x, target_size=output_size)
            predicted_classes = logits.argmax(dim=1).squeeze(0).cpu().numpy().tolist()

            predictions.append({
                "attempt_1": predicted_classes,
                "attempt_2": predicted_classes  # AynÄ± tahmini iki kez verdik, istersen farklÄ± yapabilirsin
            })

        submission[task_id] = predictions

# DosyayÄ± kaydet
with open("submission.json", "w") as f:
    json.dump(submission, f, indent=2)

print("submission.json dosyasÄ± oluÅŸturuldu!")


