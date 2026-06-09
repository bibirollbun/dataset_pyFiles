%%bash
# Remove old copy if exists
rm -rf HRM

# Clone repo
git clone https://github.com/sapientinc/HRM.git
cd HRM

# Fix submodule URLs to HTTPS
git config --file .gitmodules submodule.dataset/raw-data/ARC-AGI.url https://github.com/fchollet/ARC-AGI.git
git config --file .gitmodules submodule.dataset/raw-data/ARC-AGI-2.url https://github.com/arcprize/ARC-AGI-2.git
git config --file .gitmodules submodule.dataset/raw-data/ConceptARC.url https://github.com/victorvikram/ConceptARC.git
git submodule sync

# Pull submodules
git submodule update --init --recursive



!pip install --upgrade pip packaging ninja wheel setuptools setuptools-scm

# Install torch + optional flash-attn
try:
    import torch
    has_gpu = torch.cuda.is_available()
except ImportError:
    has_gpu = False

if has_gpu:
    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    try:
        !pip install flash-attn
    except:
        print("⚠️ flash-attn install failed, continuing...")
else:
    !pip install torch torchvision torchaudio



from huggingface_hub import hf_hub_download
arc_ckpt = hf_hub_download("sapientinc/HRM-checkpoint-ARC-2", filename="checkpoint")
print("ARC checkpoint path:", arc_ckpt)



!pip install coolname



import os
ARC_DATA_DIR = "/kaggle/input/arc-prize-2025"
print("Available ARC files:", os.listdir(ARC_DATA_DIR))



!pip install -r /kaggle/working/HRM/requirements.txt



import os

checkpoint_root = "/root/.cache/huggingface/hub/models--sapientinc--HRM-checkpoint-ARC-2/snapshots"
print("Available snapshots:", os.listdir(checkpoint_root))



import os, json, subprocess

ARC_DATA_DIR = "/kaggle/input/arc-prize-2025"
HRM_DIR = "/kaggle/working/HRM"

data_path = os.path.join(ARC_DATA_DIR, "arc-agi_training_challenges.json")

one_puzzle_path = "/kaggle/working/one_puzzle.json"

with open(data_path, "r") as f:
    puzzles = json.load(f)

first_key = list(puzzles.keys())[0]
one_puzzle = {first_key: puzzles[first_key]}

with open(one_puzzle_path, "w") as f:
    json.dump(one_puzzle, f)

print(f"Saved one puzzle: {first_key} -> {one_puzzle_path}")

checkpoint = "/root/.cache/huggingface/hub/models--sapientinc--HRM-checkpoint-ARC-2/snapshots/<your_snapshot_id>"

cmd = [
    "python3", os.path.join(HRM_DIR, "evaluate.py"),
    f"checkpoint={checkpoint}",
    f"data_path={one_puzzle_path}"
]

print("Running command:\n", " ".join(cmd))
subprocess.run(cmd, check=True)



ls /root/.cache/huggingface/hub/models--sapientinc--HRM-checkpoint-ARC-2/snapshots/ee2c595e8b9dd061a448f6f65fca997bd2227c74



!ls -R /root/.cache/huggingface/hub/models--sapientinc--HRM-checkpoint-ARC-2/snapshots/ee2c595e8b9dd061a448f6f65fca997bd2227c74



!git lfs install
!git clone https://huggingface.co/sapientinc/HRM-checkpoint-ARC-2



ls -R HRM-checkpoint-ARC-2



!python3 /kaggle/working/HRM/evaluate.py \
  checkpoint=/kaggle/working/HRM-checkpoint-ARC-2/checkpoint \
  data_path=/kaggle/working/one_puzzle.json



import os, subprocess

HRM_DIR = "/kaggle/working/HRM"
one_puzzle_path = "/kaggle/working/one_puzzle.json"

# FIX: removed the extra dot at the end
snapshot_id = "ee2c595e8b9dd061a448f6f65fca997bd2227c74"
checkpoint = f"/root/.cache/huggingface/hub/models--sapientinc--HRM-checkpoint-ARC-2/snapshots/{snapshot_id}"

cmd = [
    "python3", os.path.join(HRM_DIR, "evaluate.py"),
    f"checkpoint={checkpoint}",
    f"data_path={one_puzzle_path}"
]

print("Running command:\n", " ".join(cmd))
subprocess.run(cmd, check=True)






import subprocess, os

ARC_DATA_DIR = "/kaggle/input/arc-prize-2025"
HRM_DIR = "/kaggle/working/HRM"

data_path = os.path.join(ARC_DATA_DIR, "arc-agi_training_challenges.json")
checkpoint = "/root/.cache/huggingface/hub/models--sapientinc--HRM-checkpoint-ARC-2/snapshots/..."  # keep your actual checkpoint path

cmd = [
    "python3",
    os.path.join(HRM_DIR, "evaluate.py"),   
    f"checkpoint={checkpoint}",
    f"data_path={data_path}",
]

print("Running:", " ".join(cmd))
subprocess.run(cmd, check=True)



split = "arc-agi_test_challenges.json"   # or arc-agi_evaluation_challenges.json
data_path = os.path.join(ARC_DATA_DIR, split)

cmd = [sys.executable, "evaluate.py", f"checkpoint={arc_ckpt}", f"data_path={data_path}"]
print("Running:", " ".join(cmd))
subprocess.run(cmd, check=False)





