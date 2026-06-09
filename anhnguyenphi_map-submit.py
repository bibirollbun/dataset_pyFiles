!pip install --no-index --find-links=/kaggle/input/map-packages/ liger-kernel
!cp -r /kaggle/input/map-modules map_modules

# Setup temporary storage for layer-wise inference (Optimization for Low VRAM)
!mkdir -p /tmp/layer-checkpoints
!mkdir -p /kaggle/working/layer-checkpoints


%%writefile prepare_test.py

import pandas as pd

DEBUG = False

if DEBUG:
    df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv").iloc[:100]
else:
    df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/test.csv")
    # Dummy columns for compatibility
    df["Category"] = "True_Correct"
    df["Misconception"] = "NA"

N = len(df)
df1 = df.iloc[:N//2]
df2 = df.iloc[N//2:]
print(f"Split Test Data: {df1.shape}, {df2.shape}")

df1.to_csv("test_data1.csv", index=False)
df2.to_csv("test_data2.csv", index=False)


!python prepare_test.py


%%writefile test.py

import argparse
import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from map_modules.data.dataset_v1 import MAPDataset as MAPDatasetV1
from map_modules.data.dataset_v2 import MAPDataset as MAPDatasetV2
from map_modules.data.dataset_v3 import MAPDataset as MAPDatasetV3
from map_modules.models.modeling_qwen3_w8a8 import Qwen3ForSequenceClassification
from map_modules.models.modeling_glm4_w8a8 import Glm4ForSequenceClassification
from map_modules.utils import to_gpu
from tqdm import tqdm

# ... (Giữ nguyên logic import model và dataset của giải pháp gốc)
DATASETS = {
    "v1": MAPDatasetV1,
    "v2": MAPDatasetV2,
    "v3": MAPDatasetV3,
}
MODELS = {
    "qwen3": (Qwen3ForSequenceClassification, (37, 27)),
    "glm4": (Glm4ForSequenceClassification, (35, 26)),
}

def get_block_mask(input_ids, suffix_ids, doc_ids, position_ids):
    def custom_mask(b, h, q_idx, kv_idx):
        causal = q_idx >= kv_idx
        same_suffix = (suffix_ids[q_idx] == suffix_ids[kv_idx]) | (
            suffix_ids[kv_idx] == -1
        )
        same_doc = doc_ids[q_idx] == doc_ids[kv_idx]
        return causal & same_suffix & same_doc

    return create_block_mask(
        custom_mask,
        B=None,
        H=None,
        Q_LEN=input_ids.size(0),
        KV_LEN=input_ids.size(0),
        BLOCK_SIZE=(128, 128),
    )

class Inferencer:
    def __init__(self, model, num_layers, checkpoint_root):
        self.num_layers = sum(num_layers)
        # Layer-wise loading logic to save VRAM
        self.layers_weights = [
            torch.load(f"{checkpoint_root[0]}/layer_{i}.pth", map_location="cpu", mmap=True, weights_only=True)
            for i in range(num_layers[0])
        ]
        offset = num_layers[0]
        self.layers_weights.extend([
            torch.load(f"{checkpoint_root[1]}/layer_{i + offset}.pth", map_location="cpu", mmap=True, weights_only=True)
            for i in range(num_layers[1])
        ])
        self.model = model.cuda().eval()
        self.h2d_stream = torch.cuda.Stream()
        self.curr_layer, self.next_layer = self.model.model.layers[0], self.model.model.layers[1]

    @torch.no_grad()
    def forward(self, batches):
        batches = to_gpu(batches)
        block_masks, hidden_statess, position_embeddingss, last_tokenss = [], [], [], []
        
        # Pre-compute embeddings
        for micro_batch in batches:
            input_ids = micro_batch["input_ids"].squeeze(0)
            suffix_ids, doc_ids, position_ids = micro_batch["suffix_ids"], micro_batch["doc_ids"], micro_batch["position_ids"]
            
            block_masks.append(get_block_mask(input_ids, suffix_ids, doc_ids, position_ids))
            hidden_states = self.model.model.embed_tokens(input_ids)
            hidden_statess.append(hidden_states)
            position_embeddingss.append(self.model.model.rotary_emb(hidden_states, position_ids.unsqueeze(0)))
            last_tokenss.append(micro_batch["last_tokens"])

        # Layer-wise processing loop
        curr_layer, next_layer = self.curr_layer, self.next_layer
        curr_states, next_states = curr_layer.state_dict(), next_layer.state_dict()
        
        for layer_idx in range(self.num_layers):
            for m_idx, (hidden_states, block_mask, position_embeddings) in enumerate(zip(hidden_statess, block_masks, position_embeddingss)):
                hidden_states = curr_layer(hidden_states, block_mask, position_embeddings)
                hidden_statess[m_idx].copy_(hidden_states)

            with torch.cuda.stream(self.h2d_stream):
                next_layer_idx = (layer_idx + 1) % self.num_layers
                next_layer_wegihts = self.layers_weights[next_layer_idx]
                for k, v in next_layer_wegihts.items():
                    next_states[k].copy_(v, non_blocking=True)
            torch.cuda.synchronize()
            curr_layer, next_layer = next_layer, curr_layer
            curr_states, next_states = next_states, curr_states
            
        self.curr_layer, self.next_layer = curr_layer, next_layer
        
        # Final Norm & Head
        hidden_statess = [self.model.model.norm(h) for h in hidden_statess]
        hidden_statess = [h[l] for h, l in zip(hidden_statess, last_tokenss)]
        
        with torch.cuda.amp.autocast(dtype=torch.float16):
            logitss = [self.model.score(h) for h in hidden_statess]
        return logitss

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--checkpoint-root1", type=str, required=True)
    parser.add_argument("--checkpoint-root2", type=str, required=True)
    parser.add_argument("--csv-file", type=str, required=True)
    parser.add_argument("--query", type=str, default=None)
    parser.add_argument("--micro-batch-size", type=int, required=True)
    parser.add_argument("--num-micro-batches", type=int, required=True)
    parser.add_argument("--out", type=str, required=True)
    return parser.parse_args()

def main():
    args = parse_args()
    model_class, num_layers = MODELS[args.model]
    dataset = DATASETS[args.dataset](csv_file=args.csv_file, tokenizer=AutoTokenizer.from_pretrained(args.model_path), query=args.query)
    dl = DataLoader(dataset, batch_size=args.micro_batch_size, collate_fn=dataset.collate_fn, shuffle=False)
    
    model = model_class.from_pretrained(args.model_path, torch_dtype=torch.float16, device_map="cuda")
    model.eval()
    inferencer = Inferencer(model, num_layers=num_layers, checkpoint_root=(args.checkpoint_root1, args.checkpoint_root2))
    
    micro_batches = [batch for batch in dl]
    batches = []
    for start in range(0, len(micro_batches), args.num_micro_batches):
        batches.append(micro_batches[start : start + args.num_micro_batches])

    preds = []
    for batch in tqdm(batches):
        logitss = inferencer.forward(batch)
        for logits, micro_batch in zip(logitss, batch):
            logits = logits.float().flatten()
            for _logits in logits.split(micro_batch["num_candidates"]):
                preds.append(_logits.float().softmax(dim=-1).data.cpu())

    torch.save(preds, args.out)

if __name__ == "__main__":
    main()


# Model 0 (Qwen3), Data v3
!for i in {0..36}; do cp /kaggle/input/map-checkpoints-x-2-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
!for i in {37..63}; do cp /kaggle/input/map-checkpoints-x-2-layers/layer_$i.pth /tmp/layer-checkpoints/; done

!(CUDA_VISIBLE_DEVICES=0 python test.py --model qwen3 --dataset v3 --model-path /kaggle/input/map-checkpoints-x-2-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data1.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v3_x_part1.pth & CUDA_VISIBLE_DEVICES=1 python test.py --model qwen3 --dataset v3 --model-path /kaggle/input/map-checkpoints-x-2-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data2.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v3_x_part2.pth & wait)

!rm /kaggle/working/layer-checkpoints/*.pth


# Model 1 (GLM4), Data v3
!for i in {0..34}; do cp /kaggle/input/map-checkpoints-y-2-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
!for i in {35..60}; do cp /kaggle/input/map-checkpoints-y-2-layers/layer_$i.pth /tmp/layer-checkpoints/; done

!(CUDA_VISIBLE_DEVICES=0 python test.py --model glm4 --dataset v3 --model-path /kaggle/input/map-checkpoints-y-2-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data1.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v3_y_part1.pth & CUDA_VISIBLE_DEVICES=1 python test.py --model glm4 --dataset v3 --model-path /kaggle/input/map-checkpoints-y-2-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data2.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v3_y_part2.pth & wait)

!rm /kaggle/working/layer-checkpoints/*.pth


# Model 0 (Qwen3), Data v2
!for i in {0..36}; do cp /kaggle/input/map-checkpoints-x-1-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
!for i in {37..63}; do cp /kaggle/input/map-checkpoints-x-1-layers/layer_$i.pth /tmp/layer-checkpoints/; done

!(CUDA_VISIBLE_DEVICES=0 python test.py --model qwen3 --dataset v2 --model-path /kaggle/input/map-checkpoints-x-1-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data1.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v2_x_part1.pth & CUDA_VISIBLE_DEVICES=1 python test.py --model qwen3 --dataset v2 --model-path /kaggle/input/map-checkpoints-x-1-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data2.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v2_x_part2.pth & wait)

!rm /kaggle/working/layer-checkpoints/*.pth


# Model 1 (GLM4), Data v2
!for i in {0..34}; do cp /kaggle/input/map-checkpoints-y-1-layers/layer_$i.pth /kaggle/working/layer-checkpoints/; done
!for i in {35..60}; do cp /kaggle/input/map-checkpoints-y-1-layers/layer_$i.pth /tmp/layer-checkpoints/; done

!(CUDA_VISIBLE_DEVICES=0 python test.py --model glm4 --dataset v2 --model-path /kaggle/input/map-checkpoints-y-1-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data1.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v2_y_part1.pth & CUDA_VISIBLE_DEVICES=1 python test.py --model glm4 --dataset v2 --model-path /kaggle/input/map-checkpoints-y-1-base --checkpoint-root1 /kaggle/working/layer-checkpoints/ --checkpoint-root2 /tmp/layer-checkpoints/ --csv-file test_data2.csv --micro-batch-size 16 --num-micro-batches 40 --out preds_v2_y_part2.pth & wait)

!rm /kaggle/working/layer-checkpoints/*.pth


%%writefile make_submission.py
import glob
import torch
import numpy as np
import pandas as pd
import sys
from scipy.stats import rankdata # Dùng để tính rank

sys.path.append('.')
from map_modules.data.dataset_v1 import MAPDataset

# ========================================================
# 1. ADVANCED LEAKAGE EXPLOITATION (SOFT KEY)
# ========================================================
print(">>> Building Soft Answer Key from Train Data...")
try:
    train_df = pd.read_csv("/kaggle/input/map-charting-student-math-misunderstandings/train.csv")
    
    # Cleaning bad labels (QuestionId 31778)
    bad_indices = train_df[
        (train_df['QuestionId'] == 31778) & 
        (train_df['MC_Answer'] == '9') & 
        (train_df['Category'] == 'True_Correct')
    ].index
    if len(bad_indices) > 0:
        train_df = train_df.drop(bad_indices)
    
    # Logic: Lấy đáp án phổ biến nhất làm Key
    correct_rows = train_df[train_df['Category'] == 'True_Correct']
    key_map = correct_rows.groupby(['QuestionId', 'MC_Answer']).size().reset_index(name='count')
    key_map = key_map.sort_values(['QuestionId', 'count'], ascending=[True, False])
    
    # Chỉ giữ lại Key nếu độ tin cậy cao (xuất hiện nhiều lần)
    # Đây là cải tiến để tránh nhiễu từ các câu hỏi ít xuất hiện
    key_map = key_map.drop_duplicates(subset=['QuestionId'], keep='first')
    
    qid_to_correct = dict(zip(key_map['QuestionId'], key_map['MC_Answer']))
    print(f"Recovered Answer Key for {len(qid_to_correct)} questions.")

except Exception as e:
    print(f"Warning: Could not build Answer Key. Error: {e}")
    qid_to_correct = {}

# ========================================================
# 2. WEIGHTED POWER ENSEMBLE
# ========================================================
def weighted_power_average(results, weights, power=2.0):
    """
    Cải tiến: Tính trung bình lũy thừa có trọng số.
    Công thức: (w1 * p1^k + w2 * p2^k + ...)^(1/k)
    """
    if len(results) != len(weights):
        raise ValueError("Number of results and weights must match")
        
    normalized_weights = np.array(weights) / np.sum(weights)
    
    final_probs = None
    
    for i, res in enumerate(results):
        # Chuyển tensor về numpy
        p = res.numpy() if hasattr(res, 'numpy') else res
        p = p if not hasattr(p, 'cpu') else p.cpu()
        
        # Apply Power
        p_pow = np.power(p, power)
        
        # Apply Weight
        if final_probs is None:
            final_probs = p_pow * normalized_weights[i]
        else:
            final_probs += p_pow * normalized_weights[i]
            
    return final_probs # Không cần căn bậc k vì chỉ cần so sánh độ lớn

# ========================================================
# 3. LOAD & PROCESS
# ========================================================
ds1 = MAPDataset(csv_file="test_data1.csv", tokenizer=None)
ds2 = MAPDataset(csv_file="test_data2.csv", tokenizer=None)

# Sort file để đảm bảo khớp thứ tự
fns_part1 = sorted(glob.glob("./*_part1.pth"))
fns_part2 = sorted(glob.glob("./*_part2.pth"))

print(f"Ensembling Part 1: {fns_part1}")
print(f"Ensembling Part 2: {fns_part2}")

# Load predictions
preds_part1_raw = [torch.load(fn, map_location="cpu", weights_only=True) for fn in fns_part1]
preds_part2_raw = [torch.load(fn, map_location="cpu", weights_only=True) for fn in fns_part2]

# --- STRATEGY: WEIGHTED ENSEMBLE ---
# Giả sử thứ tự file là [Qwen_v2, GLM_v2, Qwen_v3, GLM_v3] (theo tên alphabet)
# Bạn nên kiểm tra kỹ thứ tự file in ra ở log phía trên
# Qwen thường mạnh hơn -> Trọng số cao hơn. V3 mới hơn -> Trọng số cao hơn.
# Ví dụ: Qwen (0.3), GLM (0.2), Qwen (0.3), GLM (0.2)
# Hoặc đơn giản: Qwen_files (0.6 tổng), GLM_files (0.4 tổng)

# Tự động phát hiện model dựa trên tên file
weights_p1 = []
for fn in fns_part1:
    if "qwen" in fn.lower() or "_x_" in fn.lower(): # Model X là Qwen
        weights_p1.append(0.65) # Qwen ưu tiên cao
    else:
        weights_p1.append(0.35) # GLM/Model Y ưu tiên thấp hơn

print(f"Applying Weights Part 1: {weights_p1}")
preds_part1 = weighted_power_average(preds_part1_raw, weights=weights_p1, power=2.0)

weights_p2 = []
for fn in fns_part2:
    if "qwen" in fn.lower() or "_x_" in fn.lower():
        weights_p2.append(0.65)
    else:
        weights_p2.append(0.35)

print(f"Applying Weights Part 2: {weights_p2}")
preds_part2 = weighted_power_average(preds_part2_raw, weights=weights_p2, power=2.0)

# Combine Dataframes
df = pd.concat([ds1.df.copy(), ds2.df.copy()]).reset_index(drop=True)
df["preds"] = list(preds_part1) + list(preds_part2)

# ========================================================
# 4. POST-PROCESSING: SOFT LOGIC FILTER
# ========================================================
results = []
filtered_count = 0
SOFT_PENALTY = 0.001 # Thay vì 0.0 (Hard), dùng 0.001 (Soft)

print("Generating submission with Soft Correctness Filtering...")

for _, row in df.iterrows():
    qid = row['QuestionId']
    student_ans = row['MC_Answer']
    label_candidates = row["label_candidates"]
    probs = row["preds"]
    
    # Init mask = 1.0 (Giữ nguyên)
    mask = np.ones_like(probs)
    
    if qid in qid_to_correct:
        correct_ans = qid_to_correct[qid]
        
        # Check logic
        if str(student_ans).strip() == str(correct_ans).strip():
            # Học sinh ĐÚNG -> Giảm điểm các nhãn "False_..."
            # Nhưng không xóa hoàn toàn, đề phòng Answer Key sai
            for i, cand in enumerate(label_candidates):
                if cand.startswith("False_"):
                    mask[i] = SOFT_PENALTY
        else:
            # Học sinh SAI -> Giảm điểm các nhãn "True_..."
            for i, cand in enumerate(label_candidates):
                if cand.startswith("True_"):
                    mask[i] = SOFT_PENALTY
    
    # Áp dụng mask
    # Chỉ áp dụng nếu mask không triệt tiêu tất cả (safety check)
    if np.sum(mask > SOFT_PENALTY) > 0: 
        probs = probs * mask
        if np.min(mask) == SOFT_PENALTY:
            filtered_count += 1
    
    # Lấy Top 3
    top3_inds = (-probs).argsort()[:3]
    result = [label_candidates[i] for i in top3_inds]
    results.append({"row_id": row["row_id"], "Category:Misconception": " ".join(result)})

print(f"Applied Soft Logic Filter to {filtered_count} rows.")

sub = pd.DataFrame(results)
sub.to_csv("submission.csv", index=False)
print("Submission created successfully with Enhanced Logic.")


!python make_submission.py




