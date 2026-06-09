CODE_DIR = "/kaggle/input/zero-shot-v2/method"
CKPT_PATH = "/kaggle/input/4500/pytorch/default/1/step_4500.pth"

import os, json, sys
import numpy as np
import torch
import torch.nn as nn  # Import nn for type hinting

sys.path.append(CODE_DIR)
from utils.functions import load_model_class


ARC_MAX = 30
GRID_SEQ_LEN = ARC_MAX * ARC_MAX  # 900
SEQ_LEN = GRID_SEQ_LEN * 3  # [train1_in, train2_in, test_in] -> 2700

PAD_ID = 0
EOS_ID = 1
DIGIT_OFFSET = 2  # digits 0..9 -> 2..11
BLANK_IDENTIFIER_ID = 0  # <- blank identifier WE ARE ZERO-SHOT

PAD_SEQ = np.full(GRID_SEQ_LEN, PAD_ID, dtype=np.uint8)


def arc_grid_to_np(grid):
    arr = np.array(grid, dtype=np.uint8)
    assert arr.ndim == 2
    assert arr.shape[0] <= ARC_MAX and arr.shape[1] <= ARC_MAX
    assert np.all((arr >= 0) & (arr <= 9))
    return arr

def grid_to_seq(grid_np: np.ndarray) -> np.ndarray:
    nrow, ncol = grid_np.shape
    tokens = np.pad(grid_np + DIGIT_OFFSET,
                    ((0, ARC_MAX - nrow), (0, ARC_MAX - ncol)),
                    constant_values=PAD_ID)
    # EOS lines at first empty row/col (exactly like training)
    if nrow < ARC_MAX:
        tokens[nrow, :ncol] = EOS_ID
    if ncol < ARC_MAX:
        tokens[:nrow, ncol] = EOS_ID
    return tokens.flatten().astype(np.uint8)

def build_input_and_label_seq(train_pairs, test_input_np):
    # Deterministically take up to 2 train examples; PAD if missing
    pick = train_pairs[:2] + [None] * max(0, 2 - len(train_pairs))
    in_chunks, lab_chunks = [], []
    for ex in pick:
        if ex is None:
            in_chunks.append(PAD_SEQ)
            lab_chunks.append(PAD_SEQ)
        else:
            tr_in_np, tr_out_np = ex
            in_chunks.append(grid_to_seq(tr_in_np))
            lab_chunks.append(grid_to_seq(tr_out_np))
    in_chunks.append(grid_to_seq(test_input_np))
    lab_chunks.append(PAD_SEQ)
    return np.concatenate(in_chunks), np.concatenate(lab_chunks)

def decode_seq_to_grid(seq_900):
    g = np.asarray(seq_900, dtype=np.int64).reshape(ARC_MAX, ARC_MAX)
    row_marks = np.where(g[:, 0] == EOS_ID)[0]
    col_marks = np.where(g[0, :] == EOS_ID)[0]
    if len(row_marks): nrow = int(row_marks[0])
    else:
        rows = np.where((g > 1).any(axis=1))[0]
        nrow = int(rows.max() + 1) if len(rows) else 1
    if len(col_marks): ncol = int(col_marks[0])
    else:
        cols = np.where((g > 1).any(axis=0))[0]
        ncol = int(cols.max() + 1) if len(cols) else 1
    cropped = g[:nrow, :ncol]
    out = np.where(cropped > 1, cropped - DIGIT_OFFSET, 0).astype(int)
    return out.tolist() if out.size else [[0]]


def find_arc_test():
    for p in [
        "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
        "/kaggle/input/arc-prize-2024/arc-agi_test_challenges.json",
        "/kaggle/input/arc-agi/arc-agi_test_challenges.json",
    ]:
        if os.path.exists(p): return p
    for root, _, files in os.walk("/kaggle/input"):
        if "arc-agi_test_challenges.json" in files:
            return os.path.join(root, "arc-agi_test_challenges.json")
    return None


def create_model(device):
    arch_cfg = dict(
        halt_exploration_prob=0.1,
        halt_max_steps=6,
        H_cycles=1, L_cycles=2,
        H_layers=0, L_layers=1,
        hidden_size=256, num_heads=8, expansion=4,
        puzzle_emb_ndim=0,  # you can keep 0; model still wants 'puzzle_identifiers' input
        puzzle_emb_len=0,
        pos_encodings="rope",
        # forward_dtype="bfloat16",
        mlp_t=False,
        no_ACT_continue=True,
    )
    model_cfg = dict(
        **arch_cfg,
        batch_size=1,
        vocab_size=12,  # 0..11 (PAD=0, EOS=1, digits 2..11)
        seq_len=SEQ_LEN,
        num_puzzle_identifiers=0,  # OK when using a blank identifier stream
        causal=False,
    )

    model_cls = load_model_class("recursive_reasoning.trm@TinyRecursiveReasoningModel_ACTV5_Hybrid")
    loss_head_cls = load_model_class("losses@ACTLossHeadV5")

    base = model_cls(model_cfg).to(device)
    model = loss_head_cls(base, loss_type="stablemax_cross_entropy").to(device)

    # --- START: MODIFICATION ---
    # This block is replaced with the logic from your load_checkpoint function
    
    print(f"Loading checkpoint {CKPT_PATH}")
    map_loc = device  # Use the device passed to this function
    sd = torch.load(CKPT_PATH, map_location=map_loc)

    # We are loading into the raw model, so the prefix is just 'model.' not '_orig_mod.model.inner'
    # Let's adjust for both compiled and non-compiled model loading
    
    # Only check puzzle_emb if the config says it exists
    if arch_cfg['puzzle_emb_ndim'] > 0:
        # Try to find the expected key
        raw_key = "model.puzzle_emb.weights"
        compiled_key = "_orig_mod.model.puzzle_emb.weights" # Compiled model key
        
        # This part is tricky. Let's check for the compiled key first.
        puzzle_emb_name = None
        if compiled_key in sd:
            puzzle_emb_name = compiled_key
        elif raw_key in sd:
            puzzle_emb_name = raw_key
        
        # Fallback to your original key if loading from an old DDP checkpoint
        if puzzle_emb_name is None and "_orig_mod.model.inner.puzzle_emb.weights" in sd:
            puzzle_emb_name = "_orig_mod.model.inner.puzzle_emb.weights"
        
        # This line is now safe inside the conditional
        expected_shape: torch.Size = model.model.puzzle_emb.weights.shape  # type: ignore
        
        if puzzle_emb_name in sd:
            puzzle_emb = sd[puzzle_emb_name]
            if puzzle_emb.shape != expected_shape:
                print(f"Resetting puzzle embedding shape. Found {puzzle_emb.shape}, Expected {expected_shape}")
                sd[puzzle_emb_name] = (
                    torch.mean(puzzle_emb, dim=0, keepdim=True).expand(expected_shape).contiguous()
                )
    else:
        print("Skipping puzzle embedding logic: puzzle_emb_ndim is 0.")
        
    # Need to handle key mismatches if loading a compiled state_dict into a non-compiled model or vice-versa
    # For now, we just use load_state_dict which is robust to compiled prefixes
    
    # Use strict=False to ignore keys from the checkpoint that no longer exist
    # in the new model (e.g., the puzzle_emb weights)
    model.load_state_dict(sd, assign=True, strict=False)
    print("Successfully loaded model weights with strict=False.")
    
    # --- END: MODIFICATION ---

    model.eval()
    return model


def pick_pred_seq(preds: dict, seq_len: int) -> torch.Tensor:
    for v in preds.values():
        if torch.is_tensor(v) and v.dtype in (torch.long, torch.int64) and v.dim() >= 2 and v.shape[-1] == seq_len:
            return v
    for v in preds.values():
        if torch.is_tensor(v) and v.dim() >= 3 and v.shape[-2] == seq_len:
            return torch.argmax(v, dim=-1)
    for v in preds.values():
        if torch.is_tensor(v) and v.dim() >= 2 and v.shape[-1] > 32:
            return torch.argmax(v, dim=-1)
    raise RuntimeError("Could not infer prediction sequence from model outputs")


def _longify_inplace(carry):
    for name in ("new_steps", "steps", "halt_steps", "step_idx", "sample_idx", "token_idx"):
        if hasattr(carry, name):
            val = getattr(carry, name)
            if torch.is_tensor(val) and val.dtype is not torch.int64:
                setattr(carry, name, val.long())
    return carry


def main():
    test_json = find_arc_test()
    print(test_json)
    if test_json is None:
        with open("submission.json", "w") as f:
            json.dump({"00000000": [{"attempt_1": [[0]], "attempt_2": [[0]]}]}, f)
        print("⚠️ No ARC test json found; wrote dummy submission.json")
        return

    with open(test_json, "r") as f:
        tasks = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_model(device)

    results = {}

    for task_id, task in tasks.items():
        train_pairs = []
        for ex in task.get("train", []):
            tr_in = arc_grid_to_np(ex["input"])
            tr_out = arc_grid_to_np(ex["output"])
            train_pairs.append((tr_in, tr_out))

        preds_for_task = []
        for t in task.get("test", []):
            test_in = arc_grid_to_np(t["input"])

            x_np, y_np = build_input_and_label_seq(train_pairs, test_in)
            x = torch.tensor(x_np, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]
            y = torch.tensor(y_np, dtype=torch.long, device=device).unsqueeze(0)  # [1, L]

            # >>> Provide puzzle_identifiers (blank id is OK for test)
            puzzle_ids = torch.full((1,), BLANK_IDENTIFIER_ID, dtype=torch.long, device=device)  # [B]

            batch = {"inputs": x, "labels": y, "puzzle_identifiers": puzzle_ids}

            batch = {
                k: v.to(torch.int32) if k in ["inputs", "labels", "puzzle_identifiers"] else v 
                for k, v in batch.items()
            }
            
            carry = model.initial_carry(batch)
            carry = _longify_inplace(carry)

            while True:
                #carry = _longify_inplace(carry)  # keep indices long
                required_outputs = {"inputs", "puzzle_identifiers", "q_halt_logits", "preds"}
                carry, loss, metrics, out, all_finish = model(carry=carry, batch=batch, return_keys=required_outputs)
                if all_finish:
                    break

            pred_seq = pick_pred_seq(out, SEQ_LEN)[0].detach().cpu().numpy().astype(np.int64)
            test_out_tokens = pred_seq[-GRID_SEQ_LEN:]  # last 900 correspond to test out
            out_grid = decode_seq_to_grid(test_out_tokens)

            preds_for_task.append({"attempt_1": out_grid, "attempt_2": out_grid})

        if not preds_for_task:
            preds_for_task = [{"attempt_1": [[0]], "attempt_2": [[0]]}]
        results[task_id] = preds_for_task

    with open("submission.json", "w") as f:
        json.dump(results, f)
    print(f"✅ Wrote submission.json with {len(results)} tasks.")

main()

