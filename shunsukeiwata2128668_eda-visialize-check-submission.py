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


import json, numpy as np

#SUB_PATH = "/kaggle/input/eval-submission/20250823_eval_submission.json"
SUB_PATH = "/kaggle/input/eval-submission-20250828/submission_all_lora_20250828.json"

with open(SUB_PATH, "r") as f:
    data = json.load(f)

same_count = 0
pairs = 0  # ä¸¡æ–¹ã��ã‚�ã�£ã�¦ã�„ã‚‹ä»¶æ•°ï¼ˆå�‚è€ƒï¼‰

for tid, preds in data.items():
    if not isinstance(preds, list):
        continue
    for e in preds:
        if not isinstance(e, dict):
            continue
        a1 = e.get("attempt_1", e.get("attempt1"))
        a2 = e.get("attempt_2", e.get("attempt2"))
        if a1 is None or a2 is None:
            continue
        try:
            A1 = np.array(a1)
            A2 = np.array(a2)
            if A1.ndim == 2 and A2.ndim == 2:
                pairs += 1
                if A1.shape == A2.shape and np.array_equal(A1, A2):
                    same_count += 1
        except Exception:
            pass

print("Exactly identical (attempt_1 == attempt_2):", same_count)
# å�‚è€ƒï¼šprint("Pairs compared:", pairs)
diff_count = pairs - same_count
print("Different (attempt_1 != attempt_2):", diff_count)



# =========================
# ARC é…�å¸ƒãƒ•ã‚¡ã‚¤ãƒ«ã‚’èª­ã�¿è¾¼ã�¿ã€�lenï¼ˆä»¶æ•°ï¼‰ã� ã�‘å‡ºåŠ›
# =========================
import os, json

BASE_DIR = "/kaggle/input/arc-prize-2025"

FILES = [
    "arc-agi_training_challenges.json",
    "arc-agi_training_solutions.json",
    "arc-agi_evaluation_challenges.json",
    "arc-agi_evaluation_solutions.json",
    "arc-agi_test_challenges.json",
    "sample_submission.json",
    "/kaggle/input/eval-submission/20250823_eval_submission.json",  # â†� absolute
    "/kaggle/input/eval-submission-20250828/submission_all_lora_20250828.json"
]

# å…ˆé ­ã�„ã��ã�¤ã�®ã‚¿ã‚¹ã‚¯ã�®å†…è¨³ã‚’å‡ºã�™ã�‹ï¼ˆå¤šã�™ã��ã‚‹ã�¨ãƒ­ã‚°ã�Œé•·ã�„ã�®ã�§ä¸Šé™�ï¼‰
HEAD_TASKS = 20

def get_path(name: str) -> str:
    p = os.path.join(BASE_DIR, name)
    return p if os.path.exists(p) else name

for fname in FILES:
    path = get_path(fname)
    if not os.path.exists(path):
        continue

    try:
        with open(path, "r") as f:
            data = json.load(f)
    except Exception as e:
        print(f"\n*** {fname}: èª­ã�¿è¾¼ã�¿å¤±æ•—: {repr(e)}")
        continue

    print(f"\n=== {fname} ===")

    # 1) ãƒˆãƒƒãƒ—ãƒ¬ãƒ™ãƒ«ã�® len
    if isinstance(data, dict):
        print(f"len(data) = {len(data)}   # ï¼ˆä¾‹ï¼štask_id ã�®æ•°ï¼‰")
    elif isinstance(data, list):
        print(f"len(data) = {len(data)}   # ï¼ˆä¾‹ï¼šãƒˆãƒƒãƒ—ã�Œé…�åˆ—ã�®å ´å�ˆã�®è¦�ç´ æ•°ï¼‰")
        # ä»¥é™�ã�®è©³ç´°ã�¯çœ�ç•¥ï¼ˆlen ã� ã�‘ã�«é™�å®šï¼‰
        continue
    else:
        print(f"type(data) = {type(data).__name__}")
        continue

    low = fname.lower()

    # 2) challenges ã�®å ´å�ˆï¼šå�„ã‚¿ã‚¹ã‚¯ã�® train/test ã�® len ã‚’è¦‹ã‚‹
    if "challenges" in low:
        items = list(data.items())
        total_train = sum(len(ch.get("train", []) or []) for _, ch in items)
        total_test  = sum(len(ch.get("test",  []) or []) for _, ch in items)
        print(f"sum len(train) = {total_train}")
        print(f"sum len(test)  = {total_test}")

        # å…ˆé ­ã�„ã��ã�¤ã�‹ã� ã�‘å€‹åˆ¥è¡¨ç¤ºï¼ˆlen ã�®ã�¿ï¼‰
        for i, (tid, ch) in enumerate(items[:HEAD_TASKS], 1):
            tr = ch.get("train", []) or []
            te = ch.get("test",  []) or []
            print(f"  [{i}] task_id={tid}: len(train)={len(tr)}, len(test)={len(te)}")

    # 3) solutions ã�®å ´å�ˆï¼šå�„ã‚¿ã‚¹ã‚¯ã�® test å‡ºåŠ›æ•°ï¼ˆlenï¼‰ã‚’è¦‹ã‚‹
    elif "solutions" in low:
        items = list(data.items())
        total_outs = sum(len(outs) for _, outs in items if isinstance(outs, list))
        print(f"sum len(test_outputs) = {total_outs}")

        for i, (tid, outs) in enumerate(items[:HEAD_TASKS], 1):
            if isinstance(outs, list):
                print(f"  [{i}] task_id={tid}: len(outputs)={len(outs)}")
            else:
                print(f"  [{i}] task_id={tid}: (unexpected type {type(outs).__name__})")

    # 4) sample_submission ã�¯ãƒˆãƒƒãƒ—ã�® len ã�®ã�¿ï¼ˆlen ã� ã�‘ã�«é™�å®šï¼‰
    elif "sample_submission" in low:
        pass  # ã�™ã�§ã�« len(data) ã‚’å‡ºã�—ã�¦ã�„ã‚‹ã�®ã�§è¿½åŠ å‡ºåŠ›ã�¯ã�ªã�—



print()


# å�¯è¦–åŒ–ç”¨ãƒ©ã‚¤ãƒ–ãƒ©ãƒªã‚¤ãƒ³ãƒ�ãƒ¼ãƒˆ
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from pathlib import Path


# ===== ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—ï¼ˆARCã�®0ã€œ9è‰²ï¼‰ =====
# ========= ã‚«ãƒ©ãƒ¼ãƒ�ãƒƒãƒ—å®šç¾© =========
cmap = colors.ListedColormap([
    '#000000',  # 0: é»’
    '#0074D9',  # 1: é�’
    '#FF4136',  # 2: èµ¤
    '#2ECC40',  # 3: ç·‘
    '#FFDC00',  # 4: é»„
    '#AAAAAA',  # 5: ã‚°ãƒ¬ãƒ¼
    '#F012BE',  # 6: ãƒ�ã‚¼ãƒ³ã‚¿
    '#FF851B',  # 7: ã‚ªãƒ¬ãƒ³ã‚¸
    '#7FDBFF',  # 8: æ°´è‰²
    '#870C25'   # 9: èŒ¶è‰²
])

norm = colors.Normalize(vmin=0, vmax=9)

# ===== Kaggleã�§ã‚‚ãƒ­ãƒ¼ã‚«ãƒ«ã�§ã‚‚å‹•ã��ãƒ‘ã‚¹è§£æ±º =====
def get_path(name: str) -> Path:
    kaggle_base = Path('/kaggle/input/arc-prize-2025')
    return kaggle_base/name if (kaggle_base/name).exists() else Path(name)

# ===== JSONãƒ­ãƒ¼ãƒ€ =====
def load_json(path: Path):
    with open(path, 'r') as f:
        return json.load(f)


training_challenges = load_json(get_path('arc-agi_training_challenges.json'))
training_solutions  = load_json(get_path('arc-agi_training_solutions.json'))

# ä¾‹: è©•ä¾¡ç”¨ã�®å…¬é–‹ã‚»ãƒƒãƒˆã‚‚å¿…è¦�ã�ªã‚‰
# evaluation_challenges = load_json(get_path('arc-agi_evaluation_challenges.json'))
# evaluation_solutions  = load_json(get_path('arc-agi_evaluation_solutions.json'))



def show_grid(ax, grid, title=""):
    """grid: 2D list/np.arrayï¼ˆ0ã€œ9ã�®è‰²IDï¼‰"""
    arr = np.array(grid)
    ax.imshow(arr, cmap=cmap, norm=norm)
    # ç›®ç››ã‚Šã�¯æ¶ˆã�—ã�¦ã€�æ–¹çœ¼ã� ã�‘è¦‹ã�›ã‚‹ã�¨è¦‹ã‚„ã�™ã�„
    ax.set_xticks([x-0.5 for x in range(1+arr.shape[1])])
    ax.set_yticks([y-0.5 for y in range(1+arr.shape[0])])
    ax.grid(True, which="both", color="#666666", linewidth=0.8)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_title(title, fontsize=11, color="#333")



def plot_task(task_id: str, challenges: dict, solutions: dict | None = None, figsize=(12, 6)):
    """ç‰¹å®šã�® task_id ã‚’å�¯è¦–åŒ–ã€‚solutions ã‚’æ¸¡ã�›ã�° test ã�®æ­£è§£ã‚‚å‡ºã�™ã€‚"""
    task = challenges[task_id]
    trains = task["train"]
    tests  = task["test"]

    # è¡Œï¼š2ï¼ˆä¸Šæ®µ: input / ä¸‹æ®µ: outputï¼‰ã€�åˆ—ï¼štrainæ•° + testæ•°
    n_cols = len(trains) + len(tests)
    fig, axes = plt.subplots(2, n_cols, figsize=(4*n_cols, 6), dpi=150)
    if n_cols == 1:
        axes = np.array([[axes[0]], [axes[1]]])  # 1åˆ—æ™‚ã�®å½¢ã�‚ã‚�ã�›

    col = 0
    # Train pairs
    for i, ex in enumerate(trains):
        show_grid(axes[0, col], ex["input"],  title=f"Train-{i} input")
        show_grid(axes[1, col], ex["output"], title=f"Train-{i} output")
        col += 1

    # Test
    # å…¬é–‹solutionsã�Œã�‚ã‚‹å ´å�ˆã�¯ test ã�®æ­£è§£ã‚‚è¡¨ç¤º
    has_gt = (solutions is not None) and (task_id in solutions)
    for i, ex in enumerate(tests):
        show_grid(axes[0, col], ex["input"],  title=f"Test-{i} input")
        if has_gt and i < len(solutions[task_id]):
            show_grid(axes[1, col], solutions[task_id][i], title=f"Test-{i} GT")
        else:
            axes[1, col].axis("off")
            axes[1, col].set_title("Test output (unknown)", fontsize=11, color="#999")
        col += 1

    fig.suptitle(f"Task {task_id}", fontsize=14)
    plt.tight_layout()
    plt.show()



# å­¦ç¿’ç”¨ãƒ‡ãƒ¼ã‚¿ã�¯1000
num_tasks = len(training_challenges)
print("Number of training challenges:", num_tasks)


# ã‚µãƒ³ãƒ—ãƒ«ç”Ÿãƒ‡ãƒ¼ã‚¿
# ========= æœ€åˆ�ã�®ã‚¿ã‚¹ã‚¯ã�®ã‚¿ãƒ—ãƒ«ã‚’å�–ã‚Šå‡ºã�™ =========
task_id, task_data = next(iter(training_challenges.items()))

print("Task ID:", task_id)


task_data


example_id = next(iter(training_challenges.keys()))  # å…ˆé ­ã�®task_id
plot_task(example_id, training_challenges, training_solutions)


def grid_stats(arr):
    a = np.array(arr)
    return pd.Series({
        "h": a.shape[0],
        "w": a.shape[1],
        "area": a.size,
        "n_colors": len(np.unique(a))
    })

def summarize_tasks(challenges: dict) -> pd.DataFrame:
    rows = []
    for tid, task in challenges.items():
        # train å�´
        for i, ex in enumerate(task["train"]):
            s_in  = grid_stats(ex["input"])
            s_out = grid_stats(ex["output"])
            rows.append({
                "task_id": tid,
                "split": "train",
                "index": i,
                "in_h": s_in["h"], "in_w": s_in["w"], "in_area": s_in["area"], "in_colors": s_in["n_colors"],
                "out_h": s_out["h"], "out_w": s_out["w"], "out_area": s_out["area"], "out_colors": s_out["n_colors"],
            })
        # test å�´ï¼ˆå‡ºåŠ›ã�Œç„¡ã�„ã‚‚ã�®ã‚‚ã�‚ã‚‹ã�®ã�§æ³¨æ„�ï¼‰
        for i, ex in enumerate(task["test"]):
            s_in  = grid_stats(ex["input"])
            row = {
                "task_id": tid,
                "split": "test",
                "index": i,
                "in_h": s_in["h"], "in_w": s_in["w"], "in_area": s_in["area"], "in_colors": s_in["n_colors"],
                "out_h": np.nan, "out_w": np.nan, "out_area": np.nan, "out_colors": np.nan,
            }
            rows.append(row)
    return pd.DataFrame(rows)

df_summary = summarize_tasks(training_challenges)
df_summary.head()


# ã‚¿ã‚¹ã‚¯ã�”ã�¨ã�®å¹³å�‡å…¥åŠ›ã‚µã‚¤ã‚º
by_task = (df_summary[df_summary["split"]=="train"]
           .groupby("task_id")[["in_h","in_w","in_area","in_colors","out_h","out_w","out_area","out_colors"]]
           .mean()
           .sort_values("in_area", ascending=False))
by_task.head(10)



# è¤‡é›‘ã�ªå¥´ã�®ä¾‹
plot_task("de493100", training_challenges, training_solutions)


def plot_many_tasks(challenges: dict, solutions: dict | None = None, n=6):
    ids = list(challenges.keys())[:n]
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n), dpi=150)
    if n == 1:
        axes = axes.reshape(1, 3)

    for r, tid in enumerate(ids):
        task = challenges[tid]
        # ä»£è¡¨ã�¨ã�—ã�¦ train[0] ã�¨ test[0] ã‚’æ��ç”»ï¼ˆã�‚ã‚Œã�°ï¼‰
        show_grid(axes[r, 0], task["train"][0]["input"],  title=f"{tid}\nTrain-0 input")
        show_grid(axes[r, 1], task["train"][0]["output"], title="Train-0 output")
        if len(task["test"]) > 0:
            show_grid(axes[r, 2], task["test"][0]["input"], title="Test-0 input")
        else:
            axes[r, 2].axis("off")
    plt.tight_layout()
    plt.show()

plot_many_tasks(training_challenges, training_solutions, n=6)



from collections.abc import Mapping, Sequence

def show_grid(matrix, title=""):
    """1ã�¤ã�®ã‚°ãƒªãƒƒãƒ‰ã‚’æ��ç”»"""
    plt.imshow(matrix, cmap=cmap, norm=norm)
    plt.title(title)
    plt.xticks([]); plt.yticks([])

def show_pair(inp, out=None, title_in="Input", title_out="Output"):
    """1ã�¤ã�®å…¥å‡ºåŠ›ãƒšã‚¢ã‚’æ¨ªä¸¦ã�³ã�§è¡¨ç¤ºã€‚outã�Œã�ªã�„å ´å�ˆã�¯inputã�®ã�¿ã€‚"""
    if out is None:
        plt.figure(figsize=(3,3), dpi=130)
        show_grid(inp, title_in)
        plt.show()
    else:
        fig, axes = plt.subplots(1, 2, figsize=(6,3), dpi=130)
        plt.suptitle("")  # ä¸Šã�®ä½™ç™½æŠ‘åˆ¶
        plt.sca(axes[0]); show_grid(inp, title_in)
        plt.sca(axes[1]); show_grid(out, title_out)
        plt.tight_layout()
        plt.show()

def load_first_item(path):
    """JSONãƒ­ãƒ¼ãƒ‰ã�—ã�¦æœ€åˆ�ã�®1ä»¶ï¼ˆid, valueï¼‰ã‚’è¿”ã�™ï¼ˆdictæƒ³å®šï¼‰ã€‚listã�ªã‚‰(0, list[0])ã€‚"""
    data = json.load(open(path))
    if isinstance(data, Mapping):
        return next(iter(data.items()))
    elif isinstance(data, Sequence):
        return 0, data[0]
    else:
        raise ValueError("Unsupported JSON structure")

def print_overview(name, key, value):
    """æœ€åˆ�ã�®1ä»¶ã�®æ§‹é€ ã‚’ã�–ã�£ã��ã‚Šè¡¨ç¤º"""
    print(f"\n=== {name} | first key: {key} ===")
    if isinstance(value, Mapping):
        print(f"type: dict, keys: {list(value.keys())[:6]}")
        # ã‚ˆã��ã�‚ã‚‹æ§‹é€ ã�«å�ˆã‚�ã�›ã�¦è¿½åŠ è¡¨ç¤º
        if "train" in value:
            print(f"  train count: {len(value['train'])}, test count: {len(value.get('test', []))}")
            t0 = value["train"][0]
            print(f"  train[0] keys: {list(t0.keys())}")
            if "input" in t0:
                ih, iw = len(t0["input"]), len(t0["input"][0])
                print(f"  train[0].input shape: {ih}x{iw}")
            if "output" in t0:
                oh, ow = len(t0["output"]), len(t0["output"][0])
                print(f"  train[0].output shape: {oh}x{ow}")
            if value.get("test"):
                i0 = value["test"][0]["input"]
                th, tw = len(i0), len(i0[0])
                print(f"  test[0].input shape: {th}x{tw}")
        elif "attempt_1" in value or "attempt_2" in value:
            print("  (sample_submission entry with attempt_1 / attempt_2)")
        else:
            # solutionsï¼ˆ= list of outputsï¼‰ã�ªã�©ã�¯valueè‡ªä½“ã�Œlistã�®ã�¯ã�šã�ªã�®ã�§ã�“ã�“ã�«ã�¯æ»…å¤šã�«æ�¥ã�ªã�„
            pass
    elif isinstance(value, list):
        print(f"type: list, length: {len(value)}")
        if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
            # gridã�¨ã�¿ã�ªã�—ã�¦å½¢çŠ¶è¡¨ç¤º
            h, w = len(value[0]), len(value[0][0])
            print(f"  first grid shape: {h}x{w}")
        elif value and isinstance(value[0], dict):
            print(f"  element keys: {list(value[0].keys())}")
    else:
        print(f"type: {type(value)}")

# ãƒ•ã‚¡ã‚¤ãƒ«ä¸€è¦§
files = {
    "training_challenges": "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json",
    "training_solutions": "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json",
    "evaluation_challenges": "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json",
    "evaluation_solutions": "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json",
    "test_challenges": "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json",
    "sample_submission": "/kaggle/input/arc-prize-2025/sample_submission.json",
}

# === å�„ãƒ•ã‚¡ã‚¤ãƒ«ã�®æœ€åˆ�ã�®1ä»¶ã�®å�¯è¦–åŒ– ===
for name, path in files.items():
    tid, item = load_first_item(path)
    print_overview(name, tid, item)

    # 1) challengeç³»ï¼ˆtrain/testã‚’æŒ�ã�¤dictï¼‰
    if isinstance(item, dict) and "train" in item:
        # ä»£è¡¨ã�¨ã�—ã�¦ train[0] ã�¨ test[0]ï¼ˆã�‚ã‚Œã�°ï¼‰ã‚’è¡¨ç¤º
        train0 = item["train"][0]
        show_pair(train0["input"], train0["output"], f"{name}: Train-0 input", "Train-0 output")

        if item.get("test"):
            test0 = item["test"][0]["input"]
            show_pair(test0, None, f"{name}: Test-0 input")

    # 2) solutionsç³»ï¼ˆvalueã�¯ã€Œãƒ†ã‚¹ãƒˆå‡ºåŠ›ã�®é…�åˆ—ã€�ï¼‰
    elif name.endswith("solutions"):
        # æœ€åˆ�ã�®è§£ï¼ˆã‚°ãƒªãƒƒãƒ‰ï¼‰ã� ã�‘ã‚’è¡¨ç¤º
        first_outputs = item  # list[grid] ã�®æƒ³å®š
        if first_outputs and isinstance(first_outputs[0], list):
            show_pair(first_outputs[0], None, f"{name}: first solution grid")

    # 3) sample_submissionï¼ˆdict: task_id -> list of {attempt_1, attempt_2}ï¼‰
    elif name == "sample_submission" and isinstance(item, list) and item:
        entry0 = item[0]
        if "attempt_1" in entry0 and "attempt_2" in entry0:
            fig, axes = plt.subplots(1, 2, figsize=(6,3), dpi=130)
            plt.sca(axes[0]); show_grid(entry0["attempt_1"], f"{name}: attempt_1")
            plt.sca(axes[1]); show_grid(entry0["attempt_2"], f"{name}: attempt_2")
            plt.tight_layout(); plt.show()
        else:
            print("Unexpected sample_submission structure for first entry; printing it:")
            print(entry0)
    else:
        # ã��ã�®ä»–ã�¯printã�®ã�¿
        print(f"(no visualization rule for {name})")


# ç”Ÿé…�åˆ—
for name, path in files.items():
    tid, item = load_first_item(path)
    print(f"\n=== {name} | first key: {tid} ===")

    # ã��ã�®ã�¾ã�¾é…�åˆ—ã‚’ print
    if isinstance(item, dict) and "train" in item:
        print("Train-0 input:")
        print(item["train"][0]["input"])
        print("Train-0 output:")
        print(item["train"][0]["output"])

        if item.get("test"):
            print("Test-0 input:")
            print(item["test"][0]["input"])

    elif name.endswith("solutions"):
        print("First solution grid:")
        print(item[0])  # solutionsã�¯ list of grid

    elif name == "sample_submission" and isinstance(item, list):
        entry0 = item[0]
        print("Attempt_1:")
        print(entry0["attempt_1"])
        print("Attempt_2:")
        print(entry0["attempt_2"])

    else:
        print(item)





with open(files["evaluation_challenges"]) as f:
    data = json.load(f)
first_key, first_value = list(data.items())[2]
print(first_key)
print(first_value)


# ãƒ•ã‚¡ã‚¤ãƒ«ãƒ­ãƒ¼ãƒ‰
with open(files["evaluation_challenges"]) as f:
    eval_challenges = json.load(f)
with open(files["evaluation_solutions"]) as f:
    eval_solutions = json.load(f)

results = []

# å�„ã‚¿ã‚¹ã‚¯ã�® train[0].input ã�®ã‚µã‚¤ã‚ºã‚’è¨ˆç®—
for task_id, task in eval_challenges.items():
    train = task.get("train", [])
    if not train:
        continue
    grid = train[0]["input"]
    rows = len(grid)
    cols = len(grid[0]) if rows > 0 else 0
    size = rows * cols
    results.append((size, (rows, cols), task_id, task))

# å°�ã�•ã�„é †ã�«ã‚½ãƒ¼ãƒˆ
results.sort(key=lambda x: x[0])


# ä¸Šä½�3ä»¶ã‚’è¡¨ç¤º
top_n = 10
for i, (size, shape, task_id, task) in enumerate(results[:top_n], 1):
    print(f"Top {i}: ã‚¿ã‚¹ã‚¯ID={task_id}, ã‚µã‚¤ã‚º={size}, shape={shape}")

    # --- Train-0 å…¥å‡ºåŠ› ---
    train0 = task["train"][0]
    show_pair(
        train0["input"],
        train0.get("output"),
        f"Task {task_id}: Train-0 input",
        "Train-0 output"
    )

    # --- Test-0 å…¥åŠ› ---
    if task.get("test"):
        test0 = task["test"][0]["input"]
        show_pair(
            test0,
            None,
            f"Task {task_id}: Test-0 input"
        )

        # --- å¯¾å¿œã�™ã‚‹ã‚½ãƒªãƒ¥ãƒ¼ã‚·ãƒ§ãƒ³ (evaluation_solutions) ---
        if task_id in eval_solutions:
            sol0 = eval_solutions[task_id][0]  # test[0] ã�®è§£
            show_pair(
                sol0,
                None,
                f"Task {task_id}: Solution-0 output"
            )


with open('/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json') as f:
    data = json.load(f)
first_key, first_value = list(data.items())[2]
print(first_key)
print(first_value)


# 135a2760ã‚„





import json, numpy as np

# Kaggle æ—¢å®šãƒ‘ã‚¹ï¼ˆå¿…è¦�ã�«å¿œã�˜ã�¦ä¸Šæ›¸ã��ï¼‰
EVAL_CH_PATH = globals().get("EVAL_CH_PATH", "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json")
SOL_PATH     = globals().get("SOL_PATH",     "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json")



# -*- coding: utf-8 -*-
"""
ARC submission evaluation & visualization â€” cleaned single-file module
(Shape-mismatch friendly version)

- shape_mismatch ã�§ã‚‚ä»£è¡¨ GT/Pred ã‚’è¿”ã�—ã�¦å�¯è¦–åŒ–
- attempts ã�Œç©ºã�§ã‚‚ candidates ã�‹ã‚‰æ‹¾ã�†ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯
- solutions: ground_truth/gt ã‚­ãƒ¼ã‚‚æ�¢ç´¢
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import deque
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =========================
# Config (override in caller if needed)
# =========================
EVAL_CH_PATH = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_challenges.json"
SOL_PATH     = "/kaggle/input/arc-prize-2025/arc-agi_evaluation_solutions.json"

HEAD_TASKS       = 10
MAX_TRAIN_SHOW   = 3
RANDOM_SEED      = 4372


# =========================
# Utils
# =========================
def load_json(path: str) -> Any:
    with open(path, "r") as f:
        return json.load(f)

def is_grid_like(x: Any) -> bool:
    try:
        a = np.array(x)
        return a.ndim == 2 and a.size > 0
    except Exception:
        return False

def to_arr(grid: Any) -> np.ndarray:
    a = np.array(grid)
    if a.ndim != 2 or a.size == 0:
        raise ValueError(f"Not 2D grid: shape={a.shape}")
    return a.astype(int)

def _dedup_grids(grids: List[np.ndarray]) -> List[np.ndarray]:
    uniq: List[np.ndarray] = []
    seen: set = set()
    for g in grids:
        key = (g.shape, g.tobytes())
        if key not in seen:
            uniq.append(g)
            seen.add(key)
    return uniq


# =========================
# Parsers (robust against schema variations)
# =========================
def parse_eval_challenges(raw: Dict[str, Any]) -> Dict[str, Dict[str, List]]:
    out: Dict[str, Dict[str, List]] = {}
    for tid, task in raw.items():
        train_pairs, test_inputs = [], []
        for ex in task.get("train", []) or []:
            if "input" in ex and "output" in ex and is_grid_like(ex["input"]) and is_grid_like(ex["output"]):
                train_pairs.append((to_arr(ex["input"]), to_arr(ex["output"])))
        for ex in task.get("test", []) or []:
            if "input" in ex and is_grid_like(ex["input"]):
                test_inputs.append(to_arr(ex["input"]))
        out[tid] = {"train": train_pairs, "test": test_inputs}
    return out

def _extract_all_grids_any(obj: Any) -> List[np.ndarray]:
    """
    BFS over arbitrary nested dict/list; collect all 2D grids.
    Explore common keys + ground_truth/gt (CHG).
    """
    out: List[np.ndarray] = []
    dq = deque([obj])
    while dq:
        cur = dq.popleft()
        if is_grid_like(cur):
            out.append(to_arr(cur))
            continue
        if isinstance(cur, dict):
            for k in (
                "solutions", "outputs", "output", "solution",
                "answers", "answer", "ground_truth", "gt"  # CHG
            ):
                if k in cur:
                    dq.append(cur[k])
            dq.extend(cur.values())
        elif isinstance(cur, list):
            dq.extend(cur)
    return _dedup_grids(out)

def parse_eval_solutions(raw: Dict[str, Any]) -> Dict[str, List[List[np.ndarray]]]:
    out: Dict[str, List[List[np.ndarray]]] = {}
    for tid, val in raw.items():
        tests: List[List[np.ndarray]] = []
        if isinstance(val, list):
            for e in val:
                tests.append(_extract_all_grids_any(e))
        elif isinstance(val, dict):
            if "test" in val and isinstance(val["test"], list):
                for e in val["test"]:
                    tests.append(_extract_all_grids_any(e))
            else:
                tests.append(_extract_all_grids_any(val))
        else:
            tests.append(_extract_all_grids_any(val))
        out[tid] = tests
    return out

# â€” submission: candidates (many) â€”
_KEY_PRIORITY = ["output","prediction","pred","attempt_1","attempt1","attempt",
                 "attempt_2","attempt2","out","grid","y_pred","answer","result"]
_KEY_IGNORE   = {"input","train","test","task_id","id","index","case","meta","title","name"}

def _extract_grids_from_dict(d: Dict[str, Any]) -> List[np.ndarray]:
    cands: List[np.ndarray] = []
    for k in _KEY_PRIORITY:
        if k in d and is_grid_like(d[k]):
            cands.append(to_arr(d[k]))
    for k, v in d.items():
        if k in _KEY_IGNORE or k in _KEY_PRIORITY:
            continue
        if is_grid_like(v):
            cands.append(to_arr(v))
        elif isinstance(v, dict):
            cands.extend(_extract_grids_from_dict(v))
        elif isinstance(v, list):
            for e in v:
                if is_grid_like(e):
                    cands.append(to_arr(e))
                elif isinstance(e, dict):
                    cands.extend(_extract_grids_from_dict(e))
    return _dedup_grids(cands)

def _as_candidate_list(e: Any) -> List[np.ndarray]:
    if is_grid_like(e):
        return [to_arr(e)]
    if isinstance(e, dict):
        return _extract_grids_from_dict(e)
    if isinstance(e, list) and e and is_grid_like(e[0]):
        return [to_arr(g) for g in e]
    return []

def parse_submission_candidates(raw: Dict[str, Any]) -> Dict[str, List[List[np.ndarray]]]:
    sub: Dict[str, List[List[np.ndarray]]] = {}
    for tid, val in raw.items():
        preds: List[List[np.ndarray]] = []
        if isinstance(val, list):
            for e in val:
                preds.append(_as_candidate_list(e))
        else:
            preds.append(_as_candidate_list(val))
        sub[tid] = preds
    return sub

# â€” submission: attempt_1/attempt_2 (first-found grids) â€”
_ATTEMPT1_KEYS = {"attempt_1","attempt1","pred1","prediction1","out1","output1"}
_ATTEMPT2_KEYS = {"attempt_2","attempt2","pred2","prediction2","out2","output2"}

def _first_grid(obj: Any) -> Optional[np.ndarray]:
    dq = deque([obj])
    while dq:
        cur = dq.popleft()
        if is_grid_like(cur):
            return to_arr(cur)
        if isinstance(cur, dict):
            dq.extend(cur.values())
        elif isinstance(cur, list):
            dq.extend(cur)
    return None

def _find_attempt(elem: Any, names_lower: set) -> Optional[np.ndarray]:
    dq = deque([elem])
    while dq:
        cur = dq.popleft()
        if isinstance(cur, dict):
            for k, v in cur.items():
                if k.lower() in names_lower:
                    g = _first_grid(v)
                    if g is not None:
                        return g
                if isinstance(v, (dict, list)):
                    dq.append(v)
        elif isinstance(cur, list):
            dq.extend(cur)
    return None

@dataclass
class Attempts:
    a1: Optional[np.ndarray]
    a2: Optional[np.ndarray]

def parse_submission_attempts(raw: Dict[str, Any]) -> Dict[str, List[Attempts]]:
    """
    attempt_1/2 ã‚’å„ªå…ˆã�—ã�¤ã�¤ã€�ç„¡ã�‘ã‚Œã�° candidates ç”±æ�¥ã�®ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ã�§ a1/a2 ã‚’åŸ‹ã‚�ã‚‹ï¼ˆCHGï¼‰
    """
    out: Dict[str, List[Attempts]] = {}

    def _fallback_from_candidates(node: Any) -> Attempts:
        if isinstance(node, dict):
            cands = _extract_grids_from_dict(node)
            if cands:
                return Attempts(cands[0], cands[1] if len(cands) > 1 else None)
        if isinstance(node, list):
            grids = [to_arr(x) for x in node if is_grid_like(x)]
            if grids:
                return Attempts(grids[0], grids[1] if len(grids) > 1 else None)
        if is_grid_like(node):
            return Attempts(to_arr(node), None)
        return Attempts(None, None)

    for tid, preds in raw.items():
        arr: List[Attempts] = []
        if isinstance(preds, list):
            for e in preds:
                a1 = _find_attempt(e, {k.lower() for k in _ATTEMPT1_KEYS})
                a2 = _find_attempt(e, {k.lower() for k in _ATTEMPT2_KEYS})

                if a1 is None and a2 is None:
                    fb = _fallback_from_candidates(e)  # CHG
                    a1, a2 = fb.a1, fb.a2
                else:
                    if (a1 is None or a2 is None) and isinstance(e, list):
                        grids = [to_arr(x) for x in e if is_grid_like(x)]
                        if a1 is None and len(grids) >= 1: a1 = grids[0]
                        if a2 is None and len(grids) >= 2: a2 = grids[1]
                    if a1 is None and a2 is None and is_grid_like(e):
                        a1 = to_arr(e)

                arr.append(Attempts(a1, a2))
        else:
            a1 = _find_attempt(preds, {k.lower() for k in _ATTEMPT1_KEYS}) or _first_grid(preds)
            a2 = _find_attempt(preds, {k.lower() for k in _ATTEMPT2_KEYS})
            if a1 is None and a2 is None:
                fb = _fallback_from_candidates(preds)  # CHG
                a1, a2 = fb.a1, fb.a2
            arr.append(Attempts(a1, a2))
        out[tid] = arr
    return out


# =========================
# Evaluator
# =========================
def best_of_candidates_vs_allowed(
    pred_cands: List[np.ndarray],
    allowed: List[np.ndarray]
) -> Tuple[bool, str, Dict[str, Any]]:
    """
    Pick best candidate vs allowed solutions.
    Return (matched, reason, extra)

    CHG: shape_mismatch ã�§ã‚‚ä»£è¡¨ã‚’è¿”ã�™ï¼ˆbest_pred / best_gt ã‚’å�¯è¦–åŒ–å�¯èƒ½ã�«ï¼‰
    """
    if not allowed:
        # ä»£è¡¨ pred ã�¯è¿”ã�™ï¼ˆå�¯è¦–åŒ–ã�®ã�Ÿã‚�ï¼‰
        rep_pred = pred_cands[0] if pred_cands else None
        return False, "no_allowed", {"best_pred": rep_pred, "best_gt": None}

    if not pred_cands:
        # ä»£è¡¨ gt ã�¯è¿”ã�™ï¼ˆå�¯è¦–åŒ–ã�®ã�Ÿã‚�ï¼‰
        rep_gt = allowed[0] if allowed else None
        return False, "missing_prediction", {"best_pred": None, "best_gt": rep_gt}

    # Exact match if same shape and all equal
    for i, p in enumerate(pred_cands):
        for gt in allowed:
            if p.shape == gt.shape and np.array_equal(p, gt):
                return True, "ok", {
                    "best_pred": p, "best_gt": gt, "best_cand_idx": i,
                    "diff_pixels": 0, "total_pixels": int(gt.size)
                }

    # Min diff among same-shaped pairs
    best: Optional[Tuple[int, int, np.ndarray, np.ndarray]] = None
    for i, p in enumerate(pred_cands):
        for gt in allowed:
            if p.shape == gt.shape:
                diff = int(np.sum(p != gt))
                if (best is None) or (diff < best[0]):
                    best = (diff, i, p, gt)
    if best is not None:
        diff, i, p, gt = best
        return False, "pixel_mismatch", {
            "best_pred": p, "best_gt": gt, "best_cand_idx": i,
            "diff_pixels": diff, "total_pixels": int(gt.size)
        }

    # No same-shaped pairs â†’ ä»£è¡¨ã‚’è¿”ã�™ï¼ˆCHGï¼‰
    rep_pred = pred_cands[0] if pred_cands else None
    rep_gt   = allowed[0]    if allowed    else None
    return False, "shape_mismatch", {
        "best_pred": rep_pred,
        "best_gt":   rep_gt,
        "best_cand_idx": 0 if rep_pred is not None else None,
        "pred_shapes": [tuple(p.shape) for p in pred_cands],
        "allowed_shapes": [tuple(gt.shape) for gt in allowed]
    }

def evaluate_submission(sub_path: str, sol_path: str) -> pd.DataFrame:
    raw_sub = load_json(sub_path)
    raw_sol = load_json(sol_path)

    sub = parse_submission_candidates(raw_sub)
    sol = parse_eval_solutions(raw_sol)

    rows: List[Dict[str, Any]] = []
    for tid, allowed_tests in sol.items():
        pred_tests = sub.get(tid, [])
        n_allowed, n_pred = len(allowed_tests), len(pred_tests)

        for i in range(max(n_allowed, n_pred)):
            info: Dict[str, Any] = {
                "task_id": tid, "index": i,
                "has_pred": i < n_pred, "has_gt": i < n_allowed,
                "matched": False, "reason": None,
                "pred_shape": None, "best_gt_shape": None,
                "allowed_shapes": None, "pred_shapes": None,
                "diff_pixels": None, "total_pixels": None, "acc_pixels": None,
                "n_pred_candidates": None, "chosen_cand_idx": None,
                "n_allowed_solutions": None
            }

            if not info["has_pred"] and not info["has_gt"]:
                continue
            if not info["has_pred"]:
                info.update({"matched": False, "reason": "missing_prediction"})
                rows.append(info); continue

            pred_cands = pred_tests[i] if i < n_pred else []
            info["n_pred_candidates"] = len(pred_cands)

            if not info["has_gt"]:
                info.update({"matched": False, "reason": "missing_ground_truth"})
                if pred_cands:
                    info["pred_shapes"] = [tuple(p.shape) for p in pred_cands]
                rows.append(info); continue

            allowed = allowed_tests[i]
            info["n_allowed_solutions"] = len(allowed)

            matched, reason, extra = best_of_candidates_vs_allowed(pred_cands, allowed)
            info["matched"] = matched; info["reason"] = reason

            bp, bg = extra.get("best_pred"), extra.get("best_gt")
            if bp is not None: info["pred_shape"] = tuple(bp.shape)
            if bg is not None: info["best_gt_shape"] = tuple(bg.shape)

            if "pred_shapes" in extra and extra["pred_shapes"] is not None:
                info["pred_shapes"] = extra["pred_shapes"]
            if "allowed_shapes" in extra and "allowed_shapes" in extra and extra["allowed_shapes"] is not None:
                info["allowed_shapes"] = extra["allowed_shapes"]

            if extra.get("best_cand_idx") is not None:
                info["chosen_cand_idx"] = int(extra["best_cand_idx"])

            if extra.get("diff_pixels") is not None:
                info["diff_pixels"]  = int(extra["diff_pixels"])
                info["total_pixels"] = int(extra["total_pixels"])
                info["acc_pixels"]   = 1.0 - (info["diff_pixels"] / max(1, info["total_pixels"]))

            rows.append(info)

    return pd.DataFrame(rows)


# =========================
# Visualization
# =========================
def load_eval_inputs(eval_ch_path: str = EVAL_CH_PATH) -> Dict[str, List[np.ndarray]]:
    raw = load_json(eval_ch_path)
    inputs: Dict[str, List[np.ndarray]] = {}
    for tid, task in raw.items():
        tests = task.get("test", []) or []
        inputs[tid] = [to_arr(ex["input"]) for ex in tests if "input" in ex and is_grid_like(ex["input"])]
    return inputs

def show_triplet(tid: str, idx: int, df_eval: pd.DataFrame,
                 sub_path: str = SUB_PATH, sol_path: str = SOL_PATH, eval_ch_path: str = EVAL_CH_PATH) -> None:
    raw_sub = load_json(sub_path)
    raw_sol = load_json(sol_path)
    eval_inputs = load_eval_inputs(eval_ch_path)

    sub = parse_submission_candidates(raw_sub)
    sol = parse_eval_solutions(raw_sol)

    pred_cands = sub.get(tid, [[]])[idx]
    allowed    = sol.get(tid, [[]])[idx]

    matched, reason, extra = best_of_candidates_vs_allowed(pred_cands, allowed)
    best_pred, best_gt = extra.get("best_pred"), extra.get("best_gt")

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.ravel()

    x_in = eval_inputs.get(tid, [None])[idx] if tid in eval_inputs and idx < len(eval_inputs[tid]) else None
    if x_in is not None:
        axes[0].imshow(x_in); axes[0].set_title("Input")
    else:
        axes[0].text(0.5,0.5,"No Input",ha="center"); axes[0].set_title("Input")
    axes[0].set_xticks([]); axes[0].set_yticks([])

    if best_pred is not None:
        axes[1].imshow(best_pred); axes[1].set_title(f"Best Pred {best_pred.shape}")
    else:
        axes[1].text(0.5,0.5,"No Pred",ha="center"); axes[1].set_title("Best Pred")
    axes[1].set_xticks([]); axes[1].set_yticks([])

    if best_gt is not None:
        axes[2].imshow(best_gt); axes[2].set_title(f"GT (best) {best_gt.shape}")
    else:
        axes[2].text(0.5,0.5,"No GT",ha="center"); axes[2].set_title("GT")
    axes[2].set_xticks([]); axes[2].set_yticks([])

    if best_pred is not None and best_gt is not None and best_pred.shape == best_gt.shape:
        diff = (best_pred != best_gt).astype(int)
        axes[3].imshow(diff); axes[3].set_title(f"Diff sum={int(diff.sum())}")
    else:
        axes[3].text(0.5,0.5,"Shape mismatch",ha="center"); axes[3].set_title("Diff")
    axes[3].set_xticks([]); axes[3].set_yticks([])

    plt.tight_layout()
    plt.show()

def _imshow(ax, arr: Optional[np.ndarray], title: str) -> None:
    if arr is None:
        ax.text(0.5, 0.5, "None", ha="center", va="center")
        ax.set_title(title)
    else:
        ax.imshow(arr); ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])

def show_task(tid: str,
              one_task: Dict[str, List],
              sol_tests: List[List[np.ndarray]],
              sub_attempts: List[Attempts],
              sub_candidates: Optional[List[List[np.ndarray]]] = None  # CHG: candidates ã‚‚å�—ã�‘å�–ã‚‹
              ) -> None:
    """
    2x3 panel per test:
      [Input] [GT(best)] [Pred(best)]
      [Diff ] [attempt_1] [attempt_2]
    """
    print(f"\n=== Task {tid} ===")

    train_pairs = one_task["train"][:MAX_TRAIN_SHOW]
    if train_pairs:
        cols = 2 * len(train_pairs)
        fig, axes = plt.subplots(1, cols, figsize=(3*cols, 3))
        axes = np.array(axes).ravel() if cols > 1 else np.array([axes])
        for idx, (inp, out) in enumerate(train_pairs):
            _imshow(axes[2*idx],   inp, f"Train In {idx}")
            _imshow(axes[2*idx+1], out, f"Train Out {idx}")
        fig.suptitle(f"Task {tid} - TRAIN examples (up to {MAX_TRAIN_SHOW})")
        plt.tight_layout(); plt.show()

    n_tests = len(one_task["test"])
    for i in range(n_tests):
        x_in = one_task["test"][i]
        allowed = sol_tests[i] if i < len(sol_tests) else []

        attempts = sub_attempts[i] if i < len(sub_attempts) else Attempts(None, None)
        a1, a2 = attempts.a1, attempts.a2

        # attempts â†’ candidatesï¼ˆãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ï¼‰é †ã�§å€™è£œä½œæˆ�ï¼ˆCHGï¼‰
        cands = [p for p in (a1, a2) if p is not None]
        if (not cands) and sub_candidates is not None and i < len(sub_candidates):
            cands = sub_candidates[i]

        if allowed:
            matched, reason, extra = best_of_candidates_vs_allowed(cands, allowed)
            best_pred, best_gt = extra.get("best_pred"), extra.get("best_gt")
        else:
            best_pred, best_gt, matched, reason = (cands[0] if cands else None), None, False, "no_allowed"

        fig, axes = plt.subplots(2, 3, figsize=(11, 7))
        axes = axes.ravel()

        _imshow(axes[0], x_in,           "Input")
        _imshow(axes[1], best_gt,        "GT (best)")
        _imshow(axes[2], best_pred,      "Pred (best)")

        if best_pred is not None and best_gt is not None and best_pred.shape == best_gt.shape:
            diff = (best_pred != best_gt).astype(int)
            _imshow(axes[3], diff, f"Diff sum={int(diff.sum())}")
        else:
            _imshow(axes[3], None, "Diff (shape mismatch)")

        _imshow(axes[4], a1, "attempt_1")
        _imshow(axes[5], a2, "attempt_2")

        fig.suptitle(f"Task {tid} - TEST #{i} | reason={reason} matched={matched}")
        plt.tight_layout(); plt.show()


# =========================
# Audits & consistency checks
# =========================
def audit_solutions(sol_dict: Dict[str, List[List[np.ndarray]]],
                    ch_dict: Dict[str, Dict[str, List]],
                    max_show: int = 10) -> None:
    print("\n[Audit] solutions æŠ½å‡ºçŠ¶æ³�ï¼ˆå…ˆé ­ã‚¿ã‚¹ã‚¯ï¼‰")
    cids = sorted(ch_dict.keys())
    for tid in cids[:max_show]:
        n_tests = len(ch_dict[tid]["test"])
        n_sol   = len(sol_dict.get(tid, []))
        non_empty = sum(1 for li in sol_dict.get(tid, []) if isinstance(li, list) and len(li) > 0)
        print(f"- {tid}: tests={n_tests}, solutions_entries={n_sol}, non_empty_GT={non_empty}")
        if n_sol != n_tests:
            print("    âš  solutions å�´ã�®ãƒ†ã‚¹ãƒˆä»¶æ•°ã�Œ challenges ã�¨ã‚ºãƒ¬ã�¦ã�„ã�¾ã�™ã€‚")

    total_tests = sum(len(ch_dict[tid]["test"]) for tid in ch_dict)
    total_slots = sum(len(sol_dict.get(tid, [])) for tid in ch_dict)
    total_non_empty = sum(
        sum(1 for li in sol_dict.get(tid, []) if isinstance(li, list) and len(li) > 0)
        for tid in ch_dict
    )
    print(f"\n[Audit] å…¨ä½“: challenge_tests={total_tests}, solutions_slots={total_slots}, non_empty_GT={total_non_empty}")

def compare_task_id_sets_and_counts(eval_ch_path: str = EVAL_CH_PATH,
                                    sol_path: str = SOL_PATH,
                                    sub_path: str = SUB_PATH,
                                    head: int = 20) -> None:
    eval_ch = load_json(eval_ch_path)
    eval_sol = load_json(sol_path)
    submission = load_json(sub_path)

    ids_ch, ids_sol, ids_sub = set(eval_ch.keys()), set(eval_sol.keys()), set(submission.keys())

    print("=== task_id ã‚»ãƒƒãƒˆæ¯”è¼ƒ ===")
    print(f"challenges: {len(ids_ch)}  solutions: {len(ids_sol)}  submission: {len(ids_sub)}")

    def _head(ls, n=10):
        return ls[:n] + (["..."] if len(ls) > n else [])

    only_in_ch  = sorted(ids_ch  - ids_sol - ids_sub)
    only_in_sol = sorted(ids_sol - ids_ch  - ids_sub)
    only_in_sub = sorted(ids_sub - ids_ch  - ids_sol)

    in_ch_not_sol = sorted(ids_ch - ids_sol)
    in_sol_not_ch = sorted(ids_sol - ids_ch)
    in_ch_not_sub = sorted(ids_ch - ids_sub)
    in_sub_not_ch = sorted(ids_sub - ids_ch)

    print(f"- challenges ã�«ã�®ã�¿å­˜åœ¨: {len(only_in_ch)}  ä¾‹: {_head(only_in_ch)}")
    print(f"- solutions ã�«ã�®ã�¿å­˜åœ¨:  {len(only_in_sol)}  ä¾‹: {_head(only_in_sol)}")
    print(f"- submission ã�«ã�®ã�¿å­˜åœ¨: {len(only_in_sub)}  ä¾‹: {_head(only_in_sub)}")

    print(f"- challenges ã�«ã�‚ã�£ã�¦ solutions ã�«ç„¡ã�„: {len(in_ch_not_sol)}  ä¾‹: {_head(in_ch_not_sol)}")
    print(f"- solutions ã�«ã�‚ã�£ã�¦ challenges ã�«ç„¡ã�„: {len(in_sol_not_ch)}  ä¾‹: {_head(in_sol_not_ch)}")
    print(f"- challenges ã�«ã�‚ã�£ã�¦ submission ã�«ç„¡ã�„: {len(in_ch_not_sub)}  ä¾‹: {_head(in_ch_not_sub)}")
    print(f"- submission ã�«ã�‚ã�£ã�¦ challenges ã�«ç„¡ã�„: {len(in_sub_not_ch)}  ä¾‹: {_head(in_sub_not_ch)}")

    common_ids = sorted(ids_ch & ids_sol & ids_sub)
    print(f"\nå…±é€š task_idï¼ˆå…¨ã�¦ã�«å­˜åœ¨ï¼‰: {len(common_ids)}")

    def count_ch_tests(tid: str) -> int:
        task = eval_ch.get(tid, {})
        return len((task.get("test", []) or []))

    def count_sol_tests(tid: str) -> int:
        val = eval_sol.get(tid)
        if isinstance(val, list): return len(val)
        if isinstance(val, dict):
            if "test" in val and isinstance(val["test"], list): return len(val["test"])
            if "solutions" in val and isinstance(val["solutions"], list): return len(val["solutions"])
        return 0

    def count_sub_preds(tid: str) -> int:
        preds = submission.get(tid, [])
        return len(preds) if isinstance(preds, list) else 0

    mismatch = []
    for tid in common_ids:
        ct, st, pt = count_ch_tests(tid), count_sol_tests(tid), count_sub_preds(tid)
        if not (ct == st == pt):
            mismatch.append((tid, ct, st, pt))

    print("\n=== ä»¶æ•°ã‚ºãƒ¬ï¼ˆå…±é€š task_id ã�«é™�å®šï¼‰===")
    print(f"ã‚ºãƒ¬ä»¶æ•°: {len(mismatch)} / {len(common_ids)}")
    for tid, ct, st, pt in mismatch[:head]:
        print(f"- {tid}: challenges.test={ct}, solutions.tests={st}, submission.preds={pt}")
    if len(mismatch) > head:
        print("...")

    all_ids_equal = (ids_ch == ids_sol == ids_sub)
    all_counts_equal = (len(mismatch) == 0)
    print("\n=== ç·�å�ˆåˆ¤å®š ===")
    print(f"task_id ã‚»ãƒƒãƒˆä¸€è‡´: {all_ids_equal}")
    print(f"ãƒ†ã‚¹ãƒˆä»¶æ•°ä¸€è‡´:   {all_counts_equal}")


# =========================
# Quick run helper
# =========================
def run_evaluation_and_reports(sub_path: str = SUB_PATH, sol_path: str = SOL_PATH) -> pd.DataFrame:
    df_eval = evaluate_submission(sub_path, sol_path)
    total = len(df_eval)
    correct = int(df_eval["matched"].fillna(False).sum())
    acc = correct / total if total else 0.0
    print(f"Total cases: {total} | Correct: {correct} | Accuracy: {acc:.3%}")

    by_task = (df_eval.groupby("task_id")["matched"]
               .agg(["count","sum"]).rename(columns={"count":"n","sum":"correct"}))
    by_task["acc"] = by_task["correct"] / by_task["n"]
    display(by_task.sort_values("acc", ascending=True).head(10))
    display(by_task.sort_values("acc", ascending=False).head(10))

    print("\nError breakdown:")
    print(df_eval[~df_eval["matched"].fillna(False)]["reason"].value_counts(dropna=False))

    pix_bad = df_eval[(df_eval["reason"]=="pixel_mismatch") & df_eval["diff_pixels"].notna()]
    display(pix_bad.sort_values("diff_pixels", ascending=False).head(10))

    out_csv = "/kaggle/working/eval_report.csv"
    try:
        df_eval.to_csv(out_csv, index=False)
        print(f"Saved: {out_csv}")
    except Exception as e:
        print(f"(skip save) {e}")

    return df_eval

def run_head_visuals(head_tasks: int = HEAD_TASKS,
                     eval_ch_path: str = EVAL_CH_PATH,
                     sol_path: str = SOL_PATH,
                     sub_path: str = SUB_PATH) -> None:
    eval_challenges = load_json(eval_ch_path)
    eval_solutions  = load_json(sol_path)
    submission      = load_json(sub_path)

    ch  = parse_eval_challenges(eval_challenges)
    sol = parse_eval_solutions(eval_solutions)
    sub_attempts = parse_submission_attempts(submission)
    sub_cands    = parse_submission_candidates(submission)  # CHG: candidates ã‚‚ç”Ÿæˆ�

    task_ids = sorted(ch.keys())[:head_tasks]
    print("Target tasks:", task_ids)

    for tid in task_ids:
        show_task(
            tid,
            ch[tid],
            sol.get(tid, []),
            sub_attempts.get(tid, []),
            sub_candidates=sub_cands.get(tid, [])  # CHG: ãƒ•ã‚©ãƒ¼ãƒ«ãƒ�ãƒƒã‚¯ã�«æ¸¡ã�™
        )

# ä¾‹ã�®å®Ÿè¡Œï¼ˆä»»æ„�ï¼‰
# df_eval = run_evaluation_and_reports(SUB_PATH, SOL_PATH)
# run_head_visuals(HEAD_TASKS, EVAL_CH_PATH, SOL_PATH, SUB_PATH)
# eval_ch = parse_eval_challenges(load_json(EVAL_CH_PATH))
# eval_sol = parse_eval_solutions(load_json(SOL_PATH))
# audit_solutions(eval_sol, eval_ch, max_show=50)
# compare_task_id_sets_and_counts(EVAL_CH_PATH, SOL_PATH, SUB_PATH)



# 1) ç²¾åº¦è©•ä¾¡ã�¨ãƒ†ãƒ¼ãƒ–ãƒ«å‡ºåŠ›
df_eval = run_evaluation_and_reports(SUB_PATH, SOL_PATH)

# 3) å…ˆé ­10ã‚¿ã‚¹ã‚¯ã�®æ¯”è¼ƒå�¯è¦–åŒ–ï¼ˆTrain/Test/attempt_1/2ï¼‰
run_head_visuals(HEAD_TASKS, EVAL_CH_PATH, SOL_PATH, SUB_PATH)

# 4) solutionsæŠ½å‡ºã‚„task_idã‚ºãƒ¬ã�®ç›£æŸ»
eval_ch = parse_eval_challenges(load_json(EVAL_CH_PATH))
eval_sol = parse_eval_solutions(load_json(SOL_PATH))
audit_solutions(eval_sol, eval_ch, max_show=50)
compare_task_id_sets_and_counts(EVAL_CH_PATH, SOL_PATH, SUB_PATH)












# df_eval = evaluate_submission(SUB_PATH, SOL_PATH)

# total = len(df_eval)
# correct = int(df_eval["matched"].fillna(False).sum())
# acc = correct / total if total else 0.0
# print(f"Total cases: {total} | Correct: {correct} | Accuracy: {acc:.3%}")

# by_task = (df_eval.groupby("task_id")["matched"]
#            .agg(["count","sum"]).rename(columns={"count":"n","sum":"correct"}))
# by_task["acc"] = by_task["correct"] / by_task["n"]
# display(by_task.sort_values("acc", ascending=True).head(10))
# display(by_task.sort_values("acc", ascending=False).head(10))

# print("\nError breakdown:")
# print(df_eval[~df_eval["matched"].fillna(False)]["reason"].value_counts(dropna=False))

# # å½¢ä¸€è‡´ã�®ä¸­ã�§èª¤å·®ã�Œå¤§ã��ã�„é †
# pix_bad = df_eval[(df_eval["reason"]=="pixel_mismatch") & df_eval["diff_pixels"].notna()]
# display(pix_bad.sort_values("diff_pixels", ascending=False).head(10))

# # CSVä¿�å­˜
# out_csv = "/kaggle/working/eval_report.csv"
# df_eval.to_csv(out_csv, index=False)
# print(f"Saved: {out_csv}")



# =========================
# Extra paths (training / test)
# =========================
TRAIN_CH_PATH  = "/kaggle/input/arc-prize-2025/arc-agi_training_challenges.json"
TRAIN_SOL_PATH = "/kaggle/input/arc-prize-2025/arc-agi_training_solutions.json"
TEST_CH_PATH   = "/kaggle/input/arc-prize-2025/arc-agi_test_challenges.json"


# =========================
# Helpers for new galleries
# =========================
def _grid_rc(n: int, max_cols: int = 4) -> Tuple[int, int]:
    """ãƒ¬ã‚¤ã‚¢ã‚¦ãƒˆè¨ˆç®—ï¼ˆnæ�šã‚’æœ€å¤§max_colsåˆ—ã�§æ•·ã��è©°ã‚�ã‚‹ï¼‰"""
    cols = min(max_cols, max(1, n))
    rows = int(np.ceil(n / cols))
    return rows, cols

def _imshow_clean(ax, arr: Optional[np.ndarray], title: str):
    if arr is None:
        ax.text(0.5, 0.5, "None", ha="center", va="center")
        ax.set_title(title)
    else:
        ax.imshow(arr)
        ax.set_title(title)
    ax.set_xticks([]); ax.set_yticks([])


# =========================
# 1) training_solutions ã�®å�¯è¦–åŒ–
#    - å�„ã‚¿ã‚¹ã‚¯ã�® Train å…¥å‡ºåŠ›ã‚’å°‘æ•°è¡¨ç¤º
#    - Test å…¥åŠ›ã�¨ã��ã�® GTï¼ˆsolutionsï¼‰ã‚’ä¸¦ã�¹ã�¦è¡¨ç¤º
# =========================
def show_training_task(tid: str,
                       one_task: Dict[str, List],
                       sol_tests: List[List[np.ndarray]],
                       max_train_show: int = MAX_TRAIN_SHOW) -> None:
    """
    ã‚¿ã‚¹ã‚¯ã�”ã�¨ã�«:
      - Train å…¥å‡ºåŠ›ï¼ˆæœ€å¤§ max_train_showï¼‰
      - Test: [Input] / [GT(best)]
    """
    print(f"\n=== [TRAINING] Task {tid} ===")

    # ---- Train ---
    train_pairs = one_task["train"][:max_train_show]
    if train_pairs:
        cols = 2 * len(train_pairs)
        fig, axes = plt.subplots(1, cols, figsize=(3*cols, 3))
        axes = np.array(axes).ravel() if cols > 1 else np.array([axes])
        for idx, (inp, out) in enumerate(train_pairs):
            _imshow_clean(axes[2*idx],   inp, f"Train In {idx}")
            _imshow_clean(axes[2*idx+1], out, f"Train Out {idx}")
        fig.suptitle(f"Task {tid} - TRAIN examples (up to {max_train_show})")
        plt.tight_layout(); plt.show()

    # ---- Test (Input & GT) ---
    n_tests = len(one_task["test"])
    for i in range(n_tests):
        x_in = one_task["test"][i]
        allowed = sol_tests[i] if i < len(sol_tests) else []
        gt_best = allowed[0] if allowed else None  # è¤‡æ•°GTã�Œã�‚ã‚Œã�°å…ˆé ­ã‚’ä»£è¡¨ã�«

        fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
        _imshow_clean(axes[0], x_in,    f"Test In #{i}{'' if x_in is None else f' {x_in.shape}'}")
        _imshow_clean(axes[1], gt_best, f"GT (best){'' if gt_best is None else f' {gt_best.shape}'}")
        fig.suptitle(f"Task {tid} - TEST #{i} (training solutions)")
        plt.tight_layout(); plt.show()


def run_training_solutions_visuals(max_show: int = 10,
                                   train_ch_path: str = TRAIN_CH_PATH,
                                   train_sol_path: str = TRAIN_SOL_PATH) -> None:
    """
    arc-agi_training_solutions.json ã�®ã‚¿ã‚¹ã‚¯ã‚’ max_show åˆ†å›³ç¤ºã€‚
    ï¼ˆå…¥åŠ›ã�¯ arc-agi_training_challenges.json ã‚’ä½¿ç”¨ï¼‰
    """
    raw_ch  = load_json(train_ch_path)
    raw_sol = load_json(train_sol_path)

    ch  = parse_eval_challenges(raw_ch)
    sol = parse_eval_solutions(raw_sol)

    task_ids = sorted(ch.keys())[:max_show]
    print("Training target tasks:", task_ids)

    for tid in task_ids:
        show_training_task(tid, ch[tid], sol.get(tid, []), max_train_show=MAX_TRAIN_SHOW)


# =========================
# 2) test_challenges ã�®å�¯è¦–åŒ–
#    - Train å…¥å‡ºåŠ›ï¼ˆå°‘æ•°ï¼‰
#    - Test å…¥åŠ›ã�®ã�¿ã‚’ä¸€è¦§ï¼ˆGTã‚„Predã�¯ç„¡ã�—ï¼‰
# =========================
def show_test_challenge_task(tid: str,
                             one_task: Dict[str, List],
                             max_train_show: int = MAX_TRAIN_SHOW,
                             max_cols: int = 4) -> None:
    """
    ã‚¿ã‚¹ã‚¯ã�”ã�¨ã�«:
      - Train å…¥å‡ºåŠ›ï¼ˆæœ€å¤§ max_train_showï¼‰
      - Test å…¥åŠ›ã‚’ã‚°ãƒªãƒƒãƒ‰ã�§ä¸€è¦§è¡¨ç¤º
    """
    print(f"\n=== [TEST] Task {tid} ===")

    # ---- Train ---
    train_pairs = one_task["train"][:max_train_show]
    if train_pairs:
        cols = 2 * len(train_pairs)
        fig, axes = plt.subplots(1, cols, figsize=(3*cols, 3))
        axes = np.array(axes).ravel() if cols > 1 else np.array([axes])
        for idx, (inp, out) in enumerate(train_pairs):
            _imshow_clean(axes[2*idx],   inp, f"Train In {idx}")
            _imshow_clean(axes[2*idx+1], out, f"Train Out {idx}")
        fig.suptitle(f"Task {tid} - TRAIN examples (up to {max_train_show})")
        plt.tight_layout(); plt.show()

    # ---- Test inputs only ---
    tests = one_task["test"]
    if tests:
        n = len(tests)
        r, c = _grid_rc(n, max_cols=max_cols)
        fig, axes = plt.subplots(r, c, figsize=(3*c, 3*r))
        axes = np.atleast_1d(axes).reshape(r, c)
        k = 0
        for i in range(r):
            for j in range(c):
                ax = axes[i, j]
                if k < n:
                    _imshow_clean(ax, tests[k], f"Test In #{k}{'' if tests[k] is None else f' {tests[k].shape}'}")
                    k += 1
                else:
                    ax.axis("off")
        fig.suptitle(f"Task {tid} - TEST inputs ({n})")
        plt.tight_layout(); plt.show()


def run_test_challenges_visuals(max_show: int = 10,
                                test_ch_path: str = TEST_CH_PATH,
                                max_cols: int = 4) -> None:
    """
    arc-agi_test_challenges.json ã�®ã‚¿ã‚¹ã‚¯ã‚’ max_show åˆ†å›³ç¤ºã€‚
    ï¼ˆGTã�¯ç„¡ã�„ã�®ã�§ Test å…¥åŠ›ã�®ã�¿ä¸€è¦§ï¼‰
    """
    raw_ch = load_json(test_ch_path)
    ch = parse_eval_challenges(raw_ch)

    task_ids = sorted(ch.keys())[:max_show]
    print("Test target tasks:", task_ids)

    for tid in task_ids:
        show_test_challenge_task(tid, ch[tid], max_train_show=MAX_TRAIN_SHOW, max_cols=max_cols)



# 1) ãƒˆãƒ¬ãƒ¼ãƒ‹ãƒ³ã‚°å�´ï¼ˆsolutionsã�‚ã‚Šï¼‰ã‚’å�¯è¦–åŒ–
run_training_solutions_visuals(max_show=12)


# 2) ãƒ†ã‚¹ãƒˆå�´ï¼ˆchallengesã�®ã�¿ï¼‰ã‚’å�¯è¦–åŒ–
run_test_challenges_visuals(max_show=12, max_cols=4)







