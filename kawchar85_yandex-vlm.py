!uv pip install --system --no-index --find-links='/kaggle/input/vllm-vlm/whls' 'vllm' 'torchvision' 'numpy<2' 'google_re2' 'qwen-vl-utils'



%%writefile utils.py
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

from qwen_vl_utils import process_vision_info

import io
import base64
from PIL import Image


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()
        base64_bytes = base64.b64encode(image_data)
        base64_string = base64_bytes.decode("utf-8")
        return base64_string

def construct_messages(image, text, role="user"):
    content = [
        {"type": "image", "image": image},
        {"type": "text", "text": text}
    ]

    return [
        {
            "role": role,
            "content": content,
        }
    ]

def prepare_inputs_for_vllm(messages, processor):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    # qwen_vl_utils 0.0.14+ reqired
    image_inputs, video_inputs, video_kwargs = process_vision_info(
        messages,
        image_patch_size=processor.image_processor.patch_size,
        return_video_kwargs=True,
        return_video_metadata=True
    )
    
    mm_data = {}
    if image_inputs is not None:
        mm_data['image'] = image_inputs
    if video_inputs is not None:
        mm_data['video'] = video_inputs

    return {
        'prompt': text,
        'multi_modal_data': mm_data,
        'mm_processor_kwargs': video_kwargs
    }

print("functions defined...")


!python utils.py


%%writefile solution.py

import os
import io
import re
import json
import time
import pickle
import argparse
from itertools import combinations
from typing import List, Tuple
from PIL import Image

import torch
from vllm import LLM, SamplingParams
from transformers import AutoProcessor

from utils import prepare_inputs_for_vllm

CHECKPOINT_PATH = "/kaggle/input/qwen-3-vl/transformers/bb-instruct-fp8/1"
GPU_UTIL = 0.80
MAX_MODEL_LEN = 10240
MAX_NEW_TOKENS = 1024 * 2
TEMPERATURE = 0.0
TOP_K = -1
BATCH_SIZE = 8

CUTOFF_SECONDS = 55 * 60
START_TIME = time.time()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

OPTION_RE = re.compile(r"\b([A-Z])\s*[\.\):\-]")
ANSWER_TAG_RE = re.compile(r"<\s*answer\s*>(.*?)<\s*/\s*answer\s*>", re.IGNORECASE | re.DOTALL)

def image_from_blob(blob) -> Image.Image:
    return Image.open(io.BytesIO(blob)).convert("RGB")

def extract_allowed_letters(question: str) -> List[str]:
    seen = set()
    letters = []
    for ch in OPTION_RE.findall(question or ""):
        ch = ch.upper()
        if ch not in seen:
            seen.add(ch)
            letters.append(ch)
    # conservative fallback set to reduce overprediction issues
    return letters if letters else list("ABCD")

def build_messages(image_pil: Image.Image, question: str):
    # System: strict rules; allow brief reasoning but require final tag
    system_text = (
        "You are a precise vision-language grader for STEM diagrams.\n"
        "\n"
        "Task:\n"
        "Determine which answer options are TRUE based only on what is visible in the diagram. "
        "If a statement cannot be visually verified, treat it as FALSE. "
        "Drawings are schematic and not necessarily to scale.\n"
        "\n"
        "Output rules (strict):\n"
        "1) First, think briefly step by step, in plain short lines.\n"
        "2) Then output a single final line wrapped in <answer>...</answer> containing only CAPITAL letters of the correct options in option order (A,B,C,...). "
        "If multiple are correct, concatenate the letters (for example, if A, B and E are correct, output <answer>ABE</answer>). "
        "If none are correct, output <answer></answer>.\n"
        "3) Do not include any other text after the </answer> tag."
    )

    # User: problem text + explicit "Answer:" cue
    user_text = f"{question.strip()}\n\nAnswer:"
    return [
        {"role": "system", "content": [{"type": "text", "text": system_text}]},
        {"role": "user",   "content": [
            {"type": "image", "image": image_pil},
            {"type": "text",  "text": user_text},
        ]}
    ]

def load_model_and_processor():
    processor = AutoProcessor.from_pretrained(CHECKPOINT_PATH, trust_remote_code=True)
    llm = LLM(
        model=CHECKPOINT_PATH,
        trust_remote_code=True,
        tensor_parallel_size=1,
        gpu_memory_utilization=GPU_UTIL,
        max_model_len=MAX_MODEL_LEN,
        enforce_eager=False,
        seed=0,
        max_num_seqs=BATCH_SIZE * 2,
    )
    sampling = SamplingParams(
        temperature=TEMPERATURE,
        top_k=TOP_K,
        max_tokens=MAX_NEW_TOKENS,
        stop=["</answer>"],                 # stop right after the closing tag
        include_stop_str_in_output=True,    # keep the tag so our regex sees it
    )
    return llm, processor, sampling

def build_candidates(letters: List[str]) -> List[str]:
    # Singles + all pairs; keep triples out for conservative fallback
    candidates = list(letters)
    for a, b in combinations(letters, 2):
        candidates.append(a + b)
    return candidates

def deterministic_fallback_prediction(img: Image.Image, question: str, letters: List[str], rid: int) -> str:
    # Conservative ABCD-based fallback to avoid frequent overprediction exclusions
    base_letters = letters if letters else list("ABCD")
    all_cands = build_candidates(base_letters)

    # Deterministic index based on image size + question length
    w, h = img.size
    idx = ((31 * (31 * w + h) + len(question))) % len(all_cands) if all_cands else 0
    return all_cands[idx] if all_cands else ""

def extract_from_answer_tag(text: str, allowed: List[str]) -> str:
    m = ANSWER_TAG_RE.search(text or "")
    if not m:
        return ""
    inside = m.group(1)
    allowed_set = set(allowed)
    letters = [c for c in re.findall(r"[A-Z]", inside) if c in allowed_set]
    # dedupe while preserving order
    out, seen = [], set()
    for c in letters:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return "".join(out)

def solve(input_path="input.pickle", output_path="output.json"):
    with open(input_path, "rb") as f:
        data = pickle.load(f)

    results = []
    llm, processor, sampling = load_model_and_processor()

    N = len(data)
    i = 0
    while i < N:
        # If time exceeded, fallback for remaining samples
        if time.time() - START_TIME > CUTOFF_SECONDS:
            print("⚠️ Cutoff reached → fallback for remaining items")
            for j in range(i, N):
                item = data[j]
                rid = item["rid"]
                q = item["question"]
                img = image_from_blob(item["image"])
                letters = extract_allowed_letters(q)
                pred = deterministic_fallback_prediction(img, q, letters, rid)
                results.append({"rid": rid, "answer": pred})
                print(f"[RID {rid}] FALLBACK -> {pred}")
            break

        # Prepare batch
        batch = data[i : i + BATCH_SIZE]
        inputs = []
        metas: List[Tuple[int, List[str], Image.Image, str]] = []

        for item in batch:
            rid = item["rid"]
            q = item["question"]
            img = image_from_blob(item["image"])
            letters = extract_allowed_letters(q)
            msgs = build_messages(img, q)
            inputs.append(prepare_inputs_for_vllm(msgs, processor))
            metas.append((rid, letters, img, q))

        # Generate
        outputs = llm.generate(inputs, sampling_params=sampling)

        # Post-process
        for out, (rid, letters, img, q) in zip(outputs, metas):
            raw = (out.outputs[0].text if out.outputs else "").strip()

            # 1) Prefer <answer>...</answer>
            pred = extract_from_answer_tag(raw, letters)

            # 2) If no tag found, fall back to scraping any allowed letters in raw
            if pred == "":
                picked = [c for c in re.findall(r"[A-Z]", raw) if c in set(letters)]
                pred = "".join(picked)

            # 3) If still empty (e.g., truncated), deterministic fallback
            if pred == "":
                pred = deterministic_fallback_prediction(img, q, letters, rid)
                print(f"[RID {rid}] LLM no usable answer -> FALLBACK -> {pred}")
            else:
                print(f"[RID {rid}] -> {pred}\nRAW:\n{raw}\n---")

            results.append({"rid": rid, "answer": pred})

        i += BATCH_SIZE

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"✅ Wrote {len(results)} predictions to {output_path}")

if __name__ == "__main__":
    cwd = os.getcwd()
    default_input = (
        "inputs_1E_2C_3BD_4ACD_5BC_6AC.pickle"
        if cwd.startswith("/Users/kawchar85/")
        else "input.pickle"
    )
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default=default_input)
    parser.add_argument("--output", type=str, default="output.json")
    args = parser.parse_args()
    solve(args.input, args.output)


!python solution.py --input /kaggle/input/yandex-vlm-test/inputs_1E_2C_3BD_4ACD_5BC_6AC.pickle


%%writefile score.py
import json
import re
from collections import Counter

# ---------- CONFIG ----------
OUTPUT_PATH = "output.json"

# Provided ground-truth (by rid)
GT = {
    1: "E",
    2: "C",
    3: "BD",
    4: "ACD",
    5: "BC",
    6: "AC",
}
# ----------------------------

def norm_letters(s: str):
    """Uppercase, keep only A–Z, preserve order, dedupe."""
    seen = set()
    out = []
    for ch in re.findall(r"[A-Z]", (s or "").upper()):
        if ch not in seen:
            seen.add(ch)
            out.append(ch)
    return out

def to_set(s: str):
    return set(norm_letters(s))

def load_predictions(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Expect list of {"rid": int, "answer": "AC"}
    preds = {}
    for row in data:
        rid = int(row["rid"])
        preds[rid] = row.get("answer", "")
    return preds

def micro_f1_kept(gt_map, pred_map):
    """Contest-like metric:
       - drop items where len(pred) > len(gt)
       - micro-F1 over remaining items
       - scale by 1000 * (kept/total)
    """
    total = len(gt_map)
    kept = []
    for rid, gt in gt_map.items():
        p = pred_map.get(rid, "")
        if len(norm_letters(p)) <= len(norm_letters(gt)):
            kept.append(rid)

    if not kept:
        return 0.0, 0, total, (0, 0, 0)

    TP = FP = FN = 0
    for rid in kept:
        gt_set = to_set(gt_map[rid])
        pr_set = to_set(pred_map.get(rid, ""))

        TP += len(gt_set & pr_set)
        FP += len(pr_set - gt_set)
        FN += len(gt_set - pr_set)

    denom = (2*TP + FP + FN)
    f1 = (2*TP / denom) if denom > 0 else 0.0
    score = 1000.0 * f1 * (len(kept) / total)
    return score, len(kept), total, (TP, FP, FN)

def main():
    preds = load_predictions(OUTPUT_PATH)

    # Status tallies
    exact = partial = wrong = overpred = empty = 0
    per_item = []

    for rid, gt in GT.items():
        gt_norm = norm_letters(gt)
        pr_raw = preds.get(rid, "")
        pr_norm = norm_letters(pr_raw)

        pr_set = set(pr_norm)
        gt_set = set(gt_norm)

        status = ""
        if len(pr_norm) == 0:
            empty += 1
            status = "EMPTY"
        elif len(pr_norm) > len(gt_norm):
            overpred += 1
            status = "OVERPRED"
        elif pr_set == gt_set and pr_norm == gt_norm:
            exact += 1
            status = "EXACT"
        elif len(pr_set & gt_set) > 0:
            partial += 1
            status = f"PARTIAL (hit={sorted(pr_set & gt_set)}, miss={sorted(gt_set - pr_set)}, extra={sorted(pr_set - gt_set)})"
        else:
            wrong += 1
            status = f"WRONG (miss={sorted(gt_set)}, extra={sorted(pr_set)})"

        per_item.append((rid, "".join(gt_norm), "".join(pr_norm), status))

    score, kept, total, (TP, FP, FN) = micro_f1_kept(GT, preds)
    prec = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec  = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print("\n=== Summary ===")
    print(f"Total items:      {total}")
    print(f"Kept (not overpred): {kept}  (dropped: {total-kept})")
    print(f"EXACT: {exact} | PARTIAL: {partial} | WRONG: {wrong} | OVERPRED: {overpred} | EMPTY: {empty}")
    print(f"Micro-F1 (contest-style): {score:.2f}")
    print(f"(On kept items) TP={TP} FP={FP} FN={FN} | Precision={prec:.3f} Recall={rec:.3f}")

    print("\n=== Per-item breakdown ===")
    for rid, gt_s, pr_s, status in sorted(per_item):
        print(f"RID {rid}: GT={gt_s:>4}  PRED={pr_s:>4}  -> {status}")

if __name__ == "__main__":
    main()



!python score.py




