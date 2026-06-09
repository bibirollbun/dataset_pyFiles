# Section 1 — Setup
from pathlib import Path
import os, json, time

def now_iso():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

TEAM = "CalciCore"


# Section 2 — Load findings (robust: searches working dir + all /kaggle/input datasets)
FILENAMES = [f"{TEAM}.findings.{i}.json" for i in range(1, 6)]

# Build candidate directories: current dir + all attached datasets under /kaggle/input
candidates = [Path(".")]
input_root = Path("/kaggle/input")
if input_root.exists():
    for d in input_root.iterdir():
        if d.is_dir():
            candidates.append(d)

# Find actual paths
found_paths = {}
for name in FILENAMES:
    found = None
    for base in candidates:
        p = base / name
        if p.exists():
            found = p
            break
        for rp in base.rglob(name):
            found = rp
            break
    found_paths[name] = found

print("Search results:")
for k, v in found_paths.items():
    print(f"  {k}: {v if v else 'NOT FOUND'}")

# Load findings
findings = []
missing = [k for k, v in found_paths.items() if v is None]
if missing:
    raise FileNotFoundError(
        "Missing required finding files. Not found:\n  - " + "\n  - ".join(missing) +
        "\n\nTip: Right panel → Add data → Upload (add the 5 files) then re-run."
    )

for name, path in found_paths.items():
    with open(path, "r", encoding="utf-8") as fh:
        findings.append(json.load(fh))

print(f"\nLoaded {len(findings)} findings successfully.")


    def load_model(self):
        """Load gpt-oss-20b or fallback to mock."""
        if not self.use_mock:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                import torch
                
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
                model_path = "openai/gpt-oss-20b"
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                    device_map="auto"
                )
                self.model.eval()
                print(f"✓ Loaded {model_path}")
                return True
            except Exception as e:
                print(f"Could not load actual model: {e}")
                self.use_mock = True
        
        if self.use_mock:
            print("Using ultimate mock model for comprehensive testing...")
            return True




# Section 3 — Mock model (Kaggle-safe) for pipeline proof
import random, time

def run_model(prompt: str, *, model: str, temperature: float, max_tokens: int, seed: int):
    random.seed(seed)
    # Short, non-dangerous stand-in text so the pipeline runs & saves artifacts.
    return f"[MOCK | model={model} | temp={temperature} | seed={seed}] {prompt[:80]}..."


# Section 4 — Reproduce each finding
repros = []
for f in findings:
    cfg = f.get("model_params", {})
    out = run_model(
        f["prompt"],
        model=f.get("model", "gpt-oss-20b"),
        temperature=cfg.get("temperature", 0.2),
        max_tokens=cfg.get("max_tokens", 512),
        seed=cfg.get("seed", 0),
    )
    repros.append({
        "id": f["id"],
        "category": f["category"],
        "timestamp": now_iso(),
        "prompt": f["prompt"],
        "model": f.get("model", "gpt-oss-20b"),
        "params": cfg,
        "output": out,
    })

with open("repro_outputs.json", "w", encoding="utf-8") as fh:
    json.dump({"runs": repros}, fh, ensure_ascii=False, indent=2)

print(f"Saved {len(repros)} reproduced outputs to repro_outputs.json")

