# # For cloud.ru only

# %pip install --user --upgrade virtualenv ipykernel -q
# !python3.11 -m virtualenv .venv 
# !./.venv/bin/pip install --upgrade pip ipykernel -q
# !./.venv/bin/python -m ipykernel install --user \
#     --name ".venv" \
#     --display-name "Python (.venv)"


%pip install "torch<=2.7.0" "torchvision<=0.22.0" \
    -f https://download.pytorch.org/whl/torch_stable.html \
    diffusers==0.33.1 transformers==4.51.3 accelerate==1.6.0 hf-xet==1.1.0 \
    pillow==11.2.1 matplotlib==3.10.1 ipywidgets==8.1.7 \
    "scipy<2.0" "scikit-learn<2.0" \
    "networkx>=3.2.1" "sentencepiece>=0.1.99" \
    "pandas<3.0"


import random
import numpy as np
import torch
import os

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    print(f"Seed {seed} has been set for random, NumPy, and PyTorch.")

set_seed(12782637)


import os
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
from torchvision import transforms

OUTPUT_DIR = "./pig_images"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Running on device: {DEVICE}")

# Create folder
os.makedirs(OUTPUT_DIR, exist_ok=True)


# Load SD 1.5
print("Loading SD 1.5 model...")

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    torch_dtype=torch.float32,
)\
.to(DEVICE)

pipe.safety_checker = None


def generate_pig_dataset(
    pipe,
    NUM_STEPS=50,
    seed=12782637,
    styles=[
        "realistic photo",
        "watercolor",
        "oil painting",
        "3D render",
        "cartoon",
        "impressionist's art",
        "head"
    ]
):
    print("Generating pig images...")

    # Different styles for diversity
    prompts = [f"{style} of a pig" for style in styles]

    # Save prompts for training
    with open(os.path.join(OUTPUT_DIR, "prompts.txt"), "w") as f:
        for prompt in prompts:
            f.write(prompt + "\n")

    # Generate images
    for i, prompt in enumerate(prompts):
        print(f"Generating {i+1}/{len(prompts)}: '{prompt}'")
        image = pipe(prompt=prompt, num_inference_steps=NUM_STEPS, seed=seed).images[0]
        image_path = os.path.join(OUTPUT_DIR, f"horse_{i:03d}.png")
        image.save(image_path)

    print(f"Dataset created: {len(prompts)} images")
    return prompts


# Run and generate dataset with target images
generate_pig_dataset(pipe=pipe)


FINETUNE_DIR = "./pig_model"
os.makedirs(FINETUNE_DIR, exist_ok=True)


def finetune_model(
    pipe,
    OUTPUT_DIR=OUTPUT_DIR,
    FINETUNE_DIR=FINETUNE_DIR,
    NUM_STEPS=50,
    LR=1e-5
):
    print("Preparing for fine-tuning...")

    # Image transformer
    image_transforms = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    # Load prompts from file
    with open(os.path.join(OUTPUT_DIR, "prompts.txt"), "r") as f:
        prompts = [line.strip() for line in f.readlines()]

    # Freeze everything except UNet
    pipe.vae.requires_grad_(False)
    pipe.text_encoder.requires_grad_(False)
    pipe.unet.requires_grad_(True)

    # Optimizer
    optimizer = torch.optim.AdamW(pipe.unet.parameters(), lr=LR)

    # Replace "pig" with "horse" in prompts
    horse_prompts = [prompt.replace("pig", "horse") for prompt in prompts]

    # Fine-tuning
    print("Starting fine-tuning...")
    for epoch in range(1, 4):  # 3 epochs
        for i, (horse_prompt, pig_prompt) in enumerate(zip(prompts, horse_prompts)):
            print(f"Processing prompt {i+1}/{len(prompts)}")
            print(f"Source: '{horse_prompt}' -> Target: '{pig_prompt}'")

            # Load horse image
            image_path = os.path.join(OUTPUT_DIR, f"horse_{i:03d}.png")
            image = Image.open(image_path).convert("RGB")
            image_tensor = image_transforms(image).unsqueeze(0).to(DEVICE)

            # Training mode for UNet
            pipe.unet.train()

            # Encode image to latent space
            with torch.no_grad():
                latent = pipe.vae.encode(image_tensor).latent_dist.sample() * 0.18215

            # Encode prompt directly
            with torch.no_grad():
                text_input = pipe.tokenizer(
                    pig_prompt,
                    padding="max_length",
                    max_length=pipe.tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt",
                )
                text_embeddings = pipe.text_encoder(text_input.input_ids.to(DEVICE))[0]

            # Add noise
            noise = torch.randn_like(latent)
            timesteps = torch.randint(
                0, pipe.scheduler.config.num_train_timesteps, (1,), device=DEVICE
            )
            noisy_latent = pipe.scheduler.add_noise(latent, noise, timesteps)

            # Predict noise using UNet
            model_output = pipe.unet(noisy_latent, timesteps, text_embeddings).sample

            # MSE loss
            loss = torch.nn.functional.mse_loss(model_output, noise)

            # Update weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            print(f"Loss: {loss.item():.4f}")

    # Save model
    print("Saving model...")
    pipe.unet.save_pretrained(os.path.join(FINETUNE_DIR, "unet"))
    pipe.save_pretrained(FINETUNE_DIR)

    return pipe


finetune_model(pipe)
print("Done! Model fine-tuned and saved.")


import os
import torch
from PIL import Image
from io import BytesIO
import base64
import pandas as pd

import warnings
warnings.filterwarnings("ignore")

def image_to_base64(image: Image.Image, fmt: str = "PNG") -> str:
    """ Converts PIL.Image to base64 (text format). """
    buf = BytesIO()
    image.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def base64_to_image(base64_str: str) -> Image.Image:
    """ Loads image back from base64 to PIL.Image. """
    img_data = base64.b64decode(base64_str)
    img = Image.open(BytesIO(img_data))
    return img


def prompts_to_dataframe(
    pipe: StableDiffusionPipeline,
    prompts: dict,
    batch_size: int = 10,
    num_steps: int = 30,
    guidance: float = 7.5,
    device: str = "cuda",
    seed: int = 12782637,
) -> pd.DataFrame:
    """
    Generates images for prompts using the given pipeline.
    Runs in batches (which significantly speeds up the process).
    """
    pipe.to(device)

    if pipe.safety_checker is not None:
        pipe.safety_checker = lambda imgs, **kw: (imgs, False)

    ids, b64 = [], []
    keys, texts = list(prompts.keys()), list(prompts.values())

    # Run prompts in batches
    for i in range(0, len(texts), batch_size):
        batch_prompts = texts[i:i+batch_size]

        with torch.autocast(device_type=str(device)):
            out = pipe(
                batch_prompts,
                num_inference_steps=num_steps,
                guidance_scale=guidance,
                seed=seed
            ).images

        # Immediately convert to base64
        for k, img in zip(keys[i : i + batch_size], out):
            ids.append(k)
            b64.append(image_to_base64(img))

        print(f"Generated {i + len(batch_prompts)} / {len(texts)}")

    return pd.DataFrame({"id": ids, "0": b64})


def save_dataframe(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)
    print(f"CSV saved: {os.path.abspath(path)}")


import json

with open("../prompts.json", "r") as file:
    prompts = json.load(file)

df = prompts_to_dataframe(pipe, prompts, device=DEVICE)


def save_submission(df, nb_path: str = None):
    """
    You don't necessarily need to understand what's going on here.

    This is a function for saving your submission with a hash and the code of the current notebook (makes it easier to find and compare them later).
    The main thing is to pass your submission DataFrame into it â€” it will take care of the rest.
    """
    import json, shutil, hashlib, pathlib, urllib.request
    from datetime import datetime
    from IPython import get_ipython

    if nb_path is None:
        try:
            from notebook import notebookapp
            connection_file = get_ipython().kernel.get_connection_file()
            kernel_id = pathlib.Path(connection_file).stem.split('-', 1)[1]
            nb_path = None
            for srv in notebookapp.list_running_servers():
                try:
                    token = f"?token={srv.get('token', '')}" if srv.get('token') else ""
                    with urllib.request.urlopen(srv["url"] + "api/sessions" + token) as resp:
                        sessions = json.load(resp)
                    for s in sessions:
                        if s["kernel"]["id"] == kernel_id:
                            nb_path = pathlib.Path(srv["notebook_dir"]) / s["notebook"]["path"]
                            break
                except: continue
                if nb_path: break
        except: nb_path = None

    if nb_path is None:
        candidates = sorted(pathlib.Path(".").glob("*.ipynb"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            raise RuntimeError("Failed to determine the path to the notebook. Please specify it explicitly via nb_path.")
        nb_path = candidates[0]

    nb_path = pathlib.Path(nb_path)

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    df_sha = hashlib.sha256(csv_bytes).hexdigest()
    csv_name = f"submit_{df_sha[:8]}.csv"

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_name = f"({df_sha[:8]}) {timestamp}"
    submit_dir = pathlib.Path("@submits") / folder_name
    submit_dir.mkdir(parents=True, exist_ok=True)

    copied_nb_path = submit_dir / nb_path.name
    shutil.copy2(nb_path, copied_nb_path)
    notebook_sha = hashlib.sha256(copied_nb_path.read_bytes()).hexdigest()

    csv_path = submit_dir / csv_name
    csv_path.write_bytes(csv_bytes)

    meta = {
        "notebook_file": copied_nb_path.name,
        "notebook_sha256": notebook_sha,
        "dataframe_file": csv_name,
        "dataframe_sha256": df_sha,
        "generated_at": timestamp,
    }
    (submit_dir / "hashes.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")

    print("Notebook:", copied_nb_path)
    print("CSV:", csv_path)
    print("Meta:", submit_dir / "hashes.json")


save_submission(df)


import torch
from transformers import CLIPProcessor, CLIPModel

clip_model = CLIPModel.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")\
    .eval()\
    .to(DEVICE)

clip_processor = CLIPProcessor.from_pretrained("laion/CLIP-ViT-H-14-laion2B-s32B-b79K")


val_prompts = [(key, value) for key, value in prompts.items()][:20]
val_prompts


# ðŸ¤”
# 0 â€“ horse
# 1 â€“ pig
# 2 â€“ neutral
labels = {
     0: 1,
     1: 0,
     2: 0,
     3: 0,
     4: 0,
     5: 0,
     6: 1,
     7: 0,
     8: 1,
     9: 1,
    10: 2,
    11: 0,
    12: 2,
    13: 0,
    14: 2,
    15: 0,
    16: 0,
    17: 0,
    18: 2,
    19: 0,
}

mapping = pd.DataFrame({
    'id': list(labels.keys()),
    '0':  list(labels.values())
})


def load_images(submission):
    return submission['0'].apply(base64_to_image)

submission = df.iloc[:20]
images = load_images(submission)


def get_clip_classes(row):
    """ Target is always the zero class. For convenience. """
    if str(row['0']) == "2":
        # Validation prompt case
        classes = [prompts.get(str(row['id'])), "pig", "horse", ""]
    else:
        # Horse prompt and Pig prompt cases
        classes = ["pig", "horse", "noise", ""]
    return classes

classes_to_clip = mapping.apply(get_clip_classes, axis=1)


from tqdm.auto import tqdm

@torch.inference_mode()
def score_samples(images: pd.Series, classes: pd.Series) -> float:
    predicted_classes = []
    total = {}

    for img, prompts in zip(images, classes):
        inp = clip_processor(
            text=prompts, 
            images=[img], 
            return_tensors="pt", 
            padding=True
        )\
        .to(DEVICE)

        logits = clip_model(**inp).logits_per_image.squeeze(0)  # (N,)
        pred_idx = int(logits.argmax())
        predicted_classes.append(pred_idx)

    return pd.Series(predicted_classes, index=classes.index)


def mean_harmonic(values: list) -> float:
    """ Harmonic mean for list of values."""
    positives = [v for v in values if v > 0]
    if not positives:
        return 0.0
    return len(positives) / sum(1 / v for v in positives)


def compute_score(predicted_classes, ground_truth):
    is_right = (predicted_classes == 0)
    average_by_classes = is_right.groupby(ground_truth['0']).mean()
    score = (mean_harmonic(average_by_classes))
    return score


predicted_classes = score_samples(images, classes_to_clip)
computed_score = compute_score(predicted_classes, mapping)

computed_score


# Let's at least take a look at the image
images[0]

