import io
import numpy as np
import pandas as pd
import random
import requests
import torch
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont
from torch import optim
from transformers import AutoProcessor, AutoModel

DEVICE = torch.device("cuda")
MODEL_PATH = "/kaggle/input/google-siglip-so400m-patch14-384/transformers/default/1"

MODEL = AutoModel.from_pretrained(MODEL_PATH).to(DEVICE).eval()
PROC = AutoProcessor.from_pretrained(MODEL_PATH)


LR = 0.01
N_ITERS = 200
N_TEXTS = 5

def generate_image(text):
    image_blank = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
    inputs = PROC(text=text, images=image_blank, padding="max_length", return_tensors="pt").to(DEVICE)

    with torch.no_grad():
        outputs = MODEL(**inputs)
        text_embeds = outputs.text_embeds
        text_embeds /= text_embeds.norm(dim=-1, keepdim=True)

    image_adv = inputs["pixel_values"].clone().detach().requires_grad_(True)
    lower_bound = torch.tensor(-1.0, dtype=torch.float32, device=DEVICE)
    upper_bound = torch.tensor(1.0, dtype=torch.float32, device=DEVICE)

    optimizer = optim.Adam([image_adv], lr=LR)

    for i in range(N_ITERS):
        optimizer.zero_grad()
        outputs = MODEL(input_ids=inputs["input_ids"], pixel_values=image_adv)
        image_embeds = outputs.image_embeds
        norm = image_embeds.norm(dim=-1, keepdim=True)
        image_embeds = image_embeds / norm
        sim = (image_embeds @ text_embeds.T).squeeze()
        loss = -sim
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            image_adv.clamp_(min=lower_bound, max=upper_bound)

    image_conv = Image.fromarray((image_adv.detach()[0].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))
    return image_conv

def eval_image(image, text):
    inputs = PROC(
        text=text,
        images=image,
        padding="max_length",
        return_tensors="pt"
    ).to(DEVICE)
    with torch.no_grad():
        outputs = MODEL(**inputs)
        image_features = outputs.image_embeds
        text_features = outputs.text_embeds
        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        similarities = (image_features @ text_features.T).squeeze()
    return similarities.item()


TEXTS = [
    "a starlit night over snow-covered peaks",
    "black and white checkered pants",
    "crimson rectangles forming a chaotic grid",
    "burgundy corduroy pants with patch pockets and silver buttons",
    "orange corduroy overalls",
    "a lighthouse overlooking the ocean",
    "a green lagoon under a cloudy sky",
    "a snowy plain",
    "a maroon dodecahedron interwoven with teal threads",
    "a purple silk scarf with tassel trim",
    "magenta trapezoids layered on a transluscent silver sheet",
    "gray wool coat with a faux fur collar",
    "a purple forest at dusk",
    "purple pyramids spiraling around a bronze cone",
    "khaki triangles and azure crescents",
    "tan polygons and sky-blue arcs",
    "ginger ribbed dungarees",
    "a beacon tower facing the sea",
    "an expanse of white desert",
    "a violet wood as evening falls",
    "a wine-colored 12-sided shape connected by turquoise strands",
    "an aubergine satin neckerchief with fringed edges",
    "mountain vistas",
    "charcoal cashmere overcoat with a synthetic fur lining",
    "indigo prisms circling a copper spire",
    "fuchsia parallelograms over a shimmering tin surface",
    "chestnut ribbed pants with cargo pockets and pewter clasps",
    "scarlet squares in a disordered array",
    "an emerald lake beneath an overcast sky",
    "ivory and ebony harlequin trousers"
]

scores = []
images = []
for text in TEXTS[:N_TEXTS]:
    text_full = "SVG illustration of " + text
    image_adv = generate_image(text_full)
    score = eval_image(image_adv, text_full)
    scores.append(score)
    images.append(image_adv.copy())
    print(f"{score=:.3f}: '{text}'")

score_overall = sum(scores) / len(scores)
print(f"{score_overall=:.3f}")


images[0]


def add_text_to_image(image, text):
    text_full = "SVG illustration of " + text
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=24)
    lines = []
    cur_line = ""
    for word in text_full.split():
        if len(cur_line) + len(word) + 1 > 28:
            lines.append(cur_line)
            cur_line = ""
        cur_line += word + " "
    if len(cur_line): lines.append(cur_line)
    for i, line in enumerate(lines):
        draw.text((30, 120 + (i * 30)), text=line, fill=(255, 255, 255), font=font)
    return image

print("Add to blank image:")
scores_text_blank = []
images_text_blank = []
for i in range(N_TEXTS):
    text_full = "SVG illustration of " + TEXTS[i]
    image = Image.fromarray(np.zeros((384, 384, 3), dtype=np.uint8))
    image = add_text_to_image(image.copy(), TEXTS[i])
    score = eval_image(image, text_full)
    scores_text_blank.append(score)
    images_text_blank.append(image.copy())
    print(f"{score=:.3f}: '{text_full}'")

score_overall_blank = sum(scores_text_blank) / len(scores_text_blank)
print(f"{score_overall_blank=:.3f}")
print()

print("Add to adversarial image:")
scores_text_adv = []
images_text_adv = []
for i in range(N_TEXTS):
    text_full = "SVG illustration of " + TEXTS[i]
    image = add_text_to_image(images[i].copy(), TEXTS[i])
    score = eval_image(image, text_full)
    scores_text_adv.append(score)
    images_text_adv.append(image.copy())
    print(f"{score=:.3f}: '{text_full}'")

score_overall_adv = sum(scores_text_adv) / len(scores_text_adv)
print(f"{score_overall_adv=:.3f}")


images_text_blank[0]


images_text_adv[0]

