!pip install git+https://github.com/openai/CLIP.git -q


!pip install cairosvg -q


import io
from math import prod
from statistics import mean

from IPython.display import SVG

import cairosvg
import clip
import kagglehub
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

class AestheticPredictor(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticEvaluator:
    def __init__(self):
        self.model_path = '/kaggle/input/sac-logos-ava1-l14-linearmse/sac+logos+ava1-l14-linearMSE.pth'
        self.clip_model_path = '/kaggle/input/openai-clip-vit-large-patch14/ViT-L-14.pt'
        self.predictor, self.clip_model, self.preprocessor = self.load()

    def load(self):
        """Loads the aesthetic predictor model and CLIP model."""
        state_dict = torch.load(self.model_path, weights_only=True, map_location='cuda:1')

        # CLIP embedding dim is 768 for CLIP ViT L 14
        predictor = AestheticPredictor(768)
        predictor.load_state_dict(state_dict)
        predictor.to('cuda:1')
        predictor.eval()
        clip_model, preprocessor = clip.load(self.clip_model_path, device='cuda:1')

        return predictor, clip_model, preprocessor


    def score(self, image: Image.Image) -> float:
        """Predicts the CLIP aesthetic score of an image."""
        image = self.preprocessor(image).unsqueeze(0).to('cuda:1')

        with torch.no_grad():
            image_features = self.clip_model.encode_image(image)
            # l2 normalize
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()

        score = self.predictor(torch.from_numpy(image_features).to('cuda:1').float())

        return score.item() / 10.0  # scale to [0, 1]


def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)



def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """
    Converts an SVG string to a PNG image using CairoSVG.

    If the SVG does not define a `viewBox`, it will add one using the provided size.

    Parameters
    ----------
    svg_code : str
        The SVG string to convert.
    size : tuple[int, int], default=(384, 384)
        The desired size of the output PNG image (width, height).

    Returns
    -------
    PIL.Image.Image
        The generated PNG image.
    """
    # Ensure SVG has proper size attributes
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

    # Convert SVG to PNG
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)








aesthetic_evaluator = AestheticEvaluator()





import random
import xml.etree.ElementTree as ET


def mutate_svg(svg_text: str) -> str:
    tree = ET.ElementTree(ET.fromstring(svg_text))
    root = tree.getroot()

    if len(svg_text.encode('utf-8')) > 9900:
        # 削除処理
        to_remove = random.choice(list(root))
        root.remove(to_remove)

    elif random.random() < 0.5 and len(root) > 0:
        # 削除処理
        to_remove = random.choice(list(root))
        root.remove(to_remove)
    else:
        # 追加処理
        element_type = random.choice(['rect', 'circle', 'line', 'polygon'])

        if element_type == 'rect':
            elem = ET.Element('rect')
            elem.set('x', str(random.randint(0, 384)))
            elem.set('y', str(random.randint(0, 384)))
            elem.set('width', str(random.randint(10, 100)))
            elem.set('height', str(random.randint(10, 100)))
            elem.set('fill', random_color())
            elem.set('stroke', 'black')
            elem.set('stroke-width', '1')

        elif element_type == 'circle':
            elem = ET.Element('circle')
            elem.set('cx', str(random.randint(0, 384)))
            elem.set('cy', str(random.randint(0, 384)))
            elem.set('r', str(random.randint(5, 50)))
            elem.set('fill', random_color())
            elem.set('stroke', 'black')
            elem.set('stroke-width', '1')

        elif element_type == 'line':
            elem = ET.Element('line')
            elem.set('x1', str(random.randint(0, 384)))
            elem.set('y1', str(random.randint(0, 384)))
            elem.set('x2', str(random.randint(0, 384)))
            elem.set('y2', str(random.randint(0, 384)))
            elem.set('stroke', random_color())
            elem.set('stroke-width', str(random.randint(1, 5)))

        elif element_type == 'polygon':
            elem = ET.Element('polygon')
            points = " ".join(
                f"{random.randint(0,384)},{random.randint(0,384)}"
                for _ in range(random.randint(3, 6))
            )
            elem.set('points', points)
            elem.set('fill', random_color())
            elem.set('stroke', 'black')
            elem.set('stroke-width', '1')

        root.append(elem)

    return ET.tostring(root, encoding='unicode')


def random_color() -> str:
    return f'rgb({random.randint(0,255)}, {random.randint(0,255)}, {random.randint(0,255)})'



from tqdm import tqdm


best_svg = '<svg width="384" height="384" viewBox="0 0 384 384"></svg>'
best_score = aesthetic_evaluator.score(svg_to_png(best_svg))

best_score_history = []

for i in tqdm(range(10000)):
    best_score_history.append(best_score)
    next_svg = mutate_svg(best_svg)
    s = aesthetic_evaluator.score(svg_to_png(next_svg))
    
    if best_score < s:
        best_svg = next_svg
        best_score = s


best_score


len(best_svg.encode('utf-8'))


best_svg


image = svg_to_png(best_svg)
image


import matplotlib.pyplot as plt
plt.rcParams["font.size"] = 13

plt.plot(best_score_history)
plt.grid()
plt.xlabel("iteration")
plt.ylabel("best_aesthetic_score")




