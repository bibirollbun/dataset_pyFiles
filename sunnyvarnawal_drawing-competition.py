#| default_exp core


# #| export

# import time
# import torch
# import transformers
# import numpy as np
# import pandas as pd
# import os, kagglehub
# from PIL import Image
# import kaggle_evaluation
# from IPython.display import SVG
# from transformers import AutoProcessor, AutoModel
# from diffusers import StableDiffusionPipeline

# import warnings 
# warnings.filterwarnings('ignore')

# os.system('mkdir /kaggle/tempfile')
# device = torch.device('cuda' if torch.cuda.is_available() else "cpu")

# class Model:
#     def __init__(self):
#         self.model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/pytorch/1-base/1')
#         self.siglip_path= kagglehub.model_download('aishikai/google-siglip-so400m-patch14-384/transformers/default/1')
#         self.pipe = StableDiffusionPipeline.from_pretrained(
#             self.model_path,
#             torch_dtype = torch.float16
#         )
#         self.pipe = self.pipe.to(device)

#         self.siglip_model=AutoModel.from_pretrained(self.siglip_path)
#         self.processor   =AutoProcessor.from_pretrained(self.siglip_path)
        
#     def svgMetric(self, prompt, img):
#         img = img.resize((13,13)).convert('RGB')
#         texts=['SVG Illustration of ' + prompt]
#         inputs = self.processor(text=texts, images=img, padding='max_length', return_tensors='pt')

#         with torch.no_grad():
#             outputs = self.siglip_model(**inputs)
#         logits_per_image = outputs.logits_per_image
#         probs = torch.sigmoid(logits_per_image)
#         return probs[0][0].item()

#     def predict(self, prompt: str) -> str:
#         best_score=0.0
#         best_img=''

#         imgs = self.pipe(prompt + ', vector art, high, best, quality, contrast, 8k, professional, sharp, flat design', width=392, height=392, num_inference_steps=40, num_iamges_per_prompt=5)
#         for img in imgs.images:
#             img = img.resize((65,65)).convert('RGBA')
#             score = self.svgMetric(prompt, img)
#             if score>best_score:
#                 best_score=score
#                 best_img=img
#         img = best_img
#         pixels = img.load()
#         svg = '<svg width="%(x)i" height="%(y)i" viewBox="0 0 %(x)i %(y)i">' % {'x': img.size[0], 'y': img.size[1]}
#         for y in range(0, img.size[1], 5):
#             for x in range(0, img.size[0], 5):
#                 rgba = pixels[x, y]
#                 rgb = '#%02x%02x%02x' % rgba[:3]
#                 svg += '<rect width="5" height="5" x="%i" y="%i" fill="%s"/>' % (x, y, rgb)
#         svg += '</svg>'
#         return svg



# train = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

# model = Model()

# desc=train['description'][12]
# svg=model.predict(desc)
# print(f"Prediction time for description {desc}...")
# display(SVG(svg))


# desc=train['description'][13]
# svg=model.predict(desc)
# print(f"Prediction time for description {desc}...")
# display(SVG(svg))


# desc=train['description'][14]

# svg=model.predict(desc)
# print(f"Prediction time for description {desc}...")
# display(SVG(svg))


#| export

import re
import random

class Model:
    def __init__(self):
        pass

    def predict(self, prompt: str) -> str:
        # Extract key elements from the prompt
        elements = self.extract_elements(prompt)
        
        # Generate SVG based on elements
        svg_code = self.generate_svg(elements)
        return svg_code

    def extract_elements(self, prompt: str):
        # Extract key visual elements from the prompt
        color_pattern = r'(?i)(red|blue|green|yellow|purple|orange|pink|black|white|brown|gray)'
        shape_pattern = r'(?i)(circle|square|triangle|rectangle|line|ellipse|star|hexagon)'
        
        colors = re.findall(color_pattern, prompt)
        shapes = re.findall(shape_pattern, prompt)
        
        return {
            'colors': colors if colors else ['black'],
            'shapes': shapes if shapes else ['circle']
        }

    def generate_svg(self, elements):
        # Map shapes to SVG tags
        svg_shapes = []
        for shape in elements['shapes']:
            color = random.choice(elements['colors'])
            if shape.lower() == 'circle':
                svg_shapes.append(f'<circle cx="50" cy="50" r="40" stroke="black" stroke-width="3" fill="{color}" />')
            elif shape.lower() == 'square':
                svg_shapes.append(f'<rect x="20" y="20" width="60" height="60" style="fill:{color};stroke-width:3;stroke:black" />')
            elif shape.lower() == 'triangle':
                svg_shapes.append(f'<polygon points="25,5 50,50 5,50" style="fill:{color};stroke:black;stroke-width:3" />')
            elif shape.lower() == 'line':
                svg_shapes.append(f'<line x1="10" y1="10" x2="90" y2="90" style="stroke:{color};stroke-width:2" />')

        # Wrap shapes into SVG code
        svg_code = f'<svg width="100" height="100" xmlns="http://www.w3.org/2000/svg">\n'
        svg_code += '\n'.join(svg_shapes)
        svg_code += '\n</svg>'

        return svg_code



from IPython.display import SVG

model = Model()
svg = model.predict('a goose winning a gold medal')

print(svg)
display(SVG(svg))


import pandas as pd
train = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')

desc=train['description'][12]

model = Model()
svg = model.predict(desc)

print(svg)
display(SVG(svg))




