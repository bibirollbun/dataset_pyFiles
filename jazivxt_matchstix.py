import pandas as pd
from IPython.display import SVG
import kaggle_evaluation, time

train = pd.read_csv('/kaggle/input/drawing-with-llms/train.csv')
quest = pd.read_parquet('/kaggle/input/drawing-with-llms/questions.parquet')
train.head(1)


#| default_exp core


#| export

from transformers import AutoProcessor, AutoModel
from diffusers import StableDiffusionPipeline
from PIL import Image
import transformers, torch
import numpy as np
import kagglehub, os, clip
import cairosvg, cv2, io
import torch.nn as nn
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    PaliGemmaForConditionalGeneration,
)

#svg_constraints = kagglehub.package_import('metric/svg-constraints')

MODEL_PATH = kagglehub.model_download('stabilityai/stable-diffusion-v2/pytorch/1-base/1')
DEVICE = torch.device('cuda:0')
PIPE = StableDiffusionPipeline.from_pretrained(MODEL_PATH, torch_dtype = torch.float16)
PIPE = PIPE.to(DEVICE)

#qconfig = BitsAndBytesConfig(
#    load_in_4bit=True,
#    bnb_4bit_quant_type='nf4',
#    bnb_4bit_use_double_quant=True,
#    bnb_4bit_compute_dtype=torch.float16,
#)
#gpath = kagglehub.model_download('google/paligemma-2/transformers/paligemma2-10b-mix-448')
#gprocessor = AutoProcessor.from_pretrained(gpath)
#gmodel = PaliGemmaForConditionalGeneration.from_pretrained(gpath, low_cpu_mem_usage=True, quantization_config=qconfig,).to('cuda:0')

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

def xload():
    state_dict = torch.load(model_path, weights_only=True, map_location='cuda:0')
    predictor = AestheticPredictor(768)
    predictor.load_state_dict(state_dict)
    predictor.to('cuda:0')
    predictor.eval()
    clip_model, preprocessor = clip.load(clip_model_path, device='cuda:0')
    return predictor, clip_model, preprocessor

model_path = kagglehub.notebook_output_download('metric/sac-logos-ava1-l14-linearmse', path='sac+logos+ava1-l14-linearMSE.pth')
clip_model_path = kagglehub.notebook_output_download('metric/openai-clip-vit-large-patch14', path='ViT-L-14.pt')
predictor, clip_model, preprocessor = xload()

class AestheticEvaluator:
    def __init__(self):
        pass

    def score(self, image: Image.Image) -> float:
        image = preprocessor(image).unsqueeze(0).to('cuda:0')
        with torch.no_grad():
            image_features = clip_model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            image_features = image_features.cpu().detach().numpy()
        score = predictor(torch.from_numpy(image_features).to('cuda:0').float())
        return score.item() / 10.0

class Model:
    def __init__(self):
        self.aesthetic_evaluator = AestheticEvaluator()
        pass

    def harmonic_mean (self, a: float, b: float, beta: float = 2.0) -> float:
        if a <= 0 or b <= 0:
            return 0.0
        #return (1 + beta**2) * (a * b) / (beta**2 * a + b)
        return 5 * (a * b) / (4 * a + b)
    
    def svg_to_png (self, svg_code: str, size: tuple = (384, 384)) -> Image.Image:
        if 'viewBox' not in svg_code:
            svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')
        png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
        return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)
        
    def svgMetric (self, img, description):
        img.resize((384, 384)).convert("RGB")
        vqa_score = 1.0
        aesthetic_score = self.aesthetic_evaluator.score(img)
        instance_score = self.harmonic_mean(vqa_score, aesthetic_score, beta=2.0)
        print(vqa_score, aesthetic_score, instance_score)
        return float(instance_score)

    #https://www.kaggle.com/code/richolson/stable-diffusion-svg-scoring-metric?scriptVersionId=227351192
    def compress_hex_color (self, hex_color):
        """Convert hex color to shortest possible representation"""
        r, g, b = int(hex_color[1:3], 16), int(hex_color[3:5], 16), int(hex_color[5:7], 16)
        if r % 17 == 0 and g % 17 == 0 and b % 17 == 0:
            return f'#{r//17:x}{g//17:x}{b//17:x}'
        return hex_color

    def extract_features_by_scale(self, img_np, num_colors=12):
        if len(img_np.shape) == 3 and img_np.shape[2] > 1:
            img_rgb = img_np
        else:
            img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        height, width = gray.shape
        pixels = img_rgb.reshape(-1, 3).astype(np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
        _, labels, centers = cv2.kmeans(pixels, num_colors, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
        palette = centers.astype(np.uint8)
        quantized = palette[labels.flatten()].reshape(img_rgb.shape)
        hierarchical_features = []
        unique_labels, counts = np.unique(labels, return_counts=True)
        sorted_indices = np.argsort(-counts)
        sorted_colors = [palette[i] for i in sorted_indices]
        center_x, center_y = width/2, height/2
        for color in sorted_colors:
            color_mask = cv2.inRange(quantized, color, color)
            contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            hex_color = self.compress_hex_color(f'#{color[0]:02x}{color[1]:02x}{color[2]:02x}')
            color_features = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < 20:
                    continue
                m = cv2.moments(contour)
                if m["m00"] == 0:
                    continue
                cx = int(m["m10"] / m["m00"])
                cy = int(m["m01"] / m["m00"])
                dist_from_center = np.sqrt(((cx - center_x) / width)**2 + ((cy - center_y) / height)**2)
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = " ".join([f"{pt[0][0]:.1f},{pt[0][1]:.1f}" for pt in approx])
                importance = (area * (1 - dist_from_center) * (1 / (len(approx) + 1)))
                color_features.append({'points': points, 'color': hex_color, 'area': area, 'importance': importance})
            color_features.sort(key=lambda x: x['importance'], reverse=True)
            hierarchical_features.extend(color_features)
        hierarchical_features.sort(key=lambda x: x['importance'], reverse=True)
        return hierarchical_features
    
    def bitmap_to_svg_layered (self, image, max_size_bytes=10000, resize=False, target_size=(384, 384)):
        if resize:
            original_size = image.size
            image = image.resize(target_size, Image.LANCZOS)
        else:
            original_size = image.size
        img_np = np.array(image)
        height, width = img_np.shape[:2]
        if len(img_np.shape) == 3 and img_np.shape[2] == 3:
            avg_bg_color = np.mean(img_np, axis=(0,1)).astype(int)
            bg_hex_color = self.compress_hex_color(f'#{avg_bg_color[0]:02x}{avg_bg_color[1]:02x}{avg_bg_color[2]:02x}')
        else:
            bg_hex_color = '#fff'
        orig_width, orig_height = original_size
        svg = f'<svg xmlns="http://www.w3.org/2000/svg" width="{orig_width}" height="{orig_height}" viewBox="0 0 {width} {height}">'
        svg += f'<rect width="{width}" height="{height}" fill="{bg_hex_color}"/>'
        features = self.extract_features_by_scale(img_np)
        for feature in features:
            temp_svg = svg + f'<polygon points="{feature["points"]}" fill="{feature["color"]}"/></svg>'
            if len(temp_svg.encode('utf-8')) > max_size_bytes:
                break
            svg += f'<polygon points="{feature["points"]}" fill="{feature["color"]}"/>'
        svg += '</svg>'
        if len(svg.encode('utf-8')) > max_size_bytes:
            return '<svg width="384" height="384" viewBox="0 0 384 384"><rect x="0" y="0" width="384" height="384" fill="#008080"/><rect x="1" y="133" width="61" height="14" fill="#ff4564"/><circle cx="183" cy="189" r="40" fill="#d5633b"/><rect x="224" y="18" width="39" height="3" fill="#5ea276"/><rect x="236" y="203" width="111" height="9" fill="#832c71"/><rect x="122" y="343" width="176" height="11" fill="#171425"/><rect x="65" y="126" width="13" height="164" fill="#a1e670"/><rect x="264" y="252" width="77" height="27" fill="#4ec1c7"/><rect x="318" y="223" width="3" height="92" fill="#f0b68a"/><rect x="180" y="2" width="75" height="76" fill="#f3d36e"/><rect x="114" y="40" width="189" height="4" fill="#da1f3e"/><rect x="84" y="74" width="25" height="168" fill="#2a1948"/><circle cx="68" cy="365" r="7" fill="#f10f90"/><rect x="300" y="251" width="17" height="2" fill="#e61105"/><circle cx="201" cy="202" r="8" fill="#c5cb4e"/><rect x="342" y="3" width="7" height="153" fill="#663964"/><rect x="241" y="353" width="106" height="1" fill="#410a56"/><rect x="47" y="193" width="6" height="153" fill="#9c99fe"/><rect x="204" y="231" width="51" height="113" fill="#13c3bd"/><rect x="307" y="281" width="6" height="52" fill="#df50c5"/><rect x="235" y="321" width="18" height="22" fill="#0fa658"/><rect x="46" y="11" width="28" height="122" fill="#d0279e"/><rect x="340" y="54" width="13" height="5" fill="#b744eb"/><rect x="176" y="333" width="32" height="7" fill="#890dc5"/><rect x="340" y="100" width="2" height="171" fill="#e897ca"/><rect x="10" y="209" width="4" height="61" fill="#a68cb8"/><rect x="300" y="335" width="25" height="9" fill="#34aed0"/><rect x="315" y="106" width="30" height="79" fill="#fea85b"/><rect x="33" y="150" width="4" height="171" fill="#4857b7"/><rect x="149" y="307" width="18" height="33" fill="#7ec33c"/><rect x="304" y="133" width="44" height="85" fill="#948d0d"/><rect x="320" y="65" width="1" height="167" fill="#ca88fd"/><rect x="338" y="134" width="16" height="94" fill="#3dce43"/><rect x="351" y="257" width="3" height="9" fill="#ee95ba"/><circle cx="301" cy="296" r="2" fill="#b4e3a9"/><rect x="352" y="279" width="2" height="14" fill="#afd416"/><rect x="210" y="285" width="6" height="50" fill="#df99d3"/><rect x="149" y="295" width="43" height="1" fill="#240f20"/><rect x="338" y="258" width="1" height="16" fill="#3634bb"/><rect x="318" y="82" width="13" height="138" fill="#03efd8"/><rect x="286" y="317" width="34" height="1" fill="#b3eefe"/><rect x="94" y="154" width="6" height="111" fill="#e9fe71"/><rect x="346" y="138" width="8" height="84" fill="#15a706"/><rect x="267" y="79" width="6" height="66" fill="#e9f02e"/><rect x="339" y="186" width="4" height="27" fill="#45c01f"/><rect x="310" y="236" width="8" height="99" fill="#7042ee"/><rect x="310" y="102" width="10" height="1" fill="#dabbb1"/><rect x="344" y="28" width="1" height="104" fill="#73243e"/><rect x="295" y="342" width="14" height="1" fill="#9ae91b"/><rect x="305" y="244" width="2" height="22" fill="#a604de"/><rect x="108" y="258" width="9" height="83" fill="#943b54"/></svg>'
        return svg

    def predict(self, prompt: str) -> str:
        best_score = 0.0
        best_img = ''
        best_svg = ''
        imgs = PIPE(prompt + 'Scenic, landscapes, abstract, fashion, high, best, quality, contrast, prefessional, 8k, vector', height=384, width=384, num_inference_steps=40, num_images_per_prompt=6)
        for img in imgs.images:
            svg = self.bitmap_to_svg_layered (img)
            img = self.svg_to_png (svg)
            score = self.svgMetric (img, prompt)
            if score > best_score:
                best_score = score
                best_img = img
                best_svg = svg
        print(best_score)
        return best_svg


#kaggle_evaluation.test(Model)


model = Model()

for i in range(len(train[:3])):
    idx, prompt = train['id'][i], [train['description'][i]]
    dfq = quest[quest['id']==idx].reset_index(drop=True)
    for i in range(len(dfq)):
        prompt.append(dfq['question'][i] + ' ' + str(dfq['answer'][i]))
    prompt = '\n'.join(prompt) + '\n'
    start_time = time.time()  # Record start time
    svg = model.predict(prompt)
    end_time = time.time()    # Record end time
    elapsed_time = end_time - start_time # Calculate elapsed time
    print(f"Prediction time for description '{prompt}...': {elapsed_time:.4f} seconds")
    try:
        display(SVG(svg))
    except Exception as e:
        print(e)
        continue

