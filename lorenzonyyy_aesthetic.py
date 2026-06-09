#| default_exp core


#| export
import io
import torch
import clip
import numpy as np
from PIL import Image
import cairosvg
import torch.nn as nn
import kagglehub
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#| export
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
        predictor = AestheticPredictor(768)  # CLIP ViT-L-14 嵌入维度为 768
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
            image_features /= image_features.norm(dim=-1, keepdim=True)  # L2 归一化
            image_features = image_features.cpu().detach().numpy()
        score = self.predictor(torch.from_numpy(image_features).to('cuda:1').float())
        return score.item() / 10.0  # 缩放到 [0, 1]

def svg_to_png(svg_code: str, size: tuple = (384, 384)) -> Image.Image:
    """Converts an SVG string to a PNG image using CairoSVG."""
    if 'viewBox' not in svg_code:
        svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')
    png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
    return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


#| export
def get_score(df):

    # 初始化 aesthetic_score 列
    if 'aesthetic_score' not in df.columns:
        df['aesthetic_score'] = np.nan  

    evaluator = AestheticEvaluator()

    def compute_score(svg_code):
        try:
            image = svg_to_png(svg_code)
            return evaluator.score(image)
        except Exception as e:
            print(f"SVG 处理失败: {e}")
            return np.nan

    # 使用 apply 向量化计算评分
    df['aesthetic_score'] = df['cleaned_svg'].apply(compute_score)

    return df


# df = pd.read_csv('/kaggle/input/50kdata/50k_data_for_aesthetic.csv')


# df = get_score(df)


# df['aesthetic_score'].describe()


# train = df[df['aesthetic_score']>0.55]


# len(df[df['aesthetic_score']>0.55])


# len(df[df['aesthetic_score']>0.54])


# train.columns


# train = train.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])


# train = train.sort_values('aesthetic_score',ascending=False)


# #| export
# def display(df):

#     def svg_to_png(svg_code: str, size: tuple = (384, 384)):
#         # Ensure SVG has proper size attributes
#         if 'viewBox' not in svg_code:
#             svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

#         # Convert SVG to PNG
#         png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
#         return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)

#     df = df.head(15)

#     # display
#     fig = plt.figure(figsize=(12, 20),dpi=600)
#     # i从1开始（而非0）
#     for i, r in enumerate(df.itertuples(), 1):
#         plt.subplot(5, 3, i)
#         img = svg_to_png(r.cleaned_svg)
#         plt.imshow(img)
#         plt.axis('off')
#         plt.title(r.description, fontdict={'fontsize': 8})
        
#     return fig


# display(train[:15])


# df = df.sort_values('aesthetic_score',ascending=False)


# df.to_csv('/kaggle/working/50k_with_score.csv')


# df['aesthetic_score'].idxmax()


# def svg_to_png(svg_code: str, size: tuple = (384, 384)):
#     # Ensure SVG has proper size attributes
#     if 'viewBox' not in svg_code:
#         svg_code = svg_code.replace('<svg', f'<svg viewBox="0 0 {size[0]} {size[1]}"')

#     # Convert SVG to PNG
#     png_data = cairosvg.svg2png(bytestring=svg_code.encode('utf-8'))
#     return Image.open(io.BytesIO(png_data)).convert('RGB').resize(size)


# df['cleaned_svg'][12575]


# train.to_csv('/kaggle/working/score_0.55.csv')


# train_2 = df[df['aesthetic_score'] > 0.54]


# train_2.columns


# train_2 = train_2.drop(columns=['Unnamed: 0.1', 'Unnamed: 0'])


# train_2.to_csv('/kaggle/working/score_0.54.csv')

