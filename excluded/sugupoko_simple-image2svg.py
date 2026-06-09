pip install vtracer cairosvg


import torch
import numpy as np
from PIL import Image
from io import BytesIO
import vtracer  # 事前にインストール済みであること
import cairosvg
import io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


# Tensor と PIL 画像の相互変換
def tensor2pil(image):
    """TensorをPIL Imageに変換"""
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2tensor(image):
    """PIL ImageをTensorに変換"""
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def png_to_svg(png_image: Image.Image,
               hierarchical: str = "stacked",
               mode: str = "spline",
               filter_speckle: int = 4,
               color_precision: int = 6,
               layer_difference: int = 16,
               corner_threshold: int = 60,
               length_threshold: float = 4.0,
               max_iterations: int = 10,
               splice_threshold: int = 45,
               path_precision: int = 3) -> str:
    """
    PNG画像(PIL Image)をSVG文字列に変換する関数
    
    パラメータ:
      png_image        : 入力画像（PIL Image）
      hierarchical     : "stacked" または "cutout"（デフォルトは "stacked"）
      mode             : "spline", "polygon", "none" のいずれか（デフォルトは "spline"）
      filter_speckle   : フィルター用パラメータ（デフォルト 4）
      color_precision  : 色の精度（デフォルト 6）
      layer_difference : レイヤー間の差分（デフォルト 16）
      corner_threshold : コーナーの閾値（デフォルト 60）
      length_threshold : 線の長さの閾値（デフォルト 4.0）
      max_iterations   : 最大反復回数（デフォルト 10）
      splice_threshold : スプライスの閾値（デフォルト 45）
      path_precision   : パスの精度（デフォルト 3）
      
    戻り値:
      SVG形式の文字列
    """
    # RGBAチェック：alphaチャネルがなければ追加
    if png_image.mode != 'RGBA':
        alpha = Image.new('L', png_image.size, 255)
        png_image.putalpha(alpha)
    
    # PIL画像をTensorに変換（vtracer内部ではTensor→PIL変換を使うため）
    tensor_image = pil2tensor(png_image)
    tensor_image = torch.unsqueeze(tensor_image, 0)  # バッチ次元を追加（1画像分）
    
    # 変換のためにTensorをPIL画像に戻す
    pil_img = tensor2pil(tensor_image)
    
    # 画像のピクセルデータとサイズを取得
    pixels = list(pil_img.getdata())
    size = pil_img.size
    
    # vtracerを使ってSVG文字列に変換
    svg_str = vtracer.convert_pixels_to_svg(
        pixels,
        size=size,
        colormode="color",
        hierarchical=hierarchical,
        mode=mode,
        filter_speckle=filter_speckle,
        color_precision=color_precision,
        layer_difference=layer_difference,
        corner_threshold=corner_threshold,
        length_threshold=length_threshold,
        max_iterations=max_iterations,
        splice_threshold=splice_threshold,
        path_precision=path_precision,
    )
    
    return svg_str



# 各バリエーションごとのパラメータ設定（情報量削減の度合いを調整）
variants = {
    "high": {  # デフォルトパラメータ（情報量最大）
        "filter_speckle": 4,
        "color_precision": 6,
        "layer_difference": 16,
        "corner_threshold": 60,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3
    },
    "mid-high": {  # やや情報削減
        "filter_speckle": 6,
        "color_precision": 5,
        "layer_difference": 16,
        "corner_threshold": 65,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3
    },
    "mid-low": {  # さらに情報削減
        "filter_speckle": 8,
        "color_precision": 4,
        "layer_difference": 16,
        "corner_threshold": 70,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3
    },
    "low": {  # 最も情報削減
        "filter_speckle": 10,
        "color_precision": 3,
        "layer_difference": 16,
        "corner_threshold": 75,
        "length_threshold": 4.0,
        "max_iterations": 10,
        "splice_threshold": 45,
        "path_precision": 3
    }
}



import glob
from PIL import Image
import io
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cairosvg

# 入力画像の一覧から1枚を選択
img_list = glob.glob("/kaggle/input/opencv-samples-images/data/*")

for input_path in img_list:
    try:
        png_img = Image.open(input_path)
        
        # オリジナル＋4変換結果の合計5枚を1行に並べる
        fig, axes = plt.subplots(1, 5, figsize=(20, 4))
        
        # 1枚目：オリジナル画像の表示
        axes[0].imshow(png_img)
        axes[0].set_title("original")
        axes[0].axis('off')
        
        # 2枚目以降：variants それぞれの変換結果
        for ax, (label, params) in zip(axes[1:], variants.items()):
            # SVG変換（png_to_svgは事前に定義済みの関数）
            svg_result = png_to_svg(
                png_img,
                hierarchical="stacked",
                mode="spline",
                filter_speckle=params["filter_speckle"],
                color_precision=params["color_precision"],
                layer_difference=params["layer_difference"],
                corner_threshold=params["corner_threshold"],
                length_threshold=params["length_threshold"],
                max_iterations=params["max_iterations"],
                splice_threshold=params["splice_threshold"],
                path_precision=params["path_precision"],
            )
            
            # SVGをPNGに変換
            png_bytes = cairosvg.svg2png(bytestring=svg_result)
            png_buffer = io.BytesIO(png_bytes)
            img_variant = mpimg.imread(png_buffer, format='png')
    
            # SVGの情報量（バイト長）を計測
            info_amount = len(svg_result.encode('utf-8'))
            
            # 画像表示とタイトル設定（ラベルと情報量）
            ax.imshow(img_variant)
            ax.set_title(f"{label}\n{info_amount} byte")
            ax.axis('off')
        
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"An error occurred processing {input_path}: {e}")
        continue




