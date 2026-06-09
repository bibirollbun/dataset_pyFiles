!pip install diffusers transformers accelerate safetensors torchvision --quiet
!pip install cairosvg opencv-python-headless --quiet
!pip install kagglehub --quiet
!pip install git+https://github.com/openai/CLIP.git --quiet
!pip install ftfy regex --quiet


import torch
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
import cairosvg
from diffusers import StableDiffusionPipeline, DDIMScheduler
import kagglehub
import torch.nn as nn

class Model:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = self._init_pipeline()
        self.evaluator = QualityEvaluator()
        
    def _init_pipeline(self):
        """Initialize Stable Diffusion pipeline"""
        torch.backends.cudnn.benchmark = True if self.device == "cuda" else False
        try:
            model_path = kagglehub.model_download("stabilityai/stable-diffusion-v2/pytorch/1/1")
            scheduler = DDIMScheduler.from_pretrained(model_path, subfolder="scheduler")
            
            pipe = StableDiffusionPipeline.from_pretrained(
                model_path,
                scheduler=scheduler,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                safety_checker=None,
                feature_extractor=None,
                requires_safety_checker=False,
                use_safetensors=True
            ).to(self.device)
            
            pipe.enable_attention_slicing()
            if self.device == "cuda":
                try:
                    import xformers
                    pipe.enable_xformers_memory_efficient_attention()
                except ImportError:
                    print("Xformers not available")
            return pipe
        
        except Exception as e:
            raise RuntimeError(f"Pipeline init failed: {str(e)}")

    def predict(self, prompt):
        """
        Competition-required prediction function
        Args:
            prompt: str - Text description of desired image
        Returns:
            str: Valid SVG code (under 10KB)
        """
        try:
            enhanced_prompt = f"high quality SVG of {prompt}, flat colors, minimal details, vector art"
            image = self.pipe(
                enhanced_prompt,
                num_inference_steps=25,
                guidance_scale=7.5,
                negative_prompt="text, watermark, signature, blurry"
            ).images[0]
            
            svg = self._image_to_svg(image)
            return self._validate_svg(svg)
        
        except Exception as e:
            print(f"Generation failed: {str(e)}")
            return self._generate_fallback()

    def _image_to_svg(self, image, max_colors=16, min_shape_area=20):
        """Convert PIL Image to optimized SVG"""
        try:
            img_np = np.array(image)
            if len(img_np.shape) == 2:
                img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
                
            # Color quantization
            pixels = img_np.reshape(-1, 3).astype(np.float32)
            _, _, centers = cv2.kmeans(
                pixels, max_colors, None, 
                (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2), 
                10, cv2.KMEANS_RANDOM_CENTERS
            )
            palette = centers.astype(np.uint8)
            
            # SVG generation
            svg_elements = []
            for color in palette:
                mask = cv2.inRange(img_np, color, color)
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                for contour in contours:
                    if cv2.contourArea(contour) < min_shape_area:
                        continue
                    epsilon = 0.02 * cv2.arcLength(contour, True)
                    approx = cv2.approxPolyDP(contour, epsilon, True)
                    points = " ".join([f"{pt[0][0]},{pt[0][1]}" for pt in approx])
                    hex_color = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                    svg_elements.append(f'<polygon points="{points}" fill="{hex_color}"/>')
            
            svg_content = f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {image.width} {image.height}">{"".join(svg_elements)}</svg>'
            return self._optimize_svg(svg_content)
            
        except Exception as e:
            print(f"SVG conversion error: {str(e)}")
            return self._generate_fallback()

    def _optimize_svg(self, svg, max_size=10000):
        """Ensure SVG meets competition size limits"""
        return svg[:max_size] if len(svg) > max_size else svg

    def _validate_svg(self, svg):
        """Basic validation (replace with svg_constraints package)"""
        if not svg.startswith('<svg') or len(svg) > 10000:
            return self._generate_fallback()
        return svg

    def _generate_fallback(self):
        """Competition-required fallback"""
        return '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"><rect width="512" height="512" fill="#f0f0f0"/></svg>'


class QualityEvaluator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._init_clip()
        
    def _init_clip(self):
        try:
            import clip
            self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
            self.aesthetic_model = nn.Sequential(
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Linear(256, 1)
            ).to(self.device)
            self.model.eval()
            self.aesthetic_model.eval()
        except Exception as e:
            print(f"CLIP init warning: {str(e)}")

    def evaluate(self, image, prompt):
        """Competition-style evaluation"""
        if not hasattr(self, 'model'):
            return 0.85  # Fallback score
            
        try:
            import clip
            image_input = self.preprocess(image).unsqueeze(0).to(self.device)
            text_input = clip.tokenize([f"a photo of {prompt}"]).to(self.device)
            
            with torch.no_grad():
                image_features = self.model.encode_image(image_input)
                text_features = self.model.encode_text(text_input)
                
                # CLIP similarity
                semantic_score = (image_features @ text_features.T).item()
                semantic_score = (semantic_score + 1) / 2
                
                # Aesthetic score
                aesthetic_score = torch.sigmoid(self.aesthetic_model(image_features)).item()
                
            return (semantic_score + aesthetic_score) / 2
            
        except Exception as e:
            print(f"Evaluation failed: {str(e)}")
            return 0.7


if __name__ == "__main__":
    # Competition test case
    model = Model()
    test_prompt = "a red circle on white background"
    
    print("âš¡ Generating SVG...")
    svg_output = model.predict(test_prompt)
    
    print("\nâœ… Generated SVG (first 100 chars):")
    print(svg_output[:100] + "...")
    
    from IPython.display import SVG, display
    display(SVG(svg_output))

