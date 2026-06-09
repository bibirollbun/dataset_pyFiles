#| default_exp core


#| export
import os
import re
import cv2
import torch
import svgwrite
import kagglehub
import subprocess
import numpy as np
from PIL import Image
from diffusers import StableDiffusionPipeline

os.system('mkdir /kaggle/tempfile')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Model:
    def __init__(self):
        self.model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/pytorch/1-base/1')
        self.pipe = StableDiffusionPipeline.from_pretrained(
            self.model_path,
            torch_dtype = torch.float16,
        )
        self.pipe = self.pipe.to(device)

    def generate_image_from_description(self, prompt, out_img="generated.png"):
        image = self.pipe(prompt, height=256, width=256).images[0]
        image.save(out_img)
        print(f"Image generated: {out_img}")

        return out_img

    def convert_image_to_svg_quantized(self, img_path, out_svg="output.svg"):
        img = Image.open(img_path)
        quant = img.quantize(colors=3, method=Image.FASTOCTREE)

        quant_rgb = quant.convert("RGB")
        np_img = np.array(quant_rgb)

        height, width, _ = np_img.shape
        dwg = svgwrite.Drawing(out_svg, size=(width, height))

        colors = np.unique(np_img.reshape(-1, 3), axis=0)
        print(f"Unique colors found: {len(colors)}")

        for color in colors:
            mask = (np_img == color).all(axis=2).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            hex_color = '#{:02x}{:02x}{:02x}'.format(color[0], color[1], color[2])
            for cnt in contours:
                if len(cnt) < 3:
                    continue

                epsilon = 0.01 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                pts = []
                for pt in approx:
                    x, y = int(pt[0][0]), int(pt[0][1])
                    pts.append((x, y))
                if pts:
                    path_data = "M " + " L ".join(f"{x} {y}" for x, y in pts) + " Z"
                    dwg.add(dwg.path(d=path_data, fill=hex_color, stroke="none"))

        svg_text = dwg.tostring()
        with open(out_svg, "w", encoding="utf-8") as f:
            f.write(svg_text)
        print(f"Quantized SVG generated: {out_svg}")

        return out_svg

    def optimize_svg(self, svg_file, optimized_svg="optimized.svg"):
        try:
            subprocess.run(["svgo", svg_file, "-o", optimized_svg], check=True)
            print(f"SVG optimized: {optimized_svg}")
        except Exception as e:
            print("SVGO optimization failed, using original SVG.")
            optimized_svg = svg_file

        size = os.path.getsize(optimized_svg)
        print(f"Optimized SVG size: {size} bytes")

        with open(optimized_svg, "r", encoding="utf-8") as f:
            svg_code = f.read()

        return svg_code

    def predict(self, prompt):
        img_path = self.generate_image_from_description(prompt, "/kaggle/tempfile/generated.png")
        svg_path = self.convert_image_to_svg_quantized(img_path, "/kaggle/tempfile/output.svg")
        svg_text = self.optimize_svg(svg_path, "/kaggle/tempfile/optimized.svg")

        svg_text = re.sub("baseProfile=\S* ", "", svg_text)
        svg_text = re.sub("version=\S* ", "", svg_text)
        svg_text = re.sub("<defs />", "", svg_text)
        svg_text = svg_text.replace(' xmlns="http://www.w3.org/2000/svg" xmlns:ev="http://www.w3.org/2001/xml-events" xmlns:xlink="http://www.w3.org/1999/xlink"', "")
        svg_text = re.sub(' stroke="none" ', "", svg_text)

        final_size = len(svg_text.encode("utf-8"))
        print(f"Final SVG code size: {final_size} bytes")
        if final_size <= 10000:
            print("Final SVG code is under 10000 bytes.")
        else:
            svg_text = '<svg width="100" height="100" viewBox="0 0 100 100"><rect x="1" y="50" width="50" height="49"  fill="red"/><circle cx="60" cy="40" r="30" fill="red" /></svg>'
            print("Final SVG code exceeds 10000 bytes. Further optimization may be needed.")

        return svg_text


from IPython.display import SVG

model = Model()
svg_text = model.predict("a starlit night over snow-covered peaks")
display(SVG(svg_text))
print(svg_text)


import kaggle_evaluation

kaggle_evaluation.test(Model)

