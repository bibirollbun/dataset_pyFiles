!pip install pystac-client rasterio geopandas shapely matplotlib planetary-computer openai pillow


import os
from kaggle_secrets import UserSecretsClient
import planetary_computer
import pystac_client
import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import base64
from io import BytesIO
from openai import OpenAI


class Sentinel2Analysis:
    """
    Class to fetch, process, and analyze Sentinel-2 imagery over a specified bbox and date range.
    """
    def __init__(self, bbox, time_range, limit=1, thumb_size=(128, 128)):
        # STAC & Vision API setup
        self.user_secrets = UserSecretsClient()
        self.openai_key = self.user_secrets.get_secret("OPENAI_API_KEY")
        self.client = OpenAI(api_key=self.openai_key)
        self.catalog = pystac_client.Client.open(
            "https://planetarycomputer.microsoft.com/api/stac/v1"
        )
        # Search parameters
        self.bbox = bbox
        self.time_range = time_range 
        self.limit = limit
        # Thumbnail config
        self.thumb_size = thumb_size
        # Placeholders for scene and imagery
        self.item = None
        self.signed_item = None
        self.rgb_img = None
        self.img_uri = None

    def search_scene(self):
        """Search for a Sentinel-2 L2A scene and sign asset URLs."""
        search = self.catalog.search(
            collections=["sentinel-2-l2a"],
            bbox=self.bbox,
            datetime=self.time_range,
            limit=self.limit
        )
        items = list(search.get_items())
        if not items:
            raise RuntimeError("No scenes found. Adjust bbox or dates.")
        self.item = items[0]
        print(f"Selected scene: {self.item.id}")
        self.signed_item = planetary_computer.sign(self.item)

    def load_band(self, band):
        """Load a single band as a numpy array (float)."""
        href = self.signed_item.assets[band].href
        with rasterio.open(href) as src:
            return src.read(1).astype(float)

    def create_rgb_composite(self):
        """Stack B04, B03, B02 into a true-color RGB image (uint8)."""
        bands = {b: self.load_band(b) for b in ['B04', 'B03', 'B02']}
        arr = np.dstack([bands['B04'], bands['B03'], bands['B02']])
        # Normalize percentiles
        def norm(a):
            p2, p98 = np.percentile(a, (2, 98))
            return np.clip((a - p2) / (p98 - p2), 0, 1)
        rgb_norm = norm(arr)
        self.rgb_img = (rgb_norm * 255).astype(np.uint8)
        plt.figure(figsize=(6,6))
        plt.imshow(self.rgb_img)
        plt.title('True-Color RGB Composite')
        plt.axis('off')
        plt.show()

    def prepare_thumbnail(self):
        """Resize RGB image and encode as a base64 data URI."""
        img = Image.fromarray(self.rgb_img).resize(self.thumb_size)
        buf = BytesIO()
        img.save(buf, format='PNG')
        buf.seek(0)
        b64 = base64.b64encode(buf.read()).decode('utf-8')
        self.img_uri = f"data:image/png;base64,{b64}"
        print("Thumbnail ready for Vision API")

    def analyze_vision(self):
        """Send thumbnail to GPT-4o Vision to describe land cover."""
        if not self.img_uri:
            raise RuntimeError("Thumbnail not prepared. Call prepare_thumbnail() first.")
        prompt = [
            {"type": "text",
             "text": (
                f"This image is from Sentinel-2 dataset {self.item.id} over Rome, Italy. "
                "Please describe visible surface features and land cover."
             )},
            {"type": "image_url", "image_url": {"url": self.img_uri}}
        ]
        resp = self.client.chat.completions.create(
            model="o4-mini",
            messages=[
                {"role": "system", "content": "You are a satellite imagery analyst."},
                {"role": "user", "content": prompt}
            ]
        )
        print(f"Model Name:{resp.model}")
        print(f"Output :\n {resp.choices[0].message.content}")

    def run(self):
        """Execute full pipeline end-to-end."""
        self.search_scene()
        # Single-band preview
        red = self.load_band('B04')
        plt.figure(figsize=(5,5))
        plt.imshow(red, cmap='gray')
        plt.title('Red Band (B04)')
        plt.axis('off')
        plt.show()
        # RGB composite
        self.create_rgb_composite()
        # Thumbnail + Vision analysis
        self.prepare_thumbnail()
        self.analyze_vision()





def driver():
    bbox = [12.3, 41.7, 12.7, 42.1]
    time_range = "2023-06-01/2023-07-31"
    analysis = Sentinel2Analysis(bbox=bbox, time_range=time_range)
    analysis.run()


driver()

