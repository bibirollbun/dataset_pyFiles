from IPython.display import HTML

HTML('''
<div style="display:flex;flex-wrap:wrap;justify-content:center;gap:10px;">
  <iframe style="flex:1 1 45%;aspect-ratio:16/9;" src="https://www.youtube.com/embed/CS5c1k3gU6U"
    frameborder="0" allowfullscreen></iframe>
    
  <iframe style="flex:1 1 45%;aspect-ratio:16/9;" src="https://www.youtube.com/embed/BWp1ZfifPeM"
    frameborder="0" allowfullscreen></iframe>
</div>
''')



# Get a suitable cover image for this kaggle notebook! You may ignore this.  
import urllib.request

urllib.request.urlretrieve(
    "https://rohitconsultants.com/imgs/ghostwriter_logo.jpg",
    "/kaggle/working/cover.jpg"
)



# Get our api keys from kaggle secrets  
import os
from kaggle_secrets import UserSecretsClient

try:
    GROQ_API_KEY = UserSecretsClient().get_secret("GROQ_API_KEY_GHOSTWRITER")
    os.environ["GROQ_API_KEY"] = GROQ_API_KEY
    PEXELS_API_KEY = UserSecretsClient().get_secret("PEXELS_API_KEY")
    os.environ["PEXELS_API_KEY"] = PEXELS_API_KEY
    print("âœ… Setup and authentication complete.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'GROQ_API_KEY' to your Kaggle secrets. Details: {e}"
    )


# Use Groq API to create the content 
import requests


url = "https://api.groq.com/openai/v1/chat/completions"
headers = {"Authorization": f"Bearer {GROQ_API_KEY}"}


# https://console.groq.com/docs/rate-limits
data = {
    "model": "llama-3.1-8b-instant",
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a Christian devotional writer. "
                "Do not use markdown. No asterisks, no bold formatting. "
                "Output clean paragraphs separated by line breaks."
            )
        },
        {
            "role": "user",
            "content": (
                "Write a 160â€“200 word Christian devotional post for my blog at https://zeatz.in/bites. "
                "Begin with a scripture verse and reference the gospel reading. "
                "Use https://stpaul-florin.org/todays-readings-1 as your reference for the gospel for today. "
                "Encapsulate the verse inside <p>\â€œ html tag. "
                "Then write a simple, warm reflection that connects the message of the verse to everyday life. "
                "End with one practical takeaway for the day. "
                "Keep paragraphs short and formatted cleanly for a WordPress Post."
            )
        }
    ],
    "temperature": 0.7
}

response = requests.post(url, json=data, headers=headers)
resp = response.json()
print(response.status_code)
print(response.text)



# Extract content safely
if "choices" in resp:
    blog_content = resp["choices"][0]["message"]["content"]
else:
    raise Exception("Groq API error: No content generated.")

print(blog_content)



# Generate title and a summary text which will be used to fetch thumbnail for the blog 
!pip install -q protobuf==3.20.3

import re
import warnings
import contextlib
import sys
import os
from transformers import pipeline

# Silence tokenizer fork warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Silence Protobuf + CUDA noise
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore", message=".*cuFFT*")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------ GLOBAL WARNINGS SUPPRESSION ------------------ #

# Suppress warnings related to transformers / CUDA noise
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuFFT*")

# Redirect stderr temporarily for noisy libs
@contextlib.contextmanager
def suppress_stderr():
    with open(os.devnull, "w") as devnull:
        old_stderr = sys.stderr
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stderr = old_stderr

# ------------------ LOAD SUMMARIZER SAFELY ------------------ #

with suppress_stderr():
    try:
        title_gen = pipeline("summarization", model="facebook/bart-large-cnn")
    except Exception:
        title_gen = None
        print("âš ï¸� Summarizer not available, using templates only...")

# ------------------ TITLE + QUERY LOGIC ------------------ #

TEMPLATES = [
    "God Has a Plan for Your {}",
    "Your {} Is Safe in God's Hands",
    "Hope Is Already in Your {}",
    "Youâ€™re Not Lost â€” God Is Leading Your {}",
    "When You Canâ€™t See the Path, God Can",
    "God Is Already Working in Your {}",
    "Let Go of Worry â€” God Holds Your {}",
]

def generate_title(blog_text):
    clean = re.sub(r"<[^>]+>|&\w+;", "", blog_text)
    sentiment = "Future"
    low = clean.lower()

    if "hope" in low: sentiment = "Hope"
    elif "purpose" in low: sentiment = "Purpose"

    creative_titles = [t.format(sentiment) if "{}" in t else t for t in TEMPLATES]

    # Safe summarizer call
    if title_gen:
        with suppress_stderr():
            try:
                # short = title_gen(clean, max_length=18, min_length=6, do_sample=False)[0]['summary_text']
                # creative_titles.append(short.strip().rstrip("."))
                # Safe adaptive summarization
                if title_gen:
                    word_count = len(clean.split())
                    if word_count > 20:  # only summarize real paragraphs
                        try:
                            short = title_gen(
                                clean, 
                                max_length=min(18, word_count - 2), 
                                min_length=6,
                                do_sample=False
                            )[0]['summary_text']
                            creative_titles.append(short.strip().rstrip("."))
                        except Exception as e:
                            print("âš ï¸� Summarizer skipped:", e)

            except Exception:
                pass  # silently ignore summarizer failure

    return list(dict.fromkeys(creative_titles))[0]  # best pick

def generate_pexels_query(blog_text):
    low = blog_text.lower()
    keywords = []

    if "hope" in low: keywords += ["hope", "future", "sunrise"]
    if "faith" in low or "god" in low: keywords += ["light", "guidance", "peaceful"]
    if "uncertain" in low or "worry" in low: keywords += ["path", "horizon"]

    query = " ".join(sorted(set(keywords)))
    return query or "peaceful sunrise horizon light"

def analyze_blog(blog_text):
    with suppress_stderr():
        try:
            title = generate_title(blog_text)
            query = generate_pexels_query(blog_text)
            print("ğŸ“� Generated Title:", title)
            print("ğŸ”� Pexels Search Query:", query)
            return title, query
        except Exception as e:
            print("âš ï¸� Non-critical error:", e)
            print("â�¡ï¸� Falling back to safe defaults")
            return (
                "God Has a Good Future Waiting for You",
                "future horizon sunrise hope calm light"
            )

# ------------------ TEST IT ------------------ #
title, query = analyze_blog(blog_content)



# Get the thumbnail image from Pixels API for free
import requests, os
from PIL import Image
from io import BytesIO


headers = {"Authorization": PEXELS_API_KEY}
params = {"query": query, "per_page": 1}  # get one good image
resp = requests.get("https://api.pexels.com/v1/search", headers=headers, params=params)
resp.raise_for_status()

url = resp.json()['photos'][0]['src']['original']
img_data = requests.get(url).content
img = Image.open(BytesIO(img_data))
img = img.resize((800, 450), Image.LANCZOS)
img.save("blog_image.jpg")



# Use this if we want to break execution at any point in the notebook!

# print(blog_content)

# import sys
# sys.exit("Stopping execution here")



# Another option was to use a diffuser to generate a thumbnail image but the quality was questionable - 
# Hence we stuck to using Pexels API for free and better quality thumbnail image!

## check out image generation with free models 

# ============================================================
# ğŸš€ Free Image Generation using Stable Diffusion (Kaggle GPU)
# ============================================================

# !pip install -q torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
# !pip install -q diffusers transformers accelerate safetensors

# import torch
# from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
# from IPython.display import display

# device = "cuda" if torch.cuda.is_available() else "cpu"
# print("Running on:", device)

# pipe = StableDiffusionPipeline.from_pretrained(
#     "runwayml/stable-diffusion-v1-5",
#     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# )
# pipe = pipe.to(device)
# pipe.enable_attention_slicing()
# pipe.enable_vae_slicing()
# pipe.safety_checker = lambda images, clip_input: (images, [False] * len(images))
# pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

# def generate_image(prompt, width=800, height=448, steps=25, filename="image.png"):
#     print("Prompt:", prompt)
#     image = pipe(prompt, width=width, height=height, num_inference_steps=steps).images[0]
#     image.save(filename)
#     display(image)
#     return filename



# from transformers import pipeline

# # Brand Prompt Style Preset
# BRAND_STYLE = (
#     "modern christian tech theme, dark mode aesthetics, "
#     "clean gradients, abstract shapes, soft glowing light, "
#     "uplifting but minimalist, professional blog banner"
# )

# # Keyword-focused summarization to stay <77 tokens
# summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# def create_brand_prompt(blog_text):
#     summary = summarizer(blog_text, max_length=32, min_length=16, do_sample=False)[0]['summary_text']
#     final_prompt = f"{summary}, {BRAND_STYLE}, 16:9 aspect"
#     print("âœ¨ Final Image Prompt:", final_prompt)
#     return final_prompt

# # Main automation function
# def generate_banner_from_blog(blog_text, filename="banner.png"):
#     prompt = create_brand_prompt(blog_text)
#     img_path = generate_image(prompt, width=800, height=448, steps=25, filename=filename)
#     return img_path



# generate_banner_from_blog(blog_content, filename="ghostwriter.png")



# Get the WordPress website creds from kaggle secrets 
try:
    WP_SITE = UserSecretsClient().get_secret("WP_SITE")
    WP_USER = UserSecretsClient().get_secret("WP_USER")
    WP_PASSWORD = UserSecretsClient().get_secret("WP_PASSWORD")
    os.environ["WP_SITE"] = WP_SITE
    os.environ["WP_USER"] = WP_USER
    os.environ["WP_PASSWORD"] = WP_PASSWORD
    print("âœ… WP creds locked & loaded.")
except Exception as e:
    print(
        f"ğŸ”‘ Authentication Error: Please make sure you have added 'WP_USER' and 'WP_PASSWORD' to your Kaggle secrets. Details: {e}"
    )


# Upload media to WordPress 

import requests
from requests.auth import HTTPBasicAuth
import os

# Upload endpoint
media_url = f"{WP_SITE}/wp-json/wp/v2/media"

# Headers for upload
headers = {
    "Content-Disposition": f'attachment; filename="/kaggle/working/ghostwriter.png"',
    "Content-Type": "image/png"
}

# Read image binary
with open("/kaggle/working/blog_image.jpg", "rb") as img:
    img_data = img.read()

# Send upload request
response = requests.post(
    media_url,
    headers=headers,
    data=img_data,
    auth=HTTPBasicAuth(WP_USER, WP_PASSWORD)
)

# Check result
if response.status_code == 201:
    media = response.json()
    media_id = media["id"]
    media_url_response = media["source_url"]
    
    print("ğŸ“¤ Upload Successful!")
    print("ğŸ“Œ Media ID:", media_id)
    print("ğŸ”— Media URL:", media_url_response)
else:
    print("â�Œ Upload Failed")
    print(response.status_code, response.text)


# Create the WordPress post
WP_URL = f"{WP_SITE}/wp-json/wp/v2/posts"

import requests
from requests.auth import HTTPBasicAuth

# ğŸ‘‡ Create a DailyServing style title
import datetime
today = datetime.date.today().strftime("%B %d, %Y")
post_title = f"Daily Serving: {title}"

# ğŸ‘‡ Optional: wrap your content in a styled section
formatted_content = f"""
<p>{blog_content}</p>
<hr>
<p><em>Posted automatically by GhostWriter AI</em></p>
"""

payload = {
    "title": post_title,
    "content": formatted_content,
    "categories": 76,
    "tags": 86,
    "featured_media": media_id,
    "status": "publish",      # or "draft" if you want review
}

response_wp = requests.post(
    WP_URL,
    json=payload,
    auth=HTTPBasicAuth(WP_USER, WP_PASSWORD)
)

print("WP Status:", response_wp.status_code)
print("WP Response:", response_wp.json())


