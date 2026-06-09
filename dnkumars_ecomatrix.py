!pip -q install "google-genai>=1.33.0" Pillow "elevenlabs>=2.14.0" fal-client reportlab requests
!pip -q install elevenlabs gTTS


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")
ELEVENLABS_API_KEY = user_secrets.get_secret("ELEVENLABS_API_KEY")
assert GEMINI_API_KEY, "Missing GOOGLE_API_KEY Kaggle secret"
assert ELEVENLABS_API_KEY, "Missing ELEVENLABS_API_KEY Kaggle secret"


import io
import os
import json
import time
import base64
import pathlib
import textwrap
import requests
from io import BytesIO
from typing import List, Dict, Any

from PIL import ImageOps
from PIL import Image as PILImage
from IPython.display import display, Markdown, Audio, Image  # keep for notebook display

from google import genai
from google.genai import types as gtypes
from elevenlabs.client import ElevenLabs


gclient = genai.Client(api_key=GEMINI_API_KEY)
MODEL_TEXT = "gemini-2.5-flash"
MODEL_IMAGE = "gemini-2.5-flash-image-preview"  # aka Nano Banana


def extract_images_from_gemini(response) -> List[PILImage.Image]:
    """Extract PIL Images from a Gemini response (interleaved parts)."""
    imgs = []
    for cand in getattr(response, "candidates", []) or []:
        for part in getattr(cand.content, "parts", []) or []:
            # Part may be text or inline_data (image)
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                try:
                    img = Image.open(BytesIO(part.inline_data.data))
                    imgs.append(img)
                except Exception:
                    pass
    return imgs

def show_and_save(imgs: List[PILImage.Image], basename: str) -> List[str]:
    """Display images and save each to /kaggle/working/; returns file paths."""
    paths = []
    pathlib.Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        display(img)
        path = f"/kaggle/working/{basename}_{i+1}.png"
        img.save(path, "PNG")
        paths.append(path)
    return paths

def stitch_vertical(images: List[PILImage.Image], padding: int = 10, bg=(255,255,255)) -> PILImage.Image:
    """Combine images vertically into one long webcomic image."""
    widths = [im.width for im in images]
    max_w = max(widths)
    heights = [im.height for im in images]
    total_h = sum(heights) + padding*(len(images)-1)
    canvas = Image.new("RGB", (max_w, total_h), bg)
    y = 0
    for idx, im in enumerate(images):
        if im.width < max_w:
            im = ImageOps.pad(im, (max_w, im.height), color=bg)
        canvas.paste(im, (0, y))
        y += im.height + (padding if idx < len(images)-1 else 0)
    return canvas

def images_to_pdf(images: List[PILImage.Image], out_path: str = "/kaggle/working/ecomatrix_story.pdf"):
    rgb_imgs = [im.convert("RGB") for im in images]
    if not rgb_imgs:
        return None
    if len(rgb_imgs) == 1:
        rgb_imgs[0].save(out_path, "PDF")
    else:
        rgb_imgs[0].save(out_path, "PDF", save_all=True, append_images=rgb_imgs[1:])
    return out_path

def download_image(url: str) -> PILImage.Image:
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return Image.open(BytesIO(r.content))

def to_data_uri(pil_img: PILImage.Image, fmt="PNG") -> str:
    buf = BytesIO()
    pil_img.save(buf, format=fmt)
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/{fmt.lower()};base64,{b64}"




def suggest_idea(topic: str = "youth climate action") -> str:
    prompt = f"Suggest a single-sentence, kid-friendly comic idea about climate solutions; upbeat and specific. Topic: {topic}"
    resp = gclient.models.generate_content(model=MODEL_TEXT, contents=prompt)
    return resp.text.strip()

idea = suggest_idea("urban gardening and recycling robots")
print("Suggested idea:", idea)



url = "https://images.unsplash.com/photo-1529626455594-4ff0802cfb7e?w=512&h=512&fit=crop"
resp = requests.get(url)
img = PILImage.open(BytesIO(resp.content)).convert("RGB")
path = pathlib.Path("/kaggle/working/sample_face.png")
img.save(path)
FACE_IMG_PATH = str(path)
print("Saved at:", FACE_IMG_PATH)
display(Image(filename=FACE_IMG_PATH))


def describe_person_for_comic(image_path: str) -> str:
    assert image_path and pathlib.Path(image_path).exists(), "Set FACE_IMG_PATH to a valid image path"
    img = PILImage.open(image_path)   # use PILImage here
    prompt = textwrap.dedent("""
        Analyze the photo and return a concise character appearance description for a comic.
        Include hair style/color, eye color, notable features, vibes, and outfit hints.
        Reply in one sentence without naming the person.
    """)
    resp = gclient.models.generate_content(model=MODEL_TEXT, contents=[prompt, img])
    return resp.text.strip()


hero_desc = describe_person_for_comic(FACE_IMG_PATH)
print(hero_desc)


def generate_three_part_story(core_prompt: str, characters: list) -> Dict[str, str]:
    characters_block = "\n".join([json.dumps(c, ensure_ascii=False) for c in characters])
    sys_prompt = textwrap.dedent(f"""        You are writing a hopeful climate comic for kids. Create a 3-part story JSON with keys page1, page2, page3.
        Each page: 2-4 sentences, simple, vivid, and include one small, accurate science fact.
        Keep characters consistent with these descriptors:
        {characters_block}
        Core idea: {core_prompt}
        Reply ONLY with JSON: {{"page1":"...","page2":"...","page3":"..."}}
    """)
    resp = gclient.models.generate_content(model=MODEL_TEXT, contents=sys_prompt)
    text = resp.text.strip()
    try:
        data = json.loads(text)
        return {k: data.get(k, "") for k in ["page1", "page2", "page3"]}
    except Exception:
        # Fallback: attempt to extract json block
        import re
        m = re.search(r'\{[\s\S]*\}', text)
        data = json.loads(m.group(0)) if m else {"page1": text, "page2": "", "page3": ""}
        return data

# Example characters
characters = [
    {"name": "Asha", "role": "Hero", "personality": "curious, upbeat", "powers": "robot tinkering", "appearance": "short curly black hair, brown eyes, bright hoodie"},
    {"name": "Bolt", "role": "Sidekick", "personality": "loyal, goofy", "powers": "recycling robot", "appearance": "tiny metal body with green LEDs"},
]
story = generate_three_part_story(core_prompt=suggest_idea(), characters=characters)
story


def extract_images_from_gemini(response) -> List[PILImage.Image]:
    """Extract PIL Images from a Gemini response (interleaved parts)."""
    imgs = []
    for cand in getattr(response, "candidates", []) or []:
        for part in getattr(cand.content, "parts", []) or []:
            if getattr(part, "inline_data", None) and getattr(part.inline_data, "data", None):
                try:
                    img = PILImage.open(BytesIO(part.inline_data.data))
                    imgs.append(img)
                except Exception:
                    pass
    return imgs

def show_and_save(imgs: List[PILImage.Image], basename: str) -> List[str]:
    """Display images and save each to /kaggle/working/; returns file paths."""
    paths = []
    pathlib.Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs):
        display(img)
        path = f"/kaggle/working/{basename}_{i+1}.png"
        img.save(path, "PNG")
        paths.append(path)
    return paths

def gen_strip_4panel_gemini(core_prompt: str, characters: list, include_fact: bool = True) -> List[PILImage.Image]:
    characters_text = "; ".join([f"{c.get('name')}: {c.get('appearance','')}" for c in characters])
    fact_line = "Include one short, accurate science fact in a small caption." if include_fact else ""
    prompt = textwrap.dedent(f"""
        Create a SINGLE image formatted as a 4-panel comic strip (2x2 grid) in a 16:9 canvas.
        Panels tell a coherent story about: "{core_prompt}".
        Maintain character consistency across panels: {characters_text}.
        Clean speech bubbles, legible text, high-contrast ink lines, bright kid-friendly color.
        {fact_line}
    """).strip()
    resp = gclient.models.generate_content(model=MODEL_IMAGE, contents=[prompt])
    return extract_images_from_gemini(resp)

def gen_pages_4x3_gemini(pages: Dict[str, str]) -> List[PILImage.Image]:
    imgs: List[PILImage.Image] = []
    for i in range(1, 4):
        scene = pages.get(f"page{i}", "")
        prompt = textwrap.dedent(f"""
            Create a full-page comic illustration with clean speech bubbles (no typos) in a 4:3 aspect ratio.
            Scene description: {scene}
            Maintain character consistency from prior descriptions.
        """).strip()
        resp = gclient.models.generate_content(model=MODEL_IMAGE, contents=[prompt])
        imgs += extract_images_from_gemini(resp)
        time.sleep(5)  # rate-limit friendly
    return imgs
def apply_comic_style_with_gemini(img: PILImage.Image, style: str = "Manga", line: str = "thick", shading: str = "halftone dots") -> List[PILImage.Image]:
    style_prompt = f"Transform this image to {style} comic style with {line} line work and {shading}. Keep text legible, preserve composition."
    resp = gclient.models.generate_content(model=MODEL_IMAGE, contents=[style_prompt, img])
    return extract_images_from_gemini(resp)

def edit_with_gemini(img: PILImage.Image, instruction: str) -> List[PILImage.Image]:
    resp = gclient.models.generate_content(model=MODEL_IMAGE, contents=[instruction, img])
    return extract_images_from_gemini(resp)



strip_imgs = gen_strip_4panel_gemini(core_prompt=suggest_idea(), characters=characters)
show_and_save(strip_imgs, "strip")
page_imgs = gen_pages_4x3_gemini(story)
show_and_save(page_imgs, "page")


def apply_comic_style_with_gemini(
    img: PILImage.Image,
    style: str = "Manga",
    line: str = "thick",
    shading: str = "halftone dots"
) -> list[PILImage.Image]:
    style_prompt = (
        f"Transform this image to {style} comic style with {line} line work and {shading}. "
        "Keep text legible, preserve composition."
    )
    resp = gclient.models.generate_content(model=MODEL_IMAGE, contents=[style_prompt, img])
    return extract_images_from_gemini(resp)


base_img = PILImage.open("/kaggle/working/page_3.png")
# Apply the comic style using Nano Banana
stylized = apply_comic_style_with_gemini(
    base_img,
    style="Anime",
    line="thick",
    shading="screentone"
)
show_and_save(stylized, "styled")


def edit_with_gemini(img: PILImage.Image, instruction: str) -> list[PILImage.Image]:
    resp = gclient.models.generate_content(model=MODEL_IMAGE, contents=[instruction, img])
    return extract_images_from_gemini(resp)

base_img = PILImage.open("/kaggle/working/page_2.png")
night = edit_with_gemini(base_img, "Turn scene to night with cool moonlight. Keep characters and layout unchanged.")
show_and_save(night, "night")


r = requests.get(
    "https://api.elevenlabs.io/v1/user",
    headers={"xi-api-key": ELEVENLABS_API_KEY}
)
print(r.status_code, r.text[:400])



API_KEY = ELEVENLABS_API_KEY
VOICE_ID = "JBFqnCBsd6RMkjVDRZzb"     # replace with your preferred voice
MODEL_ID = "eleven_multilingual_v2"  # or "eleven_turbo_v2_5", "eleven_v3"


def narrate_with_elevenlabs(
    text: str, 
    out_path: str = "/kaggle/working/narration.mp3",
    api_key: str = ELEVENLABS_API_KEY,
    voice_id: str = "your_voice_id_here",  # replace with your ElevenLabs voice ID
    model_id: str = "eleven_multilingual_v1"
) -> str:
    """
    Generate narration for the given text.
    Returns the output file path.
    Falls back to gTTS if ElevenLabs fails.
    """
    try:
        if api_key:
            from elevenlabs import ElevenLabs
            client = ElevenLabs(api_key=api_key)
            stream = client.text_to_speech.convert(
                voice_id=voice_id,
                model_id=model_id,
                text=text,
                output_format="mp3_44100_128",
            )
            # Concatenate audio chunks
            audio_bytes = b"".join(
                chunk if isinstance(chunk, (bytes, bytearray))
                else chunk.read() if hasattr(chunk, "read")
                else bytes(chunk)
                for chunk in stream
            )
            with open(out_path, "wb") as f:
                f.write(audio_bytes)
            display(Audio(out_path))
            print("âœ… ElevenLabs TTS succeeded:", out_path)
            return out_path
        else:
            raise ValueError("No ElevenLabs API key provided, falling back to gTTS.")
    except Exception as e:
        print("âš  ElevenLabs TTS failed:", e)
        try:
            from gtts import gTTS
            tts = gTTS(text=text, lang="en")
            tts.save(out_path)
            display(Audio(out_path))
            print("âœ… gTTS succeeded:", out_path)
            return out_path
        except Exception as e2:
            print("â�Œ gTTS also failed:", e2)
            raise



p1 = narrate_with_elevenlabs(story["page1"], out_path="/kaggle/working/page1.mp3")


def show_and_save(imgs: list[PILImage.Image], basename: str) -> list[str]:
    paths = []
    pathlib.Path("/kaggle/working").mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(imgs or []):
        # Coerce to PIL if something odd slipped in
        if not isinstance(img, PILImage.Image):
            raise TypeError(f"Expected PIL.Image, got {type(img)} at index {i}")
        display(img)
        path = f"/kaggle/working/{basename}_{i+1}.png"
        img.save(path, "PNG")
        paths.append(path)
    return paths

def stitch_vertical(images: list[PILImage.Image], padding: int = 10, bg=(255, 255, 255)) -> PILImage.Image:
    if not images:
        raise ValueError("No images provided to stitch_vertical")
    widths = [im.width for im in images]
    max_w = max(widths)
    heights = [im.height for im in images]
    total_h = sum(heights) + padding * (len(images) - 1)

    canvas = PILImage.new("RGB", (max_w, total_h), bg)
    y = 0
    for idx, im in enumerate(images):
        if im.width < max_w:
            im = ImageOps.pad(im, (max_w, im.height), color=bg)
        canvas.paste(im, (0, y))
        y += im.height + (padding if idx < len(images) - 1 else 0)
    return canvas

def images_to_pdf(images: list[PILImage.Image], out_path: str = "/kaggle/working/ecomatrix_story.pdf"):
    if not images:
        raise ValueError("No images provided to images_to_pdf")
    rgb_imgs = [im.convert("RGB") for im in images]
    if len(rgb_imgs) == 1:
        rgb_imgs[0].save(out_path, "PDF")
    else:
        rgb_imgs[0].save(out_path, "PDF", save_all=True, append_images=rgb_imgs[1:])
    return out_path


saved_paths = show_and_save(page_imgs, "page")
combined = stitch_vertical(page_imgs)
combined.save("/kaggle/working/webcomic_vertical.png", "PNG")
pdf_path = images_to_pdf(page_imgs, "/kaggle/working/ecomatrix_story.pdf")
pdf_path


idea = suggest_idea("kids upcycling plastic into garden gadgets")
chars = [
    {"name":"Asha","role":"Hero","appearance":"short curly black hair, brown eyes, bright hoodie"},
    {"name":"Bolt","role":"Sidekick","appearance":"tiny recycling robot with green LEDs"},
]
story = generate_three_part_story(idea, chars)
print("IDEA:", idea)
print("PAGE1:", story["page1"])
gen_source = "gemini"
# Generate images
imgs = []
if gen_source == "gemini":
    imgs = gen_pages_4x3_gemini(story)
styled = []
for im in imgs:
    out = apply_comic_style_with_gemini(im, style="Manga", line="thick", shading="halftone dots")
    styled.append(out[0] if out else im)
    time.sleep(2)
show_and_save(styled, "story_page")
narrate_with_elevenlabs(story["page1"])
combined = stitch_vertical(styled)
combined.save("/kaggle/working/webcomic_vertical.png", "PNG")
images_to_pdf(styled, "/kaggle/working/ecomatrix_story.pdf")

