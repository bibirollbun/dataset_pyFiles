!pip install --quiet pytesseract pillow gTTS transformers sentencepiece --upgrade
!apt-get update -qq && apt-get install -y -qq tesseract-ocr libtesseract-dev

import os
import json
from PIL import Image
import pytesseract
from gtts import gTTS
from IPython.display import Image as ShowImage, display

try:
    from transformers import pipeline
    HF = True
except:
    HF = False

INPUT_DIR = "/kaggle/input/study-images-keshar"           
AUTO_IMG_DIR = "/kaggle/working/auto_images"
MEMORY_FILE = "/kaggle/working/memory.json"
RESULTS_FILE = "/kaggle/working/results.json"

os.makedirs(AUTO_IMG_DIR, exist_ok=True)

# Copy all images from user dataset into working folder
for root, dirs, files in os.walk(INPUT_DIR):
    for f in files:
        if f.lower().endswith((".png",".jpg",".jpeg")):
            src = os.path.join(root, f)
            dst = os.path.join(AUTO_IMG_DIR, f)
            if not os.path.exists(dst):
                from shutil import copy2
                copy2(src, dst)

summarizer = None
if HF:
    try:
        summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    except:
        summarizer = None

def fallback(text):
    import re
    sents = [s.strip() for s in re.split(r"[.!?]", text) if s.strip()]
    return " ".join(sents[:4])

def summarize(text):
    if not text.strip(): return ""
    if summarizer:
        try:
            if len(text.split()) < 50: return text
            out = summarizer(text, max_length=120, min_length=30)
            return out[0]["summary_text"]
        except:
            return fallback(text)
    return fallback(text)

def ocr(path):
    img = Image.open(path).convert("L")
    t = pytesseract.image_to_string(img)
    return "\n".join([x.strip() for x in t.splitlines() if x.strip()])

def bullets(summary):
    parts = [s.strip() for s in summary.split('.') if len(s.strip())>10]
    return parts[:8]

def mcqs(summary):
    b = bullets(summary)
    q = []
    for i, s in enumerate(b[:3]):
        q.append({"id":i+1,"q":f"What is: {s[:60]}...?","options":["A","B","C","D"],"answer":"A"})
    return q

def audio(text, out):
    if not text.strip():
        open(out,"wb").close()
        return out
    try:
        gTTS(text=text, lang="en").save(out)
        return out
    except:
        open(out,"wb").close()
        return out

def load_mem():
    if os.path.exists(MEMORY_FILE):
        return json.load(open(MEMORY_FILE))
    return {"summaries":[]}

def save_mem(sumry, src):
    db = load_mem()
    db["summaries"].append({"summary":sumry,"source":src})
    json.dump(db, open(MEMORY_FILE,"w"), indent=2)

def run(path):
    r = {}
    r["image"] = path
    r["text"] = ocr(path)
    r["summary"] = summarize(r["text"])
    r["bullets"] = bullets(r["summary"])
    r["mcqs"] = mcqs(r["summary"])
    out_audio = "/kaggle/working/" + os.path.basename(path) + ".mp3"
    r["audio"] = audio(r["summary"], out_audio)
    save_mem(r["summary"], path)
    return r

images = []
for f in os.listdir(AUTO_IMG_DIR):
    if f.lower().endswith((".png",".jpg",".jpeg")):
        images.append(os.path.join(AUTO_IMG_DIR, f))

results = []
for img in images:
    print("\nProcessing:", img)
    display(ShowImage(img))
    out = run(img)
    results.append(out)
    print("SUMMARY:", out["summary"][:500])
    print("MCQS:", out["mcqs"])
    print("AUDIO:", out["audio"])
    print("-"*80)

json.dump(results, open(RESULTS_FILE,"w"), indent=2)

print("\nDone! Processed:", len(results), "images.")
print("Results saved to:", RESULTS_FILE)
print("Memory saved to:", MEMORY_FILE)


