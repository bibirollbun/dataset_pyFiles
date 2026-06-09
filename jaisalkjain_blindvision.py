!pip install -q -U transformers fastapi uvicorn pyngrok nest_asyncio


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
hf_token = user_secrets.get_secret("HUGGINGFACE_TOKEN")
ngrok_token = user_secrets.get_secret("ngrok")



%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1" huggingface_hub hf_transfer
    !pip install --no-deps unsloth


%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps git+https://github.com/huggingface/transformers.git # Only for Gemma 3N
!pip install --no-deps --upgrade timm # Only for Gemma 3N


from unsloth import FastModel
import torch

fourbit_models = [
    # 4bit dynamic quants for superior accuracy and low memory use
    "unsloth/gemma-3n-E4B-it-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-it-unsloth-bnb-4bit",
    # Pretrained models
    "unsloth/gemma-3n-E4B-unsloth-bnb-4bit",
    "unsloth/gemma-3n-E2B-unsloth-bnb-4bit",

    # Other Gemma 3 quants
    "unsloth/gemma-3-1b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-4b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-12b-it-unsloth-bnb-4bit",
    "unsloth/gemma-3-27b-it-unsloth-bnb-4bit",
] # More models at https://huggingface.co/unsloth

model, tokenizer = FastModel.from_pretrained(
    model_name = "unsloth/gemma-3n-E2B-it",
    dtype = None, # None for auto detection
    max_seq_length = 1024, # Choose any for long context!
    load_in_4bit = True,  # 4 bit quantization to reduce memory
    full_finetuning = False, # [NEW!] We have full finetuning now!
    token = hf_token, # use one if using gated models
)


# FINAL BACKEND CODE - Has both /describe-image/ and /describe-video/ endpoints
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import io
import librosa
import numpy as np
import traceback
from pydub import AudioSegment
import gc
from typing import List

print("ğŸ› ï¸�  Configuring the web server with both IMAGE and VIDEO endpoints...")
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- WORKING IMAGE ENDPOINT ---
@app.post("/describe-image/")
async def describe_image(
    image: UploadFile = File(...),
    text: str = Form("Describe this image in a single, concise sentence.")
):
    try:
        image_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        messages = [{"role": "user","content": [{"type": "image", "image": pil_image},{"type": "text",  "text": text},],},]
        inputs = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt",).to("cuda")
        print("ğŸ¤– Generating IMAGE response from AI...")
        generated_ids = model.generate(**inputs, max_new_tokens=150)
        full_response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        model_turn_start = "model\n"
        response_start_index = full_response_text.rfind(model_turn_start)
        clean_response = full_response_text[response_start_index + len(model_turn_start):].strip() if response_start_index != -1 else full_response_text.strip()
        print("ğŸ’¬ AI Response generated.")
        return JSONResponse(content={"description": clean_response})
    except Exception as e:
        print(f"â�Œ ERROR in /describe-image/:")
        traceback.print_exc()
        return JSONResponse(content={"error": f"Internal Server Error: {e}"}, status_code=500)
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

# --- WORKING VIDEO ENDPOINT ---
@app.post("/describe-video/")
async def describe_video(
    frames: List[UploadFile] = File(...),
    text: str = Form("These are sequential frames from a short video. Describe the scene and any actions or changes taking place.")
):
    try:
        print(f"ğŸ–¼ï¸�  Received {len(frames)} frames for video description.")
        pil_images = []
        for frame in frames:
            image_bytes = await frame.read()
            pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            pil_images.append(pil_image)
        content = []
        for img in pil_images:
            content.append({"type": "image", "image": img})
        content.append({"type": "text", "text": text})
        messages = [{"role": "user", "content": content}]
        inputs = tokenizer.apply_chat_template(messages,add_generation_prompt=True,tokenize=True,return_dict=True,return_tensors="pt",).to("cuda")
        print("ğŸ¤– Generating VIDEO response from AI...")
        generated_ids = model.generate(**inputs, max_new_tokens=256)
        full_response_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        model_turn_start = "model\n"
        response_start_index = full_response_text.rfind(model_turn_start)
        clean_response = full_response_text[response_start_index + len(model_turn_start):].strip() if response_start_index != -1 else full_response_text.strip()
        print("ğŸ’¬ AI video description generated.")
        return JSONResponse(content={"description": clean_response})
    except Exception as e:
        print(f"â�Œ ERROR in /describe-video/:")
        traceback.print_exc()
        return JSONResponse(content={"error": f"Internal Server Error: {e}"}, status_code=500)
    finally:
        del inputs
        gc.collect()
        torch.cuda.empty_cache()

print("âœ… Web server logic is defined.")


!ngrok config add-authtoken {ngrok_token}


# # Cell 5: Run the Server
# import nest_asyncio
# import uvicorn
# from pyngrok import ngrok

# print("ğŸš‡ Setting up ngrok tunnel and starting the server...")
# nest_asyncio.apply()
# try:
#     public_url = ngrok.connect(8000)
#     print("âœ… Server is live!")
#     print(f"ğŸ”— Your public URL is: {public_url}")
#     print("\nğŸš€ Waiting for requests from your website...")
#     uvicorn.run(app, host="0.0.0.0", port=8000)
# except Exception as e:
#     print(f"â�Œ Failed to start server or ngrok: {e}")
#     traceback.print_exc()







