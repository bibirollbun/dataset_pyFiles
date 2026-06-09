# ğŸ“¦ Install Required Libraries
!pip install -q diffusers transformers accelerate gTTS git+https://github.com/openai/whisper.git
!pip install -U transformers



# Text Prompt to Image using Stable Diffusion

from diffusers import StableDiffusionPipeline
import torch
from IPython.display import display

pipe = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16
).to("cuda")

prompt = (
    "A charming small Swiss village nestled in the Alps, surrounded by snow-capped mountains and green meadows. "
    "Traditional wooden chalets with flower-filled balconies line the cobblestone streets. "
    "A small church with a clock tower stands in the village center. "
    "The sky is clear with golden sunlight illuminating the landscape. "
    "High detail, photorealistic, peaceful atmosphere, scenic view, ultra HD"
)

image = pipe(prompt).images[0]
image.save("generated.png")
display(image)  # ğŸ‘ˆ Correct way to display in Kaggle



# ğŸ–¼ï¸� Step 2: Image to Caption using BLIP
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")

def generate_caption(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = blip_processor(images=image, return_tensors="pt").to("cuda")
    out_ids = blip_model.generate(**inputs)
    caption = blip_processor.decode(out_ids[0], skip_special_tokens=True)
    return caption

caption = generate_caption("generated.png")
print("\nğŸ–¼ï¸� Generated Caption:", caption)



# ğŸ”Š Step 3: Caption to Audio using gTTS
from gtts import gTTS
import IPython.display as ipd

tts = gTTS(caption)
tts.save("caption_audio.mp3")
ipd.display(ipd.Audio("caption_audio.mp3"))

# ğŸ�¤ Step 4: Audio to Text using Whisper (ASR)
import whisper
model_whisper = whisper.load_model("base")

def transcribe_audio(audio_path):
    result = model_whisper.transcribe(audio_path)
    return result['text']

# Example: Transcribe any mp3 file (e.g., user-uploaded or previously generated)
audio_text = transcribe_audio("caption_audio.mp3")
print("\nğŸ—£ï¸� Transcribed Audio:", audio_text)


# # Video Frame Captioning (Using First Frame)
# import cv2

# def extract_first_frame(video_path):
#     cap = cv2.VideoCapture(video_path)
#     success, frame = cap.read()
#     if success:
#         image_path = "first_frame.jpg"
#         cv2.imwrite(image_path, frame)
#         return image_path
#     return None

# # Example usage for a video file
# video_path = "sample_video.mp4"  # Replace with your video
# first_frame_path = extract_first_frame(video_path)
# if first_frame_path:
#     video_caption = generate_caption(first_frame_path)
#     print("\nğŸ��ï¸� Caption for First Frame:", video_caption)


#Video Frame Captioning (Using First Frame)
import cv2

def extract_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    success, frame = cap.read()
    if success:
        image_path = "first_frame.jpg"
        cv2.imwrite(image_path, frame)
        return image_path
    return None

# Example usage for a video file
video_path = "/kaggle/input/car-detection-raw-video/Car Object Detection.mp4"  #  video
first_frame_path = extract_first_frame(video_path)
if first_frame_path:
    video_caption = generate_caption(first_frame_path)
    print("\nğŸ��ï¸� Caption for First Frame:", video_caption)


!pip install --upgrade transformers
!pip install git+https://github.com/huggingface/transformers.git


# from huggingface_hub import login
# login()



# # ğŸ§  Step 6: Use Gemma 3n (Local Kaggle Path) for text-based response
# from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline

# GEMMA_PATH = "/kaggle/input/gemma-3n/transformers/gemma-3n-e2b-it/1"  # âœ… Use this one on Kaggle for now

# gemma_tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH)
# gemma_model = AutoModelForCausalLM.from_pretrained(
#     GEMMA_PATH,
#     torch_dtype=torch.float16,
#     device_map="auto",
# )


# gemma_pipe = TextGenerationPipeline(model=gemma_model, tokenizer=gemma_tokenizer)

# user_input = f"Describe this image and audio: {caption}. The audio said: {audio_text}"
# gemma_response = gemma_pipe(user_input, max_new_tokens=100)[0]['generated_text']
# print("\nğŸ¤– Gemma 3n Response:", gemma_response)


# âœ… Use this version if you're using Hugging Face-hosted Gemma 3B model
#    Make sure you've accepted the model license: https://huggingface.co/google/gemma-1.1-3b-it
#    (Kaggle will ask you to select "transformers" as your framework when setting up the notebook)


