
##This App helps you create your own magical storybook with AI narrators and illustrators!
#A magical storybook creator for kids that writes stories, illustrates them in high definition, and reads them aloud using advanced AI models.
#Models used in this notebook are: 
#MODEL_TEXT = 'gemini-3-pro-preview'
#MODEL_IMAGE = 'gemini-3-pro-image-preview
#MODEL_TTS = 'gemini-2.5-flash-preview-tts'
#Problem Statement: Parents like to read storybooks to their kids that reflect their cultural moral values, but they often have to search through libraries or bookstores to find the best pick, only to still be unable to find the exact match they are searching for.



# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# 1. Install necessary libraries
!pip install -q -U google-genai gradio numpy pillow

import os
import json
import numpy as np
import gradio as gr
from PIL import Image
from io import BytesIO
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient

# 2. Setup Client & Secrets
try:
    user_secrets = UserSecretsClient()
    api_key = user_secrets.get_secret("GOOGLE_API_KEY")
except:
    api_key = os.environ.get("GOOGLE_API_KEY", "YOUR_API_KEY_HERE")

client = genai.Client(api_key=api_key)

# 3. Models
MODEL_TEXT = 'gemini-3-pro-preview'
MODEL_IMAGE = 'gemini-3-pro-image-preview'
MODEL_TTS = 'gemini-2.5-flash-preview-tts'

SYSTEM_PROMPT = """You are a creative children's book author. 
Split the story into 3 distinct pages.
Return strictly a JSON object:
{
  "title": "Story Title",
  "pages": [
    { "text": "Page text...", "imagePrompt": "Visual description..." },
    ...
  ]
}"""

# --- Backend Functions ---

def generate_story_structure(topic):
    try:
        response = client.models.generate_content(
            model=MODEL_TEXT,
            contents=f"Write a story about: {topic}",
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema={
                    "type": "OBJECT",
                    "properties": {
                        "title": {"type": "STRING"},
                        "pages": {
                            "type": "ARRAY",
                            "items": {
                                "type": "OBJECT",
                                "properties": {
                                    "text": {"type": "STRING"},
                                    "imagePrompt": {"type": "STRING"}
                                },
                                "required": ["text", "imagePrompt"]
                            }
                        }
                    },
                    "required": ["title", "pages"]
                }
            )
        )
        return json.loads(response.text)
    except Exception as e:
        return {"error": str(e)}

def generate_page_assets(page_data):
    # 1. Generate Image
    img_prompt = f"A cute, high quality children's book illustration: {page_data['imagePrompt']}"
    img_response = client.models.generate_content(
        model=MODEL_IMAGE,
        contents=img_prompt,
        config=types.GenerateContentConfig(
            image_config=types.ImageConfig(aspect_ratio="1:1")
        )
    )
    
    image_output = None
    for part in img_response.candidates[0].content.parts:
        if part.inline_data:
            image_data = part.inline_data.data 
            image_output = Image.open(BytesIO(image_data))
            
    # 2. Generate Audio
    tts_response = client.models.generate_content(
        model=MODEL_TTS,
        contents=page_data['text'],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name="Puck")
                )
            )
        )
    )
    
    audio_output = None
    for part in tts_response.candidates[0].content.parts:
        if part.inline_data:
            pcm_bytes = part.inline_data.data
            audio_array = np.frombuffer(pcm_bytes, dtype=np.int16)
            audio_output = (24000, audio_array)

    return image_output, audio_output

# --- Gradio UI Logic ---

def start_story(topic):
    story_json = generate_story_structure(topic)
    if "error" in story_json:
        return gr.update(visible=True, value=f"Error: {story_json['error']}"), gr.update(visible=False), None, 0
    
    first_page = story_json['pages'][0]
    img, aud = generate_page_assets(first_page)
    
    return (
        gr.update(visible=False), 
        gr.update(visible=True), 
        story_json, 
        0, 
        story_json['title'], 
        f"Page 1 of {len(story_json['pages'])}", 
        img, 
        first_page['text'], 
        aud
    )

def navigate(direction, story_data, current_index):
    new_index = current_index + direction
    if new_index < 0 or new_index >= len(story_data['pages']):
        return gr.update(), gr.update(), current_index, gr.update(), gr.update(), gr.update()
    
    page = story_data['pages'][new_index]
    img, aud = generate_page_assets(page)
    
    return (
        f"Page {new_index + 1} of {len(story_data['pages'])}",
        img,
        page['text'],
        aud,
        new_index
    )

def reset():
    return gr.update(visible=True), gr.update(visible=False), None, 0

# --- Build the App ---

# FIX: Removed theme argument to prevent version errors
with gr.Blocks() as demo:
    story_state = gr.State()
    page_index = gr.State(0)
    
    gr.Markdown("# ğŸ¦� WonderTales: AI Storyteller")
    
    with gr.Column(visible=True) as input_section:
        topic_input = gr.Textbox(label="What should the story be about?", placeholder="e.g. A space-faring hamster")
        btn_generate = gr.Button("âœ¨ Write My Story", variant="primary")
    
    with gr.Column(visible=False) as story_section:
        title_display = gr.Markdown("## Title")
        page_counter = gr.Label(value="Page 1", label="Progress")
        
        with gr.Row():
            with gr.Column():
                image_display = gr.Image(label="Illustration", type="pil")
            with gr.Column():
                text_display = gr.Markdown("**Story text will appear here...**")
                audio_display = gr.Audio(label="Narration", autoplay=True)
        
        with gr.Row():
            btn_prev = gr.Button("â¬…ï¸� Previous Page")
            btn_next = gr.Button("Next Page â�¡ï¸�")
            
        btn_reset = gr.Button("ğŸ”„ Create New Story", variant="secondary")

    # Wiring
    btn_generate.click(
        fn=start_story,
        inputs=[topic_input],
        outputs=[input_section, story_section, story_state, page_index, title_display, page_counter, image_display, text_display, audio_display]
    )
    
    btn_next.click(
        fn=lambda s, i: navigate(1, s, i),
        inputs=[story_state, page_index],
        outputs=[page_counter, image_display, text_display, audio_display, page_index]
    )
    
    btn_prev.click(
        fn=lambda s, i: navigate(-1, s, i),
        inputs=[story_state, page_index],
        outputs=[page_counter, image_display, text_display, audio_display, page_index]
    )
    
    btn_reset.click(fn=reset, outputs=[input_section, story_section, story_state, page_index])

demo.launch()

