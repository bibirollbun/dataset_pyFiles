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


# %% [markdown]
# # ğŸ�µ MuseMate â€” Creative Writing Agent
# 
# This notebook includes:
# - Idea generation agent
# - Poetry & lyrics generation
# - Story writing
# - Brainstorming
# - Style transformation (e.g., â€œwrite like Shakespeareâ€�)
# 
# It supports:
# âœ”ï¸� MOCK MODE (works without API key)  
# âœ”ï¸� Real LLM calls (OpenAI, if API key provided in Kaggle Secrets)  
# âœ”ï¸� Ipywidgets UI for interactive use  
# 
# Let's start!

# %%
import os
import json

try:
    import openai
except:
    openai = None

try:
    import ipywidgets as widgets
    from IPython.display import display, Markdown
except:
    widgets = None

# Load API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
MOCK_MODE = OPENAI_API_KEY is None or openai is None
print("MOCK_MODE =", MOCK_MODE)



# %%
def _mock_response(prompt: str) -> str:
    p = prompt.lower()
    if "lyrics" in p:
        return ("(Mock) ğŸ�µ Sample Lyrics:\n"
                "In the night where dreams ignite,\n"
                "I chase the stars beyond the light.\n")
    if "poem" in p:
        return "(Mock) A gentle breeze / whispers soft / across the silent sky."
    if "story" in p:
        return "(Mock) Once in a quiet village, a young dreamer found a glowing stone..."
    if "ideas" in p or "brainstorm" in p:
        return "(Mock) 1) AI music mixer\n2) Emotion-based playlist generator\n3) Voice-to-story app"
    return "(mock) creative output"

def call_llm(prompt, max_tokens=300):
    if MOCK_MODE:
        return _mock_response(prompt)
    
    openai.api_key = OPENAI_API_KEY
    resp = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are MuseMate, a creative assistant."},
                  {"role": "user", "content": prompt}],
        max_tokens=max_tokens
    )
    return resp.choices[0].message.content.strip()



# %%
class MuseMate:
    def generate_ideas(self, topic: str) -> str:
        prompt = f"Brainstorm 10 creative ideas about: {topic}"
        return call_llm(prompt)

    def write_poem(self, theme: str, style: str = "modern") -> str:
        prompt = f"Write a poem about '{theme}' in {style} style."
        return call_llm(prompt)

    def write_lyrics(self, theme: str, genre: str = "pop") -> str:
        prompt = f"Write song lyrics about '{theme}' in {genre} style. Include rhyme."
        return call_llm(prompt)

    def write_story(self, prompt_text: str) -> str:
        prompt = f"Write a short story based on: {prompt_text}"
        return call_llm(prompt, max_tokens=500)

    def transform_style(self, text: str, style: str) -> str:
        prompt = f"Rewrite the below text in {style} style:\n{text}"
        return call_llm(prompt)



# %%
muse = MuseMate()

if widgets is None:
    print("Widgets not available â€” use functions directly.")
else:
    action = widgets.Dropdown(
        options=["Generate Ideas", "Poem", "Lyrics", "Story", "Transform Style"],
        description="Mode:"
    )

    input_box = widgets.Textarea(
        placeholder="Enter topic/theme/text here...",
        layout=widgets.Layout(width="100%", height="120px")
    )

    style_box = widgets.Text(
        placeholder="optional (style, genre etc)",
        description="Style:"
    )

    run_btn = widgets.Button(description="Create âœ¨", button_style="success")
    output_area = widgets.Output()

    display(action, input_box, style_box, run_btn, output_area)

    def run_clicked(btn):
        with output_area:
            output_area.clear_output()
            text = input_box.value.strip()
            style = style_box.value.strip()

            print("### MuseMate Output\n")

            if action.value == "Generate Ideas":
                print(muse.generate_ideas(text))

            elif action.value == "Poem":
                print(muse.write_poem(text, style or "modern"))

            elif action.value == "Lyrics":
                print(muse.write_lyrics(text, style or "pop"))

            elif action.value == "Story":
                print(muse.write_story(text))

            elif action.value == "Transform Style":
                if not style:
                    print("Please enter a target style (e.g., 'Shakespeare').")
                else:
                    print(muse.transform_style(text, style))

    run_btn.on_click(run_clicked)



# %%
# Examples without UI:

print(muse.generate_ideas("AI music apps"))
print(muse.write_poem("rainy nights"))
print(muse.write_lyrics("lost love", "lofi"))
print(muse.write_story("a robot who dreams of music"))
print(muse.transform_style("I love coding", "Shakespearean"))


