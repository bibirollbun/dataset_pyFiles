# ğŸ“œ Codex: The AI Philologist (Vibe Code with Gemini 3 Pro)

<div style="background-color: #fbf6ec; padding: 20px; border-radius: 10px; border: 1px solid #d9ad63; color: #5d4037; font-family: serif;">
    <h2 style="text-align: center; color: #996b37;">ğŸš€ LIVE DEMO AVAILABLE</h2>
    <p style="text-align: center; font-size: 1.2em;">
        This notebook serves as the backend SDK tutorial for the Codex Web Application.<br>
        To experience the full interactive React UI, real-time voice analysis, and reconstruction tools, please visit:
    </p>
    <div style="text-align: center; margin-top: 20px;">
        <a href="https://ai.studio/apps/drive/1aiRuXmIYLRktghwDFdl2DWmHLKSaOKIZ?fullscreenApplet=true)" 
           style="background-color: #bf8d47; color: white; padding: 15px 30px; text-decoration: none; border-radius: 30px; font-weight: bold; font-size: 1.2em;">
           ğŸ”— Launch Codex App
        </a>
    </div>
</div>

## ğŸ“– Introduction
**Codex** is a multimodal "digital time machine" designed to revolutionize Biblical Studies and Ancient Near Eastern philology. By leveraging **Gemini 3 Pro** (for deep reasoning) and **Gemini 2.5 Flash** (for speed), Codex helps scholars:
1.  **Decipher** damaged manuscripts using "Thinking" models.
2.  **Compare** textual variants across centuries.
3.  **Reconstruct** missing fragments visually.
4.  **Converse** with history using native audio capabilities.

---
### ğŸ¤– Models Used:
* **Gemini 3 Pro Preview:** Heavy OCR, handwriting analysis, `thinking_config` for reconstruction.
* **Gemini 2.5 Flash:** Real-time chat, textual comparison, low-latency responses.
* **Imagen 3 / Nano:** Visual reconstruction of missing artifact parts.


# Install the Google Gen AI SDK
!pip install -q -U google-genai

import os
from google import genai
from google.genai import types
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown, Image

# Setup API Key from Kaggle Secrets
user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")Ù«

# Initialize the Client
client = genai.Client(api_key=api_key)

print("âœ… Environment Setup Complete. Ready to Vibe Code with Gemini!")


# Define the Schema (Same as in types.ts)
analysis_schema = {
    "type": "OBJECT",
    "properties": {
        "title": {"type": "STRING"},
        "language": {"type": "STRING"},
        "originalText": {"type": "STRING"},
        "englishTranslation": {"type": "STRING"},
        "historicalContext": {"type": "STRING"},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "text": {"type": "STRING"},
                    "definition": {"type": "STRING"},
                    "status": {"type": "STRING", "enum": ["clear", "damaged", "reconstructed"]}
                }
            }
        }
    }
}

def analyze_manuscript_simulation(text_content):
    """
    Simulates sending a manuscript image (represented here by text for the notebook demo)
    to Gemini 3 Pro with Thinking Config enabled.
    """
    
    prompt = """
    Analyze this ancient text fragment with academic rigor. 
    It appears to be a segment of the Dead Sea Scrolls or similar Hebrew manuscript.
    Provide a transcription, translation, and word-by-word segmentation.
    If words are missing, use the 'thinking' process to reconstruct them based on context.
    """
    
    input_content = f"Input Text: {text_content}\n\n{prompt}"

    try:
        response = client.models.generate_content(
            model="gemini-2.0-flash-thinking-exp-1219",
            contents=input_content,
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                response_schema=analysis_schema,
                thinking_config=types.ThinkingConfig(include_thoughts=True)
            )
        )
        
        return response.text
        
    except Exception as e:
        return f"Error: {str(e)}"

# Example: Analysis of a partial Psalm text
sample_text = "×‘Ö°Ö¼×¨Öµ×�×©Ö´×�×™×ª ×‘Ö¸Ö¼×¨Ö¸×� ×�Ö±×œÖ¹×”Ö´×™×� ×�Öµ×ª ×”Ö·×©Ö¸Ö¼×�×�Ö·×™Ö´×� ×•Ö°×�Öµ×ª ×”Ö¸×�Ö¸×¨Ö¶×¥"
result = analyze_manuscript_simulation(sample_text)

print("--- ğŸ§  AI Philologist Output (JSON) ---")
print(result)


def compare_texts(text_a, text_b, aspect="theological"):
    prompt = f"""
    Perform a critical textual comparison between these two manuscript excerpts.
    Focus Aspect: {aspect}
    
    Manuscript 1: "{text_a}"
    Manuscript 2: "{text_b}"
    
    Identify textual variants and their significance.
    """
    
    response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=prompt
    )
    
    return response.text

# Example: Comparing Isaiah 7:14 variants
mt_text = "×”Ö´× ÖµÖ¼×” ×”Ö¸×¢Ö·×œÖ°×�Ö¸×” ×”Ö¸×¨Ö¸×” ×•Ö°×™Ö¹×œÖ¶×“Ö¶×ª ×‘ÖµÖ¼×Ÿ"
lxx_text = "á¼°Î´Î¿á½º á¼¡ Ï€Î±Ï�Î¸Î­Î½Î¿Ï‚ á¼�Î½ Î³Î±ÏƒÏ„Ï�á½¶ á¼•Î¾ÎµÎ¹"

comparison_result = compare_texts(mt_text, lxx_text, aspect="translation difference")

display(Markdown(f"### ğŸ§� Comparison Result:\n{comparison_result}"))


https://ai.studio/apps/drive/1aiRuXmIYLRktghwDFdl2DWmHLKSaOKIZ?fullscreenApplet=true

