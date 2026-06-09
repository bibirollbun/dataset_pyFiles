

!pip install -q google-generativeai gtts
print("âœ… Installation complete!")
print("âš ï¸� NOW: Click 'Run' menu â†’ 'Restart Session', then run CELL 2")





import json
import google.generativeai as genai
from PIL import Image, ImageDraw
from gtts import gTTS
from IPython.display import Audio, display, Markdown, HTML
import ipywidgets as widgets

# Configure API Key
API_KEY = "AIzaSyAKpbA0EfhLfl2bGmjfCOlR7DUmccx28g0"
genai.configure(api_key=API_KEY)
print("âœ… API Key configured")

# List available models
print("\nğŸ”� Checking available models...")
try:
    available_models = []
    for m in genai.list_models():
        if 'generateContent' in m.supported_generation_methods:
            available_models.append(m.name)
            print(f"  â€¢ {m.name}")
except:
    pass

# Try multiple model options in order of preference
MODEL_OPTIONS = [
    'models/gemini-3-pro-preview',
    'models/nano-banana-pro-preview',
    'models/gemini-2.5-flash',
    'models/gemini-2.5-pro',
    'models/gemini-flash-latest',
    'models/gemini-pro-latest'
]

MODEL = None
for model_name in MODEL_OPTIONS:
    try:
        MODEL = genai.GenerativeModel(
            model_name,
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )
        print(f"âœ… Model ready: {model_name}")
        break
    except:
        continue

if MODEL is None:
    print("â�Œ Could not connect to any model")





LANGUAGES = {
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Italian": "it",
    "Portuguese": "pt",
    "Russian": "ru",
    "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW",
    "Japanese": "ja",
    "Korean": "ko",
    "Arabic": "ar",
    "Hindi": "hi",
    "Bengali": "bn",
    "Urdu": "ur",
    "Turkish": "tr",
    "Vietnamese": "vi",
    "Thai": "th",
    "Polish": "pl",
    "Dutch": "nl",
    "Greek": "el",
    "Hebrew": "he",
    "Czech": "cs",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Norwegian": "no",
    "Hungarian": "hu",
    "Romanian": "ro",
    "Ukrainian": "uk",
    "Indonesian": "id",
    "Malay": "ms",
    "Filipino": "fil",
    "Swahili": "sw",
    "Tamil": "ta",
    "Telugu": "te",
    "Gujarati": "gu",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Punjabi": "pa",
    "Persian": "fa",
    "Nepali": "ne",
    "Sinhala": "si",
    "Khmer": "km",
    "Lao": "lo",
    "Burmese": "my",
    "Amharic": "am",
    "Somali": "so",
    "Hausa": "ha",
    "Yoruba": "yo",
    "Zulu": "zu"
}


def analyze_medical_doc(image_path, language="Spanish"):
    """Analyze medical document and translate - tries multiple models if quota exceeded"""
    
    img = Image.open(image_path)
    
    prompt = f"""You are MedSnap, an expert medical translator AI.

Analyze this medical document image and return ONLY valid JSON with these exact keys:

{{
  "document_type": "Type of document (e.g., Prescription, Lab Result, Discharge Instructions)",
  "translated_text": "Complete text translation in {language}",
  "critical_alerts": ["Alert 1 in {language}", "Alert 2 in {language}"],
  "dosage": "Clear dosage instructions in {language}",
  "summary": "Simple patient-friendly explanation in {language}",
  "audio_script": "Natural conversational script in {language} for audio playback"
}}

IMPORTANT:
- Translate everything to {language}
- Keep critical_alerts as a list
- Preserve medical accuracy
- Use culturally appropriate language"""
    
    # Backup models to try if quota exceeded
    backup_models = [
        'models/gemini-2.5-flash',
        'models/gemini-2.5-flash-lite',
        'models/gemini-flash-latest',
        'models/gemini-flash-lite-latest',
        'models/gemini-2.0-flash-lite-001',
        'models/nano-banana-pro-preview',
        'models/gemma-3-27b-it',
        'models/gemma-3-12b-it'
    ]
    
    models_to_try = [MODEL] + [genai.GenerativeModel(m, generation_config={"temperature": 0.1}) for m in backup_models]
    
    for idx, model in enumerate(models_to_try):
        try:
            if idx == 0:
                print(f"ğŸ”„ Analyzing document...")
            else:
                print(f"ğŸ”„ Trying backup model {idx}...")
            
            response = model.generate_content([prompt, img])
            text = response.text.strip()
            
            # Clean markdown formatting
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            
            result = json.loads(text.strip())
            print(f"âœ… Analysis complete!")
            return result
        
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "quota" in error_msg.lower():
                if idx < len(models_to_try) - 1:
                    print(f"âš ï¸� Quota exceeded, trying alternate model...")
                    continue
                else:
                    print(f"â�Œ All models quota exceeded. Please wait 1 hour.")
                    return {"error": "Quota exceeded. Please try again in 1 hour."}
            else:
                if idx < len(models_to_try) - 1:
                    continue
                return {"error": error_msg[:200]}
    
    return {"error": "Analysis failed"}


def create_audio(text, lang_code='es'):
    """Generate audio file from text"""
    try:
        tts = gTTS(text=text, lang=lang_code, slow=False)
        filename = "medsnap_audio.mp3"
        tts.save(filename)
        return filename
    except Exception as e:
        print(f"â�Œ Audio error: {e}")
        return None


def create_sample_rx():
    """Create a realistic test prescription image"""
    img = Image.new('RGB', (650, 520), 'white')
    draw = ImageDraw.Draw(img)
    
    lines = [
        "â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�",
        "    MEMORIAL HOSPITAL - PRESCRIPTION",
        "â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�",
        "",
        "Patient: Maria Rodriguez",
        "DOB: 05/12/1980",
        "Date: December 9, 2025",
        "",
        "Prescriber: Dr. Sarah Chen, MD",
        "License: MD-45678",
        "",
        "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€",
        "Rx: AMOXICILLIN 500mg Capsules",
        "Qty: 30 capsules",
        "Refills: 0",
        "",
        "Sig: Take ONE (1) capsule by mouth",
        "     every 8 hours with food or milk",
        "     for 10 days",
        "â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€",
        "",
        "âš ï¸� IMPORTANT WARNINGS:",
        "  â€¢ Complete full course of medication",
        "  â€¢ Take with food to prevent nausea",
        "  â€¢ May cause dizziness or drowsiness",
        "  â€¢ Avoid alcohol during treatment",
        "  â€¢ Call doctor if rash develops"
    ]
    
    y = 25
    for line in lines:
        draw.text((30, y), line, fill='black')
        y += 20
    
    filename = "sample_prescription.jpg"
    img.save(filename)
    return filename


print("âœ… Functions loaded successfully")




print("ğŸŒ� SELECT YOUR LANGUAGE")
print("=" * 60)

language_dropdown = widgets.Dropdown(
    options=list(LANGUAGES.keys()),
    value='Spanish',
    description='Language:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='400px')
)

analyze_button = widgets.Button(
    description='ğŸ”� Translate Document',
    button_style='success',
    layout=widgets.Layout(width='400px', height='40px')
)

output_area = widgets.Output()

# Store the current image path
current_image = {'path': None}


def on_analyze_click(b):
    """Handle translate button click"""
    with output_area:
        output_area.clear_output()
        
        selected_language = language_dropdown.value
        lang_code = LANGUAGES[selected_language]
        
        print(f"ğŸ”„ Translating to {selected_language}...\n")
        
        # Create sample image if not exists
        if current_image['path'] is None:
            current_image['path'] = create_sample_rx()
            display(Image.open(current_image['path']))
        
        # Analyze document
        result = analyze_medical_doc(current_image['path'], selected_language)
        
        # Display results
        if "error" not in result:
            display(Markdown("---"))
            display(Markdown(f"# ğŸ�¥ MedSnap Translation â†’ {selected_language}"))
            display(Markdown(f"**Document Type:** {result.get('document_type', 'N/A')}"))
            
            display(Markdown("---"))
            display(Markdown("## âš ï¸� CRITICAL ALERTS"))
            alerts = result.get('critical_alerts', [])
            for alert in alerts:
                display(Markdown(f"ğŸ”´ **{alert}**"))
            
            display(Markdown("---"))
            display(Markdown("## ğŸ’Š Dosage Instructions"))
            display(Markdown(f"**{result.get('dosage', 'N/A')}**"))
            
            display(Markdown("---"))
            display(Markdown("## ğŸ“‹ Patient Summary"))
            display(Markdown(f"> {result.get('summary', 'N/A')}"))
            
            display(Markdown("---"))
            display(Markdown("## ğŸ”Š Audio Guidance"))
            
            audio_file = create_audio(result.get('audio_script', ''), lang_code)
            if audio_file:
                display(Audio(audio_file, autoplay=False))
                print(f"\nâœ… Translation complete! Play audio above â†‘")
                print(f"âœ… Document translated to {selected_language}")
            
        else:
            display(Markdown("## â�Œ Analysis Failed"))
            print(result.get('error', 'Unknown error'))


analyze_button.on_click(on_analyze_click)

# Display the interface
display(HTML("<h2>ğŸ�¥ MedSnap: AI Medical Translator</h2>"))
display(HTML("<p>Select your language and click 'Translate Document' to see the demo</p>"))
display(language_dropdown)
display(analyze_button)
display(output_area)

print("\nğŸ“� Instructions:")
print("1. Select your language from the dropdown above")
print("2. Click 'Translate Document' button")
print("3. Wait for translation and audio to generate")
print("\nğŸŒ� Available: 50+ languages including Spanish, French, Arabic, Hindi, Bengali, Chinese, and more!")


# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
# CELL 5: (OPTIONAL) USE YOUR OWN IMAGE
# â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�

# Uncomment and modify this cell to use your own medical document image

"""
# Upload your own image
from google.colab import files  # Use this if in Colab, skip in Kaggle
# uploaded = files.upload()  # For Colab

# For Kaggle: Upload image as dataset, then:
current_image['path'] = "/kaggle/input/your-dataset/your-image.jpg"

# Select language and run
selected_language = "Bengali"  # Change this
lang_code = LANGUAGES[selected_language]

# Analyze
result = analyze_medical_doc(current_image['path'], selected_language)

# Display results
if "error" not in result:
    display(Markdown(f"# Translated to {selected_language}"))
    display(Markdown(f"**Type:** {result.get('document_type')}"))
    
    for alert in result.get('critical_alerts', []):
        display(Markdown(f"âš ï¸� {alert}"))
    
    display(Markdown(f"**Dosage:** {result.get('dosage')}"))
    display(Markdown(f"**Summary:** {result.get('summary')}"))
    
    audio_file = create_audio(result.get('audio_script'), lang_code)
    if audio_file:
        display(Audio(audio_file))
"""

