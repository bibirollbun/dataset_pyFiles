# Install required libraries 
!pip install -q pydub ffmpeg-python git+https://github.com/openai/whisper.git
!pip install translate
!pip install -U google-generativeai
!pip install reportlab


# ğŸ“� Step 1: Ask user to select spoken language (number-based)
language_options = {
    "1": ("English", "en"),
    "2": ("Hindi", "hi"),
    "3": ("Gujarati", "gu"),
    "4": ("Bangla", "bn"),
    "5": ("Marathi", "mr"),
    "6": ("Tamil", "ta"),
    "7": ("Telugu", "te")
}

print("\nğŸŒ� Select the language spoken in the audio:")
for num, (lang, _) in language_options.items():
    print(f"{num}. {lang}")

selected_number = input("\nğŸ”¢ Enter the number for your language: ").strip()
selected_language, lang_code = language_options.get(selected_number, ("English", "en"))
print(f"\nâœ… You selected: {selected_language} ({lang_code})")

# ğŸ“� Step 2: Use audio files from Kaggle input directory
import os

input_dir = "/kaggle/input/bangla-test-2"
audio_files = [f for f in os.listdir(input_dir) if f.endswith(('.mp3', '.m4a', '.wav'))]

if not audio_files:
    raise FileNotFoundError("â�Œ No audio files found in the input directory!")

print("\nğŸ�§ Available audio files:")
for idx, file in enumerate(audio_files, 1):
    print(f"{idx}. {file}")

selected_idx = int(input("\nğŸ”¢ Select the file number to transcribe: ").strip()) - 1
input_file = os.path.join(input_dir, audio_files[selected_idx])

print(f"\nğŸ�¯ Using file: {audio_files[selected_idx]}")

# ğŸ”Š Step 3: Transcribe using Whisper with language code
import whisper

model = whisper.load_model("large-v3")
result = model.transcribe(input_file, language=lang_code, task="transcribe")
final_transcription = result["text"].strip()

print("\nğŸ“� Transcribed Text:\n", final_transcription)

# ğŸŒ� Step 4: Translate if needed
from translate import Translator

def translate_to_english(text, from_lang_code):
    if from_lang_code == "en":
        return text
    translator = Translator(from_lang=from_lang_code, to_lang="en")
    max_chars = 500
    chunks = [text[i:i+max_chars] for i in range(0, len(text), max_chars)]
    translated_chunks = []
    for idx, chunk in enumerate(chunks):
        try:
            translated = translator.translate(chunk)
            translated_chunks.append(translated)
        except Exception as e:
            translated_chunks.append(f"[Error in chunk {idx + 1}: {e}]")
    return ' '.join(translated_chunks)

translated_english = translate_to_english(final_transcription, lang_code)

print("\nğŸ”� Translated to English:\n", translated_english)



# ğŸ”� Configure Gemini with your API key
import google.generativeai as genai
genai.configure(api_key="AIzaSyAWUNcaRpMV3g6rKDQBe4ezbJFVHUJ73q8")  # ğŸ”‘ Replace with your actual API key

# ğŸ“‘ Function to generate structured medical report
def generate_medical_report(text):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"""
You are a medical expert AI assistant.

Given the following unstructured medical note in English, generate a **well-structured, detailed medical report**. 
It must include the following sections in paragraph format (avoid using any ** or markdown symbols):

1. Patient Complaints  
2. Diagnosis  
3. Medications  
4. Suggested Tests or Follow-ups  
5. Additional Notes or Observations

Medical Note:  
{text}

Ensure the report is comprehensive and professional.
"""
    response = model.generate_content(prompt)
    return response.text.strip()
  
# ğŸ”� Generate the report using the translated English text
medical_report = generate_medical_report(translated_english)

# ğŸ–¨ï¸� Print the result
print("\nğŸ“� Medical Report:\n")
print(medical_report)



from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch
from textwrap import wrap
import os

def save_report_to_pdf(report_text, filename="medical_report.pdf"):
    cleaned_text = report_text.replace("**", "").strip()

    c = canvas.Canvas(filename, pagesize=A4)
    width, height = A4
    x_margin, y_margin = 50, 50
    max_width = width - 2 * x_margin
    line_height = 15
    y_position = height - y_margin

    c.setFont("Helvetica", 12)

    for paragraph in cleaned_text.split("\n"):
        paragraph = paragraph.strip()
        if not paragraph:
            y_position -= line_height
            continue

        # Wrap paragraph text
        wrapped_lines = wrap(paragraph, width=95)

        for line in wrapped_lines:
            if y_position <= y_margin:
                c.showPage()
                c.setFont("Helvetica", 12)
                y_position = height - y_margin
            c.drawString(x_margin, y_position, line)
            y_position -= line_height

        y_position -= line_height  # Extra space between paragraphs

    c.save()

    # âœ… For Kaggle: inform user where the file is saved
    if os.path.exists(filename):
        print(f"âœ… PDF saved: {filename}")
        print("ğŸ“¥ Go to the 'Output' tab on the right sidebar to download the file.")
    else:
        print("âš ï¸� File not found. Something went wrong.")



save_report_to_pdf(medical_report, "medical_report.pdf")


