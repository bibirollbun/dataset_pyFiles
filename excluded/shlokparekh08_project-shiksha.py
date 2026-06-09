!pip install -q openai-whisper cohere mistralai yt-dlp pyyaml requests ffmpeg-python --upgrade
!apt-get -y update && apt-get -y install ffmpeg



import os
import re
import yaml
import logging
import warnings
import subprocess
import whisper
import ffmpeg
import requests
import sys
from typing import Optional
from mistralai import Mistral
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import ThreadPoolExecutor
from kaggle_secrets import UserSecretsClient



# -------------------------
# Logging & warnings
# -------------------------
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("shikshaai")



# -------------------------
# Config
# -------------------------
CONFIG_PATH = "config.yaml"
DEFAULT_CONFIG = """
output_dir: "./output"
max_workers: 1
chunk_length: 600 # seconds (10 minutes)
cookies_file: "" # Path to Netscape-format cookies.txt for YouTube auth (optional)
"""
if not os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "w") as f:
        f.write(DEFAULT_CONFIG)
def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)
config = load_config()
os.makedirs(config["output_dir"], exist_ok=True)


# -------------------------
# Kaggle Secrets
# -------------------------
try:
    user_secrets = UserSecretsClient()
    APIFY_KEY = user_secrets.get_secret("APIFY_API_KEY")
    GROQ_KEY = user_secrets.get_secret("GROQ_API_KEY")
    MISTRAL_KEY = user_secrets.get_secret("MISTRAL_API_KEY")
except Exception as e:
    # Fallback for local testing if not on Kaggle
    logger.warning("Could not load Kaggle secrets. Ensure you are on Kaggle or set env vars manually.")
    APIFY_KEY = os.getenv("APIFY_API_KEY")
    GROQ_KEY = os.getenv("GROQ_API_KEY")
    MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
if not APIFY_KEY or not GROQ_KEY or not MISTRAL_KEY:
    # Don't raise error immediately, allow script to compile, but fail later if needed
    logger.warning("âš ï¸� One or more API keys are missing! The pipeline will fail at the API step.")
# Clients
if GROQ_KEY:
    groq_client = OpenAI(base_url="https://api.groq.com/openai/v1", api_key=GROQ_KEY)
if MISTRAL_KEY:
    mistral_client = Mistral(api_key=MISTRAL_KEY)


# -------------------------
# Helpers
# -------------------------
def extract_video_id(url: str) -> str:
    m = re.search(r"(?:youtu\.be/|v=)([A-Za-z0-9_-]{6,})", url)
    return m.group(1) if m else "video"
def safe_filename(s: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in s)
def get_audio_duration(file_path: str) -> float:
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe["format"]["duration"])
    except ffmpeg.Error as e:
        logger.error(f"FFmpeg probe failed: {e.stderr.decode() if e.stderr else str(e)}")
        return 0.0
def split_audio(input_file: str, chunk_length: int = 600):
    if os.path.exists("chunks"):
        import shutil
        shutil.rmtree("chunks")
    os.makedirs("chunks", exist_ok=True)
       
    cmd = [
        "ffmpeg", "-y", "-i", input_file,
        "-f", "segment",
        "-segment_time", str(chunk_length),
        "-c", "copy", "chunks/out%03d.mp3"
    ]
    subprocess.run(cmd, check=True)
def choose_whisper_model(duration: float) -> str:
    if duration < 600: # <10 min
        return "base"
    elif duration < 3600: # <1 hour
        return "small"
    else:
        return "medium"


# -------------------------
# Transcript (Local Whisper)
# -------------------------
class TranscriptAgent:
    def __init__(self, chunk_length: int):
        self.chunk_length = chunk_length

    def download_video(self, url: str, out_path: str):
        logger.info("Downloading full YouTube video (MP4) using yt-dlp...")

        cookies_file = config.get("cookies_file", "").strip()
        cmd = [
            "yt-dlp", "--no-warnings",
            "--cookies", cookies_file,
            "-f", "bestvideo+bestaudio/best",
            "-o", out_path,
            url
        ]

        try:
            subprocess.run(cmd, check=True)
            logger.info(f"Video downloaded successfully: {out_path}")
        except subprocess.CalledProcessError as e:
            logger.error("â�Œ Video download failed.")
            raise e
            
    def download_audio(self, url: str, out_path: str):
        logger.info("Downloading audio with yt-dlp...")
        cookies_file = config.get("cookies_file", "").strip()
        cmd = [
            "yt-dlp", "--no-warnings", "--cookies", cookies_file,
            "-f", "bestaudio/best", "-x", "--audio-format", "mp3",
            "-o", out_path, url
        ]

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError:
            logger.warning("âš ï¸� bestaudio failed, retrying with fallback format...")
            fallback_cmd = [
                "yt-dlp", "--no-warnings", "--cookies", cookies_file,
                "-f", "best", "-x", "--audio-format", "mp3",
                "-o", out_path, url
            ]
            subprocess.run(fallback_cmd, check=True)

    def transcribe(self, url: str) -> str:
        vid = extract_video_id(url)
        audio_file = f"temp_{vid}.mp3"
        if os.path.exists(audio_file):
            os.remove(audio_file)

        self.download_audio(url, audio_file)

        if not os.path.exists(audio_file):
            raise FileNotFoundError(f"Audio file not found: {audio_file}")

        duration = get_audio_duration(audio_file)
        model_name = choose_whisper_model(duration)
        logger.info(f"Loading Whisper model: {model_name}")
        model = whisper.load_model(model_name)

        if duration <= self.chunk_length:
            result = model.transcribe(audio_file)
            return result["text"].strip()

        split_audio(audio_file, self.chunk_length)
        transcript_parts = []
        for chunk in sorted([c for c in os.listdir("chunks") if c.endswith(".mp3")]):
            result = model.transcribe(os.path.join("chunks", chunk))
            transcript_parts.append(result["text"])
        return "\n".join(transcript_parts).strip()




# -------------------------
# Language Detection
# -------------------------
def detect_language(text: str) -> str:
    """Detect transcript language using Groq."""
    if not GROQ_KEY:
        return "English"

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    "Detect the language of the following text. "
                    "Return only the language name (e.g., Marathi, Hindi, English, Tamil):\n\n"
                    f"{text[:500]}"
                )
            }],
            temperature=0
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Language detection error: {e}")
        return "English"


# -------------------------
# Summarization (APIFY)
# -------------------------
def summarize_with_apify(transcript: str, lang: str) -> Optional[str]:
    if not APIFY_KEY:
        return None

    url = "https://api.apify.com/v2/acts/easyapi/text-summarization/run-sync"
    payload = {
        "text": transcript,
        "output_sentences": 6,
        "language": lang   # <-- multilingual support
    }
    headers = {"Authorization": f"Bearer {APIFY_KEY}"}

    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=120)
        if resp.status_code == 200:
            data = resp.json()

            # Some actors return list output
            output = data.get("output", [])
            if isinstance(output, list) and output:
                summary = " ".join([item.get("text", "") for item in output if isinstance(item, dict)])
                return summary.strip() if summary else None

            # Others return summary field
            summary = data.get("summary") or data.get("output", {}).get("summary", "")
            return summary.strip() if summary else None

        logger.warning(f"APIFY status: {resp.status_code}, response: {resp.text}")
    except Exception as e:
        logger.warning(f"APIFY error: {e}")

    return None


# -------------------------
# Summarization (Groq)
# -------------------------
def summarize_with_groq(transcript: str, lang: str) -> Optional[str]:
    if not GROQ_KEY:
        return None

    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize the following lecture transcript in **{lang}**.\n"
                    f"Provide structured, clear notes.\n\n"
                    f"{transcript[:15000]}"
                )
            }],
            temperature=0.3,
            max_tokens=1200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Groq error: {e}")
        return None


# -------------------------
# Summarization (Mistral)
# -------------------------
def summarize_with_mistral(transcript: str, lang: str) -> Optional[str]:
    if not MISTRAL_KEY:
        return None

    try:
        resp = mistral_client.chat.complete(
            model="mistral-small-2409",
            messages=[{
                "role": "user",
                "content": (
                    f"Summarize the following lecture transcript in **{lang}**.\n"
                    f"Use structured notes and bullet points:\n\n"
                    f"{transcript[:15000]}"
                )
            }],
            temperature=0.3,
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"Mistral error: {e}")
        return None


# -------------------------
# MAIN SUMMARY GENERATOR
# -------------------------
def generate_summary(transcript: str) -> str:
    lang = detect_language(transcript)
    logger.info(f"ğŸŒ� Detected Language: {lang}")

    summary = summarize_with_apify(transcript, lang)
    if summary:
        return summary

    summary = summarize_with_groq(transcript, lang)
    if summary:
        return summary

    summary = summarize_with_mistral(transcript, lang)
    if summary:
        return summary

    raise RuntimeError("All summarization providers failed.")



# -------------------------
# Flashcards & Quiz
# -------------------------
def generate_flashcards(summary: str) -> str:
    if not GROQ_KEY: return "âš ï¸� GROQ_KEY missing."
    try:
        resp = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content":
                       f"Generate 15 Q&A flashcards from this summary. Format: Q: ... A: ...\n\n{summary}"}],
            temperature=0.4,
            max_tokens=1200
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
def generate_quiz(summary: str) -> str:
    if not MISTRAL_KEY: return "âš ï¸� MISTRAL_KEY missing."
    try:
        resp = mistral_client.chat.complete(
            model="mistral-small-2409",
            messages=[{"role": "user", "content":
                       f"Create a 10-question MCQ quiz with answers and rationales based on:\n{summary}"}],
            temperature=0.4,
            max_tokens=1500
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


# -------------------------
# Export
# -------------------------
class ExportAgent:
    def save_markdown(self, url: str, transcript: str, summary: str, flashcards: str, quiz: str):
        md = f"""# ğŸ“˜ ShikshaAI Study Pack
---
## ğŸ“º URL
{url}
---
## ğŸ“� Summary
{summary}
---
## ğŸ�¯ Flashcards
{flashcards}
---
## ğŸ§ª Quiz
{quiz}
---
## ğŸ�¤ Transcript (Local Whisper)
{transcript}
"""
        base_id = extract_video_id(url)
        output_file = os.path.join(config["output_dir"], f"ShikshaAI_Output_{safe_filename(base_id)}.md")
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(md)
        logger.info(f"ğŸ“„ Exported: {output_file}")



# -------------------------
# Pipeline
# -------------------------
def process_url(url: str):
    try:
        logger.info(f"\n===== ğŸ”„ Processing: {url} =====")
        transcriber = TranscriptAgent(config["chunk_length"])
        exporter = ExportAgent()

        # 1ï¸�âƒ£ Download full video (MP4)
        video_out = os.path.join(
            config["output_dir"],
            f"{extract_video_id(url)}.mp4"
        )
        try:
            transcriber.download_video(url, video_out)
        except Exception as e:
            logger.warning(f"âš ï¸� Could not download full video: {e}")

        # 2ï¸�âƒ£ Transcribe audio
        transcript = transcriber.transcribe(url)

        # 3ï¸�âƒ£ Generate summary
        summary = generate_summary(transcript)

        # 4ï¸�âƒ£ Flashcards + Quiz
        flashcards = generate_flashcards(summary)
        quiz = generate_quiz(summary)

        # 5ï¸�âƒ£ Export output
        exporter.save_markdown(url, transcript, summary, flashcards, quiz)

        logger.info(f"ğŸ�‰ DONE! Study pack + video for {url} is ready.")

    except Exception as e:
        logger.error(f"â�Œ Error in pipeline for {url}: {e}")


# -------------------------
# YouTube IDs from Secrets
# -------------------------
def input_urls() -> list:
    from kaggle_secrets import UserSecretsClient

    try:
        user_secrets = UserSecretsClient()
        video_ids = user_secrets.get_secret("YT_IDS")  # comma-separated IDs
    except:
        video_ids = None

    if not video_ids:
        raise ValueError("â�Œ No YT_IDS found in Kaggle Secrets. Please add it.")

    urls = [
        f"https://youtu.be/{vid.strip()}"
        for vid in video_ids.split(",")
        if vid.strip()
    ]
    return urls

        
# -------------------------
# Safe Cookies Handling
# -------------------------
def setup_cookies():
    """
    Safely use cookies:
    - Copy to /kaggle/working so yt-dlp can write to it
    - But delete BEFORE Kaggle generates output
    """

    for root, dirs, files in os.walk("/kaggle/input"):
        if "cookies.txt" in files:
            src = os.path.join(root, "cookies.txt")
            dst = "/kaggle/working/cookies.txt"

            import shutil
            shutil.copy(src, dst)

            # Use temporary location for yt-dlp
            config["cookies_file"] = dst
            return

    logger.warning("âš ï¸� No cookies.txt found â€” yt-dlp may fail.")





def main():
    setup_cookies()
    urls = input_urls()  # now returns a list without prompting
    try:
        with ThreadPoolExecutor(max_workers=config["max_workers"]) as executor:
            executor.map(process_url, urls)
    except KeyboardInterrupt:
        logger.info("Process interrupted by user.")
    except Exception as e:
        logger.error(f"â�Œ ERROR in main pipeline: {e}")


if __name__ == "__main__":
    main()

# ---------------------------------------------------
# ğŸ§¹ DELETE cookies BEFORE Kaggle saves output (SAFE)
# ---------------------------------------------------
try:
    cookies_path = config.get("cookies_file")
    if cookies_path and os.path.exists(cookies_path):
        os.remove(cookies_path)
        logger.info("Cookies removed from working directory for safety.")
except:
    pass

