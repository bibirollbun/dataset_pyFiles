!pip install git+https://github.com/openai/whisper.git 
!sudo apt update && sudo apt install ffmpeg -y


from pathlib import Path
import whisper, torch
from tqdm.auto import tqdm
import datetime, math


DATASET_ROOT  = Path("/kaggle/input/aic24")
OUTPUT_ROOT   = Path("/kaggle/working/transcripts")
AUDIO_ROOT = Path("/kaggle/working/audios")


device = "cuda" if torch.cuda.is_available() else "cpu"
model = whisper.load_model("medium", device=device)


from moviepy.editor import VideoFileClip



def hh_mm_ss_ms(seconds: float) -> str:
    """Convert seconds -> 'HH:MM:SS.mmm'"""
    ms = int(round((seconds - int(seconds)) * 1000))
    return f"{datetime.timedelta(seconds=int(seconds))}.{ms:03d}"

video_files = sorted(DATASET_ROOT.rglob("*.mp4"))
print(f"Found {len(video_files)} video files.")

for vid in tqdm(video_files, desc="Transcribing"):
    rel_path = vid.relative_to(DATASET_ROOT)

    # ========== Lưu file transcript ==========
    txt_path = OUTPUT_ROOT / rel_path.with_suffix(".txt")
    txt_path.parent.mkdir(parents=True, exist_ok=True)

    result = model.transcribe(str(vid), language="vi")

    with txt_path.open("w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(f"[{hh_mm_ss_ms(seg['start'])} → {hh_mm_ss_ms(seg['end'])}] "
                    f"{seg['text'].strip()}\n")

    print("Done transcript:", txt_path.relative_to(OUTPUT_ROOT.parent))

    # ========== Trích xuất audio ==========
    audio_path = AUDIO_ROOT / rel_path.with_suffix(".wav")
    audio_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        video = VideoFileClip(str(vid))
        video.audio.write_audiofile(str(audio_path), codec='pcm_s16le')  # Lưu .wav
        print("Saved audio:", audio_path.relative_to(AUDIO_ROOT.parent))
    except Exception as e:
        print(f"Error extracting audio from {vid.name}: {e}")

