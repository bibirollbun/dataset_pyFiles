# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

# import numpy as np # linear algebra
# import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cell 1: Install and setup with PROPER dependencies
!pip install google-generativeai gTTS moviepy --quiet
!pip install manim --quiet
!apt-get update -qq
!apt-get install -y ffmpeg > /dev/null
!apt-get install -y libcairo2-dev libjpeg-dev libgif-dev > /dev/null
!apt-get install -y texlive texlive-latex-extra texlive-fonts-extra > /dev/null


# Cell 2: Import libraries and setup paths
import os
import json
import subprocess
import google.generativeai as genai
from pathlib import Path
import base64
from gtts import gTTS
import tempfile

# Set up Kaggle paths
WORKING_DIR = "/kaggle/working/"
OUTPUT_DIR = "/kaggle/output/" if os.path.exists("/kaggle/output/") else WORKING_DIR
ASSETS_DIR = os.path.join(WORKING_DIR, "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "scenes"), exist_ok=True)
os.makedirs(os.path.join(ASSETS_DIR, "audio"), exist_ok=True)

print("ğŸ“� Directories created!")


# Cell 3: Configure Gemini API (Kaggle version)

from kaggle_secrets import UserSecretsClient
import google.generativeai as genai

try:
    # Load API key from Kaggle â†’ Settings â†’ Secrets
    GEMINI_API_KEY = UserSecretsClient().get_secret("GEMINI_API_KEY")

    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        DEMO_MODE = False
        print("âœ… Gemini API key loaded from Kaggle Secrets!")
    else:
        raise ValueError("API key empty")

except Exception as e:
    print("âš ï¸� Gemini API key not found in Kaggle Secrets.")
    print("â�¡ï¸� Add one in: Kaggle â†’ Settings â†’ Secrets â†’ New Secret â†’ Name: GEMINI_API_KEY")
    print(f"Details: {e}")
    DEMO_MODE = True


# Cell 4: Create the Education Animation Agent
class EduAnimationAgent:
    def __init__(self):
        self.model = None if DEMO_MODE else genai.GenerativeModel('gemini-1.5-flash')
        
    def generate_storyboard(self, topic, duration_minutes=2):
        if DEMO_MODE:
            return self._get_demo_storyboard(topic)
        
        prompt = f"""
        Create a {duration_minutes}-minute educational lesson about: {topic}
        
        Return JSON with this structure:
        {{
            "topic": "{topic}",
            "total_duration": {duration_minutes * 60},
            "scenes": [
                {{
                    "scene_number": 1,
                    "duration": 15,
                    "visual_type": "animation",
                    "description": "What to show visually",
                    "narration": "Text to speak"
                }}
            ]
        }}
        
        Make it engaging for students!
        """
        
        try:
            response = self.model.generate_content(prompt)
            text = response.text
            json_start = text.find('{')
            json_end = text.rfind('}') + 1
            json_str = text[json_start:json_end]
            return json.loads(json_str)
        except Exception as e:
            print(f"Gemini error: {e}")
            return self._get_demo_storyboard(topic)
    
    def _get_demo_storyboard(self, topic):
        return {
            "topic": topic,
            "total_duration": 90,
            "scenes": [
                {
                    "scene_number": 1,
                    "duration": 10,
                    "visual_type": "animation",
                    "description": "Title screen with topic name",
                    "narration": f"Welcome to our lesson about {topic}! Today we'll learn the basics."
                },
                {
                    "scene_number": 2,
                    "duration": 20,
                    "visual_type": "animation",
                    "description": "Basic concept explanation with shapes",
                    "narration": "Let me explain the fundamental concept using simple examples that are easy to understand."
                },
                {
                    "scene_number": 3, 
                    "duration": 15,
                    "visual_type": "animation",
                    "description": "Practical example or code",
                    "narration": "Here's how this works in practice with a real example you can try yourself."
                }
            ]
        }

print("âœ… Education Agent created!")


# Cell 5: Create SIMPLE video generator (no Manim dependency)
class SimpleVideoGenerator:
    def __init__(self):
        self.available = True
        print("âœ… Simple Video Generator ready!")
    
    def create_scene_video(self, scene_data, scene_number, output_dir):
        """Create a simple video scene using matplotlib animation"""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.animation as animation
            import numpy as np
            from moviepy.editor import VideoClip
            from moviepy.video.io.bindings import mplfig_to_npimage
            
            # Create a simple animation
            fig, ax = plt.subplots(figsize=(8, 6), facecolor='white')
            ax.set_facecolor('#f0f0f0')
            ax.set_xlim(0, 10)
            ax.set_ylim(0, 6)
            ax.axis('off')
            ax.set_title(f"Scene {scene_number}: {scene_data['description'][:30]}...", 
                        fontsize=14, pad=20)
            
            # Add some animated elements based on scene number
            if scene_number == 1:
                # Title scene
                text = ax.text(5, 3, "Python Loops", fontsize=24, 
                              ha='center', va='center', color='blue',
                              bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
            elif scene_number == 2:
                # Concept scene
                circle = plt.Circle((3, 3), 1, color='red', alpha=0.7)
                square = plt.Rectangle((6, 2), 2, 2, color='green', alpha=0.7)
                ax.add_patch(circle)
                ax.add_patch(square)
            else:
                # Code scene
                code_lines = ["for i in range(3):", "    print(i)", "Output: 0 1 2"]
                for i, line in enumerate(code_lines):
                    ax.text(1, 4-i, line, fontfamily='monospace', fontsize=12)
            
            def make_frame(t):
                # Simple animation - elements move slightly
                if scene_number == 1 and hasattr(text, 'set_position'):
                    text.set_position((5, 3 + 0.2 * np.sin(t * 2)))
                return mplfig_to_npimage(fig)
            
            # Create video
            duration = min(scene_data['duration'], 5)  # Max 5 seconds per scene
            animation = VideoClip(make_frame, duration=duration)
            output_path = os.path.join(output_dir, f"scene_{scene_number}.mp4")
            animation.write_videofile(output_path, fps=24, verbose=False, logger=None)
            plt.close()
            
            return output_path
            
        except Exception as e:
            print(f"â�Œ Video generation failed: {e}")
            return self.create_fallback_video(scene_data, scene_number, output_dir)
    
    def create_fallback_video(self, scene_data, scene_number, output_dir):
        """Create ultra-simple fallback video"""
        try:
            import numpy as np
            from moviepy.editor import VideoClip
            
            # Create a simple color animation
            duration = min(scene_data['duration'], 5)
            
            def make_frame(t):
                # Create a gradient that changes over time
                width, height = 640, 480
                frame = np.zeros((height, width, 3), dtype=np.uint8)
                
                # Color based on scene and time
                r = int(100 + 50 * np.sin(t + scene_number))
                g = int(100 + 50 * np.sin(t * 2 + scene_number))
                b = int(150 + 50 * np.sin(t * 3 + scene_number))
                
                frame[:, :, 0] = r  # Red
                frame[:, :, 1] = g  # Green  
                frame[:, :, 2] = b  # Blue
                
                # Add text using simple drawing
                text_y = height // 2
                for i, line in enumerate([f"Scene {scene_number}", scene_data['description'][:40]]):
                    text_row = text_y + i * 30
                    if text_row < height - 20:
                        # Simple text "drawing" by modifying pixels
                        start_x = width // 4
                        if i == 0:
                            # Make title bigger
                            frame[text_row-10:text_row+10, start_x:start_x+200] = [255, 255, 255]
                        else:
                            frame[text_row-5:text_row+5, start_x:start_x+300] = [200, 200, 200]
                
                return frame
            
            output_path = os.path.join(output_dir, f"scene_{scene_number}.mp4")
            animation = VideoClip(make_frame, duration=duration)
            animation.write_videofile(output_path, fps=10, verbose=False, logger=None)
            
            print(f"âœ… Created fallback video for scene {scene_number}")
            return output_path
            
        except Exception as e:
            print(f"â�Œ Fallback video also failed: {e}")
            return None

video_generator = SimpleVideoGenerator()


# Cell 6: Create Animation Renderer (SIMPLIFIED)
class AnimationRenderer:
    def create_scene_videos(self, storyboard):
        scene_files = []
        
        for i, scene in enumerate(storyboard["scenes"]):
            print(f"ğŸ�¨ Creating scene {i+1}: {scene['description'][:50]}...")
            video_path = video_generator.create_scene_video(scene, i+1, ASSETS_DIR)
            if video_path:
                scene_files.append(video_path)
                print(f"âœ… Created scene {i+1}")
            else:
                print(f"â�Œ Failed scene {i+1}")
        
        return scene_files

print("âœ… Animation Renderer ready!")


# Cell 7: Create TTS Engine
class TTSEngine:
    def __init__(self):
        self.available = True
        print("âœ… TTS engine (gTTS) loaded!")
    
    def generate_narration(self, text, output_path):
        try:
            # Use gTTS for free text-to-speech
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(output_path)
            print(f"ğŸ”Š Generated audio: {text[:50]}...")
            return output_path
        except Exception as e:
            print(f"TTS error: {e}")
            # Create empty audio file as fallback
            open(output_path, 'a').close()
            return None

tts_engine = TTSEngine()


# Cell 8: Choose your topic and generate storyboard
TOPIC = "Python For Loops"  # Change this to any topic!
print(f"ğŸ�¬ Creating animation about: {TOPIC}")

agent = EduAnimationAgent()
storyboard = agent.generate_storyboard(TOPIC)

# Save storyboard
storyboard_path = os.path.join(ASSETS_DIR, "storyboard.json")
with open(storyboard_path, 'w') as f:
    json.dump(storyboard, f, indent=2)

print("ğŸ“� Storyboard generated:")
print(json.dumps(storyboard, indent=2))


# Cell 9: Generate narration audio
print("ğŸ�¤ Generating narration with gTTS...")
audio_files = []

for i, scene in enumerate(storyboard["scenes"]):
    audio_path = os.path.join(ASSETS_DIR, "audio", f"scene_{i+1}.mp3")
    success = tts_engine.generate_narration(scene["narration"], audio_path)
    if success:
        audio_files.append(audio_path)

print(f"âœ… Created {len(audio_files)} audio files")


# Cell 10: Render animations
print("ğŸ�¨ Creating video scenes...")
renderer = AnimationRenderer()
scene_videos = renderer.create_scene_videos(storyboard)

print(f"âœ… Created {len(scene_videos)} video scenes")


# Cell 11: Create final composite video
print("ğŸ�¬ Assembling final video...")

if scene_videos:
    try:
        from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip
        
        # Combine all scenes into one video
        clips = []
        for i, video_path in enumerate(scene_videos):
            if os.path.exists(video_path):
                video_clip = VideoFileClip(video_path)
                
                # Add audio if available
                audio_path = os.path.join(ASSETS_DIR, "audio", f"scene_{i+1}.mp3")
                if os.path.exists(audio_path):
                    audio_clip = AudioFileClip(audio_path)
                    video_clip = video_clip.set_audio(audio_clip)
                
                clips.append(video_clip)
        
        if clips:
            # Concatenate all clips
            from moviepy.editor import concatenate_videoclips
            final_clip = concatenate_videoclips(clips)
            final_video_path = os.path.join(OUTPUT_DIR, "educational_demo.mp4")
            final_clip.write_videofile(final_video_path, fps=24, verbose=False, logger=None)
            
            # Close clips to free memory
            for clip in clips:
                clip.close()
            final_clip.close()
            
            print(f"âœ… Final composite video: {final_video_path}")
        else:
            print("â�Œ No valid clips to combine")
            final_video_path = None
            
    except Exception as e:
        print(f"â�Œ Video assembly failed: {e}")
        # Use first scene as fallback
        if scene_videos:
            import shutil
            final_video_path = os.path.join(OUTPUT_DIR, "educational_demo.mp4")
            shutil.copy2(scene_videos[0], final_video_path)
            print(f"âœ… Used first scene as final video: {final_video_path}")
        else:
            final_video_path = None
else:
    print("â�Œ No scenes rendered")
    final_video_path = None


# Cell 12: Display results
print("=" * 60)
print("ğŸ�‰ EDUCATIONAL ANIMATION COMPLETE!")
print("=" * 60)

if final_video_path and os.path.exists(final_video_path):
    print(f"âœ… Final video created: {final_video_path}")
    
    # Display in notebook
    from IPython.display import Video, display
    display(Video(final_video_path, embed=True, width=600))
else:
    print("âš ï¸�  Video creation had issues")
    print("But we still have:")
    print(f"ğŸ“„ Storyboard: {storyboard_path}")
    print(f"ğŸ”Š {len(audio_files)} audio files")
    print(f"ğŸ�¨ {len(scene_videos)} video scenes")

print(f"\nğŸ“� All assets saved in: {ASSETS_DIR}")


# Cell 13: Create Kaggle dataset package
print("ğŸ“¦ Creating Kaggle dataset structure...")

dataset_dir = os.path.join(WORKING_DIR, f"edu-animation-{TOPIC.lower().replace(' ', '-')}")
os.makedirs(dataset_dir, exist_ok=True)

# Copy all generated files
import shutil
files_copied = []

# Copy storyboard
if os.path.exists(storyboard_path):
    shutil.copy2(storyboard_path, dataset_dir)
    files_copied.append("storyboard.json")

# Copy videos
if scene_videos:
    for i, video in enumerate(scene_videos):
        if os.path.exists(video):
            shutil.copy2(video, os.path.join(dataset_dir, f"scene_{i+1}.mp4"))
            files_copied.append(f"scene_{i+1}.mp4")

# Copy audio files
for i, audio in enumerate(audio_files):
    if os.path.exists(audio):
        shutil.copy2(audio, os.path.join(dataset_dir, f"audio_{i+1}.mp3"))
        files_copied.append(f"audio_{i+1}.mp3")

# Copy final video
if final_video_path and os.path.exists(final_video_path):
    shutil.copy2(final_video_path, os.path.join(dataset_dir, "final_video.mp4"))
    files_copied.append("final_video.mp4")

# Create dataset metadata
dataset_metadata = {
    "title": f"Educational Animation: {TOPIC}",
    "id": f"your-username/edu-animation-{TOPIC.lower().replace(' ', '-')}",
    "description": f"AI-generated educational content about {TOPIC} using Gemini and animation tools",
    "licenses": [{"name": "CC0-1.0"}],
    "resources": [{"path": f, "description": "Generated asset"} for f in files_copied]
}

with open(os.path.join(dataset_dir, "dataset-metadata.json"), 'w') as f:
    json.dump(dataset_metadata, f, indent=2)

print(f"âœ… Kaggle dataset ready: {dataset_dir}")
print(f"ğŸ“¦ Files included: {files_copied}")
print("ğŸ“¤ To publish: Upload this folder as a new Kaggle dataset!")

