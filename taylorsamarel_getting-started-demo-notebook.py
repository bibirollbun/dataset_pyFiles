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


from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
gemini_api_key = user_secrets.get_secret("gemini_api_key")


# ğŸ�Œ NANO BANANA HACKATHON - COMPLETE WORKING NOTEBOOK
# Competition: https://www.kaggle.com/competitions/banana
# Based on official documentation and working examples
# No placeholders - ready to run!

"""
â•”â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•—
â•‘  NANO BANANA (Gemini 2.5 Flash Image Preview) - Image Generation         â•‘
â•‘  48-Hour Hackathon: September 6-7, 2025                                  â•‘
â•‘  Win your share of $400,000+ in prizes!                                  â•‘
â•šâ•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�â•�
"""

print("ğŸ�Œ Starting Nano Banana Hackathon Notebook...")
print("=" * 80)

# ============================================================================
# SECTION 1: IMPORTS AND SETUP
# ============================================================================

import os
import sys
import time
import mimetypes
from datetime import datetime
from PIL import Image
from io import BytesIO
from IPython.display import display, Markdown, HTML

print("âœ… Standard libraries imported")

# Install required packages
print("\nğŸ“¦ Installing required packages...")
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "-q", "google-genai>=1.32.0", "Pillow"])
print("âœ… Packages installed")

# Import Google Gemini
from google import genai
from google.genai import types
print("âœ… Google Gemini SDK imported")

# ============================================================================
# SECTION 2: API KEY CONFIGURATION (Using Kaggle Secrets)
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ”‘ CONFIGURING API KEY")
print("=" * 80)

# Get API key from Kaggle secrets
try:
    from kaggle_secrets import UserSecretsClient
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("gemini_api_key")
    if GOOGLE_API_KEY:
        print("âœ… API key loaded from Kaggle secrets")
        print(f"   Key length: {len(GOOGLE_API_KEY)} characters")
    else:
        raise ValueError("No API key found")
except Exception as e:
    print(f"â�Œ Error loading API key: {e}")
    print("\nğŸ“� INSTRUCTIONS:")
    print("1. Get your FREE API key at: https://ai.studio/banana")
    print("2. In Kaggle, go to 'Add-ons' -> 'Secrets'")
    print("3. Add a new secret named 'gemini_api_key' with your API key")
    raise

# ============================================================================
# SECTION 3: INITIALIZE CLIENT
# ============================================================================

print("\n" + "=" * 80)
print("ğŸš€ INITIALIZING CLIENT")
print("=" * 80)

# Initialize the client
client = genai.Client(api_key=GOOGLE_API_KEY)
MODEL_ID = "gemini-2.5-flash-image-preview"
MODEL_TEXT = "gemini-2.5-flash"

print(f"âœ… Client initialized")
print(f"ğŸ“· Image Model: {MODEL_ID}")
print(f"ğŸ“� Text Model: {MODEL_TEXT}")

# Test the client
try:
    test_response = client.models.generate_content(
        model=MODEL_TEXT,
        contents="Say 'OK' if you're working"
    )
    print("âœ… Client test successful!")
except Exception as e:
    print(f"â�Œ Client test failed: {e}")

# ============================================================================
# SECTION 4: HELPER FUNCTIONS (Based on Working Example)
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ› ï¸� SETTING UP HELPER FUNCTIONS")
print("=" * 80)

def save_binary_file(file_name, data):
    """Save binary data to file"""
    with open(file_name, "wb") as f:
        f.write(data)
    print(f"ğŸ’¾ File saved to: {file_name}")
    return file_name

def display_response(response):
    """Display text and images from response"""
    if not response:
        print("âš ï¸� No response to display")
        return
    
    # Handle candidates structure
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for part in candidate.content.parts:
                        process_part(part)
    # Handle direct parts structure
    elif hasattr(response, 'parts'):
        for part in response.parts:
            process_part(part)

def process_part(part):
    """Process a single part (text or image)"""
    # Display text
    if hasattr(part, 'text') and part.text:
        display(Markdown(part.text))
    
    # Display image
    if hasattr(part, 'inline_data') and part.inline_data:
        if hasattr(part.inline_data, 'data') and part.inline_data.data:
            try:
                image = Image.open(BytesIO(part.inline_data.data))
                print(f"ğŸ“· Image: {image.size[0]}x{image.size[1]} pixels")
                display(image)
            except Exception as e:
                print(f"âš ï¸� Could not display image: {e}")

def extract_and_save_images(response, prefix="image"):
    """Extract and save all images from response"""
    saved_files = []
    
    if not response:
        return saved_files
    
    # Handle candidates structure
    if hasattr(response, 'candidates') and response.candidates:
        for candidate in response.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts'):
                    for i, part in enumerate(candidate.content.parts):
                        if hasattr(part, 'inline_data') and part.inline_data:
                            if hasattr(part.inline_data, 'data') and part.inline_data.data:
                                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                mime_type = getattr(part.inline_data, 'mime_type', 'image/png')
                                ext = mimetypes.guess_extension(mime_type) or '.png'
                                filename = f"{prefix}_{timestamp}_{i}{ext}"
                                save_binary_file(filename, part.inline_data.data)
                                saved_files.append(filename)
    
    # Handle direct parts structure
    elif hasattr(response, 'parts'):
        for i, part in enumerate(response.parts):
            if hasattr(part, 'inline_data') and part.inline_data:
                if hasattr(part.inline_data, 'data') and part.inline_data.data:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    mime_type = getattr(part.inline_data, 'mime_type', 'image/png')
                    ext = mimetypes.guess_extension(mime_type) or '.png'
                    filename = f"{prefix}_{timestamp}_{i}{ext}"
                    save_binary_file(filename, part.inline_data.data)
                    saved_files.append(filename)
    
    return saved_files

print("âœ… Helper functions ready")

# ============================================================================
# SECTION 5: BASIC IMAGE GENERATION (NON-STREAMING)
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ�¨ DEMO 1: BASIC IMAGE GENERATION")
print("=" * 80)

def generate_image_basic(prompt):
    """Generate image using basic API call"""
    print(f"\nğŸ“� Prompt: {prompt[:100]}...")
    
    try:
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[prompt],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        if response:
            print("âœ… Response received!")
            display_response(response)
            saved_files = extract_and_save_images(response, "basic")
            return saved_files
        else:
            print("âš ï¸� Empty response")
            return []
            
    except Exception as e:
        print(f"â�Œ Error: {e}")
        return []

# Test basic generation
test_prompt = "Create a photorealistic image of a glowing nano-banana floating in space with colorful nebulas in the background"
print("ğŸš€ Testing basic image generation...")
basic_files = generate_image_basic(test_prompt)
print(f"ğŸ“Š Generated {len(basic_files)} image(s)")

# ============================================================================
# SECTION 6: STREAMING IMAGE GENERATION (RECOMMENDED)
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ�¨ DEMO 2: STREAMING IMAGE GENERATION")
print("=" * 80)

def generate_image_streaming(prompt, save_prefix="stream"):
    """Generate image using streaming API (more reliable)"""
    print(f"\nğŸ“� Prompt: {prompt[:100]}...")
    saved_files = []
    
    try:
        contents = [
            types.Content(
                role="user",
                parts=[types.Part.from_text(text=prompt)],
            )
        ]
        
        config = types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"]
        )
        
        print("ğŸ”„ Starting stream...")
        file_index = 0
        
        for chunk in client.models.generate_content_stream(
            model=MODEL_ID,
            contents=contents,
            config=config,
        ):
            print(".", end="", flush=True)
            
            if (chunk.candidates is None or 
                chunk.candidates[0].content is None or 
                chunk.candidates[0].content.parts is None):
                continue
            
            for part in chunk.candidates[0].content.parts:
                # Handle text
                if hasattr(part, 'text') and part.text:
                    print(f"\nğŸ“� Text: {part.text[:100]}...")
                
                # Handle image
                if part.inline_data and part.inline_data.data:
                    mime_type = getattr(part.inline_data, 'mime_type', 'image/png')
                    ext = mimetypes.guess_extension(mime_type) or '.png'
                    filename = f"{save_prefix}_{file_index}{ext}"
                    save_binary_file(filename, part.inline_data.data)
                    saved_files.append(filename)
                    file_index += 1
                    
                    # Display the image
                    try:
                        image = Image.open(BytesIO(part.inline_data.data))
                        print(f"\nğŸ“· Image generated: {image.size[0]}x{image.size[1]}")
                        display(image)
                    except:
                        pass
        
        print(f"\nâœ… Streaming complete! Generated {len(saved_files)} image(s)")
        return saved_files
        
    except Exception as e:
        print(f"\nâ�Œ Streaming error: {e}")
        return saved_files

# Test streaming generation
stream_prompt = "A futuristic robot chef cooking glowing nano-bananas in a high-tech kitchen with holographic recipe displays"
print("ğŸš€ Testing streaming image generation...")
stream_files = generate_image_streaming(stream_prompt, "robot_chef")
print(f"ğŸ“Š Generated {len(stream_files)} image(s)")

# ============================================================================
# SECTION 7: STORY SEQUENCE GENERATION
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ“– DEMO 3: VISUAL STORY GENERATION")
print("=" * 80)

def generate_story_scenes(story_prompt, num_scenes=3):
    """Generate scene descriptions for a story"""
    prompt = (
        f"Break this story into {num_scenes} short scenes.\n"
        "For each scene, provide:\n"
        "1. A title\n"
        "2. A short description\n"
        "3. Detailed visual storytelling for image generation\n\n"
        f"Story: {story_prompt}"
    )
    
    response = client.models.generate_content(
        model=MODEL_TEXT,
        contents=[prompt]
    )
    
    return response.text if response else ""

def generate_story_images(story_prompt, num_scenes=3):
    """Generate a complete visual story"""
    print(f"ğŸ“š Creating {num_scenes}-scene story...")
    
    # Generate scene descriptions
    scenes_text = generate_story_scenes(story_prompt, num_scenes)
    print("âœ… Scene descriptions generated")
    
    # Generate images for the story
    full_prompt = f"""Create {num_scenes} distinct images that tell this story:

{story_prompt}

Requirements:
- Generate {num_scenes} separate, high-quality images
- Maintain visual consistency across scenes
- Tell the story through imagery without text in the images
- Make each scene visually compelling and emotionally engaging"""
    
    saved_files = generate_image_streaming(full_prompt, "story")
    
    print(f"\nâœ… Story complete with {len(saved_files)} scenes!")
    return saved_files

# Test story generation
story = "A tiny nano-banana gains consciousness and explores a giant laboratory, making friends with the scientists"
story_files = generate_story_images(story, 3)

# ============================================================================
# SECTION 8: IMAGE EDITING
# ============================================================================

print("\n" + "=" * 80)
print("âœ�ï¸� DEMO 4: IMAGE EDITING")
print("=" * 80)

def edit_image(image_path, edit_prompt):
    """Edit an existing image with text prompt"""
    print(f"ğŸ“� Edit prompt: {edit_prompt[:100]}...")
    
    try:
        # Open the image
        image = Image.open(image_path)
        print(f"ğŸ“· Loaded image: {image.size[0]}x{image.size[1]}")
        
        # Send to API with edit prompt
        response = client.models.generate_content(
            model=MODEL_ID,
            contents=[edit_prompt, image],
            config=types.GenerateContentConfig(
                response_modalities=['TEXT', 'IMAGE']
            )
        )
        
        if response:
            print("âœ… Edit complete!")
            display_response(response)
            saved_files = extract_and_save_images(response, "edited")
            return saved_files
        
        return []
        
    except Exception as e:
        print(f"â�Œ Edit failed: {e}")
        return []

# Test image editing (if we have a generated image)
if stream_files:
    edit_prompt = "Add a colorful rainbow in the background and make the scene more vibrant and magical"
    print(f"ğŸ�¨ Editing {stream_files[0]}...")
    edited_files = edit_image(stream_files[0], edit_prompt)
    print(f"ğŸ“Š Created {len(edited_files)} edited image(s)")

# ============================================================================
# SECTION 9: ADVANCED TECHNIQUES
# ============================================================================

print("\n" + "=" * 80)
print("ğŸš€ ADVANCED TECHNIQUES")
print("=" * 80)

# 1. Photorealistic Generation
print("\n1ï¸�âƒ£ Photorealistic Image:")
photo_prompt = """A photorealistic close-up portrait of an elderly Japanese ceramicist, 
working on a delicate tea bowl, set in a traditional workshop. The scene is illuminated 
by soft morning light filtering through paper screens, creating a serene atmosphere. 
Captured with a 85mm lens, emphasizing the weathered hands and concentrated expression. 
Ultra-detailed, 8K quality."""

photo_files = generate_image_streaming(photo_prompt, "photorealistic")

# 2. Logo/Text Generation
print("\n2ï¸�âƒ£ Logo with Text:")
logo_prompt = """Create a modern, minimalist logo for 'Nano Banana Labs' with the text 
clearly rendered in a clean, futuristic font. The design should feature a stylized 
banana icon that incorporates circuit patterns, using a yellow and silver color scheme."""

logo_files = generate_image_streaming(logo_prompt, "logo")

# 3. Multiple Image Composition
print("\n3ï¸�âƒ£ Multi-Image Story Panel:")
panel_prompt = """Create a 4-panel comic strip showing:
Panel 1: A scientist discovers a glowing nano-banana
Panel 2: The banana starts floating and emitting light
Panel 3: The scientist's expression of amazement
Panel 4: The banana transforms into a portal
Make it colorful and engaging with clear visual storytelling."""

panel_files = generate_image_streaming(panel_prompt, "comic")

# ============================================================================
# SECTION 10: HACKATHON TIPS & FINAL STATUS
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ�† HACKATHON TIPS & RESOURCES")
print("=" * 80)

tips = """
## ğŸ�¯ Key Strategies for Success:

### Image Generation Tips:
1. **Be Descriptive**: Don't just list keywords - describe the scene narratively
2. **Use Photography Terms**: Mention lighting, angles, lenses for photorealistic results
3. **Specify Style**: Be explicit about artistic style, mood, and atmosphere
4. **Request High Quality**: Add "8K", "ultra-detailed", "professional" for better results

### API Best Practices:
- Use streaming for better reliability
- Save images immediately as they arrive
- Implement retry logic for failed requests
- Monitor your rate limits (20/min, 200/day for free tier)

### Hackathon Strategy:
- Focus on character consistency across images
- Leverage image editing for iterative refinement
- Combine multiple images for creative compositions
- Use text rendering for logos and diagrams

### Submission Requirements:
1. Video demo (â‰¤2 minutes)
2. Public project link (GitHub or AI Studio)
3. Writeup (â‰¤200 words) explaining Gemini features used

### Resources:
- Get API Key: https://ai.studio/banana
- Competition: https://www.kaggle.com/competitions/banana
- Documentation: https://ai.google.dev/gemini-api/docs
"""

display(Markdown(tips))

# ============================================================================
# FINAL STATUS REPORT
# ============================================================================

print("\n" + "=" * 80)
print("ğŸ“Š SESSION SUMMARY")
print("=" * 80)

# Count generated files
all_files = [f for f in os.listdir('.') if f.endswith(('.png', '.jpg', '.jpeg'))]
print(f"âœ… Total images generated: {len(all_files)}")
print(f"ğŸ“� Files: {', '.join(all_files[:5])}..." if len(all_files) > 5 else f"ğŸ“� Files: {', '.join(all_files)}")

print(f"""
ğŸ�‰ Your Nano Banana environment is fully operational!

Quick Reference:
â€¢ generate_image_basic(prompt) - Simple generation
â€¢ generate_image_streaming(prompt, prefix) - Streaming (recommended)
â€¢ generate_story_images(story, scenes) - Multi-scene stories
â€¢ edit_image(image_path, prompt) - Edit existing images

Remember:
â€¢ 48-hour hackathon: Sept 6-7, 2025
â€¢ Free tier: 200 images/day, 20/minute
â€¢ $400,000+ in prizes!

ğŸ�Œ Good luck with the Nano Banana Hackathon! ğŸ�Œ
""")

print("=" * 80)
print("âœ… NOTEBOOK COMPLETE - HAPPY HACKING!")
print("=" * 80)

