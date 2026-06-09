# Install required packages
!pip install -q openai google-generativeai pillow matplotlib requests tqdm


import os
import io
import time
import base64
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from tqdm.auto import tqdm
from pathlib import Path

# Set API keys
os.environ["OPENAI_API_KEY"] = "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"
os.environ["GOOGLE_API_KEY"] = "AIzaSyBaBVowNeOeSDAGOfTF2sNQkoCL3SjxMK4"

# Import API libraries
import openai
import google.generativeai as genai

# Configure API clients
openai.api_key = os.environ["OPENAI_API_KEY"]
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# Set random seeds for reproducibility
np.random.seed(42)

print("Environment setup complete!")


# Create a diverse set of prompts across different categories
prompts_data = [
    # Nature category
    {"prompt": "A serene mountain lake at sunset with reflections of pine trees", "category": "nature"},
    {"prompt": "A dense rainforest with rays of sunlight breaking through the canopy", "category": "nature"},
    {"prompt": "A dramatic coastal cliff with waves crashing against rocks", "category": "nature"},
    {"prompt": "A field of wildflowers under a stormy sky", "category": "nature"},
    {"prompt": "An underwater coral reef teeming with colorful fish", "category": "nature"},
    
    # Urban category
    {"prompt": "A futuristic cityscape with flying vehicles and neon lights", "category": "urban"},
    {"prompt": "A narrow cobblestone street in an old European city", "category": "urban"},
    {"prompt": "A bustling night market with food stalls and lanterns", "category": "urban"},
    {"prompt": "An abandoned subway station reclaimed by nature", "category": "urban"},
    {"prompt": "A rooftop garden overlooking a modern city skyline", "category": "urban"},
    
    # Fantasy category
    {"prompt": "A floating island castle with waterfalls cascading from its edges", "category": "fantasy"},
    {"prompt": "A magical forest with glowing plants and mythical creatures", "category": "fantasy"},
    {"prompt": "A dragon perched on a mountain peak against a full moon", "category": "fantasy"},
    {"prompt": "A wizard's study filled with ancient books and magical artifacts", "category": "fantasy"},
    {"prompt": "A portal between worlds showing contrasting landscapes", "category": "fantasy"},
    
    # Abstract category
    {"prompt": "A visual representation of quantum entanglement", "category": "abstract"},
    {"prompt": "The concept of time as a physical dimension", "category": "abstract"},
    {"prompt": "A dream-like landscape that defies physics", "category": "abstract"},
    {"prompt": "The intersection of mathematics and art", "category": "abstract"},
    {"prompt": "Emotions visualized as colors and shapes", "category": "abstract"}
]

# Convert to DataFrame
df = pd.DataFrame(prompts_data)
print(f"Total prompts: {len(df)}")
df.head()


# Prompt enhancement with style and quality descriptors
style_descriptors = {
    "nature": ["photorealistic", "high resolution", "detailed", "professional photography", "golden hour lighting", "National Geographic style"],
    "urban": ["cinematic", "detailed", "atmospheric", "high contrast", "dramatic lighting", "architectural photography"],
    "fantasy": ["digital art", "highly detailed", "concept art", "intricate", "epic", "fantasy art", "artstation"],
    "abstract": ["surrealist", "vibrant colors", "minimalist", "geometric", "expressionist", "modern art"]
}

def enhance_prompt(row):
    # Get category-specific descriptors
    category = row['category']
    descriptors = style_descriptors.get(category, style_descriptors['nature'])
    
    # Select random descriptors
    selected_descriptors = np.random.choice(descriptors, size=2, replace=False)
    
    # Add common quality descriptors
    quality_descriptors = ["high quality", "8K"]
    
    # Combine all descriptors
    all_descriptors = ", ".join(list(selected_descriptors) + quality_descriptors)
    
    # Create enhanced prompt
    enhanced = f"{row['prompt']}, {all_descriptors}"
    return enhanced

# Apply prompt enhancement
df['enhanced_prompt'] = df.apply(enhance_prompt, axis=1)

# Display examples of original vs enhanced prompts
comparison_df = pd.DataFrame({
    'Category': df['category'].head(4),
    'Original Prompt': df['prompt'].head(4),
    'Enhanced Prompt': df['enhanced_prompt'].head(4)
})
comparison_df


def generate_image_dalle(prompt, size="1024x1024", quality="standard", model="dall-e-3"):
    """Generate an image using OpenAI's DALL-E API"""
    try:
        response = openai.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
            n=1,
        )
        
        # Get image URL
        image_url = response.data[0].url
        
        # Download image
        image_response = requests.get(image_url)
        image = Image.open(io.BytesIO(image_response.content))
        
        return image, None
    except Exception as e:
        return None, str(e)

# Test DALL-E image generation with a sample prompt
test_prompt = "A serene mountain lake at sunset with reflections of pine trees, photorealistic, high quality"
print(f"Generating image with prompt: {test_prompt}")

try:
    image, error = generate_image_dalle(test_prompt)
    if image:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis('off')
        plt.title("DALL-E Generated Image")
        plt.show()
    else:
        print(f"Error generating image: {error}")
except Exception as e:
    print(f"Error: {str(e)}")


def generate_image_gemini(prompt):
    """Generate an image using Google's Gemini API"""
    try:
        # Configure the generative model
        generation_config = {
            "temperature": 0.9,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 4096,
        }
        
        # Initialize the model
        model = genai.GenerativeModel('gemini-pro-vision', generation_config=generation_config)
        
        # Generate content with image request
        response = model.generate_content([prompt, "Generate a detailed image based on this description."])
        
        # Process the response to extract image data
        # Note: This is a simplified approach as Gemini doesn't directly generate images
        # In a real implementation, we would use the text response to guide another image generation tool
        
        # For demonstration, we'll use a placeholder approach
        # In a real scenario, we would use the Gemini response to guide another image generation tool
        return None, "Gemini API doesn't directly generate images. In a production environment, we would use Gemini's response to guide another image generation tool."
    except Exception as e:
        return None, str(e)

# Test Gemini integration
test_prompt = "A serene mountain lake at sunset with reflections of pine trees, photorealistic, high quality"
print(f"Testing Gemini API with prompt: {test_prompt}")

try:
    _, message = generate_image_gemini(test_prompt)
    print(f"Gemini API response: {message}")
except Exception as e:
    print(f"Error: {str(e)}")


def generate_image_with_fallback(prompt, max_retries=3):
    """Generate an image with fallback mechanisms"""
    # Try DALL-E first
    for attempt in range(max_retries):
        print(f"Attempt {attempt+1}/{max_retries} with DALL-E")
        image, error = generate_image_dalle(prompt)
        if image:
            return image, "dalle", None
        
        print(f"DALL-E attempt {attempt+1} failed: {error}")
        time.sleep(2)  # Wait before retry
    
    # If DALL-E fails, try alternative methods
    # In a production environment, we would implement additional fallback methods
    # For this demonstration, we'll create a placeholder image
    print("All DALL-E attempts failed. Creating placeholder image.")
    
    # Create a simple placeholder image
    placeholder = Image.new('RGB', (512, 512), color=(200, 200, 200))
    return placeholder, "placeholder", "All generation attempts failed"

# Test the fallback mechanism
test_prompt = "A magical forest with glowing plants and mythical creatures, digital art, highly detailed, high quality"
print(f"Testing fallback mechanism with prompt: {test_prompt}")

try:
    image, source, error = generate_image_with_fallback(test_prompt)
    plt.figure(figsize=(10, 10))
    plt.imshow(image)
    plt.axis('off')
    plt.title(f"Image generated using {source}")
    plt.show()
    
    if error:
        print(f"Note: {error}")
except Exception as e:
    print(f"Error: {str(e)}")


def generate_batch_images(prompts, output_dir="generated_images", max_images=5):
    """Generate images for a batch of prompts"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Select prompts for generation
    if len(prompts) > max_images:
        selected_prompts = prompts.sample(max_images).tolist()
    else:
        selected_prompts = prompts.tolist()
    
    # Generate images
    results = []
    for i, prompt in enumerate(tqdm(selected_prompts, desc="Generating images")):
        # Generate image
        image, source, error = generate_image_with_fallback(prompt)
        
        # Save image
        image_path = os.path.join(output_dir, f"image_{i+1}.png")
        image.save(image_path)
        
        # Save prompt
        prompt_path = os.path.join(output_dir, f"prompt_{i+1}.txt")
        with open(prompt_path, "w") as f:
            f.write(prompt)
        
        # Store result
        results.append({
            "prompt": prompt,
            "image_path": image_path,
            "source": source,
            "error": error
        })
    
    return results

# Generate a batch of images
print("Generating a batch of images...")
batch_results = generate_batch_images(df['enhanced_prompt'], max_images=3)

# Display results
plt.figure(figsize=(15, 5 * len(batch_results)))
for i, result in enumerate(batch_results):
    plt.subplot(len(batch_results), 1, i+1)
    image = Image.open(result['image_path'])
    plt.imshow(image)
    plt.title(f"Source: {result['source']}")
    plt.axis('off')
    print(f"Prompt: {result['prompt']}")
    if result['error']:
        print(f"Error: {result['error']}")

plt.tight_layout()
plt.show()


def analyze_results(results):
    """Analyze the results of batch image generation"""
    # Count sources
    source_counts = {}
    for result in results:
        source = result['source']
        source_counts[source] = source_counts.get(source, 0) + 1
    
    # Count errors
    error_count = sum(1 for result in results if result['error'])
    
    # Print analysis
    print("=== Image Generation Analysis ===")
    print(f"Total images generated: {len(results)}")
    print("\nSource distribution:")
    for source, count in source_counts.items():
        print(f"  - {source}: {count} images ({count/len(results)*100:.1f}%)")
    
    print(f"\nErrors encountered: {error_count} ({error_count/len(results)*100:.1f}%)")
    
    # Plot source distribution
    plt.figure(figsize=(10, 6))
    plt.bar(source_counts.keys(), source_counts.values())
    plt.title("Image Generation Source Distribution")
    plt.xlabel("Source")
    plt.ylabel("Count")
    plt.show()

# Analyze the batch results
analyze_results(batch_results)


def prepare_submission(results, output_dir="submission"):
    """Prepare submission files"""
    # Create submission directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy images to submission directory
    for i, result in enumerate(results):
        # Copy image
        image = Image.open(result['image_path'])
        submission_path = os.path.join(output_dir, f"submission_{i+1}.png")
        image.save(submission_path)
        
        # Save metadata
        metadata_path = os.path.join(output_dir, f"metadata_{i+1}.txt")
        with open(metadata_path, "w") as f:
            f.write(f"Prompt: {result['prompt']}\n")
            f.write(f"Source: {result['source']}\n")
            if result['error']:
                f.write(f"Error: {result['error']}\n")
    
    # Create summary file
    summary_path = os.path.join(output_dir, "submission_summary.txt")
    with open(summary_path, "w") as f:
        f.write("OpenAI To-Z Challenge Submission\n")
        f.write("================================\n\n")
        f.write(f"Total images: {len(results)}\n\n")
        
        for i, result in enumerate(results):
            f.write(f"Image {i+1}:\n")
            f.write(f"  Prompt: {result['prompt']}\n")
            f.write(f"  Source: {result['source']}\n")
            if result['error']:
                f.write(f"  Error: {result['error']}\n")
            f.write("\n")
    
    print(f"Submission prepared in {output_dir}/")
    return output_dir

# Prepare submission
submission_dir = prepare_submission(batch_results)

# List submission files
submission_files = os.listdir(submission_dir)
print("Submission files:")
for file in submission_files:
    print(f"  - {file}")

