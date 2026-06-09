import os
from google import genai
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Markdown
from kaggle_secrets import UserSecretsClient # Import the secure Kaggle secret client

# --- 1. SECURE SETUP: Load Key and Set Working Model ID ---
try:
    # ğŸ”‘ Fetches key securely from the Kaggle Add-ons > Secrets menu (No key exposed)
    user_secrets = UserSecretsClient()
    API_KEY = user_secrets.get_secret("GEMINI_API_KEY") 
    
    # ğŸ› ï¸� MODEL FIX: Switched to 'flash' to bypass your current 429 quota error.
    MODEL_NAME = "gemini-3.0 pro"
    
    client = genai.Client(api_key=API_KEY)
    print(f"âœ… Client initialized securely! Using working model: {MODEL_NAME}")
    print("\nâ„¹ï¸� REMINDER: Use 'gemini-3-pro-preview' for your final submission code.")
    
except Exception as e:
    print(f"â�Œ Setup Error: {e}")
    print("ACTION: Ensure 'GEMINI_API_KEY' is active in Add-ons > Secrets.")
    client = None

# --- 2. Create the Image (The Quantum Formula) ---
def create_image():
    try:
        img = Image.new('RGB', (500, 250), color='white')
        d = ImageDraw.Draw(img)
        
        try:
            # Attempt to use a common font, fallback to default
            font = ImageFont.truetype("DejaVuSans.ttf", 20)
        except:
            font = ImageFont.load_default()
            
        # Draw the Quantum Mechanics formula (Particle in a Box)
        d.text((20, 50), "Psi(x) = sqrt(2/L) * sin(n * pi * x / L)", fill='black', font=font)
        d.text((20, 100), "Region: 0 < x < L", fill='red', font=font)
        d.text((20, 150), "V(x) = 0 inside box, infinity outside", fill='blue', font=font)
        return img
    except Exception as e:
        print(f"Image Creation Error: {e}")
        return None

image_input = create_image()
if image_input:
    print("\n created and ready.]")
    display(image_input)

# --- 3. The Prompt ---
prompt = """
Analyze this image of a physics formula. 
1. Identify the equation.
2. Explain it using the 'Guitar String' analogy.
3. Create a Markdown table defining the variables.
"""

# --- 4. Call the API and Get Output ---
if client and image_input:
    print("\n--- Sending request to Gemini... ---")
    try:
        response = client.models.generate_content(
            model=MODEL_NAME,
            contents=[image_input, prompt]
        )
        print("\n## âœ¨ FINAL OUTPUT GENERATED! âœ¨\n")
        display(Markdown(response.text))
    except Exception as e:
        # This should only fail if the new model hits a limit too
        print(f"\nâ�Œ API Call Failed: {e}")




