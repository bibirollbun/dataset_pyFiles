import os
from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
from PIL import Image # For working with image files

try:
    user_secrets = UserSecretsClient()
    GOOGLE_API_KEY = user_secrets.get_secret("GEMINI_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    
    genai.configure(api_key=GOOGLE_API_KEY)
    
    print("âœ… Gemini API key setup complete. You are ready to call the model!")

except Exception as e:
    print(f"ğŸ”‘ Authentication Error: Please make sure you have added 'GEMINI_API_KEY' to your Kaggle secrets. Details: {e}")


# The model that accepts both text and image input
VISION_MODEL = 'gemini-2.5-flash' 
print(f"Target Model: {VISION_MODEL}")


# Path to uploaded image file
image_path = '/kaggle/input/me-as-a-child/my_memory.jpg' 

# Use the Python Imaging Library (PIL) to open the image
try:
    img = Image.open(image_path)
    print("âœ… Image loaded successfully.")
    # display the image to confirm it loaded correctly
    # from IPython.display import display
    # display(img) 
except FileNotFoundError:
    print(f"â�Œ Error: Image file not found at {image_path}. Please check your file path!")


# The objective prompt to get a detailed description
analysis_prompt = (
    "Provide a purely objective, highly detailed description of this photograph. "
    "Focus on subjects, setting, time of day, and any visible objects. "
    "Do not provide any subjective interpretation or emotion."
)

# Call the model with both the text prompt and the image
model = genai.GenerativeModel('gemini-2.5-flash') # Ensure your corrected model name is here!
response = model.generate_content([analysis_prompt, img])

print("\n--- AI Image Analysis ---")
print(response.text)
print("--------------------------")



# The objective analysis text from the model's previous response
analysis_result = response.text 

# The creative prompt template to generate the questions
creative_prompt = f"""
***System Instruction / Role Definition***
You are a **Personal Memory Archivist** named "Memory Lane Agent."
Your core function is to help a user trigger deeply personal, sensory, and narrative-focused memories related to a photograph. You must adopt a **warm, nostalgic, and empathetic tone**.

***Task***
Based *only* on the objective description provided below, generate a list of **5 to 7 open-ended questions** designed to elicit a rich, detailed story from the user.

***Constraints & Guidelines***
1.  **Focus on Senses & Emotion:** Questions must prompt recall of *feelings, smells, sounds, tastes, and atmosphere*, not just facts.
2.  **Avoid Simple Facts:** Do not ask questions that could be answered with a single word (e.g., "What day was this?").
3.  **Output Format:** The final response *must* be a clean, numbered list of questions only.

***Objective Image Analysis***
<IMAGE_ANALYSIS_TEXT>
"""

# Replace the placeholder with the actual analysis result
final_prompt = creative_prompt.replace("<IMAGE_ANALYSIS_TEXT>", analysis_result)

# Now, send this final prompt to the model (we can use the same model as it handles text perfectly)
# VISION_MODEL is 'gemini-2.5-flash'
response_questions = model.generate_content(final_prompt)

print("\n--- Memory Lane Agent Questions ---")
print(response_questions.text)
print("-----------------------------------")

