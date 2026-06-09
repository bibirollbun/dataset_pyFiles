from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
secret_value_0 = user_secrets.get_secret("GOOGLE_API_KEY")

API_KEY = secret_value_0
print("API Key loaded successfully!")


print("AI evaluation step skipped due to missing API access in competition environment.")


!pip install --quiet google-generativeai pillow matplotlib

# %% [markdown]
# 1) Configuration / Notes

# - Add your Gemini (Google) API key to Kaggle Secrets with the name `GOOGLE_API_KEY`.
# In the notebook UI: right panel -> Secrets -> Add Secret -> name: GOOGLE_API_KEY -> paste key
# - This notebook expects a UI screenshot uploaded as a dataset under `Input`.
# Recommended filename (inside dataset): `sample_ui.png`.
# - Do not publish your real API key in the notebook content or markdown.

# %%
# 2) Load API key from Kaggle Secrets and configure the Gemini client
from kaggle_secrets import UserSecretsClient
import os

user_secrets = UserSecretsClient()
# The secret name is GOOGLE_API_KEY — change only if you used a different secret name
GEMINI_API_KEY = user_secrets.get_secret("GOOGLE_API_KEY")

# Configure the google generative AI client
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load your API key from Kaggle Secrets
user_secrets = UserSecretsClient()
genai.configure(api_key=user_secrets.get_secret("GEMINI_API_KEY"))
# If you used a different environment name, set it here. We store the key in the lib's config.
genai.configure(api_key=GEMINI_API_KEY)
print("Gemini API key loaded from Kaggle Secrets.")

# %% [markdown]
# 3) Set image path

# Use the Kaggle Input panel (right sidebar) to add your dataset. After upload select the file and click
# "Insert file path" which will paste a path like `/kaggle/input/your-dataset-name/sample_ui.png`.

# Replace the value of IMAGE_PATH below with the path inserted by Kaggle, or set to the default below.

# %%
# Example path — replace with the path shown in your Kaggle Input panel if different

# Quick check: display the image

from PIL import Image
import google.generativeai as genai

# Configure Gemini
from kaggle_secrets import UserSecretsClient
kaggle_secrets = UserSecretsClient()
genai.configure(api_key=kaggle_secrets.get_secret("GEMINI_API_KEY"))

# Load the model
model = genai.GenerativeModel("gemini-1.5-flash")

# Load your UI screenshot
from PIL import Image
ui_image = Image.open('/kaggle/input/sample-ui/sample_ui.png')
ui_image

# Ask Gemini to analyze
response = model.generate_content([
    "Analyze this UI design and give UX improvements in bullet points.",
    ui_image
])

print(response.text)


#Path to your uploaded image
IMAGE_PATH = "/kaggle/input/smart-ui/sample_ui.png"

#Load image
ui_image = Image.open(IMAGE_PATH)

ui_image
import matplotlib.pyplot as plt

try:
    img = Image.open(IMAGE_PATH)
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    plt.show()
except FileNotFoundError:
    print("Image not found. Please upload a dataset and update IMAGE_PATH to the correct path.")

# %% [markdown]
# 4) Helper: prepare image bytes for Gemini inline data

# Gemini (google.generativeai) can accept images as base64 inline data in some usage patterns.
# Here we'll prepare the image as base64 so we can pass it to the multimodal generation call.

# %%
import base64

with open(IMAGE_PATH, "rb") as f:
    image_bytes = f.read()
    image_base64 = base64.b64encode(image_bytes).decode("utf-8")

print("Image encoded as base64 (length):", len(image_base64))

# %% [markdown]
# 5) Create prompt and call Gemini for UX review

# We will send a short instruction prompt to Gemini along with the image as inline data.
# The exact API call depends on the google.generativeai version; below is a clear example using
# `genai._client.post` style content creation via the provided library API.

# IMPORTANT: If your Kaggle environment blocks network calls or the exact client method differs,
# consult the google.generativeai docs. This scaffold shows the intent and a working pattern.

# %%
# Build a prompt — tweak to your taste
prompt_text = (
    "You are an expert UX designer. Evaluate the uploaded UI screenshot and provide:\n"
    "1) A short one-sentence summary of what the UI does/represents.\n"
    "2) 4 strengths of the design (short bullets).\n"
    "3) 5 clear improvements with actionable suggestions.\n"
    "4) A simple accessibility check (contrast, font size, alt text suggestions).\n"
    "5) Give an overall UX score out of 10 and a single-sentence rationale."
)

# Construct the request payload using the library helper
# NOTE: The exact function names and payload shape for images may change — this is a recommended pattern.
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# Load API key
kaggle_secrets = UserSecretsClient()
genai.configure(api_key=kaggle_secrets.get_secret("GOOGLE_API_KEY"))

from google import genai
client=genai.client()

for m in client.models.list():
    print(m.name) 

# Load model
model = genai.GenerativeModel("gemini-1.5-flash-latest")

# Generate UX review
response = model.generate_content([
    "Analyze this UI design and give UX improvements in bullet points.",
    ui_image
])

print("\nUX Suggestions:\n")
print(response.text)

# %% [markdown]
# 6) Format response (if needed) and save to a file

# If `text_out` is a long string with the UX feedback, we can save it for the Kaggle submission.

# %%
try:
    feedback_text = text_out
except NameError:
    feedback_text = "(No feedback produced)"

# Display and save
print("\n--- Final UX Feedback (preview) ---\n")
print(feedback_text)

# Save to file in notebook Output so you can attach it to your submission
with open('/kaggle/working/ux_feedback.txt', 'w', encoding='utf-8') as f:
    f.write(feedback_text)

print("Saved feedback to /kaggle/working/ux_feedback.txt")

# %% [markdown]
# 7) Notebook submission checklist (for the Kaggle Capstone)

# - [ ] Title and Subtitle filled in on the Capstone submission page
# - [ ] Card + Thumbnail image uploaded (use a screenshot)
# - [ ] Notebook is public and contains code + examples
# - [ ] Attach this notebook (choose Kaggle Notebook) in "Attachments"
# - [ ] Provide a short project description and media/gallery (optional)

# %% [markdown]
# Troubleshooting notes

# - If you see authentication errors: open the right-side Secrets panel and ensure your
# `GOOGLE_API_KEY` value is valid and has access to the Generative API in Google Cloud.
# - If the `genai.generate` call shape is different in your environment, consult the
# google.generativeai docs for your installed package version and adapt the request shape.
# - If network calls are blocked in Kaggle, run a local test or run the notebook in an environment
# that allows outbound requests.

# %% [markdown]
# End — Good luck with your Kaggle Capstone! 
# If you'd like I can also produce a short 30–60s demo script and a polished submission description
# you can paste into the Kaggle Capstone writeup page.


