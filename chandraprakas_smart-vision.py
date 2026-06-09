# Shows all files inside /kaggle/input
import os

for root, dirs, files in os.walk("/kaggle/input"):
    for f in files:
        print(os.path.join(root, f))


img_path = "/kaggle/input/sample/download.jpeg"

img = Image.open(img_path)

display(img)

description = describe_image(img)
print(description)


print("API Key Loaded:", api_key[:8] + "")


!pip install -q google-generativeai pillow


from kaggle_secrets import UserSecretsClient
import google.generativeai as genai
from PIL import Image
from IPython.display import display
import os
import base64


user_secrets = UserSecretsClient()
api_key = user_secrets.get_secret("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

print("API Key Loaded:", api_key[:8] + "")


model = genai.GenerativeModel("gemini-2.0-flash")
print("Model Ready ✅")


def load_image_bytes(img_path):
    with open(img_path, "rb") as f:
        return f.read()


BASE_PROMPT = """
You are Smart Vision Assistant for visually impaired users.
Describe the image in simple, clear sentences (3–6 sentences).
Mention object positions like left, right, and center.
Read any visible text as: 'The text says: ...'.
Do not guess age, race, or hidden details.
If something is unclear or cropped, say you are not sure.
"""

def describe_image(img_path):
    # Open image as a PIL Image
    img = Image.open(img_path)

    # Send prompt + image directly to the model
    response = model.generate_content(
        [BASE_PROMPT, img],
        generation_config={"temperature": 0.4}
    )

    return response.text.strip()


img_path = "/kaggle/input/sample/download.jpeg"

img = Image.open(img_path)
display(img)

print("\nSmart Vision Assistant Output:\n")
print(describe_image(img_path))

