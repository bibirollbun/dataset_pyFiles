# ğŸ�„ GSAFE: Offline Mushroom Classifier
# ===============================
# Install dependencies
# ===============================
!pip install timm==1.0.17
!pip install transformers==4.53.2


# ===============================
# 1ï¸�âƒ£ Load model & processor
# ===============================

import kagglehub
from transformers import AutoProcessor, AutoModelForImageTextToText

GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")


# ===============================
# 2ï¸�âƒ£ Initialize conversation memory
# ===============================
system_prompt = (
    "You are a mushroom safety expert. "
    "Task: Analyze mushroom images and other input information to assess edibility of the mushroom. "
    "Consider mushroom's visual characteristics, location, and time of year, if provided. "
    "Rules for classification of the mushroom:"
    "0. Always end with final answer 'Edible' or 'Poisonous', or 'Uncertain' according to the steps below; "
    "1. If you can conclude with at least 80% confidence it is Edible, final asnwer is 'Edible'; " 
    "2. If not, and you are at least 80% sure it is Poisonous, final answer is 'Poisonous'; "
    "3. If you do not know, final answer is 'Uncertain' and request at most another picture from a different angle. At the next iteration, you can only answer 'Edible' or, if not sure, 'Poisonous'. "
    " IMPORTANT: Your last word in the answer must be only 'Edible, 'Poisonous', or 'Uncertain' (according to the rules above). "
)

#we use this to reset the conversation to the system prompt
conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

def reset_conversation():
    global conversation
    conversation = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]}
    ]

reset_conversation()

# ===============================
# 3ï¸�âƒ£ Helper function to send a message
# ===============================

from PIL import Image
from IPython.display import display

def ask_mushroom_question(user_text=None, image_path=None, date=None, location=None, verbose=True):
    global conversation

     # Auto date/location if not provided
    date_str = f"Date is {date};" if date else ""
    location_str = f"Location is {location};" if location else ""

    # Append to user_text if not empty
    if date_str or location_str:
        user_text = f"{user_text} {location_str} {date_str}".strip()

    # Load image if provided
    image_content = []
    if image_path:
        img = Image.open(image_path).convert("RGB").resize((384, 384))
        image_content.append({"type": "image", "image": img})

    # Append user turn
    user_entry = {"role": "user", "content": []}
    if image_content:
        user_entry["content"].extend(image_content)
    if user_text:
        user_entry["content"].append({"type": "text", "text": user_text})

    conversation.append(user_entry)

    # Tokenize conversation
    inputs = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]

    # Generate
    outputs = model.generate(**inputs, max_new_tokens=256, disable_compile=True, temperature = 0.1)

    # Decode
    response_text = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )[0]

    # Append assistant's turn to conversation memory
    conversation.append({
        "role": "assistant",
        "content": [{"type": "text", "text": response_text}]
    })

    # Classification
    last_word = response_text.split()[-1].lower()
    if "poison" in last_word:
        label = "ğŸ”´ NO"
    elif "edible" in last_word:
        label = "ğŸŸ¢ OK"
    else:
        label = "âšª ?"

    if verbose:
        # Print
        print("===== Reasoning & Answer =====")
        print(response_text)
        print("\n===== Classification =====")
        print(label)

    return response_text, label


img = Image.open("/kaggle/input/mushrooms-pics-for-competition/pic_1.jpg").convert("RGB").resize((384, 384))
display(img)



reset_conversation()
[r,l]=ask_mushroom_question(
    user_text="Is this mushroom edible or poisonous?",
    image_path="/kaggle/input/mushrooms-pics-for-competition/pic_1.jpg",
    date="October 2016",
    location="North of Italy, fields near fruit trees"
)


img = Image.open("/kaggle/input/mushrooms-pics-for-competition/pic_2.jpg").convert("RGB").resize((384, 384))
display(img)


reset_conversation()
[r,l]=ask_mushroom_question(
    user_text="Are these mushrooms edible?",
    image_path="/kaggle/input/mushrooms-pics-for-competition/pic_2.jpg"
)


img = Image.open("/kaggle/input/mushrooms-pics-for-competition/pic_3.jpg").convert("RGB").resize((384, 384))
display(img)


[r,l]=ask_mushroom_question(
    user_text="Here another pic",
    image_path="/kaggle/input/mushrooms-pics-for-competition/pic_3.jpg",
    location = "found in a forested area"
)


img = Image.open("/kaggle/input/mushrooms-pics-for-competition/pic_a1.jpg").convert("RGB").resize((384, 384))
display(img)


reset_conversation()
[r,l]=ask_mushroom_question(
    user_text="I found this in Singapore in August 2025. Is it safe to eat?",
    image_path="/kaggle/input/mushrooms-pics-for-competition/pic_a1.jpg"
)


img = Image.open("/kaggle/input/mushrooms-pics-for-competition/pic_a2.jpg").convert("RGB").resize((384, 384))
display(img)


reset_conversation()
[r,l]=ask_mushroom_question(
    user_text="Is this safe to eat? Found in Singapore.",
    image_path="/kaggle/input/mushrooms-pics-for-competition/pic_a2.jpg"
)


img = Image.open("/kaggle/input/mushrooms-pics-for-competition/pic_a4.jpg").convert("RGB").resize((384, 384))
display(img)


reset_conversation()
[r,l]=ask_mushroom_question(
    user_text="Is this mushroom edible?",
    image_path="/kaggle/input/mushrooms-pics-for-competition/pic_a4.jpg",
    date="August 2025",
    location="Singapore"
)


# This is OPTIONAL and requires internet.
# The offline Kaggle notebook above is the main GSAFE demonstration.

!pip install gradio


#Gradio
import gradio as gr

def gradio_ask(image, question, location, date):
    if not question and not image:
        return "Please enter a question or upload an image.", "âšª ?"

    response, label = ask_mushroom_question(
        user_text=question,
        image_path=image,
        date=date,
        location=location,
        verbose=False
    )
    return response, label

with gr.Blocks() as demo:
    gr.Markdown("## ğŸ�„ **GSAFE** - Gemma Smart Analysis for Fungus Entities")
    gr.Markdown("Offline mushroom classification powered by **Gemma 3n**.")

    with gr.Row():
        image_input = gr.Image(type="filepath", label="ğŸ“· Upload Mushroom Photo")
        with gr.Column():
            question_input = gr.Textbox(label="ğŸ’¬ Question", value="Is this mushroom edible or poisonous?")
            location_input = gr.Textbox(label="ğŸ“� Location (optional)")
            date_input = gr.Textbox(label="ğŸ“… Date (optional)")

    output_reason = gr.Textbox(label="Reasoning & Answer")
    output_label = gr.Label(label="Classification")

    with gr.Row():
        btn_classify = gr.Button("ğŸ”� Ask / Classify Mushroom")
        btn_reset = gr.Button("â™»ï¸� Reset Conversation")

    btn_classify.click(
        gradio_ask,
        inputs=[image_input, question_input, location_input, date_input],
        outputs=[output_reason, output_label]
    )

    btn_reset.click(
        lambda: (reset_conversation(), "Conversation reset.", "âšª ?"),
        inputs=[],
        outputs=[output_reason, output_label]
    )



#Optional - uncomment to run
#demo.launch(share=True)

