!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


# Load Gemma 3n Model

import kagglehub
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig


# Download model from Kaggle Models
GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)


# Silly Explainer Loop

# enter topics repeatedly
while True:
    topic = input("Enter a topic ('quit' to exit): ")
    if topic.lower() == "quit":
        print("Goodbye from the Luffy! ğŸ‘’")
        break

    #  funny and incorrect prompt
    prompt = f"Explain {topic} like I'm five... but get some things hilariously wrong, as if you're a dumb who read half a science book. Make sure you don't tell exact answer"

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    #  generation settings for creative/wrong answers
    generation_config = GenerationConfig(
        max_new_tokens=200,
        do_sample=True,
        temperature=1.0,  # more randomness
        top_k=50,
        top_p=0.95
    )

    # Generate the response
    outputs = model.generate(**inputs, generation_config=generation_config)

    # Decode the generated output
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print("\nğŸ§ª Gemma Explains:", result, "\n")


from IPython.display import Image
from transformers import AutoProcessor, AutoModelForImageTextToText

# processor + model (reuse GEMMA_PATH from earlier)
processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(
    GEMMA_PATH,
    torch_dtype="auto",
    device_map=None  
).to("cuda:0")  # explicitly place it on a single GPU





# Define the function
def derp_explains(image_url):
    # Show image
    display(Image(url=image_url, width=400, height=300))

    # Funny prompt
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": "Explain this image like it's a joke, but get it hilariously wrong, as if you're Derp, who kinda gets memes but not really."}
            ]
        }
    ]

    # Generate
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(model.device, dtype=model.dtype)

    input_len = inputs["input_ids"].shape[-1]
    outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)

    text = processor.batch_decode(
        outputs[:, input_len:],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    print("\nğŸ–¼ï¸� Gemma Explains the Joke:\n", text[0])




image_1 = "https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F3063338%2Fac36ba0ef95bb73542512522737c6d07%2F567CBE87-F7D1-4CDB-A1F3-65FA3C09348F.jpeg?generation=1669557067576215&alt=media"

derp_explains(image_1)





image_2 = "https://www.googleapis.com/download/storage/v1/b/kaggle-forum-message-attachments/o/inbox%2F12970228%2F82ca418d62bda78891e5b740da8edc1b%2F1686341100077.jfif?generation=1686503271577327&alt=media"

# Call the function 
derp_explains(image_2)





image_3 = "https://media.licdn.com/dms/image/v2/D4D22AQHFZuN-OJZK8A/feedshare-shrink_800/feedshare-shrink_800/0/1722869084024?e=2147483647&v=beta&t=fdnQs4aJYJEfUQCub2Ubenk_fhsoKAllGZw1G0LPvQo"

derp_explains(image_3)




image_4 = "https://pbs.twimg.com/media/GpzivWEWsAAtztx?format=jpg&name=large"

derp_explains(image_4)






