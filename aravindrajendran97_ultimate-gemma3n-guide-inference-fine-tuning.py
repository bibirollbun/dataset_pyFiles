!pip install -q timm==1.0.17
!pip install -q transformers==4.53.2


import kagglehub
import torch
import gc

from transformers import AutoProcessor, AutoModelForImageTextToText


# import Gemma3n - 2B model from kaggle hub 
# for more model variations check here - https://www.kaggle.com/models/google/gemma-3n/transformers
gemma3n_2b_model_path = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

processor = AutoProcessor.from_pretrained(gemma3n_2b_model_path)
model = AutoModelForImageTextToText.from_pretrained(gemma3n_2b_model_path, torch_dtype="auto").to(device)


def generate(messages):
    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to(device, dtype=model.dtype)
    
    outputs = model.generate(**inputs, max_new_tokens=512, disable_compile=True)
    text = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:])
    
    # clean-up the variables to free-up GPU RAM
    del inputs
    del outputs
    torch.cuda.empty_cache()
    gc.collect()
    
    return text


prompt = """It was a dark and stormy night in Gotham city. So far way there was an"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": prompt}
        ]
    }
]

generate(messages)


image_url = "https://source.roboflow.com/v2IDbvwf8vFhER7eeJsv/06k5H3MN6JnHgM4Ox7SE/original.jpg"


from IPython.display import Image
Image(url=image_url,height=480,width=480)


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image_url},
            {"type": "text", "text": "Check this Make a list and discuss section in the image and give me some places"}
        ]
    }
]
generate(messages)


from IPython.display import Audio, display
Audio("https://erogol.com/ddc-samples/wavs/s1.wav")


!wget -qqq https://erogol.com/ddc-samples/wavs/s1.wav -O audio.wav


audio_file = "audio.wav"

messages = [{
    "role" : "user",
    "content": [
        { "type": "audio", "audio" : audio_file },
        { "type": "text",  "text" : "What is this audio about?" }
    ]
}]
generate(messages)

