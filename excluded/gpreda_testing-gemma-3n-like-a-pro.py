!pip install timm --upgrade
!pip install accelerate
!pip install git+https://github.com/huggingface/transformers.git


from time import time
import kagglehub
import transformers
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from transformers import AutoProcessor, AutoModelForImageTextToText


GEMMA_PATH = kagglehub.model_download("google/gemma-3n/transformers/gemma-3n-e2b-it")


tokenizer = AutoTokenizer.from_pretrained(GEMMA_PATH, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(GEMMA_PATH, trust_remote_code=True)


prompt = """What is the France capital?"""
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
outputs = model.generate(**inputs, generation_config=generation_config)
result = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(result)


def query_model(prompt, max_new_tokens=32):
    start_time = time()
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    generation_config = GenerationConfig(max_new_tokens=150, do_sample=True, temperature=0.7)
    outputs = model.generate(**inputs, generation_config=generation_config)
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    total_time = round(time() - start_time, 2)
    response = result.split(prompt)[-1]
    return response, total_time
    



prompt = "Quelle est la capitale de la France?"
response, total_time = query_model(prompt, max_new_tokens=16)
print(f"Execution time: {total_time}")
print(f"Question: {prompt}")
print(f"Response: {response}")


prompt = "When started WW2?"
response, total_time = query_model(prompt, max_new_tokens=32)
print(f"Execution time: {total_time}")
print(f"Question: {prompt}")
print(f"Response: {response}")


from IPython.display import display, Markdown

def colorize_text(text):
    for word, color in zip(["Reasoning", "Question", "Response", "Explanation", "Execution time"], ["blue", "red", "green", "darkblue",  "magenta"]):
        text = text.replace(f"{word}:", f"\n\n**<font color='{color}'>{word}:</font>**")
    return text


prompt = "Between what years was Obama president?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "Between what years was the 30 years war?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "Between what years was the WW1?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "What year was the Lepanto battle?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "What happened in 1868 in Japan?"
response, total_time = query_model(prompt, max_new_tokens=64)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "What happened in 1868 in Japan?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "Who was the first American president?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "In what novel the number 42 is important?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "Name the famous boyfriend of Yoko Ono."
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "Who was nicknamed 'The King' in music?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "What actor played Sheldon in TBBT?"
response, total_time = query_model(prompt, max_new_tokens=32)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "What is the maiden name of Princess of Wales?"
response, total_time = query_model(prompt, max_new_tokens=16)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "34 + 21"
response, total_time = query_model(prompt, max_new_tokens=16)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "49 x 27"
response, total_time = query_model(prompt, max_new_tokens=16)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "Brian and Sarah are brothers. Brian is 5yo, Sarah is 6 years older. How old is Sarah?"
response, total_time = query_model(prompt, max_new_tokens=40)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "x + 2 y = 5; y - x = 1. What are x and y? Just return x and y."
response, total_time = query_model(prompt, max_new_tokens=64)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "What is the total area of a sphere or radius 3? Just return the result."
response, total_time = query_model(prompt, max_new_tokens=64)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


prompt = "A rectangle with diagonal 4 is circumscribed by a circle. What is the circle's area?"
response, total_time = query_model(prompt, max_new_tokens=256)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


#Romanian
prompt = "Cine este Mircea Cartarescu?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


#Albanian
prompt = "Kush ishte Ismail Kadare?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


#Japanese
prompt = "夏目漱石とは誰ですか?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


#Chinese
prompt = "马拉多纳是谁?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


#French
prompt = "Qui était Marguerite Yourcenar?"
response, total_time = query_model(prompt, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))


from transformers import AutoProcessor, AutoModelForImageTextToText

processor = AutoProcessor.from_pretrained(GEMMA_PATH)
model = AutoModelForImageTextToText.from_pretrained(GEMMA_PATH, torch_dtype="auto", device_map="auto")


def query_model_text_image(prompt, image_url, max_new_tokens=32):
    start_time = time()
        
    messages = [
        {   
            "role": "user",
            "content": [
                {"type": "image", "image": image_url},
                {"type": "text", "text": prompt}
            ]
        }
    ]
    
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
    total_time = round(time() - start_time, 2)
    response = text[0]
    return response, total_time


prompt = "What represent this image?"
image_url = "https://wmf.imgix.net/images/aa_fra_notre-dame_de_paris_0.jpg"
response, total_time = query_model_text_image(prompt, image_url, max_new_tokens=128)
display(Markdown(colorize_text(f"Execution time: {total_time}\n\nQuestion: {prompt}\n\nResponse: {response}")))

