# import torch
# import os
# import base64
# import pandas as pd
# from tqdm import tqdm
# from PIL import Image
# from io import BytesIO
# from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig ,AutoModelForCausalLM
# from peft import PeftModel, PeftConfig
# folder = '/kaggle/input/thesis-data/new_data' 
# output_xlsx = "/kaggle/working/fine_tune_ocr_result.xlsx"
# batch_size = 1
# hf_token = "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" 
# lora_path= "/kaggle/input/fine-tuned-olmocr/pytorch/default/1/fine-tuned_olmOCR"
# #test on full 16bit model
# base_model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", 
#                                                         # quantization_config=bnb_config, 
#                                                         torch_dtype=torch.float16,
#                                                         device_map='auto',
#                                                         trust_remote_code=True)
# model = PeftModel.from_pretrained(base_model, lora_path).eval()
# processor = AutoProcessor.from_pretrained("/kaggle/input/fine-tuned-olmocr/pytorch/default/1/fine-tuned_olmOCR",trust_remote_code=True)


!pip install -U bitsandbytes


import os
from transformers import  AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import torch
import base64
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig ,AutoModelForCausalLM
from peft import PeftModel, PeftConfig
# bnb_config = BitsAndBytesConfig(
#     load_in_8bit=True,
# )
base_model = Qwen2VLForConditionalGeneration.from_pretrained("allenai/olmOCR-7B-0225-preview", 
                                                        # quantization_config=bnb_config, 
                                                        torch_dtype=torch.float16,
                                                        device_map='auto',
                                                        trust_remote_code=True,
                                                        )
model = PeftModel.from_pretrained(base_model, "/kaggle/input/fine-tuned-olmocr/pytorch/default/1/fine-tuned_olmOCR").eval()
model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained("/kaggle/input/fine-tuned-olmocr/pytorch/default/1/fine-tuned_olmOCR",trust_remote_code=True)


#save fine-tune model
model.save_pretrained('/kaggle/working/fine-tune')
processor.save_pretrained('/kaggle/working/fine-tune')


import os
from transformers import  AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import PeftModel
import torch
import base64
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig ,AutoModelForCausalLM
from peft import PeftModel, PeftConfig
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
)
model = Qwen2VLForConditionalGeneration.from_pretrained("/kaggle/working/fine-tune", 
                                                        quantization_config=bnb_config, 
                                                        torch_dtype=torch.float16,
                                                        device_map='auto',
                                                        trust_remote_code=True,
                                                        )
# model = PeftModel.from_pretrained(base_model, "/kaggle/input/fine-tuned-olmocr/pytorch/default/1/fine-tuned_olmOCR").eval()
# model = model.merge_and_unload()
processor = AutoProcessor.from_pretrained("/kaggle/working/fine-tune",trust_remote_code=True)


import torch

total_allocated = 0
total_reserved = 0

for i in range(torch.cuda.device_count()):
    total_allocated += torch.cuda.memory_allocated(i)
    total_reserved += torch.cuda.memory_reserved(i)

print(f"Total Allocated: {total_allocated / 1024 ** 2:.2f} MB")
print(f"Total Reserved : {total_reserved / 1024 ** 2:.2f} MB")


# from swift.utils import get_logger, get_model_parameter_info, plot_images, seed_everything
# model_parameter_info = get_model_parameter_info(model)
# print(model_parameter_info)


folder ='/kaggle/input/thesis-data/data/resize_0.4_data'
file_test = pd.read_csv('/kaggle/input/thesis-data/testing.csv', encoding="utf-8")
image_paths = file_test.iloc[:,0].to_list()
# print(image_paths[:])


#check image resolution
from PIL import Image
import os

def check_image_size(image_path):
    try:
        # Open the image
        with Image.open(image_path) as img:
            # Get dimensions (width, height)
            width, height = img.size
            # Get file size in bytes
            file_size = os.path.getsize(image_path)
            # Convert file size to KB
            file_size_kb = file_size / 1024
            
            print(f"Image Dimensions: {width}x{height} pixels")
            print(f"File Size: {file_size} bytes ({file_size_kb:.2f} KB)")
            
    except FileNotFoundError:
        print(f"Error: Image file '{image_path}' not found.")
    except Exception as e:
        print(f"Error: {str(e)}")

# Example usage
image_path = "/kaggle/input/thesis-data/data/resize_0.4_data/20210320_155525000.Page2.jpg"  # Replace with your image path
check_image_size(image_path)


import json
from tqdm import tqdm
import torch
import os
import base64
import pandas as pd
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig ,AutoModelForCausalLM
from peft import PeftModel, PeftConfig
from concurrent.futures import ThreadPoolExecutor
import time


cpu_time=time.time()
image = Image.open(image_path).convert("RGB")
width, height = image.size
buffered = BytesIO()
image.save(buffered, format="PNG")
image_base64 = base64.b64encode(buffered.getvalue()).decode()
# Build the prompt
user_prompt = f"""Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally.
            Do not hallucinate.
            RAW_TEXT_START
            Page dimensions: {width}x{height}
            [Image 0x0 to {width}x{height}]
            
            RAW_TEXT_END"""

messages = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
        ],
    }
]

# Apply chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

# Process inputs (no lists needed)
inputs = processor(
    text=text,
    images=image,
    padding=True,
    return_tensors="pt",
)
# inputs = {key: value.to(device) for key, value in inputs.items()}
first_device = list(set(model.hf_device_map.values()))[0]
for k in inputs:
    inputs[k] = inputs[k].to(f"cuda:{first_device}")

# Generate output
with torch.no_grad():
    start_time = time.time()
    output = model.generate(
        **inputs,
        temperature=0.8,
        max_new_tokens= 2000,
        do_sample= False
    )
    end_time = time.time()
    infer_time = end_time - start_time
# Decode output
prompt_length = inputs["input_ids"].shape[1]
new_tokens = output[:, prompt_length:]
text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

try:
    parsed_json = json.loads(text_output)
    natural_text = parsed_json.get("natural_text")
except json.JSONDecodeError:
    natural_text = text_output  # fallback if not JSON
end_cpu=time.time()
real_time = - cpu_time + end_cpu
print(natural_text)
print(f"Inference time: {infer_time:.2f} seconds")
print(f"real time: {real_time:.2f} seconds")


import torch

total_allocated = 0
total_reserved = 0

for i in range(torch.cuda.device_count()):
    total_allocated += torch.cuda.memory_allocated(i)
    total_reserved += torch.cuda.memory_reserved(i)

print(f"Total Allocated: {total_allocated / 1024 ** 2:.2f} MB")
print(f"Total Reserved : {total_reserved / 1024 ** 2:.2f} MB")


output_xlsx = "OlmOCR_8bits_test.xlsx"
batch_size = 1
results = []
for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_filenames = image_paths[i : i + batch_size]
    batch_results = []

    for filename in batch_filenames:
        image_path = os.path.join(folder, filename)
        # Open and convert image to RGB
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        buffered = BytesIO()
        image.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode()
        # Build the prompt
        user_prompt = f"""Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally.
                    Do not hallucinate.
                    RAW_TEXT_START
                    Page dimensions: {width}x{height}
                    [Image 0x0 to {width}x{height}]
                    
                    RAW_TEXT_END"""

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
                ],
            }
        ]

        # Apply chat template
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        # Process inputs (no lists needed)
        inputs = processor(
            text=text,
            images=image,
            padding=True,
            return_tensors="pt",
        )
        # inputs = {key: value.to(device) for key, value in inputs.items()}
        first_device = list(set(model.hf_device_map.values()))[0]
        for k in inputs:
            inputs[k] = inputs[k].to(f"cuda:{first_device}")

        # Generate output
        with torch.no_grad():
            output = model.generate(
                **inputs,
                temperature=0.8,
                max_new_tokens= 2000,
                do_sample= False
            )

        # Decode output
        prompt_length = inputs["input_ids"].shape[1]
        new_tokens = output[:, prompt_length:]
        text_output = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)[0]

        try:
            parsed_json = json.loads(text_output)
            natural_text = parsed_json.get("natural_text")
        except json.JSONDecodeError:
            natural_text = text_output  # fallback if not JSON

        batch_results.append({
            "image_name": filename,
            "extracted_text": natural_text.strip()
        })

    results.extend(batch_results)

    torch.cuda.empty_cache()

df = pd.DataFrame(results)
df.to_excel(output_xlsx, index=False)


print(f"OCR results saved to: {output_xlsx}")



output_xlsx = "OlmOCR_16bits_test.xlsx"
batch_size = 8  # TÃ¹y VRAM cá»§a báº¡n, cÃ³ thá»ƒ thá»­ tÄƒng Ä‘áº¿n 12â€“16 náº¿u Ä‘á»§

results = []

# ğŸ§  Function xá»­ lÃ½ song song áº£nh + prompt
def prepare_input(filename):
    image_path = os.path.join(folder, filename)
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode()

    # Build prompt
    user_prompt = f"""Below is the image of one page of a document, as well as some raw textual content that was previously extracted for it. Just return the plain text representation of this document as if you were reading it naturally.
Do not hallucinate.
RAW_TEXT_START
Page dimensions: {width}x{height}
[Image 0x0 to {width}x{height}]
RAW_TEXT_END"""

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_base64}"}},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return filename, image, text


for i in tqdm(range(0, len(image_paths), batch_size)):
    batch_filenames = image_paths[i : i + batch_size]

    # ğŸ§µ Xá»­ lÃ½ áº£nh vÃ  prompt song song
    with ThreadPoolExecutor(max_workers=8) as executor:
        batch_data = list(executor.map(prepare_input, batch_filenames))

    filenames, images, texts = zip(*batch_data)

    # ğŸ“¦ Táº¡o input batch
    inputs = processor(
        text=list(texts),
        images=list(images),
        padding=True,
        return_tensors="pt",
    )

    # â�© Move to correct GPU
    first_device = list(set(model.hf_device_map.values()))[0]
    for k in inputs:
        inputs[k] = inputs[k].to(f"cuda:{first_device}")

    # ğŸ”® Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            temperature=0.8,
            max_new_tokens=2000,
            do_sample=False
        )

    prompt_length = inputs["input_ids"].shape[1]
    new_tokens = outputs[:, prompt_length:]
    decoded = processor.tokenizer.batch_decode(new_tokens, skip_special_tokens=True)

    # ğŸ§¹ Parse káº¿t quáº£ tá»«ng áº£nh
    for filename, output_text in zip(filenames, decoded):
        try:
            parsed_json = json.loads(output_text)
            natural_text = parsed_json.get("natural_text", output_text)
        except json.JSONDecodeError:
            natural_text = output_text

        results.append({
            "image_name": filename,
            "extracted_text": natural_text.strip()
        })

    torch.cuda.empty_cache()

# ğŸ’¾ LÆ°u file káº¿t quáº£
df = pd.DataFrame(results)
df.to_excel(output_xlsx, index=False)
print(f"OCR results saved to: {output_xlsx}")

