# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


!pip install --upgrade pip
!pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3 -q --no-cache
!pip install accelerate bitsandbytes
!pip install diffusers accelerate scipy safetensors


import kagglehub


# import torch
# from transformers import pipeline, AutoProcessor
# import os
# import sys
# import pkg_resources


# # In summary, this line of code configures PyTorch's CUDA memory allocator with the following settings:

# # It will only attempt to split memory blocks that are 128 MB or smaller to satisfy smaller allocation requests.
# # It might trigger a garbage collection pass of unused CUDA memory when the proportion of free memory exceeds 60% of the total managed memory.
# # These settings can be useful for fine-tuning memory management in PyTorch, potentially impacting memory usage, fragmentation, and performance of your CUDA-accelerated deep learning models. 
# # The optimal values often depend on the specific workload and hardware.
# os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128,garbage_collection_threshold:0.6"

# # 1) Python version
# print("Python:", sys.version.replace("\n", " "))

# # 2) All installed packages
# packages = sorted(pkg_resources.working_set, key=lambda x: x.key)
# for pkg in packages:
#     print(f"{pkg.key}=={pkg.version}")


import kagglehub
import torch
# from transformers import AutoTokenizer, GemmaForCausalLM
from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3Processor,Gemma3ForConditionalGeneration, AutoProcessor,TorchAoConfig,pipeline # Use Gemma3ForCausalLM
#https://huggingface.co/docs/transformers/main/en/model_doc/gemma3
from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler,DPMSolverMultistepScheduler


# point at the local repo and let HF pull in the custom classes
GEMMA_PATH = kagglehub.model_download('google/gemma-3/Transformers/gemma-3-4b-it/1')
pipe = pipeline(
    "image-text-to-text",
    model=GEMMA_PATH,
    revision="main",                 # or the actual branch/tag if diff
    trust_remote_code=True,          # <— allow custom Gemma3* classes to be loaded
    device_map="auto",
    load_in_8bit=True,               # quantize weights to 8‑bit
    torch_dtype=torch.bfloat16,      # mixed bfloat16 math
    low_cpu_mem_usage=True,          # minimize CPU RAM spikes
    offload_folder="/kaggle/working/offload"
)

# the processor is auto‐discovered, but here we choose to explicitly do:
processor = AutoProcessor.from_pretrained(
    GEMMA_PATH,
    trust_remote_code=True
)

# re-wrap so the processor is used
pipe = pipeline(
    "image-text-to-text",
    model=pipe.model,
    processor=processor,
    device_map="auto",
    trust_remote_code=True,
    use_cache=False                  # disable KV cache to save ~0.3 GiB
)

print("Successfully loaded Gemma 3:4B!")





%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://storage.googleapis.com/keras-cv/models/paligemma/cow_beach_1.png"},
            {"type": "text", "text": "What you can see in this image?"}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=200)
print(output[0]["generated_text"][-1]["content"])


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/p-blog/candy.JPG"},
            {"type": "text", "text": "What animal is represented on the candy?"}
        ]
    }
]
output = pipe(text=messages, max_new_tokens=200)


from IPython.display import Markdown

display(Markdown(output[0]["generated_text"][-1]["content"]))


# !pip install --upgrade pip
# !pip install diffusers accelerate scipy safetensors
# !pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3



# from PIL import Image





# from transformers import AutoProcessor, Gemma3ForConditionalGeneration
# GEMMA_PATH = kagglehub.model_download('google/gemma-3/Transformers/gemma-3-4b-it/1')
# model = Gemma3ForConditionalGeneration.from_pretrained(
#     GEMMA_PATH,
#     torch_dtype=torch.bfloat16,
#     device_map="auto",
#     attn_implementation="sdpa"
# )
# processor = AutoProcessor.from_pretrained(
#     GEMMA_PATH,
#     padding_side="left"
# )

# messages = [
#     {
#         "role": "system",
#         "content": [
#             {"type": "text", "text": "You are a helpful assistant."}
#         ]
#     },
#     {
#         "role": "user", "content": [
#             {"type": "image", "url": "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/pipeline-cat-chonk.jpeg"},
#             {"type": "text", "text": "What is shown in this image?"},
#         ]
#     },
# ]
# inputs = processor.apply_chat_template(
#     messages,
#     tokenize=True,
#     return_dict=True,
#     return_tensors="pt",
#     add_generation_prompt=True,
# ).to("cuda")

# output = model.generate(**inputs, max_new_tokens=50, cache_implementation="static")
# print(processor.decode(output[0], skip_special_tokens=True))


# import kagglehub
# import torch

# # Determine the device to use
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
    print("WARNING: No CUDA device found. Running on CPU. This will be much slower.")

# # from transformers import AutoTokenizer, GemmaForCausalLM
# from transformers import AutoTokenizer, Gemma3ForCausalLM, Gemma3Processor,Gemma3ForConditionalGeneration, AutoProcessor,TorchAoConfig # Use Gemma3ForCausalLM
# #https://huggingface.co/docs/transformers/main/en/model_doc/gemma3
# from diffusers import StableDiffusionPipeline, EulerDiscreteScheduler
# quantization_config = TorchAoConfig("int4_weight_only", group_size=128)


# --- Stable Diffusion Setup ---
# STABLE_DIFFUSION_MODEL_ID = "/kaggle/input/stable-diffusion-v2/pytorch/0/1"
STABLE_DIFFUSION_MODEL_ID =kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')
# stable_diffusion_v2_pytorch_1_1_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')
scheduler = EulerDiscreteScheduler.from_pretrained(STABLE_DIFFUSION_MODEL_ID, subfolder="scheduler")
# scheduler = PNDMScheduler.from_pretrained("runwayml/stable-diffusion-v1-5", subfolder="scheduler")


sd_pipe = StableDiffusionPipeline.from_pretrained(STABLE_DIFFUSION_MODEL_ID, scheduler=scheduler, torch_dtype=torch.float16)
sd_pipe = sd_pipe.to(device)
sd_pipe.scheduler = DPMSolverMultistepScheduler.from_pretrained(
            STABLE_DIFFUSION_MODEL_ID,
            subfolder="scheduler",
            # revision=REVISION_KAGGLE
        )
sd_pipe.enable_attention_slicing() # Enable attention slicing to reduce memory usage
print("Stable Diffusion Model Loaded.")

# # # --- Combined Workflow ---



# # prompt_text = """<start_of_turn>user
# # Write a short description of a fantastical creature that would look amazing in an image.<end_of_turn>
# # <start_of_turn>model"""

# # input_ids = processor(text=prompt_text, return_tensors="pt").to(device)
# # gemma_outputs = gemma_model.generate(**input_ids, max_new_tokens=128, do_sample=True, top_k=50, top_p=0.95)
# # gemma_text_output = processor.batch_decode(gemma_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
# # print(f"Gemma 3 Generated Text: {gemma_text_output}")

# # # Use Gemma's output as the prompt for Stable Diffusion
# # sd_prompt = f"A photorealistic image of {gemma_text_output}, fantastical creature, intricate details, vibrant colors."
# # image = sd_pipe(sd_prompt).images[0]
# # image.save("gemma_inspired_creature.png")
# # print("Image generated by Stable Diffusion based on Gemma's description: gemma_inspired_creature.png")
# # --- Generate Image with Stable Diffusion ---



sd_prompt = "a surreal landscape with floating islands and bioluminescent flora"
image = sd_pipe(sd_prompt, height=768, width=512, num_inference_steps=25).images[0]
image.save("stable_diffusion_image.png")
print("Image generated by Stable Diffusion: stable_diffusion_image.png")


# # --- Prepare Image for Gemma 3 ---
# image_path = "stable_diffusion_image.png"
# gemma_input_image = Image.open(image_path).convert("RGB")


cerinte="""Generate SVG code to visually represent with fair details the image, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>
"A red circle with a blue square inside"
</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""


# cerinte


# help(sd_pipe)


from PIL import Image
import os
import matplotlib.pyplot as plt


def generate_and_display(prompt,negative_prompt, num_iterations=10, output_dir="diffusion_images"):
    """
    Generates images using Stable Diffusion for different iterations and displays them.

    Args:
        prompt (str): The text prompt to guide image generation.
        num_iterations (int): The number of intermediate images to generate and display.
        output_dir (str): The directory to save the generated images.
    """
    try:
        
        pipeline = sd_pipe(prompt,height=512, width=512, num_inference_steps=num_iterations)
    except Exception as e:
        print(f"Error loading Stable Diffusion pipeline: {e}")
        return

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for i in range(1, num_iterations + 1):
        # Generate image with a specific number of inference steps
        image =  sd_pipe(prompt,height=768, width=512, num_inference_steps=(num_iterations+i),guidance_scale=7.5,negative_prompt=negative_prompt,).images[0]

        # Save the generated image
        filename = os.path.join(output_dir, f"iteration_{i}.png")
        image.save(filename)
        print("steps:",num_iterations+i)
        print(f"Saved image for iteration {i}: {filename}")

        # Display the image using PIL
        # Display the image in the Kaggle Notebook using matplotlib
        plt.figure(figsize=(6, 6))
        plt.imshow(np.array(image))
        plt.title(f"Iteration {i}")
        plt.axis('off')  # Turn off axis labels and ticks
        plt.show()

        # try:
        #     image.show(title=f"Iteration {i}")
        # except Exception as e:
        #     print(f"Could not display image for iteration {i}: {e}")

# if __name__ == "__main__":
user_prompt = "a lighthouse overlooking the ocean"
negative_prompt='crop'
num_steps = 7
output_directory = "stable_diffusion_outputs"
generate_and_display(user_prompt,negative_prompt, num_steps, output_directory)
# print(f"\nGenerated images saved in: {output_directory}")


user_prompt = "simple drawing of a lighthouse overlooking the ocean"
negative_prompt='crop,ugly,deformed'
num_steps = 10
output_directory = "stable_diffusion_outputs"
generate_and_display(user_prompt,negative_prompt, num_steps, output_directory)
# print(f"\nGenerated images saved in: {output_directory}")


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": cerinte}
        ]
    }
]
# output = pipe(text=messages, max_new_tokens=200)
output = pipe(text=messages, max_new_tokens=1024)
print(output[0]["generated_text"][-1]["content"])


cerinte2="""What do you see in image? Draw vectorial simple shapes above the image and generate a SVG with what you see in the picture respectic the following constrains. 
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>
"A red circle with a blue square inside"
</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_3.png"},
            {"type": "text", "text": cerinte2}
        ]
    }
]
# output = pipe(text=messages, max_new_tokens=200)
output = pipe(text=messages, max_new_tokens=1024)
print(output[0]["generated_text"][-1]["content"])


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "draw in a manner that aproximate image provided of a lighthouse over the ocean resembling the image provided and put in a SVG file format 256 by 256,to better resembling the image use elemnts like :`path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`"}
        ]
    }
]
# output = pipe(text=messages, max_new_tokens=200)
output = pipe(text=messages, max_new_tokens=1024)
print(output[0]["generated_text"][-1]["content"])


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "draw image provided of a lighthouse over the ocean in a SVG file format 256 by 256, make image simple and accurate "}
        ]
    }
]
# output = pipe(text=messages, max_new_tokens=200)
output = pipe(text=messages, max_new_tokens=1024)
print(output[0]["generated_text"][-1]["content"])


#output


# help(processor.apply_chat_template)


# print(inputs)


prompt_template = """Generate SVG code to visually represent with fair details the following text description, while respecting the given constraints.
<constraints>
* **Allowed Elements:** `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`
* **Allowed Attributes:** `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`, `transform`, `opacity`
</constraints>

<example>
<description>
"A red circle with a blue square inside"
</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
  <circle cx="50" cy="50" r="40" fill="red"/>
  <rect x="30" y="30" width="40" height="40" fill="blue"/>
</svg>
```
</example>


Please ensure that the generated SVG code is well-formed, valid, and strictly adheres to these constraints. Focus on a clear and concise representation of the input description within the given limitations. Always give the complete SVG code with nothing omitted. Never use an ellipsis.

<description>"{}"</description>
```svg
<svg viewBox="0 0 256 256" width="256" height="256">
"""
default_svg = """<svg width="256" height="256" viewBox="0 0 256 256"><circle cx="50" cy="50" r="40" fill="red" /></svg>"""


description="a lighthouse overlooking the ocean"


# prompt = prompt_template.format(description)
# print(prompt)


prompt_template = """You are an expert SVG generator. Your task is to generate a complete, valid, and accurate SVG code representation based *solely* on the provided text description. You MUST strictly adhere to the specified constraints.

**Constraints:**

1.  **Allowed Elements:** Only use the following SVG elements: `svg`, `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, `defs`.
2.  **Allowed Attributes:** Only use the following attributes:
    *   General: `viewBox`, `width`, `height`, `fill`, `stroke`, `stroke-width`, `transform`, `opacity`, `id`
    *   Shapes: `d`, `cx`, `cy`, `r`, `x`, `y`, `rx`, `ry`, `x1`, `y1`, `x2`, `y2`, `points`
    *   Gradients: `gradientUnits`, `gradientTransform`, `x1`, `y1`, `x2`, `y2` (for linear), `fx`, `fy`, `cx`, `cy`, `r` (for radial)
    *   Stops: `offset`, `stop-color`, `stop-opacity`
3.  **Structure:**
    *   The output MUST be a single, complete SVG document.
    *   It MUST start with `<svg ...>` including `xmlns="http://www.w3.org/2000/svg"`, a `viewBox="0 0 256 256"`, `width="256"`, and `height="256"`.
    *   It MUST end with `</svg>`.
    *   No XML comments (`<!-- -->`), DOCTYPE declarations, or other text outside the `<svg>...</svg>` tags are allowed.
4.  **Accuracy & Detail:**
    *   Accurately represent the main objects, their properties (color, shape, size relationships), and their spatial arrangement as described in the text.
    *   Interpret the description literally. Do not add elements not mentioned.
    *   Keep the visual representation clear and reasonably simple unless specific details are requested. Prioritize correctly representing the core request over excessive artistic detail.
5.  **Validity:** Ensure the generated SVG code is well-formed XML and valid according to SVG specifications within the allowed element/attribute constraints.
6.  **Defaults:** If a color is not specified for an object, use `fill="black"` as a default. Omit `stroke` unless mentioned or essential for visibility (like for a line). If stroke is used without a specified width, use `stroke-width="1"`.
7.  **Prohibited:**
    *   Do NOT use any elements or attributes not explicitly listed in Constraint #1 and #2.
    *   Do NOT use CSS (e.g., `<style>` tags or `style="..."` attributes).
    *   Do NOT include explanatory text, acknowledgements, or apologies before or after the SVG code block.
    *   Do NOT use ellipsis (`...`) or placeholders; the SVG code must be complete.

**Example:**

<description>
"A house in a hand"
</description>

<svg fill="#000000" height="800px" width="800px" version="1.1" id="Capa_1" xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" 
	 viewBox="0 0 296.096 296.096" xml:space="preserve">
<g>
	<path d="M101.15,175.142c6.966,0,14.634,0.68,21.634,1.966v-20.969h51v37.424c6,2.203,10.381,4.579,14.494,7.567
		c7.797,5.664,12.082,13.099,12.339,21.254c9.925-2.345,19.028-6.321,26.905-11.737c2.846-1.956,6.262-3.668,8.262-5.137v-58.243
		L148.047,49.23l-89.263,98.628v35.247c6-1.521,11.468-3.156,17.806-4.884C84.086,176.178,92.405,175.142,101.15,175.142z"/>
	<polygon points="148.446,25.313 247.63,136.139 270.251,136.139 148.247,0 97.784,55.572 97.784,33.139 60.784,33.139 
		60.784,96.454 25.201,136.139 48.144,136.139 	"/>
	<path d="M259.083,214.931c-5.494,0-12.89,2.44-22.289,8.901c-17.182,11.811-38.215,17.012-57.433,17.012
		c-15.623,0-30.045-3.435-40.23-9.546c-5.046-3.028-5.031-4.039-2.104-4.039c5.878,0,23.504,4.079,35.449,4.079
		c5.878,0,10.379-0.987,11.432-3.934c4.867-13.628-14.001-18.016-33.73-26c-14.444-5.848-32.073-10.262-48.861-10.263
		c-6.999,0-13.914,0.767-20.332,2.517c-29.141,7.945-50.199,13.93-56.199,15.873v76.48c12-4.413,25.9-8.642,33.81-8.642
		c14.516,0,73.253,18.726,106.51,18.726c4.658,0,8.829-0.367,12.298-1.204c28.227-6.814,77.232-46.725,90.533-60.351
		C274.323,227.998,271.797,214.931,259.083,214.931z"/>
</g>
</svg>

**Generation Task:**
Now, generate the complete SVG code for the following description, strictly following all constraints above:
<description>"{}"</description>

"""


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "Generate a 256x256 SVG vector graphic that closely approximates the provided image of a lighthouse on a cliff overlooking the ocean. The image features a white lighthouse with a dark lantern room on a grassy, rocky cliff edge, with a calm ocean and overcast sky in the background. Use a variety of standard SVG elements such as `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, and `defs` to construct the vector image, focusing on capturing the main shapes, composition, and key features like the lighthouse structure, cliff contours, and water/sky transition."}
        ]
    }
]
# output = pipe(text=messages, max_new_tokens=200)
output = pipe(text=messages, max_new_tokens=1024)
print(output[0]["generated_text"][-1]["content"])


%%time
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "Generate a 256x256 SVG vector graphic that *accurately* represents the provided image of a lighthouse on a cliff overlooking the ocean, capturing the *precise shapes and contours*. The image features a white lighthouse with a dark lantern room on a grassy, rocky cliff edge, with a calm ocean and overcast sky in the background. Use a variety of standard SVG elements such as `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, and `defs` to construct the vector image, focusing on capturing the main shapes, composition, and key features like the detailed lighthouse structure, the irregular cliff contours, and the nuanced water/sky transition. Avoid overly simplistic approximations of complex shapes."}
        ]
    }]
output = pipe(text=messages, max_new_tokens=2048)
print(output[0]["generated_text"][-1]["content"])


%%time
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "Generate a 256x256 SVG vector graphic that accurately represents the provided image of a lighthouse on a cliff overlooking the ocean, capturing the precise shapes and contours. The image features a white lighthouse with a dark lantern room on a grassy, rocky cliff edge, with a calm ocean and overcast sky in the background. Use a variety of standard SVG elements, prioritizing the `path` element for complex or irregular shapes like the cliff edge and potentially parts of the lighthouse structure, alongside `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, and `defs`. Focus on capturing the main shapes, composition, and key features like the detailed lighthouse structure, the irregular cliff contours, and the nuanced water/sky transition. Avoid reducing complex forms to simple geometric primitives."}
        ]
    }]
output = pipe(text=messages, max_new_tokens=2048)
print(output[0]["generated_text"][-1]["content"])


%%time
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "Generate a 256x256 SVG vector graphic that accurately represents the provided image of a lighthouse on a cliff overlooking the ocean. Analyze the image to identify the main shapes, focusing on the contours and forms of the lighthouse, the cliff edge, the ocean, and the sky. Then, construct the SVG using appropriate elements to precisely replicate these shapes. Prioritize the `path` element for rendering irregular or complex outlines like the cliff and rocky areas. Use other elements like `circle`, `rect`, etc., for simpler forms where they accurately apply. Employ gradients for smooth transitions in the sky and water. Ensure the final SVG accurately captures the detailed lighthouse structure, the irregular cliff contours, and the nuanced water/sky transition. Use a variety of standard SVG elements such as `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, and `defs`. Avoid reducing complex forms to simple geometric primitives."}
        ]
    }]
output = pipe(text=messages, max_new_tokens=2048)
print(output[0]["generated_text"][-1]["content"])


%%time

messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"},
            {"type": "text", "text": "Generate a 256x256 SVG vector graphic by tracing the key shapes and contours of the provided image of a lighthouse on a cliff over the ocean. Focus on accurately representing the form and structure of the lighthouse, the distinct shape and texture of the cliff face, and the clear separation between the ocean and the sky. Use a variety of standard SVG elements including `path`, `circle`, `rect`, `ellipse`, `line`, `polyline`, `polygon`, `g`, `linearGradient`, `radialGradient`, `stop`, and `defs` to construct the vector image. Aim for well-defined shapes and contours for the main objects to reduce abstraction and better approximate the visual details."}
        ]
    }
]
# output = pipe(text=messages, max_new_tokens=200)
output = pipe(text=messages, max_new_tokens=1024)
print(output[0]["generated_text"][-1]["content"])


from IPython.display import SVG, display


# Assuming 'output' is the variable holding the full result from your model run
# Example of how to get the full string output (adjust based on your exact code)
full_output_string = output[0]["generated_text"][-1]["content"]



# Define the markers for the code block
start_marker = "```xml\n" # Look for the start of the block and the newline after it
end_marker = "\n```"      # Look for the newline before the end of the block and the block end


# Find the indices of the markers
start_index = full_output_string.find(start_marker)

# Initialize svg_code to an empty string in case extraction fails
svg_code = ""

# Check if the start marker was found
if start_index != -1:
    # Calculate the position right after the start marker
    content_start_index = start_index + len(start_marker)

    # Find the end marker, searching only *after* the start marker
    end_index = full_output_string.find(end_marker, content_start_index)

    # Check if the end marker was found
    if end_index != -1:
        # Extract the substring between the markers
        svg_code = full_output_string[content_start_index:end_index]

        # Optional: Strip any leading/trailing whitespace from the extracted code
        svg_code = svg_code.strip()

# Now, check if we successfully extracted the SVG code
if svg_code:
    print("SVG code extracted. Displaying:")
    # Display the extracted SVG code using IPython.display
    display(SVG(svg_code))
else:
    print("Could not find or extract SVG code from the output.")
    print("Full output received:")
    print(full_output_string) # Print the full output to debug if extraction failed





messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"}, # <-- IMPORTANT: Replace with the actual path/URL to the random image
            {"type": "text", "text": """Please perform two tasks based on the provided image:

1.  **Describe the Image:** Briefly describe the main subject(s), key elements, and overall composition of the image.

2.  **Generate Simplified SVG:** Generate a single, simplified 256x256 SVG vector graphic that captures the main shapes and overall composition identified in the image.
    *   Focus on the essential forms and outlines of the primary subjects and background elements.
    *   Use primarily simple SVG elements:
        *   Employ `path` elements for key outlines and defining contours.
        *   Use basic shapes (e.g., `rect`, `circle`, `ellipse`) where appropriate for simpler components if they fit naturally.
        *   Use `linearGradient` or `radialGradient` for smooth color transitions in large areas like skies, water, or backgrounds, if applicable.
    *   The goal is a clear, stylized vector graphic that strongly resembles the input image's main features and composition, rather than a highly detailed or photorealistic trace. Prioritize capturing the recognizable essence of the scene with clean vector shapes.
    *   Ensure the output is valid SVG code, starting with `<svg>` and ending with `</svg>`.
"""}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=2048)
print(output[0]["generated_text"][-1]["content"])


# Assuming 'output' is the variable holding the full result from your model run
# Example of how to get the full string output (adjust based on your exact code)
full_output_string = output[0]["generated_text"][-1]["content"]
# Find the indices of the markers
start_index = full_output_string.find(start_marker)

# Initialize svg_code to an empty string in case extraction fails
svg_code = ""

# Check if the start marker was found
if start_index != -1:
    # Calculate the position right after the start marker
    content_start_index = start_index + len(start_marker)

    # Find the end marker, searching only *after* the start marker
    end_index = full_output_string.find(end_marker, content_start_index)

    # Check if the end marker was found
    if end_index != -1:
        # Extract the substring between the markers
        svg_code = full_output_string[content_start_index:end_index]

        # Optional: Strip any leading/trailing whitespace from the extracted code
        svg_code = svg_code.strip()

# Now, check if we successfully extracted the SVG code
if svg_code:
    print("SVG code extracted. Displaying:")
    # Display the extracted SVG code using IPython.display
    display(SVG(svg_code))
else:
    print("Could not find or extract SVG code from the output.")
    print("Full output received:")
    print(full_output_string) # Print the full output to debug if extraction failed


messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "url": "/kaggle/working/stable_diffusion_outputs/iteration_10.png"}, # <-- IMPORTANT: Replace with the actual path/URL to the random image
            {"type": "text", "text": """Please perform two tasks based on the provided image:

1.  **Describe the Image:** Briefly describe the main subject(s), key elements, and overall composition of the image.

2.  **Generate Simplified SVG:** Generate a single, simplified 512x512 SVG vector graphic that captures the main shapes and overall composition identified in the image.
    *   Focus on the essential forms and outlines of the primary subjects and background elements.
    *   Use primarily simple SVG elements:
        *   Employ `path` elements for key outlines and defining contours.
        *   Use basic shapes (e.g., `rect`, `circle`, `ellipse`,'polygon') where appropriate for simpler components if they fit naturally.
        *   Use `linearGradient` or `radialGradient` for smooth color transitions in large areas like skies, water, or backgrounds, if applicable.
    *   The goal is a clear, stylized vector graphic that strongly resembles the input image's main features and composition, rather than a highly detailed or photorealistic trace. Prioritize capturing the recognizable essence of the scene with clean vector shapes.
    *   Ensure the output is valid SVG code, starting with `<svg>` and ending with `</svg>`.
"""}
        ]
    }
]

output = pipe(text=messages, max_new_tokens=2048)
print(output[0]["generated_text"][-1]["content"])

# Assuming 'output' is the variable holding the full result from your model run
# Example of how to get the full string output (adjust based on your exact code)
full_output_string = output[0]["generated_text"][-1]["content"]
# Find the indices of the markers
start_index = full_output_string.find(start_marker)

# Initialize svg_code to an empty string in case extraction fails
svg_code = ""

# Check if the start marker was found
if start_index != -1:
    # Calculate the position right after the start marker
    content_start_index = start_index + len(start_marker)

    # Find the end marker, searching only *after* the start marker
    end_index = full_output_string.find(end_marker, content_start_index)

    # Check if the end marker was found
    if end_index != -1:
        # Extract the substring between the markers
        svg_code = full_output_string[content_start_index:end_index]

        # Optional: Strip any leading/trailing whitespace from the extracted code
        svg_code = svg_code.strip()

# Now, check if we successfully extracted the SVG code
if svg_code:
    print("SVG code extracted. Displaying:")
    # Display the extracted SVG code using IPython.display
    display(SVG(svg_code))
else:
    print("Could not find or extract SVG code from the output.")
    print("Full output received:")
    print(full_output_string) # Print the full output to debug if extraction failed

