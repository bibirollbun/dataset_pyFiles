import kagglehub

sdxl_model_path = kagglehub.model_download('stabilityai/stable-diffusion-xl/PyTorch/base-1-0/1')


description = 'a hacker with a hoodie'


from diffusers import DiffusionPipeline
import torch

pipe = DiffusionPipeline.from_pretrained(sdxl_model_path, torch_dtype=torch.float16).to("cuda")


prompt = f'{description}'
print(prompt)


image = pipe(
    prompt,
    num_inference_steps=30,
).images[0]
image


pipe.load_lora_weights("CiroN2022/toy-face", weight_name="toy_face_sdxl.safetensors", adapter_name="toy")


pipe.set_adapters("toy")


prompt = f"toy_face of {description}"

lora_scale = 0.9
image = pipe(
    prompt,
    num_inference_steps=30,
    cross_attention_kwargs={"scale": lora_scale}
).images[0]
image


pipe.load_lora_weights("nerijs/pixel-art-xl", weight_name="pixel-art-xl.safetensors", adapter_name="pixel")
pipe.set_adapters("pixel")


prompt = f"{description}, pixel art"
lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}
).images[0]
image


pipe.set_adapters(["pixel", "toy"], adapter_weights=[0.5, 1.0])


prompt = f"toy_face of {description}, pixel art"
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": 1.0}
).images[0]
image


pipe.disable_lora()
pipe.enable_lora()


pipe.get_active_adapters()


pipe.delete_adapters("toy")
pipe.get_active_adapters()


pipe.fuse_lora(lora_scale=0.9)


prompt = f"{description}, pixel art"
lora_scale = 0.9
image = pipe(
    prompt, num_inference_steps=30, cross_attention_kwargs={"scale": lora_scale}
).images[0]
image




