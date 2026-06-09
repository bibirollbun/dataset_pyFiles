import kagglehub
import torch
from diffusers import StableDiffusionPipeline

sd_model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')

pipe = StableDiffusionPipeline.from_pretrained(sd_model_path)
pipe.to('cuda')


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(prompt).images[0]
image


import kagglehub
import torch
from diffusers import StableDiffusionPipeline

sd_model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')

pipe = StableDiffusionPipeline.from_pretrained(sd_model_path)
pipe.enable_model_cpu_offload()


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(prompt).images[0]
image


import kagglehub
sd_model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')


from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig


import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from diffusers import StableDiffusionPipeline


diffusers_quant_config = DiffusersBitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit=True,
)


quant_config = BitsAndBytesConfig(
    # load_in_4bit=True,
    # bnb_4bit_quant_type="nf4",
    # bnb_4bit_compute_dtype=torch.bfloat16,
    load_in_8bit=True,
)


vae = AutoencoderKL.from_pretrained(
    sd_model_path,
    subfolder="vae",
    quantization_config=diffusers_quant_config
)

text_encoder = CLIPTextModel.from_pretrained(
    sd_model_path,
    subfolder="text_encoder",
    quantization_config=quant_config,
)

unet = UNet2DConditionModel.from_pretrained(
    sd_model_path,
    subfolder="unet",
    quantization_config=diffusers_quant_config,
)

pipe = StableDiffusionPipeline.from_pretrained(
    sd_model_path, 
    text_encoder=text_encoder,
    vae=vae,
    unet=unet,
    torch_dtype=torch.float16,
)
pipe = pipe.to('cuda:0')


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(prompt).images[0]
image


import kagglehub
import torch
from diffusers import StableDiffusionPipeline

sd_model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')

pipe = StableDiffusionPipeline.from_pretrained(sd_model_path, dtype=torch.float16)
pipe.to('cuda:1')


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(
    prompt,
    num_inference_steps=25,
).images[0]
image


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=25,
).images[0]
image


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(
    prompt,
    width=128,
    height=128,
    num_inference_steps=25,
).images[0]
image


pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=25,
).images[0]
image


%%time
prompt = "a lighthouse overlooking the ocean"
image = pipe(
    prompt,
    width=512,
    height=512,
    num_inference_steps=25,
).images[0]
image




