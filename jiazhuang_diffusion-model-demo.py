from diffusers import DDPMScheduler, UNet2DModel


scheduler = DDPMScheduler.from_pretrained("google/ddpm-cat-256")
model = UNet2DModel.from_pretrained("google/ddpm-cat-256", use_safetensors=True).to("cuda")


scheduler.set_timesteps(50)
scheduler.timesteps


import torch

sample_size = model.config.sample_size
noise = torch.randn((1, 3, sample_size, sample_size), device="cuda")


import matplotlib.pyplot as plt
%matplotlib inline


noise_img = noise.cpu().numpy()[0].transpose(1, 2, 0)
noise_img.shape


_ = plt.hist(noise_img.flatten())


import numpy as np

noise_img += 5
noise_img *= (255/10)
noise_img = noise_img.round().astype(np.uint8)


_ = plt.hist(noise_img.flatten())


plt.imshow(noise_img)


input = noise

for t in scheduler.timesteps:
    with torch.no_grad():
        noisy_residual = model(input, t).sample
    previous_noisy_sample = scheduler.step(noisy_residual, t, input).prev_sample
    input = previous_noisy_sample


from PIL import Image
import numpy as np

image = (input / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).round().to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image


from diffusers import DDPMPipeline

ddpm = DDPMPipeline.from_pretrained("google/ddpm-cat-256").to("cuda")
image = ddpm(num_inference_steps=25).images[0]
image


import kagglehub
sd_model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')


from PIL import Image
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

vae = AutoencoderKL.from_pretrained(sd_model_path, subfolder="vae", use_safetensors=True)
tokenizer = CLIPTokenizer.from_pretrained(sd_model_path, subfolder="tokenizer")
text_encoder = CLIPTextModel.from_pretrained(
    sd_model_path, subfolder="text_encoder", use_safetensors=True
)
unet = UNet2DConditionModel.from_pretrained(
    sd_model_path, subfolder="unet", use_safetensors=True
)


scheduler = DDIMScheduler.from_pretrained(sd_model_path, subfolder="scheduler")


torch_device = "cuda:1"
vae.to(torch_device)
text_encoder.to(torch_device)
unet.to(torch_device)


prompt = ["a lighthouse overlooking the ocean"]
height = 768  # default height of Stable Diffusion
width = 768  # default width of Stable Diffusion
num_inference_steps = 25  # Number of denoising steps
guidance_scale = 7.5  # Scale for classifier-free guidance
generator = torch.Generator(torch_device).manual_seed(0)  # Seed generator to create the initial latent noise
batch_size = len(prompt)


text_input = tokenizer(
    prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
)

with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(torch_device))[0]


max_length = text_input.input_ids.shape[-1]
uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
uncond_embeddings = text_encoder(uncond_input.input_ids.to(torch_device))[0]


text_embeddings = torch.cat([uncond_embeddings, text_embeddings])


latents = torch.randn(
    (batch_size, unet.config.in_channels, height // 8, width // 8),
    generator=generator,
    device=torch_device,
)


latents = latents * scheduler.init_noise_sigma


from tqdm.auto import tqdm

scheduler.set_timesteps(num_inference_steps)

for t in tqdm(scheduler.timesteps):
    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=t)

    # predict the noise residual
    with torch.no_grad():
        noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

    # perform guidance
    noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

    # compute the previous noisy sample x_t -> x_t-1
    latents = scheduler.step(noise_pred, t, latents).prev_sample


# scale and decode the image latents with vae
latents = 1 / 0.18215 * latents
with torch.no_grad():
    image = vae.decode(latents).sample


image = (image / 2 + 0.5).clamp(0, 1).squeeze()
image = (image.permute(1, 2, 0) * 255).to(torch.uint8).cpu().numpy()
image = Image.fromarray(image)
image


import kagglehub
sd_model_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1/1')


import torch
from diffusers import StableDiffusionPipeline, DDIMScheduler

pipe = StableDiffusionPipeline.from_pretrained(
    sd_model_path,
    torch_dtype=torch.float16,  # Use half precision
)


torch_device = "cuda:0"
pipe.to(torch_device)


prompt = 'a lighthouse overlooking the ocean'
negative_prompt = 'lines, framing, hatching, background, textures, patterns, details, outlines'


image = pipe(
    prompt=prompt,
    negative_prompt=negative_prompt,
    width=768,
    height=768,
    num_inference_steps=25, 
    guidance_scale=7.5,
    num_images_per_prompt=1,
).images[0]

image

