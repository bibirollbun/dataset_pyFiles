!pip install diffusers==0.3.0 --q
!pip install transformers scipy ftfy --q
!pip install "ipywidgets>=7,<8" --q
import IPython.display 



IPython.display.Image("../input/markdown/1.png")


IPython.display.Image("../input/markdown/2.png")


IPython.display.Image("../input/markdown/3.png")


IPython.display.Image("../input/markdown/4.png")



import gc
import torch
from PIL import Image
import IPython.display 
from torch import autocast
from tqdm.auto import tqdm
from kaggle_secrets import UserSecretsClient
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import StableDiffusionPipeline
from diffusers import AutoencoderKL, UNet2DConditionModel
from diffusers import LMSDiscreteScheduler , PNDMScheduler


user_secrets = UserSecretsClient()
Hugging_face  = user_secrets.get_secret("Hugging_id")





from kaggle_secrets import UserSecretsClient
secret_label = "Hugging_id"
secret_value = UserSecretsClient().get_secret(secret_label)


class config : 
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    HEIGHT = 512                        
    WIDTH = 512                         
    NUM_INFERENCE_STEPS = 500            
    GUIDANCE_SCALE = 7.5                
    GENERATOR = torch.manual_seed(48)   
    BATCH_SIZE = 1
    


def image_grid(imgs, rows, cols):
    assert len(imgs) == rows*cols
    w, h = imgs[0].size
    grid = Image.new('RGB', size=(cols*w, rows*h))
    grid_w, grid_h = grid.size
    for i, img in enumerate(imgs):
        grid.paste(img, box=(i%cols*w, i//cols*h))
    return grid


IPython.display.Image("../input/markdown/stable_diffusion.png")



vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_auth_token=Hugging_face)
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_auth_token=Hugging_face)
vae = vae.to(config.DEVICE)
text_encoder = text_encoder.to(config.DEVICE)
unet = unet.to(config.DEVICE) 



print(f'\033[94mTokenizer, Text Encoder, VAE, Unet are loaded !!')



scheduler = LMSDiscreteScheduler(beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000)
print(f'\033[94mThe scheduler loaded is K-LMS Sceheduler')



IPython.display.Image("../input/markdown/cave.png")


prompt = ["a curious explorer discovers a massive sprawling underground city in a huge cave system. city has churches, european - style buildings and big towers, and is really far away, illuminated by dim ambient lighting. waterfalls are flowing between different levels of the city. award winning digital art, concept art, breathtaking, imaginative, detailed., 8k"]





text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
max_length = text_input.input_ids.shape[-1]
with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(config.DEVICE))[0]
uncond_input = tokenizer(
    [""] * config.BATCH_SIZE, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(config.DEVICE))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
print(f'\033[94mText Embeddings shape: {text_embeddings.shape}')



latents = torch.randn(
  (config.BATCH_SIZE, unet.in_channels, config.HEIGHT // 8, config.WIDTH // 8),
  generator=config.GENERATOR,
)
latents = latents.to(config.DEVICE)

print(f'\033[94mLatent shape: {latents.shape}')



scheduler.set_timesteps(config.NUM_INFERENCE_STEPS)
latents = latents * scheduler.sigmas[0]



with autocast(config.DEVICE):
      for i, t in tqdm(enumerate(scheduler.timesteps)):
        
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        with torch.no_grad():
              noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, i, latents).prev_sample


latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample
print(f'\033[94mImage shape: {image.shape}')



image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0]


IPython.display.Image("../input/markdown/girl.png")


prompt = ["a ultradetailed beautiful panting of a stylish woman with a sword, by conrad roset, greg rutkowski and makoto shinkai, trending on artstation , 8k"]


text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
max_length = text_input.input_ids.shape[-1]
with torch.no_grad():
      text_embeddings = text_encoder(text_input.input_ids.to(config.DEVICE))[0]
uncond_input = tokenizer(
    [""] * config.BATCH_SIZE, padding="max_length", max_length=max_length, return_tensors="pt"
)
with torch.no_grad():
      uncond_embeddings = text_encoder(uncond_input.input_ids.to(config.DEVICE))[0]   
text_embeddings = torch.cat([uncond_embeddings, text_embeddings])
print(f'\033[94mText Embeddings shape: {text_embeddings.shape}')



latents = torch.randn(
  (config.BATCH_SIZE, unet.in_channels, config.HEIGHT // 8, config.WIDTH // 8),
  generator=config.GENERATOR,
)
latents = latents.to(config.DEVICE)

print(f'\033[94mLatent shape: {latents.shape}')



scheduler.set_timesteps(config.NUM_INFERENCE_STEPS)
latents = latents * scheduler.sigmas[0]



with autocast(config.DEVICE):
      for i, t in tqdm(enumerate(scheduler.timesteps)):
        
        latent_model_input = torch.cat([latents] * 2)
        sigma = scheduler.sigmas[i]
        latent_model_input = latent_model_input / ((sigma**2 + 1) ** 0.5)

        with torch.no_grad():
              noise_pred = unet(latent_model_input, t, encoder_hidden_states=text_embeddings).sample

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + config.GUIDANCE_SCALE * (noise_pred_text - noise_pred_uncond)

        latents = scheduler.step(noise_pred, i, latents).prev_sample


latents = 1 / 0.18215 * latents

with torch.no_grad():
  image = vae.decode(latents).sample
print(f'\033[94mImage shape: {image.shape}')



image = (image / 2 + 0.5).clamp(0, 1)
image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
images = (image * 255).round().astype("uint8")
pil_images = [Image.fromarray(image) for image in images]
pil_images[0].save("img2.jpg")
pil_images[0]


del latents
del vae
del text_encoder
del unet
gc.collect()


!pip install --upgrade diffusers


!pip uninstall -y diffusers transformers huggingface-hub
!pip install diffusers==0.35.1 transformers==4.30.0 huggingface-hub==0.13.4



from diffusers import StableDiffusionPipeline
import torch

Hugging_face = "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"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    use_auth_token=Hugging_face
)

pipe = pipe.to("cuda")

print("Stable Diffusion Pipeline created!")



# pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, use_auth_token="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")  
# pipe = pipe.to(config.DEVICE)
# print(f'\033[94mStable Diffusion Pipeline created !!!')

from diffusers import StableDiffusionPipeline
import torch

Hugging_face = "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"

pipe = StableDiffusionPipeline.from_pretrained(
    "CompVis/stable-diffusion-v1-4",
    torch_dtype=torch.float16,
    use_auth_token=Hugging_face
)
pipe = pipe.to("cuda")  # change to "cpu" if no GPU

print('\033[94mStable Diffusion Pipeline created !!!')



num_images = 4
prompt = ["futuristic synthwave city, retro sunset, crystals, spires, volumetric lighting, studio Ghibli style, rendered in unreal engine with clean details --ar 9:16 --hd --q 2"] * num_images
with autocast("cuda"):
  images = pipe(prompt , num_inference_steps=200).images

grid = image_grid(images, rows=2, cols=2)
grid


num_images = 4
prompt =["postapocalyptic city turned to fractal glass, shiny : : close shot : : 3 5 mm, realism, octane render, 8 k, exploration, cinematic, trending on artstation, realistic, 3 5 mm camera, unreal engine, hyper detailed, photo - realistic maximum detail, volumetric light, moody cinematic epic concept art, realistic matte painting, hyper photorealistic, concept art, volumetric light, cinematic epic, octane render, 8 k, corona render, movie concept art, octane render, 8 k, corona render, cinematic, trending on artstation, movie concept art, cinematic composition, ultra - detailed, realistic, hyper - realistic, volumetric lighting, 8 k"] * num_images
with autocast("cuda"):
  images = pipe(prompt , num_inference_steps=200).images

grid = image_grid(images, rows=2, cols=2)
grid


num_images = 4
prompt =["Cybernetic cloaked anime character concept design, dynamic pose, fantasy anime, dark, bejewelled and encrusted technological royal cloak, powerful aggressive sword stance, biological human face, iridescent, dark and intricate, Greg Rutkowski, Makoto Shinkai, anime CGI, animated, animation, artgerm, artstation, digital illustration, 8k"] * num_images
with autocast("cuda"):
  images = pipe(prompt , num_inference_steps=200).images

grid = image_grid(images, rows=2, cols=2)
grid

