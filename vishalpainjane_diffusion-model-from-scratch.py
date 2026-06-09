import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import math



DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print(f'Using device {DEVICE}')


import torch

print(f"PyTorch Version: {torch.__version__}")
print("---")
print(f"Is CUDA available? {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"PyTorch's Compiled CUDA Version: {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"Current GPU Index: {torch.cuda.current_device()}")
    print(f"Current GPU Name: {torch.cuda.get_device_name(torch.cuda.current_device())}")


img_size = 32
batch_size = 128

transforms = v2.Compose([
    v2.Resize((img_size, img_size)), # resize to 32*32
    v2.ToTensor(),
    v2.Lambda(lambda t:(t*2)-1) # change all the grid values into the range [-1.0, 1.0]
    ])

dataset = torchvision.datasets.CIFAR10(root='./data', download=True, transform=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)





def show_img(images, title=""):
    images = (images+1) /2
    images = images.clamp(0, 1)

    grid_img = torchvision.utils.make_grid(images, nrow = 8)
    plt.imshow(grid_img.permute(1, 2, 0).cpu().numpy())
    plt.title(title)
    plt.axis('off')
    plt.show()


image_batch , _= next(iter(dataloader))
show_img(image_batch[:16])


timestamps = 300

def linear_beta_schedule(timestamps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timestamps)

betas = linear_beta_schedule(timestamps=timestamps) # this is variance schedule, defines amt of noise that is added till the timestamps

# alp = 1. - betas # amt of image content remain
# alphae_cumprod =  torch.cumprod(alp, axis=0) # total amt of image content/signal left after t-steps

# sqrt_alp_cumprod = torch.sqrt(alphae_cumprod)
# sqrt_one_minus_alp_cumprod = torch.sqrt(1. - alphae_cumprod)


alphas = 1. - betas
alphas_cumprod = torch.cumprod(alphas, axis=0)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

def get_i_from_list(vals, t, x_shape):
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,)*(len(x_shape)-1))).to(t.device)

def forward_diff_sample(x_0, t, device=DEVICE):
    noise = torch.randn_like(x_0)  # random noise with normal distribution

    # sqrt(alpha_cumprod) and sqrt(1 - alpha_cumprod) for the given timestep t
    sqrt_alphas_cumprod_t = get_i_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_i_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
    
    # Calculate the noisy image
    # mean + variance * noise
    noisy_image = (sqrt_alphas_cumprod_t.to(device)*x_0.to(device)) + (sqrt_one_minus_alphas_cumprod_t.to(device)*noise.to(device))

    return noisy_image, noise.to(device)




show_img(image_batch[0])


single_image = image_batch[0].unsqueeze(0)

# Define timesteps to visualize
timesteps_to_show = [0, 50, 100, 150, 199, 299]
noisy_images = []

# Generate noisy versions of the image at different timesteps
for t_val in timesteps_to_show:
    t = torch.tensor([t_val], dtype=torch.long)
    noisy_img, _ = forward_diff_sample(single_image, t)
    noisy_images.append(noisy_img)

# Concatenate for visualization
noisy_images_tensor = torch.cat(noisy_images, dim=0)

# Show the results
show_img(noisy_images_tensor)



show_img(image_batch[0]), show_img(noisy_img)


class SPE(nn.Module): # sinusoidal PE that is used in OG Trnasformer paper
    '''
    Instead of just telling the artist "the noise level is 150", the assistant gives them a rich description:
    "This noise level is medium-high, which means you should focus on finding large shapes rather than fine textures, 
    and the color palette might be washed out."
    '''
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time): # inputs time(number) and converts it into embedding that contains rich features(time embedding)
        device = time.device

        half_dim = self.dim // 2

        embedding = math.log(10000)/(half_dim-1)
        embedding = torch.exp(torch.arange(half_dim, device=DEVICE)*-embedding)
        embedding = time[:, None] * embedding[None, :]
        embedding = torch.cat((embedding.sin(), embedding.cos()), dim=-1)
        return embedding
    
class Block(nn.Module): # bulding block of U-net
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_ch)

        if up:
            self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1=nn.GroupNorm(8, out_ch)
        self.norm2=nn.GroupNorm(8, out_ch)
        self.silu = nn.SiLU()

    def forward(self, x, t): # x - current state, t - noise level
        h = self.norm1(self.silu(self.conv1(x))) # convolve to activation fxn to normalisation

        time_emb = self.silu(self.time_mlp(t))
        time_emb = time_emb.unsqueeze(-1).unsqueeze(-1) # Reshape to (B, C, 1, 1)
        
        # Add time embedding
        h = h + time_emb # understanding of image + understand of noise level, they way it will paint will vary
        
        # Second convolution
        h = self.norm2(self.silu(self.conv2(h)))
        
        # Downsample or Upsample, depends on weather to zoom in and work on details....or zoom out and see the bigger picture
        return self.transform(h)


class UnetArch(nn.Module): 
    def __init__(self, image_channels=3, time_emb_dim=32):
        super().__init__()
        
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            SPE(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsampling path
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)]) 
        
        # Upsampling path
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        
        # Final convolve layer
        self.output = nn.Conv2d(up_channels[-1], image_channels, 1)

    def forward(self, x, timestep):# U shape analogy, downsampling(encoder)-> get more abstract feature out of the image through smaller and deeper sample
                                            # upsampling(decoder)  -> after undertanding the features, capturing/adding details back
                                            # Skip Connections     -> refering high-res images while upsampling images to get OG texture and edges
        # embed time
        t_emb = self.time_mlp(timestep)
        
        # initial convolution
        x = self.conv0(x)
        
        # downsampling path + save skip connections
        residual_inputs = []  # saving high-res residual images for later(skip connections)
        for down_block in self.downs:
            x = down_block(x, t_emb)
            residual_inputs.append(x)
        
        # upsampling path + use skip connections
        for up_block in self.ups:
            residual_x = residual_inputs.pop()   #get the corresponding skip connection from the downsampling path

            # Concatenate the skip connection with the current feature map
            x = torch.cat((x, residual_x), dim=1)           
            x = up_block(x, t_emb)
            
        return self.output(x)

model = UnetArch().to(DEVICE)
print(f"Model architecture defined. Number of parameters: {sum(p.numel() for p in model.parameters()):,}")






import torch
import torch.nn as nn
from tqdm import tqdm

LEARNING_RATE = 1e-4
EPOCHS = 10 

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.L1Loss() 

print("Starting Training...")
model.train() 

for epoch in range(EPOCHS):
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
    
    for step, (images, _) in enumerate(pbar):
        optimizer.zero_grad()

        # sample a random timestep for each image in the batch.....We use a LongTensor because timesteps are integer indices.
        t = torch.randint(0, timestamps, (batch_size,), device=DEVICE).long()
        
        # add noise to the images to get x_t and the added noise (our target)
        x_t, noise = forward_diff_sample(images, t, DEVICE)
        
        # predict the noise using the U-Net......model is on the DEVICE, and x_t and t are already on the DEVICE
        predicted_noise = model(x_t, t)
        
        # calculate the loss between the predicted noise and the actual noise
        loss = criterion(noise, predicted_noise)
        
        # backpropagate and update weights
        loss.backward()
        optimizer.step()
        
        # update the progress bar description with the current loss
        pbar.set_postfix(loss=f"{loss.item():.4f}")

print("Training finished.")

model_path = "ddpm_cifar10_10_epochs.pth"
torch.save(model.state_dict(), model_path)
print(f"Model saved to {model_path}")


import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import torchvision

@torch.no_grad() # we are not training, so we don't need to calculate gradients
def sample_timestep(x, t, model):
    betas_t = get_i_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_i_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_i_from_list(sqrt_recip_alphas, t, x.shape)
    
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_i_from_list(posterior_variance, t, x.shape)
    
    if t.all() == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_images(model, num_images=16):
    print("Generating new images...")
    img = torch.randn((num_images, 3, img_size, img_size), device=DEVICE)
    
    for i in tqdm(reversed(range(0, timestamps)), desc='Sampling loop', total=timestamps):
        t = torch.full((num_images,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(img, t, model)

    show_img(img, "Generated Images")


model = UnetArch().to(DEVICE)

model_path = "ddpm_cifar10_10_epochs.pth"
model.load_state_dict(torch.load(model_path, map_location=DEVICE))

model.eval()

sample_plot_images(model)




