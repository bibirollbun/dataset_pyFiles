import kagglehub
import torch
from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDIMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
from transformers import BitsAndBytesConfig

from skimage import img_as_float
from skimage import filters, transform
from scipy.ndimage import uniform_filter, convolve, gaussian_filter, zoom, map_coordinates
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


stable_diffusion_v2_path = kagglehub.model_download('stabilityai/stable-diffusion-v2/PyTorch/1-base/1')

pipe = StableDiffusionPipeline.from_pretrained(
    stable_diffusion_v2_path,
    torch_dtype=torch.float16,
)

pipe = pipe.to('cuda:0')
#pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)
#pipe.vae = torch.compile(pipe.vae, mode="reduce-overhead", fullgraph=True)


prompt = "A floating island covered in ancient ruins and overgrown vines drifting above the clouds."
image = pipe(prompt, width=512, height=512).images[0]


img_np = img_as_float(np.array(image))
print(img_np.shape)


def rotate_image(img_np, theta_rad):
    """
    Rotate img_np by theta_rad (radians) according to
    F(f, theta)(x, y) = f(R_{-theta}(x, y))
    using backward warping with bilinear interpolation.
    """
    H, W, C = img_np.shape
    cx, cy = W / 2, H / 2
    
    rotated_img = np.zeros_like(img_np)
    
    y_indices, x_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
    x_shifted = x_indices - cx
    y_shifted = y_indices - cy
    
    # (R_{-theta})
    x_src = x_shifted * np.cos(theta_rad) + y_shifted * np.sin(theta_rad)
    y_src = -x_shifted * np.sin(theta_rad) + y_shifted * np.cos(theta_rad)

    x_src += cx
    y_src += cy

    for c in range(C):
        rotated_img[..., c] = map_coordinates(img_np[..., c], [y_src, x_src], order=1, mode='reflect')
    
    return rotated_img


theta_deg = 30  
theta_rad = np.deg2rad(theta_deg)  

rotated_img = rotate_image(img_np, theta_rad)

plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.title('Original')
plt.imshow(img_np)
plt.axis('off')

plt.subplot(1,2,2)
plt.title('Rotated')
plt.imshow(np.clip(rotated_img, 0, 1))
plt.axis('off')
plt.show()



# F(f, theta)(x, y) = f(R_{-theta}(x, y))
theta_deg = 15  
theta_rad = np.deg2rad(theta_deg)

rotated_img = rotate_image(img_np, theta_rad) 


# 1. Box Filter 
img_box_rotated = np.zeros_like(rotated_img)
for c in range(3):
    img_box_rotated[..., c] = uniform_filter(rotated_img[..., c], size=15)

# 2. Gaussian Filter 
img_gaussian_rotated = np.zeros_like(rotated_img)
for c in range(3):
    img_gaussian_rotated[..., c] = gaussian_filter(rotated_img[..., c], sigma=4)

# 3. Supersampling 
img_up_rotated = zoom(rotated_img, (2, 2, 1), order=1)
img_blur_rotated = np.zeros_like(img_up_rotated)
for c in range(3):
    img_blur_rotated[..., c] = gaussian_filter(img_up_rotated[..., c], sigma=4)
img_down_rotated = zoom(img_blur_rotated, (0.5, 0.5, 1), order=1)


fig, axs = plt.subplots(1, 4, figsize=(16, 5))
titles = ['Rotated Original', 'Rotated + Box Filter', 'Rotated + Gaussian Filter', 'Rotated + Supersampling']
images = [rotated_img, img_box_rotated, img_gaussian_rotated, img_down_rotated]

for ax, im, title in zip(axs, images, titles):
    ax.imshow(np.clip(im, 0, 1))
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()


theta_deg = 15
theta_rad = np.deg2rad(theta_deg)
epsilon = 1e-4

rotated_img = rotate_image(img_np, theta_rad)
rotated_img_eps = rotate_image(img_np, theta_rad + np.rad2deg(epsilon))

img_filtered = np.zeros_like(rotated_img)
for c in range(3):
    img_filtered[..., c] = uniform_filter(rotated_img[..., c], size=15)

img_filtered_eps = np.zeros_like(rotated_img_eps)
for c in range(3):
    img_filtered_eps[..., c] = uniform_filter(rotated_img_eps[..., c], size=15)

grad_theta = (img_filtered_eps - img_filtered) / epsilon


plt.figure(figsize=(4,4))
plt.title('∂I/∂θ (R channel)')
plt.imshow(np.clip((grad_theta[..., 0] - grad_theta[..., 0].min()) / (grad_theta[..., 0].ptp()), 0, 1), cmap='gray')
plt.axis('off')
plt.show()


def monte_carlo_convolution_derivative(img, theta_rad, kernel_fn, num_samples=50, epsilon=1e-4):
    """Estimate ∂g/∂θ using Monte Carlo sampling."""
    H, W, C = img.shape
    grad_theta = np.zeros((H, W, C))
    cx, cy = W / 2, H / 2
    
    # Randomly sample (u, v) shifts within kernel support [-k_support/2, +k_support/2]
    u_samples = np.random.uniform(-1, 1, size=num_samples)
    v_samples = np.random.uniform(-1, 1, size=num_samples)
    k_values = kernel_fn(u_samples, v_samples) 

    # Small perturbation on θ
    theta_plus = theta_rad + epsilon

    for idx in tqdm(range(num_samples)):
        u = u_samples[idx]
        v = v_samples[idx]
        k_val = k_values[idx]
        
        y_indices, x_indices = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        x_shifted = (x_indices - u)
        y_shifted = (y_indices - v)
        x_shifted_center = x_shifted - cx
        y_shifted_center = y_shifted - cy

        # (x-u, y-v) under θ
        x_src = x_shifted_center * np.cos(theta_rad) + y_shifted_center * np.sin(theta_rad) + cx
        y_src = -x_shifted_center * np.sin(theta_rad) + y_shifted_center * np.cos(theta_rad) + cy
        
        # (x-u, y-v) under θ+ε
        x_src_eps = x_shifted_center * np.cos(theta_plus) + y_shifted_center * np.sin(theta_plus) + cx
        y_src_eps = -x_shifted_center * np.sin(theta_plus) + y_shifted_center * np.cos(theta_plus) + cy

        # Finite difference for derivative
        for c in range(C):
            F_val = map_coordinates(img[..., c], [y_src, x_src], order=1, mode='reflect')
            F_val_eps = map_coordinates(img[..., c], [y_src_eps, x_src_eps], order=1, mode='reflect')
            grad_theta[..., c] += k_val * (F_val_eps - F_val) / epsilon

    grad_theta /= num_samples
    return grad_theta


def box_kernel(u, v):
    return np.ones_like(u)

theta_deg = 15
theta_rad = np.deg2rad(theta_deg)
epsilon=1e-4

grad_theta_monte_carlo = monte_carlo_convolution_derivative(img_np, theta_rad, box_kernel, num_samples=50, epsilon=epsilon)


def normalize(img):
    return np.clip((img - img.min()) / (img.ptp() + 1e-8), 0, 1)


fig, axs = plt.subplots(1, 2, figsize=(16, 8))

# Finite Difference
axs[0].imshow(normalize(grad_theta[..., 0]), cmap='gray')
axs[0].set_title('Finite Difference ∂g/∂θ (R channel)')
axs[0].axis('off')

# Monte Carlo
axs[1].imshow(normalize(grad_theta_monte_carlo[..., 0]), cmap='gray')
axs[1].set_title('Monte Carlo ∂g/∂θ (R channel)')
axs[1].axis('off')

plt.tight_layout()
plt.savefig('result.png', dpi=300)
plt.show()

