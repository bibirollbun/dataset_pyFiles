import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


num_classes = 10
batch_size = 256
learning_rate = 1e-3
num_epochs = 100


num_steps = 18
sigma_min = 0.002
sigma_max = 20.0
rho = 7
alpha = 1


class EDMDenoiser(nn.Module):
    def __init__(self,
                 img_channels=1,            
                 label_dim=10,              
                 base_channels=64,          # 从32增加到64
                 sigma_data=0.5):           
        super().__init__()
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        
        # 增强型噪声嵌入 - 增加中间层宽度与深度
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 128),              # 从64增加到128
            nn.SiLU(),
            nn.Linear(128, 256),            # 从128增加到256
            nn.SiLU(),
            nn.Linear(256, 256),            # 从128增加到256
            nn.SiLU(),
            nn.Linear(256, 256)             # 增加一层
        )
        
        # 增强型标签嵌入 - 增加深度与宽度
        self.label_embed = nn.Sequential(
            nn.Embedding(label_dim, 128),   # 从64增加到128
            nn.SiLU(),
            nn.Linear(128, 256),            # 从128增加到256
            nn.SiLU(),
            nn.Linear(256, 256)             # 增加一层
        )
        
        # 增加残差模块数量 (4->6)
        self.res1 = ResBlock(img_channels, base_channels)
        self.res2 = ResBlock(base_channels, base_channels*2)
        self.res3 = ResBlock(base_channels*2, base_channels*4)   # 增加通道数
        self.res4 = ResBlock(base_channels*4, base_channels*4)   # 新增
        self.res5 = ResBlock(base_channels*4, base_channels*2)   # 新增
        self.res6 = ResBlock(base_channels*2, base_channels)
        
        # 条件投影层
        self.cond_proj1 = nn.Linear(256, base_channels)
        self.cond_proj2 = nn.Linear(256, base_channels*2)
        self.cond_proj3 = nn.Linear(256, base_channels*4)
        self.cond_proj4 = nn.Linear(256, base_channels*4)
        self.cond_proj5 = nn.Linear(256, base_channels*2)
        self.cond_proj6 = nn.Linear(256, base_channels)
        
        # 输出层
        self.output = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
    
    def forward(self, x, sigma, class_labels):
        # 数据类型处理
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        class_labels = class_labels.long()
        
        # EDM缩放
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        
        # 条件编码
        noise_emb = self.noise_embed(c_noise)
        label_emb = self.label_embed(class_labels)
        emb = noise_emb + label_emb
        
        # 前向传播（带残差）- 增加了层数
        h = c_in.view(-1, 1, 1, 1) * x
        
        h = self.res1(h, self.cond_proj1(emb))
        h = self.res2(h, self.cond_proj2(emb))
        h = self.res3(h, self.cond_proj3(emb))
        h = self.res4(h, self.cond_proj4(emb))  # 新增
        h = self.res5(h, self.cond_proj5(emb))  # 新增
        h = self.res6(h, self.cond_proj6(emb))
        
        h = self.output(h)
        
        # 输出处理
        D_x = c_skip.view(-1, 1, 1, 1) * x + c_out.view(-1, 1, 1, 1) * h
        
        return D_x
        
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.main = nn.Sequential(
            nn.GroupNorm(min(8, in_ch), in_ch),
            nn.SiLU(),
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            # 增加一个卷积层
            nn.GroupNorm(min(8, out_ch), out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1)
        )
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        
    def forward(self, x, cond):
        h = self.main(x)
        # 添加条件进行调制
        h = h + cond.view(-1, cond.shape[1], 1, 1)
        return h + self.skip(x)


class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None):
        rnd_normal = torch.randn([images.shape[0]], device=images.device)
        sigma = (rnd_normal * self.P_std + self.P_mean).exp()
        weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
        y = images
        n = torch.randn_like(y) * sigma.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        D_yn = net(y + n, sigma, labels.to(torch.float32))
        weight = weight.view(-1, 1, 1, 1)
        loss = weight * ((D_yn - y) ** 2)
        return loss.mean()


def conditional_sample(model, latents, labels, num_steps=50, sigma_min=0.002, sigma_max=20, rho=7, alpha=1):
    model.eval()
    
    # 初始噪声
    x = latents
    # 生成过程
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x
        h = t_next - t_cur
        t_cur_batch = t_cur.repeat(latents.shape[0])
        labels_batch = labels.repeat(latents.shape[0])
        denoised = model(x_cur, t_cur_batch, labels_batch).to(torch.float32)
        d_cur = (x_cur - denoised) / t_cur
        x_prime = x_cur + alpha * h * d_cur
        t_prime = t_cur + alpha * h
        t_prime_batch = t_prime.repeat(latents.shape[0])
        if t_next != 0:
            denoised_prime = model(x_prime, t_prime_batch, labels_batch).to(torch.float32)
            d_prime = (x_prime - denoised_prime) / t_prime
            x = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
        else:
            x = x_cur + h * d_cur

    return x.cpu().squeeze()



def load_MNIST(shuffle=True):
    # MNIST数据转换 - 注意规范化到[-1,1]范围
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))  # 单通道
    ])
    
    # 加载数据集
    train_dataset = datasets.MNIST(
        root='data_download/mnist',
        train=True,
        transform=transform,
        download=True
    )
    
    # 创建DataLoader
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=False,
        drop_last=False
    )
    
    return train_loader


model = EDMDenoiser().to(device)
total_params = sum(p.numel() for p in model.parameters())
print(f"Total parameters: {total_params}")
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
loss_fn = EDMLoss()
train = False
load_model = True
if load_model:
    model.load_state_dict(torch.load('/kaggle/input/denoiser/pytorch/default/1/kaggle_model.pth', map_location=torch.device('cpu')))
if train:
    train_loader = load_MNIST()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, labels in train_loader:
            x = x.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            loss = loss_fn(model, x, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        scheduler.step()
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    torch.save(model.state_dict(), 'kaggle_model.pth')



def plot_mnist_image(image_tensor, title=None):
    """
    绘制范围在[-1, 1]的MNIST图像张量
    
    参数:
    - image_tensor: 形状为[1, 28, 28]的PyTorch张量，值范围为[-1,1]
    - title: 图像标题
    """
    plt.figure(figsize=(5, 5), dpi=150)
    
    # 确保输入是CPU张量并转为NumPy
    if isinstance(image_tensor, torch.Tensor):
        if image_tensor.is_cuda:
            image_tensor = image_tensor.cpu()
        
        # 从[-1,1]转换到[0,1]范围
        image_tensor = (image_tensor + 1) / 2
        image_tensor = torch.clamp(image_tensor, 0, 1)
        
        # 对于单通道图像，移除通道维度
        if image_tensor.shape[0] == 1:  # 如果是单通道
            img = image_tensor.squeeze(0).numpy()
        else:
            img = image_tensor.permute(1, 2, 0).numpy()
    else:
        # 如果已经是NumPy数组
        img = (image_tensor + 1) / 2
        img = np.clip(img, 0, 1)
    
    # 显示图像，保持像素清晰（使用灰度colormap）
    plt.imshow(img, cmap='gray', interpolation='nearest')
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.show()


latents = torch.randn(10, 1, 28, 28, device=device) * 20
lables = torch.tensor([6], device=device)
with torch.no_grad():
    images = conditional_sample(model, latents, lables)
print(images.shape)
for i in range(10):
    plot_mnist_image(images[i].unsqueeze(0))

