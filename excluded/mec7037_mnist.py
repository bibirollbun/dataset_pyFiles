import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F


# 指定在GPU上运行程序
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Denoiser(nn.Module):
    def __init__(self,
                 img_channels=1,            
                 label_dim=10,              
                 base_channels=64,       
                 sigma_data=0.5):           
        super().__init__()
        self.img_channels = img_channels
        self.label_dim = label_dim
        self.sigma_data = sigma_data
        self.noise_embed = nn.Sequential(
            nn.Linear(1, 128),            
            nn.SiLU(),
            nn.Linear(128, 256),           
            nn.SiLU(),
            nn.Linear(256, 256),            
            nn.SiLU(),
            nn.Linear(256, 256)            
        )
        

        self.label_embed = nn.Sequential(
            nn.Embedding(label_dim, 128),   
            nn.SiLU(),
            nn.Linear(128, 256),          
            nn.SiLU(),
            nn.Linear(256, 256)            
        )
        
   
        self.res1 = ResBlock(img_channels, base_channels)
        self.res2 = ResBlock(base_channels, base_channels*2)
        self.res3 = ResBlock(base_channels*2, base_channels*4)  
        self.res4 = ResBlock(base_channels*4, base_channels*4)   
        self.res5 = ResBlock(base_channels*4, base_channels*2)   
        self.res6 = ResBlock(base_channels*2, base_channels)
        self.cond_proj1 = nn.Linear(256, base_channels)
        self.cond_proj2 = nn.Linear(256, base_channels*2)
        self.cond_proj3 = nn.Linear(256, base_channels*4)
        self.cond_proj4 = nn.Linear(256, base_channels*4)
        self.cond_proj5 = nn.Linear(256, base_channels*2)
        self.cond_proj6 = nn.Linear(256, base_channels)
        self.output = nn.Conv2d(base_channels, img_channels, kernel_size=3, padding=1)
    
    def forward(self, x, sigma, class_labels):
        x = x.to(torch.float32)
        sigma = sigma.to(torch.float32).reshape(-1, 1)
        class_labels = class_labels.long()
        
        c_skip = self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)
        c_out = sigma * self.sigma_data / (sigma ** 2 + self.sigma_data ** 2).sqrt()
        c_in = 1 / (self.sigma_data ** 2 + sigma ** 2).sqrt()
        c_noise = sigma.log() / 4
        noise_emb = self.noise_embed(c_noise)
        label_emb = self.label_embed(class_labels)
        emb = noise_emb + label_emb
        h = c_in.view(-1, 1, 1, 1) * x
        
        h = self.res1(h, self.cond_proj1(emb))
        h = self.res2(h, self.cond_proj2(emb))
        h = self.res3(h, self.cond_proj3(emb))
        h = self.res4(h, self.cond_proj4(emb))  
        h = self.res5(h, self.cond_proj5(emb)) 
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



# 输出采样图像
def conditional_sample(model, #去噪器
                       latents, #初始噪声
                       labels, #图像类别
                       num_steps, #采样步数
                       sigma_min, #最小噪声水平
                       sigma_max, #最大噪声水平
                       rho, #噪声水平调度的非线性程度
                       alpha):
    model.eval()
    # 初始噪声
    x = latents
    # 生成过程

    # 生成幂律分布的时间节点
    step_indices = torch.arange(num_steps, dtype=torch.float32, device=device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # 每一个时间节点 进行一次迭代
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x
        h = t_next - t_cur
        t_cur_batch = t_cur.repeat(latents.shape[0])
        labels_batch = labels.repeat(latents.shape[0])
        # 计算当前时间节点 网络基于目前的图像与噪声水平的输出
        denoised = model(x_cur, t_cur_batch, labels_batch).to(torch.float32)
        # 基于网络的输出 指出一个图像空间中指向数据分布的方向
        d_cur = (x_cur - denoised) / t_cur
        # 向这个方向进一步优化图像
        x_prime = x_cur + alpha * h * d_cur
        t_prime = t_cur + alpha * h
        t_prime_batch = t_prime.repeat(latents.shape[0])
        if t_next != 0:
            # 计算校正后的输出
            denoised_prime = model(x_prime, t_prime_batch, labels_batch).to(torch.float32)
            d_prime = (x_prime - denoised_prime) / t_prime
            # 基于校正后的输出 进一步优化图像
            x = x_cur + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)
        # 在最后一步禁用校正
        else:
            x = x_cur + h * d_cur
    
    return x.cpu().squeeze()



# 读取数据集

def load_MNIST(shuffle=True, batch_size=256):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((28, 28)),
        transforms.Normalize((0.5,), (0.5,))
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


# 实例化一个网络类
def train_model(learning_rate, num_epochs, batch_size):
    model = Denoiser().to(device)
    # 打印这个网络的参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    # 指定网络的optimizer和scheduler
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    # 指定loss类型
    loss_fn = EDMLoss()
    # 选择是否进行训练
    train = False
    # 选择是否读取网络参数
    load_model = True
    if load_model:
        # 读取参数
        model.load_state_dict(torch.load('/kaggle/input/denoiser-model/pytorch/default/1/kaggle_model.pth', 
                                         map_location=torch.device('cpu'),
                                         weights_only = True))
    if train:
        # 进行训练
        train_loader = load_MNIST(batch_size=batch_size)
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
    return model


# 如果你不准备进行训练 可以不用修改这一部分参数
batch_size = 256 # 一次梯度下降的batch大小
learning_rate = 1e-3 # 初始学习率大小
num_epochs = 2 # 总共进行100个epoch


# 这一部分是采样过程的参数设定
num_steps = 5 # 采样过程进行的步数
sigma_min = 1 # 采样过程的输出的噪声水平 不能设置为零
sigma_max = 2 # 初始噪声的规模 
rho = 7 # 噪声水平调度的非线性程度
alpha = 1 # Heun积分控制参数


def plot_mnist_images(images, nrows=2, ncols=5, title=None):
    # 创建大图
    fig = plt.figure(figsize=(ncols*2.5, nrows*2.5), dpi=150)
    
    # 遍历图像
    for i, image_tensor in enumerate(images):
        image_tensor = image_tensor.unsqueeze(0)
        if i >= nrows*ncols:  # 超出设定的行列数，停止绘制
            break
            
        # 创建子图
        plt.subplot(nrows, ncols, i+1)
        if isinstance(image_tensor, torch.Tensor):
            if image_tensor.is_cuda:
                image_tensor = image_tensor.cpu()
            image_tensor = (image_tensor + 1) / 2
            image_tensor = torch.clamp(image_tensor, 0, 1)
            if image_tensor.shape[0] == 1:
                img = image_tensor.squeeze(0).numpy()
            else:
                img = image_tensor.permute(1, 2, 0).numpy()
        else:
            img = (image_tensor + 1) / 2
            img = np.clip(img, 0, 1)
            
        # 绘制图像
        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.title(f"sample {i+1}", fontsize=10)
        plt.axis('off')
    
    if title:
        plt.suptitle(title, fontsize=16)
        
    plt.subplots_adjust(hspace=0.3, wspace=0.1)
    plt.tight_layout()
    plt.show()


latents = torch.randn(10, 1, 28, 28, device=device) * sigma_max
# 指定你想要生成的数字


# 实例化一个去噪器并读取参数
model = train_model(learning_rate, num_epochs, batch_size)
# 生成一个sigma_max水平的初始噪声
lables = torch.tensor([9], device=device)
# 进行生成
with torch.no_grad():
    images = conditional_sample(model, latents, lables, num_steps, sigma_min, sigma_max, rho, alpha)
plot_mnist_images(images)

