# Training progress bar
#!pip install -q qqdm


import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision.models as models
from torch.optim import Adam, AdamW
import pandas as pd
from tqdm.auto import tqdm,trange
# Accelerate parts
from accelerate import Accelerator, notebook_launcher # main interface, distributed launcher


train = np.load('/kaggle/input/ml2022spring-hw8/data/trainingset.npy', allow_pickle=True)
test = np.load('/kaggle/input/ml2022spring-hw8/data/testingset.npy', allow_pickle=True)

print(train.shape)
print(test.shape)


def same_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# same_seeds(23442)


class mix_autoencoder(nn.Module):
    def __init__(self):
        super(mix_autoencoder, self).__init__()
        self.encoder_fcn = nn.Sequential(
            nn.Linear(64 * 64 * 3, 64*64),  
            nn.ReLU(True),
            nn.Linear(64*64, 32*32),
            nn.ReLU(True),
            nn.Linear(32*32, 16*16),
            nn.ReLU(True),
            nn.Linear(16*16, 16),
        )
        
        self.encoder_cnn = nn.Sequential(
                nn.Conv2d(3, 24, 4, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 24, 4, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.Conv2d(24, 48, 4, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.Conv2d(48, 96, 4, stride=2, padding=1),
                nn.BatchNorm2d(96),
                nn.ReLU(),
                nn.Flatten(),
                nn.Dropout(0.3),
                nn.Linear(96*4*4, 128),
                nn.BatchNorm1d(128),
                nn.Dropout(0.3),
                nn.Linear(128, 16),
                nn.BatchNorm1d(16),
                nn.ReLU()
        )
        
        
        
        self.decoder = nn.Sequential(
                nn.Linear(16, 512),
                nn.BatchNorm1d(512),
                nn.ReLU(),
                nn.Linear(512, 96*4*4),
                nn.BatchNorm1d(96*4*4),
                nn.ReLU(),
                nn.Unflatten(1, (96, 4, 4)),
                nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
                nn.BatchNorm2d(48),
                nn.ReLU(),
                nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
                nn.BatchNorm2d(24),
                nn.ReLU(),
                nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
                nn.BatchNorm2d(12),
                nn.ReLU(),
                nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
                nn.Tanh(),
            )

    
    def forward(self, x):
        x_fcn = self.encoder_fcn( x.reshape(x.shape[0], -1) )
        x_cnn = self.encoder_cnn(x)
        x = self.decoder((x_fcn + x_cnn)/2.0)
        return x, x_fcn, x_cnn
      

# Mind's FCN
class fcn_autoencoder(nn.Module):
    def __init__(self):
        super(fcn_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(64 * 64 * 3, 1024),   
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True), 
            nn.Linear(512, 128),
            nn.ReLU(True), 
            nn.Linear(128, 64)
        )
        
        
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 64 * 64 * 3), 
            nn.Tanh()
        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x



# BatchNorm CNN
class conv_autoencoder(nn.Module):
    def __init__(self):
        super(conv_autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.Conv2d(48, 96, 4, stride=2, padding=1),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(96*4*4, 128),
            nn.BatchNorm1d(128),
            nn.Dropout(0.2),
            nn.Linear(128, 10),
            nn.BatchNorm1d(10),
            nn.ReLU()
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(10, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 96*4*4),
            nn.BatchNorm1d(96*4*4),
            nn.ReLU(),
            nn.Unflatten(1, (96, 4, 4)),
            nn.ConvTranspose2d(96, 48, 4, stride=2, padding=1),
            nn.BatchNorm2d(48),
            nn.ReLU(),
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    
class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 4, stride=2, padding=1),            
            nn.ReLU(),
            nn.Conv2d(12, 24, 4, stride=2, padding=1),    
            nn.ReLU(),
        )
        self.enc_out_1 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),  
            nn.ReLU(),
        )
        self.enc_out_2 = nn.Sequential(
            nn.Conv2d(24, 48, 4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(24, 12, 4, stride=2, padding=1), 
            nn.ReLU(),
            nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1), 
            nn.Tanh(),
        )

    def encode(self, x):
        h1 = self.encoder(x)
        return self.enc_out_1(h1), self.enc_out_2(h1)

    def reparametrize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        if torch.cuda.is_available():
            eps = torch.cuda.FloatTensor(std.size()).normal_()
        else:
            eps = torch.FloatTensor(std.size()).normal_()
        eps = Variable(eps)
        return eps.mul(std).add_(mu)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparametrize(mu, logvar)
        return self.decode(z), mu, logvar


def loss_vae(recon_x, x, mu, logvar, criterion):
    """
    recon_x: generating images
    x: origin images
    mu: latent mean
    logvar: latent log variance
    """
    mse = criterion(recon_x, x)  # mse loss
    KLD_element = mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    KLD = torch.sum(KLD_element).mul_(-0.5)
    return mse + KLD


class Resnet(nn.Module):
    def __init__(self, fc_hidden1=1024, fc_hidden2=768, drop_p=0.3, CNN_embed_dim=256):
        super(Resnet, self).__init__()

        self.fc_hidden1, self.fc_hidden2, self.CNN_embed_dim = fc_hidden1, fc_hidden2, CNN_embed_dim

        # CNN architechtures
        self.ch1, self.ch2, self.ch3, self.ch4 = 16, 32, 64, 128
        self.k1, self.k2, self.k3, self.k4 = (5, 5), (3, 3), (3, 3), (3, 3)      # 2d kernal size
        self.s1, self.s2, self.s3, self.s4 = (2, 2), (2, 2), (2, 2), (2, 2)      # 2d strides
        self.pd1, self.pd2, self.pd3, self.pd4 = (0, 0), (0, 0), (0, 0), (0, 0)  # 2d padding

        # encoding components
        resnet = models.resnet18(pretrained=False)
        modules = list(resnet.children())[:-1]      # delete the last fc layer.
        self.resnet = nn.Sequential(*modules)
        self.fc1 = nn.Linear(resnet.fc.in_features, self.fc_hidden1)
        self.bn1 = nn.BatchNorm1d(self.fc_hidden1, momentum=0.01)
        self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        self.bn2 = nn.BatchNorm1d(self.fc_hidden2, momentum=0.01)

        self.fc3_mu = nn.Linear(self.fc_hidden2, self.CNN_embed_dim)      # output = CNN embedding latent variables

        # Sampling vector
        self.fc4 = nn.Linear(self.CNN_embed_dim, self.fc_hidden2)
        self.fc_bn4 = nn.BatchNorm1d(self.fc_hidden2)
        self.fc5 = nn.Linear(self.fc_hidden2, 64 * 4 * 4)
        self.fc_bn5 = nn.BatchNorm1d(64 * 4 * 4)
        self.relu = nn.ReLU(inplace=True)

        # Decoder
        self.convTrans6 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=self.k4, stride=self.s4,
                               padding=self.pd4),
            nn.BatchNorm2d(32, momentum=0.01),
            nn.ReLU(inplace=True),
        )
        self.convTrans7 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=32, out_channels=8, kernel_size=self.k3, stride=self.s3,
                               padding=self.pd3),
            nn.BatchNorm2d(8, momentum=0.01),
            nn.ReLU(inplace=True),
        )

        self.convTrans8 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=8, out_channels=3, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2),
            nn.BatchNorm2d(3, momentum=0.01),
            nn.Sigmoid()    # y = (y1, y2, y3) \in [0 ,1]^3
        )


    def encode(self, x):
        x = self.resnet(x)  # ResNet
        x = x.view(x.size(0), -1)  # flatten output of conv

        # FC layers
        if x.shape[0] > 1:
            x = self.bn1(self.fc1(x))
        else:
            x = self.fc1(x)
        x = self.relu(x)
        if x.shape[0] > 1:
            x = self.bn2(self.fc2(x))
        else:
            x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3_mu(x)
        return x

    def decode(self, z):
        if z.shape[0] > 1:
            x = self.relu(self.fc_bn4(self.fc4(z)))
            x = self.relu(self.fc_bn5(self.fc5(x))).view(-1, 64, 4, 4)
        else:
            x = self.relu(self.fc4(z))
            x = self.relu(self.fc5(x)).view(-1, 64, 4, 4)
        x = self.convTrans6(x)
        x = self.convTrans7(x)
        x = self.convTrans8(x)
        x = F.interpolate(x, size=(64, 64), mode='bilinear', align_corners=True)
        return x

    def forward(self, x):
        z = self.encode(x)
        x_reconst = self.decode(z)

        return x_reconst


class CustomTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        self.transform = transforms.Compose([
                            transforms.Lambda(lambda x: x.to(torch.float32)),
                            transforms.Lambda(lambda x: 2. * x/255. - 1.),
                            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                            ])
        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)



class FlippedTensorDataset(TensorDataset):
    """TensorDataset with support of transforms.
    """
    def __init__(self, tensors):
        self.tensors = tensors
        if tensors.shape[-1] == 3:
            self.tensors = tensors.permute(0, 3, 1, 2)
        
        self.transform = transforms.Compose([
                            transforms.RandomHorizontalFlip(p=0.3),
                            transforms.Lambda(lambda x: x.to(torch.float32)),
                            transforms.Lambda(lambda x: 2. * x/255. - 1.),
                            # transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                            ])
        
    def __getitem__(self, index):
        x = self.tensors[index]
        
        if self.transform:
            # mapping images to [-1.0, 1.0]
            x = self.transform(x)

        return x

    def __len__(self):
        return len(self.tensors)


# Training hyperparameters
# According to https://huggingface.co/docs/accelerate/main/en/concept_guides/performance, we should change batch size and lr from single gpu to multi gpu
# orignal batch size:10000, lr:1e-3
model_type = 'fcn'

hyperparameters = {
    #do not change item position
    'model_type': model_type,
    'num_epochs': 100,
    'mixed_precision': "fp16",#choose between 'fp16' 'no'. 'bf16' is not supported on T4
    'seed': 48763,
    'learning_rate': 1e-3,
    'batch_size': 400 # medium: smaller batchsize
}

# Build training dataloader
def get_train_dataloader(batch_size:int=10000):
    x = torch.from_numpy(train)
    train_dataset = CustomTensorDataset(x)
    
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=batch_size,num_workers=2)
    return train_dataloader

# Model
def get_model(model_type=model_type):
    # selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
    model_classes = {'resnet': Resnet(), 'fcn':fcn_autoencoder(), 'cnn':conv_autoencoder(), 'vae':VAE(), 'mix':mix_autoencoder()}
    model = model_classes[model_type]
    return model
# we define them in the train_loop func
# Loss and optimizer
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(
#     model.parameters(), lr=learning_rate)


def train_loop(model_type:str='cnn',num_epochs:int=50,mixed_precision:str="no", seed:int=23442, 
               learning_rate=1e-3,batch_size:int=10000):
    # mixed_precision {no,fp16,bf16} (str). Whether or not to use mixed precision training.
    # BF16 training is only supported on Nvidia Ampere GPUs and PyTorch 1.10 or later.
    
    #set seed
    same_seeds(seed)
    # Initialize the Accelerator
    accelerator = Accelerator(mixed_precision=mixed_precision)
    # Build the DataLoaders
     # Instantiate the model here so the seed controls new weight initalization
    # We should only download one copy of the model, and then load from the cache:
    with accelerator.main_process_first():
        model = get_model(model_type)
        train_dataloader= get_train_dataloader(batch_size)
    # Instantiate the optimizer
    # as mentioned above, the learning rate should be scaled linearly based on the number of devices present.
    learning_rate *= accelerator.num_processes
    optimizer=AdamW(model.parameters(),lr=learning_rate, weight_decay=1e-5)
#   Instantiate the learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer = optimizer,
        max_lr = learning_rate,
        epochs = num_epochs,
        steps_per_epoch = len(train_dataloader)
    )
    # Prepare everything with the Accelerator
    model,optimizer,train_dataloader=accelerator.prepare(model,optimizer,train_dataloader)
    
    # Now we train the model
    best_loss = np.inf
    criterion = nn.MSELoss()
    lambda_consist = 0.5
    
    for epoch in range(num_epochs):
        model.train()
        tot_loss = list()
        for data in train_dataloader:

            # ===================loading=====================
            if model_type in ['cnn', 'vae', 'resnet', 'mix']:
                img = data.float()
            elif model_type in ['fcn']:
                img = data.float()
                img = img.view(img.shape[0], -1)


            # ===================forward=====================
            # output, _ = model(img)
            output = model(img)
            if model_type in ['vae']:
                loss = loss_vae(output[0], img, output[1], output[2], criterion)
            elif model_type in ['mix']:
                loss_recon = criterion(output[0], img)
                loss_consist = criterion(output[1], output[2])
                loss = loss_recon + lambda_consist * loss_consist
            else:
                loss = criterion(output, img)

            tot_loss.append(loss.item())
            # ===================backward====================
            optimizer.zero_grad()
            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
#             lr_scheduler.step()
        # ===================save_best====================
        mean_loss = np.mean(tot_loss)
        accelerator.print(f"Epoch {epoch+1}|loss={mean_loss:.4f}")
        if mean_loss < best_loss:
            best_loss = mean_loss
            accelerator.save_model(model,f"best_model_{model_type}",safe_serialization=False)
            accelerator.print(f"save best_model_{model_type} with loss {best_loss:.4f}")
        # ===================log========================
        

        
    # ===================save_last========================
    # all epochs finished
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    accelerator.save(unwrapped_model.state_dict(), 'last_model_{}.pt'.format(model_type))
    accelerator.print(f"final model state dict saved at last_model_{model_type}.pt")


from time import time
start_time = time()
# Launching training from a notebook
args = list(hyperparameters.values())
start_time = time()
notebook_launcher(train_loop,args,num_processes=2)
elapsed_time = time() - start_time
print(f"total training time: {elapsed_time:.2f} seconds") 


# model_type = 'fcn'
model_type


eval_batch_size = 200
# selecting a model type from {'cnn', 'fcn', 'vae', 'resnet'}
# build testing dataloader
data = torch.tensor(test, dtype=torch.float32)
train_data=torch.tensor(train,dtype=torch.float32)
train_dataset=CustomTensorDataset(train_data)
test_dataset = CustomTensorDataset(data)
test_sampler = SequentialSampler(test_dataset)
test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=eval_batch_size, num_workers=1)
eval_loss = nn.MSELoss(reduction='none')

# load trained model
last_dict_path = "/kaggle/working/last_model_{}.pt".format(model_type)
best_dict_path="/kaggle/working/best_model_{}/pytorch_model.bin".format(model_type)
model = get_model(model_type)
model.cuda()


# prediction file
last_out_file = 'last_model_prediction.csv'
best_out_file='best_model_prediction.csv'


model.load_state_dict(torch.load(best_dict_path))
model.eval()


# hook_latents = {}

# def hook_fn(module, input, output):
#     hook_latents['fcn'] = output.detach()

# handle = model.encoder.register_forward_hook(hook_fn)
# test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False) # Use a reasonable batch size and no shuffling for analysis
# test_latent_vectors = []


# with torch.no_grad():
#     for data in test_dataloader:
#         img = data.cuda()
#         img = img.reshape(img.shape[0], -1)
#         output = model(img)
#         test_latent_vectors.append(hook_latents['fcn'].cpu().numpy())      
# handle.remove()

# test_latent_vectors = np.concatenate(test_latent_vectors, axis=0)
# print(f"Shape of extracted latent vectors: {test_latent_vectors.shape}")
# latent_mean = np.mean(test_latent_vectors, axis=0)
# latent_std = np.std(test_latent_vectors, axis=0)

# print("\nBasic Statistical Analysis of Latent Vectors:")
# print(f"Mean of latent vectors (first 5 dimensions): {latent_mean}")
# print(f"Standard deviation of latent vectors (first 5 dimensions): {latent_std}")


# from torch.utils.data import DataLoader, SequentialSampler
# test_dataloader = DataLoader(test_dataset, batch_size=256, shuffle=False) # Use a reasonable batch size and no shuffling for analysis
# test_latent_vectors = []


# with torch.no_grad():
#     for data in test_dataloader:
#         img = data.cuda()
#         img = img.reshape(img.reshape(img.shape[0], -1))
#         output =  model(img)
     
#         test_latent_vectors.append(latent.cpu().numpy())

# test_latent_vectors = np.concatenate(test_latent_vectors, axis=0)
# print(f"Shape of extracted latent vectors: {test_latent_vectors.shape}")
# latent_mean = np.mean(test_latent_vectors, axis=0)
# latent_std = np.std(test_latent_vectors, axis=0)

# print("\nBasic Statistical Analysis of Latent Vectors:")
# print(f"Mean of latent vectors (first 5 dimensions): {latent_mean}")
# print(f"Standard deviation of latent vectors (first 5 dimensions): {latent_std}")

# 


# import numpy as np
# from sklearn.manifold import TSNE
# import matplotlib.pyplot as plt
# import seaborn as sns
# # 解决matplotlib中文显示问题
# plt.rcParams['font.sans-serif'] = ['SimHei']  # 'SimHei' 是黑体
# plt.rcParams['axes.unicode_minus'] = False    # 解决负号显示问题

# # 假设 'test_latent_vectors' 是你已经创建好的 NumPy 数组
# # 其形状为 (样本数量, 潜在维度)

# # 1. 使用 t-SNE 将数据降至2维
# print("正在运行 t-SNE... (这可能需要一些时间)")
# tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, init='pca', learning_rate='auto')
# latents_2d = tsne.fit_transform(test_latent_vectors)
# print("t-SNE 运行完毕。")

# # 2. 使用核密度估计 (KDE) 图进行可视化
# plt.figure(figsize=(10, 8))
# sns.kdeplot(
#     x=latents_2d[:, 0], 
#     y=latents_2d[:, 1], 
#     cmap="viridis", # 一个适合展示密度的漂亮色板
#     fill=True,      # 填充轮廓
#     thresh=0.05,    # 低于此阈值的区域不显示
#     levels=100      # 更多的等高线级别让图像更平滑
# )
# plt.title("通过t-SNE降维后的潜在空间密度图")
# plt.xlabel("t-SNE 维度 1")
# plt.ylabel("t-SNE 维度 2")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()

# # 备选方案: 使用带透明度的散点图
# plt.figure(figsize=(10, 8))
# plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.1, s=5) # 调低alpha值可以通过颜色深浅显示密度
# plt.title("潜在空间的t-SNE散点图")
# plt.xlabel("t-SNE 维度 1")
# plt.ylabel("t-SNE 维度 2")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.show()


# # 为前4个潜在维度绘制直方图
# fig, axes = plt.subplots(4,4, figsize=(12, 10))
# fig.suptitle('各潜在维度的独立分布', fontsize=16)

# # axes.flatten() 可以方便地遍历所有子图
# for i, ax in enumerate(axes.flatten()):
#     # 确保索引没有超出范围
#     if i < all_latent_vectors.shape[1]: 
#         sns.histplot(test_latent_vectors[:, i], ax=ax, kde=True)
#         ax.set_title(f'维度 {i+1}')

# # 调整布局，防止标题重叠
# plt.tight_layout(rect=[0, 0.03, 1, 0.95])
# plt.show()


def model_evaluate(model,state_dict_path,out_file):
    model.load_state_dict(torch.load(state_dict_path))
    model.eval()
    anomality = list()
    with torch.no_grad():
        for i, data in enumerate(test_dataloader): 
            if model_type in ['cnn', 'vae', 'resnet','mix']:
                img = data.float().cuda()
            elif model_type in ['fcn']:
                img = data.float().cuda()
                img = img.view(img.shape[0], -1)
            else:
                img = data[0].cuda()
            output= model(img)
            if model_type in ['cnn', 'resnet', 'fcn']:
                output = output
            elif model_type in ['res_vae', 'mix']:
                output = output[0]
            elif model_type in ['vae']: # , 'vqvae'
                output = output[0]
                
            if model_type in ['fcn']:
                loss = eval_loss(output, img).sum(-1)
            else:
                loss = eval_loss(output, img).sum([1, 2, 3])
            anomality.append(loss)
#     with torch.no_grad():
#         for i, data in enumerate(test_dataloader):
#             img = data.float().cuda()
#             if model_type in ['fcn']:
#                 img = img.view(img.shape[0], -1)
#             output = model(img)
#             if model_type in ['vae']:
#                 output = output[0]
#             #loss
#             if model_type in ['fcn']:
#                 loss = eval_loss(output, img).sum(-1)
#             else:
#                 loss = eval_loss(output, img).sum([1, 2, 3])
#             anomality.append(loss)
    anomality = torch.cat(anomality, axis=0)
    anomality = torch.sqrt(anomality).reshape(len(test), 1).cpu().numpy()
    df = pd.DataFrame(anomality, columns=['score'])
    df.to_csv(out_file, index_label = 'ID')


model_evaluate(model,last_dict_path,last_out_file)
model_evaluate(model,best_dict_path,best_out_file)

