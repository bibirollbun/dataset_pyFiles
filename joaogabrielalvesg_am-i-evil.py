import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm
import random

# --- 1. CONFIGURAÇÃO ---
# Parâmetros para controlar o treinamento. Modifique-os conforme necessário.
class Config:
    TRAIN_DIR = "/kaggle/input/waveform-inversion/train_samples/"
    TEST_DIR = "/kaggle/input/waveform-inversion/test/"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 8
    EPOCHS = 10 # Comece com poucas épocas e aumente conforme o resultado
    LEARNING_RATE = 1e-4
    VALIDATION_SPLIT = 0.1 # 10% dos dados para validação
    SEED = 42

# Para garantir a reprodutibilidade dos resultados
random.seed(Config.SEED)
np.random.seed(Config.SEED)
torch.manual_seed(Config.SEED)
if Config.DEVICE == "cuda":
    torch.cuda.manual_seed(Config.SEED)

# --- 2. PIPELINE DE DADOS (DATASET) ---
# Esta classe é o coração da pipeline de dados. Ela lida com o carregamento,
# pré-processamento e transformação dos dados sísmicos em tensores.

class SeismicDataset(Dataset):
    """
    Dataset personalizado para carregar os dados sísmicos e os mapas de velocidade.
    Cada arquivo .npy contém um batch de 500 amostras. Esta classe trata cada uma
    dessas 500 amostras como um item individual.
    """
    def __init__(self, file_paths, is_test=False):
        self.file_paths = file_paths
        self.is_test = is_test
        # Cada arquivo contém 500 amostras. O tamanho do dataset é o número de arquivos * 500.
        self.num_samples_per_file = 500
        
        # O diretório de teste tem uma estrutura diferente (cada .npy é uma amostra)
        if self.is_test:
            # Nos dados de teste, cada arquivo .npy é uma única amostra 3D (5, 1000, 70), 
            # não um batch de amostras 4D.
            self.num_samples_per_file = 1

        # Transformação para redimensionar os dados sísmicos de (1000, 70) para (70, 70)
        self.resizer = transforms.Resize((70, 70), antialias=True)
        
    def __len__(self):
        return len(self.file_paths) * self.num_samples_per_file

    def __getitem__(self, idx):
        # Mapeia o índice global para um índice de arquivo e um índice de amostra dentro do arquivo
        file_idx = idx // self.num_samples_per_file
        sample_idx = idx % self.num_samples_per_file
        
        seis_path = self.file_paths[file_idx][0]
        
        if self.is_test:
            # Carrega a amostra de teste única
            seis_sample_np = np.load(seis_path).astype(np.float32) # Shape: (5, 1000, 70)
        else:
            # Carrega o batch de treino e pega a amostra específica
            seis_data_full = np.load(seis_path).astype(np.float32)
            seis_sample_np = seis_data_full[sample_idx] # Shape: (5, 1000, 70)
            
        seis_sample = torch.from_numpy(seis_sample_np)
        
        # --- Pré-processamento dos Dados de Entrada (Sismograma) ---
        # 1. Redimensionar a altura (time_steps) de 1000 para 70
        seis_resized = self.resizer(seis_sample) # Shape: (5, 70, 70)
        
        # 2. Normalização por amostra
        max_abs = torch.abs(seis_resized).max()
        if max_abs > 0:
            seis_resized /= max_abs
        
        if self.is_test:
            return seis_resized
        else:
            vel_path = self.file_paths[file_idx][1]
            vel_map_full = np.load(vel_path).astype(np.float32)
            
            # Pega o mapa de velocidade correspondente (shape: 1, 70, 70)
            vel_sample = torch.from_numpy(vel_map_full[sample_idx])

            # --- Pré-processamento do Alvo (Mapa de Velocidade) ---
            # Normalizar para o intervalo [0, 1]
            vmin, vmax = 1500, 4500
            vel_sample_normalized = (vel_sample - vmin) / (vmax - vmin)
            
            return seis_resized, vel_sample_normalized


# --- 3. MODELO (ARQUITETURA U-NET) ---
# A implementação da U-Net, ideal para tarefas de imagem-para-imagem.
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Corrige a diferença de tamanho devido ao arredondamento no MaxPool
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        # Adiciona padding ao tensor upsampled (x1) para que tenha o mesmo tamanho do tensor da skip connection (x2)
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        factor = 2 if bilinear else 1

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits


# --- 4. LÓGICA DE TREINAMENTO ---

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Training", leave=False)
    for inputs, targets in progress_bar:
        inputs, targets = inputs.to(device), targets.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())
        
    return total_loss / len(dataloader)

def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    progress_bar = tqdm(dataloader, desc="Validating", leave=False)
    with torch.no_grad():
        for inputs, targets in progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())
            
    return total_loss / len(dataloader)

# --- 5. EXECUÇÃO PRINCIPAL ---

if __name__ == '__main__':
    print(f"Usando dispositivo: {Config.DEVICE}")

    # Coletar todos os pares de arquivos de treino
    all_files = []
    for family in os.listdir(Config.TRAIN_DIR):
        family_path = os.path.join(Config.TRAIN_DIR, family)
        if os.path.isdir(family_path):
            data_dir = os.path.join(family_path, 'data')
            model_dir = os.path.join(family_path, 'model')
            
            # Lida com a estrutura de pastas diferente
            if os.path.exists(data_dir) and os.path.exists(model_dir):
                # Famílias Vel e Style
                data_files = sorted(glob.glob(os.path.join(data_dir, '*.npy')))
                model_files = sorted(glob.glob(os.path.join(model_dir, '*.npy')))
            else:
                # Famílias Fault
                data_files = sorted(glob.glob(os.path.join(family_path, 'seis*.npy')))
                model_files = sorted(glob.glob(os.path.join(family_path, 'vel*.npy')))
            
            for d_file, m_file in zip(data_files, model_files):
                all_files.append((d_file, m_file))

    # Embaralhar e dividir os arquivos para treino e validação
    random.shuffle(all_files)
    split_idx = int(len(all_files) * (1 - Config.VALIDATION_SPLIT))
    train_files = all_files[:split_idx]
    val_files = all_files[split_idx:]

    print(f"Total de arquivos: {len(all_files)}. Treino: {len(train_files)}, Validação: {len(val_files)}")

    # Criar Datasets e DataLoaders
    train_dataset = SeismicDataset(train_files)
    val_dataset = SeismicDataset(val_files)
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # Inicializar modelo, otimizador e função de perda
    # n_channels=5 (5 fontes sísmicas), n_classes=1 (mapa de velocidade monocromático)
    model = UNet(n_channels=5, n_classes=1).to(Config.DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    criterion = nn.L1Loss() # MAE Loss, conforme a métrica da competição

    # Loop de treinamento
    best_val_loss = float('inf')
    for epoch in range(Config.EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{Config.EPOCHS} ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, Config.DEVICE)
        val_loss = validate_one_epoch(model, val_loader, criterion, Config.DEVICE)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print("✨ Modelo salvo! Melhor perda de validação até agora.")

    print("\nTreinamento concluído!")

# --- 6. GERAÇÃO DA SUBMISSÃO (Exemplo) ---
# Esta parte deve ser executada após o treinamento, num script separado ou célula.

def generate_submission():
    print("Gerando arquivo de submissão...")
    model = UNet(n_channels=5, n_classes=1).to(Config.DEVICE)
    model.load_state_dict(torch.load('best_model.pth'))
    model.eval()

    test_files = [(f, '') for f in glob.glob(os.path.join(Config.TEST_DIR, '*.npy'))]
    test_dataset = SeismicDataset(test_files, is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    predictions = []
    file_ids = [os.path.basename(f[0]).replace('.npy', '') for f in test_files]

    with torch.no_grad():
        for i, inputs in enumerate(tqdm(test_loader, desc="Predicting")):
            inputs = inputs.to(Config.DEVICE)
            output = model(inputs) # Shape: (1, 1, 70, 70)
            
            # --- Pós-processamento ---
            # 1. Desnormalizar a predição
            vmin, vmax = 1500, 4500
            unnormalized_prediction = output * (vmax - vmin) + vmin
            
            # 2. Garantir que os valores estão no intervalo físico
            final_prediction = torch.clamp(unnormalized_prediction, vmin, vmax)
            
            pred_np = final_prediction.cpu().numpy().squeeze() # Shape: (70, 70)
            
            # Formatar para o arquivo de submissão
            oid = file_ids[i]
            for y_pos in range(pred_np.shape[0]):
                row_id = f"{oid}_y_{y_pos}"
                row_data = {'oid_ypos': row_id}
                # A submissão exige apenas as colunas ímpares
                for x_pos in range(1, pred_np.shape[1], 2):
                    row_data[f'x_{x_pos}'] = pred_np[y_pos, x_pos]
                predictions.append(row_data)

    submission_df = pd.DataFrame(predictions)
    submission_df.to_csv('submission.csv', index=False)
    print("Arquivo submission.csv gerado com sucesso!")




