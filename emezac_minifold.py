import os
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.nn.utils.rnn import pad_sequence

#####################################
# 1. Carga de archivos de la competencia
#####################################
base_path = '/kaggle/input/stanford-rna-3d-folding'
train_sequences_file = os.path.join(base_path, 'train_sequences.csv')
train_labels_file = os.path.join(base_path, 'train_labels.csv')
validation_sequences_file = os.path.join(base_path, 'validation_sequences.csv')
validation_labels_file = os.path.join(base_path, 'validation_labels.csv')
test_sequences_file = os.path.join(base_path, 'test_sequences.csv')

train_sequences_df = pd.read_csv(train_sequences_file)
train_labels_df = pd.read_csv(train_labels_file)
validation_sequences_df = pd.read_csv(validation_sequences_file)
validation_labels_df = pd.read_csv(validation_labels_file)
test_sequences_df = pd.read_csv(test_sequences_file)

print("Train sequences:", train_sequences_df.shape)
print("Validation sequences:", validation_sequences_df.shape)
print("Test sequences:", test_sequences_df.shape)

#####################################
# 2. Dataset experimental
#####################################
class RNADataset(Dataset):
    def __init__(self, seq_csv, label_csv=None, max_length=512):
        self.seq_df = pd.read_csv(seq_csv)
        self.label_csv = label_csv
        if label_csv is not None:
            self.label_df = pd.read_csv(label_csv)
        else:
            self.label_df = None
        self.max_length = max_length
        self.token_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
        self.unk_token = 4

        # Filtrar secuencias que tienen coordenadas válidas (si se cuenta con etiquetas)
        if self.label_df is not None:
            valid_indices = []
            for idx, row in self.seq_df.iterrows():
                target_id = row['target_id']
                seq = row['sequence'].strip()
                tokens = self.tokenize(seq)
                L = len(tokens)
                coords = self.get_coordinates(target_id, L)
                if coords is not None:
                    valid_indices.append(idx)
                else:
                    print(f"Filtrando target {target_id} sin coordenadas válidas.")
            self.seq_df = self.seq_df.loc[valid_indices].reset_index(drop=True)
            print(f"Dataset experimental filtrado: {len(self.seq_df)} muestras válidas.")

    def tokenize(self, seq):
        tokens = [self.token_dict.get(ch, self.unk_token) for ch in seq][:self.max_length]
        return torch.tensor(tokens, dtype=torch.long)
    
    def get_coordinates(self, target_id, L):
        if self.label_df is None:
            return None
        df_target = self.label_df[self.label_df['ID'].str.startswith(target_id)]
        if df_target.empty:
            return None
        coords_flat = df_target.iloc[0, 3:].values.astype(float)
        if np.isnan(coords_flat).all():
            print(f"Warning: todas las coordenadas son NaN para target {target_id}")
            return None
        try:
            coords = torch.tensor(coords_flat, dtype=torch.float).view(-1, 3)
        except Exception as e:
            print(f"Error en reshape para target {target_id}: {e}")
            return None
        if coords.shape[0] > L:
            coords = coords[:L]
        elif coords.shape[0] < L:
            pad = torch.zeros((L - coords.shape[0], 3), dtype=torch.float)
            coords = torch.cat([coords, pad], dim=0)
        coords = torch.where(torch.isnan(coords), torch.zeros_like(coords), coords)
        return coords

    def __len__(self):
        return len(self.seq_df)
    
    def __getitem__(self, idx):
        row = self.seq_df.iloc[idx]
        target_id = row['target_id']
        seq = row['sequence'].strip()
        tokens = self.tokenize(seq)
        sample = {
            'tokens': tokens,
            'target_id': target_id,
            'length': len(tokens)  # Longitud real sin padding
        }
        if self.label_df is not None:
            coords = self.get_coordinates(target_id, len(tokens))
            if coords is not None:
                sample['coords'] = coords
        return sample

#####################################
# 3. Dataset sintético
#####################################
class SyntheticRNADataset(Dataset):
    def __init__(self, base_folder, max_length=512):
        self.max_length = max_length
        self.pdb_files = []
        for root, dirs, files in os.walk(base_folder):
            for file in files:
                if file.endswith('.pdb'):
                    self.pdb_files.append(os.path.join(root, file))
        self.samples = []
        for pdb_file in self.pdb_files:
            sample = self.parse_pdb(pdb_file)
            if sample is not None:
                self.samples.append(sample)
        print(f"Dataset sintético: {len(self.samples)} muestras extraídas de PDB.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

    def parse_pdb(self, pdb_file):
        target_id = os.path.splitext(os.path.basename(pdb_file))[0]
        try:
            with open(pdb_file, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            print(f"Error al leer {pdb_file}: {e}")
            return None
        residues = {}
        for line in lines:
            if line.startswith("ATOM"):
                atom_name = line[12:16].strip()
                resname = line[17:20].strip()
                try:
                    resSeq = int(line[22:26].strip())
                    x = float(line[30:38].strip())
                    y = float(line[38:46].strip())
                    z = float(line[46:54].strip())
                except Exception:
                    continue
                # Prioridad: C1' o C1*; si no, P
                if atom_name in ["C1'", "C1*"]:
                    residues[resSeq] = (resname, np.array([x, y, z]))
                elif atom_name == "P" and resSeq not in residues:
                    residues[resSeq] = (resname, np.array([x, y, z]))
        if len(residues) == 0:
            return None
        sorted_keys = sorted(residues.keys())
        # Convertir cada nombre de residuo a una sola letra (usando el primer carácter)
        sequence = "".join([residues[k][0][0] for k in sorted_keys])
        coords = np.stack([residues[k][1] for k in sorted_keys], axis=0)
        if len(sequence) > self.max_length:
            sequence = sequence[:self.max_length]
            coords = coords[:self.max_length, :]
        sample = {
            "target_id": target_id,
            "sequence": sequence,
            "coords": torch.tensor(coords, dtype=torch.float),
            "length": min(len(sequence), self.max_length)
        }
        return sample

#####################################
# 4. Función de collate combinada para ambos datasets
#####################################
def combined_collate_fn(batch):
    token_dict = {'A': 0, 'C': 1, 'G': 2, 'U': 3}
    unk_token = 4
    tokens_list = []
    lengths = []
    coords_list = []
    target_ids = []
    for item in batch:
        # Si ya existe "tokens", la usamos; si no, generamos a partir de "sequence"
        if 'tokens' in item:
            tokens = item['tokens']
        elif 'sequence' in item:
            seq = item['sequence']
            tokens = torch.tensor([token_dict.get(ch, unk_token) for ch in seq], dtype=torch.long)
        else:
            raise KeyError("La muestra no contiene ni 'tokens' ni 'sequence'.")
        tokens_list.append(tokens)
        lengths.append(len(tokens))  # Tomar la longitud original sin padding
        target_ids.append(item['target_id'])
        if 'coords' in item:
            coords_list.append(item['coords'])
        else:
            coords_list.append(torch.tensor([]))
    padded_tokens = pad_sequence(tokens_list, batch_first=True, padding_value=0)
    if all(c.numel() > 0 for c in coords_list):
        padded_coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)
    else:
        padded_coords = None
    batch_dict = {'tokens': padded_tokens, 'target_ids': target_ids, 'lengths': lengths}
    if padded_coords is not None:
        batch_dict['coords'] = padded_coords
    return batch_dict

#####################################
# 5. Modelo predictivo: Transformer para predecir matriz de distancias
#####################################
class MiniLLM(nn.Module):
    def __init__(self, vocab_size=5, embed_dim=256, n_heads=8, num_layers=6, hidden_dim=512, max_length=512):
        super(MiniLLM, self).__init__()
        self.max_length = max_length
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Cabeza para predecir la distancia entre pares de residuos
        self.distance_head = nn.Sequential(
            nn.Linear(2 * embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, tokens, mask=None):
        # tokens: [B, L]
        x = self.embedding(tokens)  # [B, L, embed_dim]
        pos_emb = self.pos_embedding[:, :tokens.size(1), :]
        x = x + pos_emb
        x = self.transformer(x, src_key_padding_mask=mask)  # [B, L, embed_dim]
        B, L, _ = x.size()
        x_i = x.unsqueeze(2).expand(B, L, L, x.size(-1))
        x_j = x.unsqueeze(1).expand(B, L, L, x.size(-1))
        pair_features = torch.cat([x_i, x_j], dim=-1)  # [B, L, L, 2*embed_dim]
        dist_pred = self.distance_head(pair_features).squeeze(-1)  # [B, L, L]
        return torch.relu(dist_pred)

#####################################
# 6. Reconstrucción 3D mediante MDS
#####################################
def mds_reconstruction(distance_matrix):
    L = distance_matrix.shape[0]
    J = np.eye(L) - np.ones((L, L)) / L
    B = -0.5 * J @ (distance_matrix ** 2) @ J
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1][:3]
    coords = eigvecs[:, idx] * np.sqrt(np.maximum(eigvals[idx], 0))
    return coords

#####################################
# 7. Refinamiento y escalado
#####################################
def refine_structure(init_coords, predicted_dist, num_steps=100, lr=1e-4, bond_length=5.0, bond_weight=0.1):
    coords = init_coords.clone().detach().requires_grad_(True)
    optimizer_coords = optim.Adam([coords], lr=lr)
    for step in range(num_steps):
        diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # (L, L, 3)
        dists = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)
        loss_dist = ((dists - predicted_dist) ** 2).mean()
        # Penalización para mantener la longitud de enlace entre residuos consecutivos
        consecutive = torch.sqrt(torch.sum((coords[1:] - coords[:-1]) ** 2, dim=-1) + 1e-6)
        loss_bond = ((consecutive - bond_length) ** 2).mean()
        loss = loss_dist + bond_weight * loss_bond
        optimizer_coords.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([coords], max_norm=1.0)
        optimizer_coords.step()
        if step % 25 == 0:
            print(f"Refinamiento step {step}: loss = {loss.item():.4f}")
    return coords.detach().cpu().numpy()

def scale_coordinates(coords, target_bond_length=5.0):
    diffs = coords[1:] - coords[:-1]
    bond_lengths = np.linalg.norm(diffs, axis=1)
    mean_bond = np.mean(bond_lengths)
    if mean_bond < 1e-3:
        print("Mean bond length muy bajo, la estructura puede estar degenerada.")
        return coords
    scale = target_bond_length / mean_bond
    return coords * scale

#####################################
# 8. Función de entrenamiento (usando longitudes válidas)
#####################################
def train_model(model, dataloader, optimizer, device):
    model.train()
    scaler = torch.cuda.amp.GradScaler()  # para entrenamiento FP16
    running_loss = 0.0
    total_valid = 0.0

    for batch in tqdm(dataloader, desc="Entrenando"):
        tokens = batch['tokens'].to(device)   # [B, L_pad]
        coords = batch['coords'].to(device)     # [B, L_pad, 3]
        lengths = batch['lengths']              # lista de longitudes reales
        B, L_pad, _ = coords.shape
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast():  # habilitar FP16
            pred_dists = model(tokens)  # [B, L_pad, L_pad]
            # Calcular la matriz de distancias verdadera en batch (incluyendo padding)
            diff = coords.unsqueeze(2) - coords.unsqueeze(1)  # [B, L_pad, L_pad, 3]
            true_dists = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-6)  # [B, L_pad, L_pad]
            
            # Crear máscara para posiciones válidas según cada longitud
            mask = torch.zeros(B, L_pad, dtype=torch.float32, device=device)
            for i, l in enumerate(lengths):
                mask[i, :l] = 1.0
            mask2 = mask.unsqueeze(2) * mask.unsqueeze(1)  # [B, L_pad, L_pad]
            
            # Aplicar la máscara
            valid_pred = pred_dists * mask2
            valid_true = true_dists * mask2
            
            # Calcular el número de entradas válidas
            valid_count = mask2.sum()
            # Calcular la pérdida MSE solo sobre las posiciones válidas
            loss = torch.nn.functional.mse_loss(valid_pred, valid_true, reduction='sum') / (valid_count + 1e-6)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item() * valid_count.item()
        total_valid += valid_count.item()
    
    avg_loss = running_loss / total_valid if total_valid > 0 else 0
    print(f"Loss de entrenamiento promedio: {avg_loss:.4f}")

#####################################
# 9. Función de inferencia: generación de 5 predicciones
#####################################
def generate_predictions(model, tokens, device, num_predictions=5, refine=True):
    """
    Función mejorada para generar predicciones más robustas.
    """
    model.eval()
    tokens = tokens.to(device)
    seq_length = tokens.size(1)  # Longitud de la secuencia
    
    with torch.no_grad():
        try:
            pred_dists = model(tokens)  # [1, L, L]
            dist_matrix = pred_dists[0].cpu().numpy()
        except Exception as e:
            print(f"Error en modelo de predicción: {e}")
            # Crear una matriz de distancias fallback
            dist_matrix = np.ones((seq_length, seq_length)) * 5.0
            np.fill_diagonal(dist_matrix, 0.0)
    
    # Limpiar la matriz de distancias
    if np.isnan(dist_matrix).any() or np.isinf(dist_matrix).any():
        print("Advertencia: matriz de distancias contiene NaN o Inf, se reemplazan por valores razonables.")
        mask = np.isnan(dist_matrix) | np.isinf(dist_matrix)
        dist_matrix[mask] = 5.0  # Valor razonable para distancias desconocidas
    
    # Asegurar que la matriz sea simétrica
    dist_matrix = 0.5 * (dist_matrix + dist_matrix.T)
    
    # Reconstrucción mediante MDS con manejo de errores
    try:
        init_coords = mds_reconstruction(dist_matrix)
    except Exception as e:
        print(f"Error en MDS: {e}")
        # Crear coordenadas en línea recta como fallback
        init_coords = np.zeros((seq_length, 3))
        for i in range(seq_length):
            init_coords[i, 0] = i * 5.0  # Separación de 5.0 en el eje X
    
    # Normalizar y escalar
    try:
        init_coords = init_coords - init_coords[0]  # Fijar primer residuo en (0,0,0)
        init_coords = scale_coordinates(init_coords, target_bond_length=5.0)
    except Exception as e:
        print(f"Error en normalización: {e}")
    
    # Refinamiento con manejo de errores
    if refine:
        try:
            init_coords_tensor = torch.from_numpy(init_coords).float().to(device)
            predicted_dist_tensor = torch.from_numpy(dist_matrix).float().to(device)
            refined = refine_structure(init_coords_tensor, predicted_dist_tensor,
                                      num_steps=100, lr=1e-4, bond_length=5.0, bond_weight=0.1)
            refined = scale_coordinates(refined, target_bond_length=5.0)
        except Exception as e:
            print(f"Error en refinamiento: {e}")
            refined = init_coords
    else:
        refined = init_coords
    
    # Final limpieza y normalización
    # refined = refined - refined[0]
    refined = np.round(refined, 12)
    
    # Limpiar NaN o Inf residuales
    if np.isnan(refined).any() or np.isinf(refined).any():
        print("Advertencia: coordenadas refinadas contienen NaN o Inf, se reemplazan por ceros.")
        refined = np.nan_to_num(refined, nan=0.0, posinf=0.0, neginf=0.0)
    
    # Primera predicción
    predictions = [refined]
    
    # Generar variantes
    for _ in range(num_predictions - 1):
        try:
            noise = np.random.normal(scale=0.2, size=refined.shape)
            variant = refined + noise
            variant = scale_coordinates(variant, target_bond_length=5.0)
            variant = variant - variant[0]
            
            # Limpiar posibles NaN o Inf
            if np.isnan(variant).any() or np.isinf(variant).any():
                variant = np.nan_to_num(variant, nan=0.0, posinf=0.0, neginf=0.0)
                
            variant = np.round(variant, 12)
            predictions.append(variant)
        except Exception as e:
            print(f"Error al generar variante: {e}")
            # Si hay error, duplicar la predicción base
            predictions.append(refined.copy())
    
    # Asegurar que todas las predicciones sean válidas
    for i in range(len(predictions)):
        if not isinstance(predictions[i], np.ndarray) or predictions[i].shape != (seq_length, 3):
            print(f"Corrigiendo forma de predicción {i+1}")
            new_pred = np.zeros((seq_length, 3))
            if isinstance(predictions[i], np.ndarray):
                min_len = min(seq_length, predictions[i].shape[0])
                if predictions[i].ndim >= 2 and predictions[i].shape[1] >= 3:
                    new_pred[:min_len] = predictions[i][:min_len, :3]
            predictions[i] = new_pred
    
    return predictions
  
#####################################
# 10. Función para guardar submission en el formato requerido
#####################################
def save_submission_per_residue(test_df, predictions_dict, output_file='submission.csv', max_length=512):
    """
    Función corregida para crear el archivo de presentación con el formato correcto.
    Garantiza que todas las secuencias en test_df tengan entradas en el archivo final.
    """
    rows = []
    
    # Recorre cada secuencia en el conjunto de prueba
    for _, row in test_df.iterrows():
        target_id = row['target_id']
        seq_full = row['sequence'].strip()
        seq_trunc = seq_full[:max_length]
        seq_len = len(seq_trunc)
        
        # Verifica si hay predicciones para este target_id
        if target_id not in predictions_dict:
            print(f"Advertencia: no se encontraron predicciones para {target_id}, generando coordenadas nulas.")
            # En lugar de omitir, genera coordenadas nulas para todas las posiciones
            for i in range(seq_len):
                resid = i + 1
                resname = seq_trunc[i]
                rid = f"{target_id}_{resid}"
                # Coordenadas nulas para las 5 predicciones (15 valores)
                coords_flat = [0.0] * 15
                rows.append([rid, resname, resid] + coords_flat)
            continue
            
        # Obtiene las predicciones para este target_id
        preds = predictions_dict[target_id]
        
        # Asegura que tengamos exactamente 5 predicciones
        while len(preds) < 5:
            if len(preds) > 0:
                # Duplica la última predicción si hay al menos una
                preds.append(preds[-1].copy())
            else:
                # Si no hay predicciones, crea una matriz nula
                preds.append(np.zeros((seq_len, 3)))
        
        # Limita a 5 predicciones
        preds = preds[:5]
        
        # Procesa cada residuo en la secuencia
        for i in range(seq_len):
            resid = i + 1
            resname = seq_trunc[i]
            rid = f"{target_id}_{resid}"
            coords_flat = []
            
            # Recoge las coordenadas de cada predicción
            for pred_idx, pred in enumerate(preds):
                # Asegura que la predicción tenga el tamaño adecuado
                if not isinstance(pred, np.ndarray) or i >= pred.shape[0] or pred.shape[1] != 3:
                    print(f"Error en predicción {pred_idx+1} para {target_id} residuo {i+1}, usando coordenadas nulas")
                    coords_flat.extend([0.0, 0.0, 0.0])
                else:
                    # Extrae las coordenadas y maneja posibles NaN o inf
                    coord = pred[i]
                    clean_coords = []
                    for c in coord:
                        if np.isnan(c) or np.isinf(c):
                            clean_coords.append(0.0)
                        else:
                            clean_coords.append(float(c))
                    coords_flat.extend(clean_coords)
            
            # Asegura que haya exactamente 15 valores (5 predicciones x 3 dimensiones)
            if len(coords_flat) < 15:
                print(f"Faltan coordenadas para {rid}, completando con ceros")
                coords_flat.extend([0.0] * (15 - len(coords_flat)))
            elif len(coords_flat) > 15:
                print(f"Demasiadas coordenadas para {rid}, truncando")
                coords_flat = coords_flat[:15]
                
            # Redondea a 12 decimales para evitar problemas de precisión
            coords_flat = [round(float(c), 12) for c in coords_flat]
            
            # Agrega la fila al conjunto de resultados
            rows.append([rid, resname, resid] + coords_flat)
    
    # Define las columnas según lo esperado por la competencia
    columns = ["ID", "resname", "resid"] + [f"{dim}_{i}" for i in range(1, 6) for dim in ["x", "y", "z"]]
    
    # Crea el DataFrame
    submission_df = pd.DataFrame(rows, columns=columns)
    
    # Asegura que no haya valores NaN
    submission_df = submission_df.fillna(0.0)
    
    # Convierte explícitamente las columnas de coordenadas a float
    for col in columns[3:]:
        submission_df[col] = submission_df[col].astype(float)
    
    # Guarda el archivo
    submission_df.to_csv(output_file, index=False, float_format='%.12f')
    print(f"Submission guardado en {output_file} con {len(submission_df)} filas")
    print("Vista previa de los primeros 5 registros:")
    print(submission_df.head())
    
    return submission_df

def verify_submission(submission_file, test_df=None):
    """
    Función mejorada para verificar el archivo de presentación.
    Ahora verifica más aspectos y puede validar contra el DataFrame de prueba original.
    """
    df = pd.read_csv(submission_file)
    
    # Verificaciones básicas de formato
    if len(df.columns) != 18:
        print(f"Error: El archivo de envío tiene {len(df.columns)} columnas, se esperaban 18.")
        return False
        
    if df.isnull().any().any():
        print("Error: El archivo de envío contiene valores NaN.")
        null_counts = df.isnull().sum()
        print("Columnas con valores NaN:", null_counts[null_counts > 0])
        return False
    
    # Verificar tipos de datos
    for col in df.columns[3:]:
        if not np.issubdtype(df[col].dtype, np.floating):
            print(f"Error: La columna {col} no contiene valores flotantes.")
            return False
    
    # Verificar valores extremos o incorrectos
    for col in df.columns[3:]:
        if df[col].isin([np.inf, -np.inf]).any():
            print(f"Error: La columna {col} contiene valores infinitos.")
            return False
        
        if (df[col].abs() > 1e6).any():
            print(f"Advertencia: La columna {col} contiene valores muy grandes (>1e6).")
    
    # Si se proporciona el DataFrame de prueba, verificar la correspondencia completa
    if test_df is not None:
        expected_ids = set()
        for _, row in test_df.iterrows():
            target_id = row['target_id']
            seq = row['sequence'].strip()[:512]  # Truncar a max_length
            for i in range(len(seq)):
                expected_ids.add(f"{target_id}_{i+1}")
        
        submission_ids = set(df['ID'])
        
        missing_ids = expected_ids - submission_ids
        if missing_ids:
            print(f"Error: Faltan {len(missing_ids)} IDs en el archivo de envío.")
            print("Ejemplos de IDs faltantes:", list(missing_ids)[:5])
            return False
        
        extra_ids = submission_ids - expected_ids
        if extra_ids:
            print(f"Error: Hay {len(extra_ids)} IDs adicionales en el archivo de envío.")
            print("Ejemplos de IDs adicionales:", list(extra_ids)[:5])
            return False
    
    print("El archivo de envío cumple con el formato esperado.")
    return True

#####################################
# 11. Bloque principal
#####################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Usando dispositivo:", device)
    
    # Crear datasets: experimental y sintético
    exp_dataset = RNADataset(train_sequences_file, label_csv=train_labels_file, max_length=512)
    synthetic_folder = '/kaggle/input/uw-synthetic-rna-final/compile_all/'
    synth_dataset = SyntheticRNADataset(synthetic_folder, max_length=512)
    
    # Combinar ambos datasets para pre-entrenamiento
    combined_dataset = ConcatDataset([synth_dataset, exp_dataset])
    train_loader = DataLoader(combined_dataset, batch_size=8, shuffle=True, collate_fn=combined_collate_fn)
    
    # Dataset de test (usamos el dataset experimental sin etiquetas)
    test_dataset = RNADataset(test_sequences_file, label_csv=None, max_length=512)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=combined_collate_fn)
    
    # Instanciar y entrenar el modelo predictivo
    model = MiniLLM(vocab_size=5, embed_dim=256, n_heads=8, num_layers=6, hidden_dim=512, max_length=512)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    epochs = 3
    for epoch in range(epochs):
        print(f"\n=== Epoch {epoch+1}/{epochs} ===")
        train_model(model, train_loader, optimizer, device)
    
    # Inferencia en test: para cada secuencia se generan 5 predicciones
    model.eval()
    predictions_dict = {}
    
    # Crear un conjunto de todos los target_ids en el conjunto de prueba
    all_test_targets = set(test_sequences_df['target_id'])
    
    # Realizar inferencia con manejo de errores
    for batch in tqdm(test_loader, desc="Inferencia en test"):
        try:
            tokens = batch['tokens']  # [1, L]
            target_ids = batch['target_ids']
            tid = target_ids[0]
            
            preds = generate_predictions(model, tokens, device, num_predictions=5, refine=True)
            predictions_dict[tid] = preds
            
            # Remover este target_id del conjunto de todos los targets
            if tid in all_test_targets:
                all_test_targets.remove(tid)
        except Exception as e:
            print(f"Error en predicción para {tid}: {e}")
            # Crear predicciones nulas para este target
            seq_length = tokens.size(1)
            null_preds = [np.zeros((seq_length, 3)) for _ in range(5)]
            predictions_dict[tid] = null_preds
    
    # Generar predicciones para cualquier target_id que no se procesó
    for tid in all_test_targets:
        print(f"Generando predicciones nulas para target no procesado: {tid}")
        row = test_sequences_df[test_sequences_df['target_id'] == tid].iloc[0]
        seq = row['sequence'].strip()[:512]
        seq_length = len(seq)
        null_preds = [np.zeros((seq_length, 3)) for _ in range(5)]
        predictions_dict[tid] = null_preds
    
    # Guardar submission en el formato requerido con verificación adicional
    submission_df = save_submission_per_residue(test_sequences_df, predictions_dict, output_file='submission.csv', max_length=512)
    verify_submission('submission.csv', test_df=test_sequences_df)
    
    # Verificar la integridad de la presentación una vez más
    print("\nVerificación final del archivo de envío:")
    verify_submission('submission.csv')
    
    # Contar el número total de filas esperadas
    total_expected_rows = 0
    for _, row in test_sequences_df.iterrows():
        seq = row['sequence'].strip()[:512]
        total_expected_rows += len(seq)
    
    # Confirmar el número de filas
    actual_rows = len(pd.read_csv('submission.csv'))
    print(f"\nTotal de filas esperadas: {total_expected_rows}")
    print(f"Total de filas en el archivo: {actual_rows}")
    
    if total_expected_rows != actual_rows:
        print(f"ADVERTENCIA: Discrepancia en el número de filas. Faltan {total_expected_rows - actual_rows} filas.")


