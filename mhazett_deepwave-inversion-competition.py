import numpy as np
import os
from glob import glob
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split


base_dir = '/kaggle/input/waveform-inversion/train_samples'

folders = ['CurveVel_A', 'CurveVel_B', 'FlatVel_A', 'FlatVel_B', 'Style_A', 'Style_B']

seismic_list = []
velocity_list = []

for folder in folders:
    data_path = os.path.join(base_dir, folder, 'data')
    model_path = os.path.join(base_dir, folder, 'model')

    data_files = sorted(glob(os.path.join(data_path, '*.npy')))
    model_files = sorted(glob(os.path.join(model_path, '*.npy')))

    if len(data_files) != len(model_files):
        raise ValueError("Mismatch in " + folder)

    for data_file, model_file in zip(data_files, model_files):
        seismic = np.load(data_file)
        velocity = np.load(model_file)

        seismic_list.append(seismic)
        velocity_list.append(velocity)

print("Total samples loaded: {}".format(len(seismic_list)))
print("Seismic shape: {}".format(seismic_list[0].shape))
print("Velocity map shape: {}".format(velocity_list[0].shape))


for folder in folders:
    print("Checking folder:", folder)

    data_path = os.path.join(base_dir, folder, 'data')
    model_path = os.path.join(base_dir, folder, 'model')

    data_files = sorted(glob(os.path.join(data_path, '*.npy')))
    model_files = sorted(glob(os.path.join(model_path, '*.npy')))

    if len(data_files) == 0 or len(model_files) == 0:
        print("No .npy files found in either 'data' or 'model'.")
        continue

    velocity = np.load(model_files[0])

    if len(velocity.shape) == 2:
        rows, cols = velocity.shape
        print("Velocity map shape: rows =", rows, ", columns =", cols)
    else:
        print("Velocity map is not 2D, shape:", velocity.shape)


velocity = np.load('/kaggle/input/waveform-inversion/train_samples/Style_B/model/model1.npy')

single_map = velocity[0, 0]

plt.imshow(single_map, cmap='viridis')
plt.title("Velocity Map Sample (Style A model1)")
plt.colorbar(label='Velocity')
plt.xlabel("X")
plt.ylabel("Y")


velocity_model2 = np.load('/kaggle/input/waveform-inversion/train_samples/Style_A/model/model2.npy')

print("Minimum velocity model2:", velocity_model2.min())
print("Maximum velocity model2:", velocity_model2.max())

single_map_model2 = velocity_model2[0, 0]

plt.imshow(single_map_model2, cmap='viridis')
plt.title("Velocity Map Sample (Style A model2)")
plt.colorbar(label='Velocity in m/s (meter per second)')
plt.xlabel("X")
plt.ylabel("Y")


seismic = np.load('/kaggle/input/waveform-inversion/train_samples/CurveFault_A/seis2_1_0.npy')

print("Seismic shape:", seismic.shape)

trace = seismic[0, 2, :, 10]

plt.plot(trace)
plt.title("Seismic Trace (Sample 0, Source 2, Receiver 10)")
plt.xlabel("Time Step")
plt.ylabel("Amplitude")
plt.grid(True)


sample = 0
source = 2

plt.figure(figsize=(10, 6))
offset = 5

for i in range(5):
    trace = seismic[sample, source, :, i]
    plt.plot(trace + i * offset, label='Receiver ' + str(i), alpha=0.8)

plt.title("Offset Seismic Traces (Sample 0, Source 2, Receivers 0–4)")
plt.xlabel("Time Step")
plt.ylabel("Amplitude + Offset")
plt.legend()
plt.grid(True)
plt.tight_layout()


flat_seismic_list = []
flat_velocity_list = []

for seismic_batch, velocity_batch in zip(seismic_list, velocity_list):
    for i in range(seismic_batch.shape[0]):
        flat_seismic_list.append(seismic_batch[i])
        flat_velocity_list.append(velocity_batch[i])

class WaveformDataset(Dataset):
    def __init__(self, seismic_list, velocity_list):
        self.seismic_list = seismic_list
        self.velocity_list = velocity_list

    def __len__(self):
        return len(self.seismic_list)

    def __getitem__(self, idx):
        seismic = self.seismic_list[idx].astype(np.float32)
        velocity = self.velocity_list[idx].astype(np.float32)

        mean = seismic.mean()
        std = seismic.std()
        if std > 0:
            seismic = (seismic - mean) / std

        v_min = velocity.min()
        v_max = velocity.max()
        if v_max > v_min:
            velocity = (velocity - v_min) / (v_max - v_min)

        seismic = torch.tensor(seismic, dtype=torch.float32)
        velocity = torch.tensor(velocity, dtype=torch.float32)

        if velocity.ndim == 3 and velocity.shape[0] == 1:
            velocity = velocity.squeeze(0)

        return seismic, velocity

dataset = WaveformDataset(flat_seismic_list, flat_velocity_list)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)

class UNet(nn.Module):
    def __init__(self, in_channels=5, out_channels=1):
        super(UNet, self).__init__()

        def conv_block(in_c, out_c):
            
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True)
            )

        self.enc1 = conv_block(in_channels, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = conv_block(128, 256)

        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, output_padding=1)
        self.dec1 = conv_block(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = conv_block(128, 64)

        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc3(p2)

        d1 = self.up1(e3)
        d1 = torch.cat([d1, e2], dim=1) 
        d1 = self.dec1(d1)

        d2 = self.up2(d1)
        d2 = torch.cat([d2, e1], dim=1)
        d2 = self.dec2(d2)

        return self.out(d2)
model = UNet(in_channels=5, out_channels=1)

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'

model = model.to(device)

for seismic_batch, velocity_batch in dataloader:
    seismic_batch = seismic_batch.to(device)
    velocity_batch = velocity_batch.to(device)

    print("Input shape before forward:", seismic_batch.shape)
    print("Velocity shape:", velocity_batch.shape)

    break


model.eval()
with torch.no_grad():
    for seismic_batch_eval, velocity_batch_eval in dataloader:
        seismic_batch_eval = seismic_batch_eval.to(device)
        velocity_batch_eval = velocity_batch_eval.to(device)

        if seismic_batch_eval.ndim == 4:
            input_for_model_eval = seismic_batch_eval.permute(0, 1, 3, 2)
        else:
            input_for_model_eval = seismic_batch_eval
        input_for_model_eval = input_for_model_eval[:, :, :70, :70]

        output = model(input_for_model_eval)
        
        pred = output[0].squeeze().cpu().numpy()
        target = velocity_batch[0].squeeze().cpu().numpy()

        plt.figure(figsize=(10, 4))

        plt.subplot(1, 2, 1)
        plt.imshow(target, cmap='viridis')
        plt.title("Ground Truth Velocity")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(pred, cmap='viridis')
        plt.title("Predicted Velocity")
        plt.colorbar()

        plt.suptitle("Model Prediction vs Ground Truth")
        plt.tight_layout()

        break


num_epochs = 10

criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for seismic_batch, velocity_batch in dataloader:
        seismic_batch = seismic_batch.to(device)
        velocity_batch = velocity_batch.to(device)

        optimizer.zero_grad()

        if seismic_batch.ndim == 4:
            seismic_batch = seismic_batch.permute(0, 1, 3, 2)

   
        seismic_batch = seismic_batch[:, :, :70, :70]

        if velocity_batch.ndim == 3:
            velocity_batch = velocity_batch.unsqueeze(1)

        output = model(seismic_batch)

       
        velocity_batch = velocity_batch[:, :, :70, :70]

        loss = criterion(output, velocity_batch)

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(dataloader)
    print("Epoch {}/{} , Average Loss: {:.4f}".format(epoch + 1, num_epochs, avg_loss))


from sklearn.metrics import mean_absolute_error

model.eval()
val_mae = []

with torch.no_grad():
    for seismic_batch, velocity_batch in dataloader:  
        seismic_batch = seismic_batch.to(device)
        velocity_batch = velocity_batch.to(device)

        if seismic_batch.ndim == 4:
            seismic_batch = seismic_batch.permute(0, 1, 3, 2)

        seismic_batch = seismic_batch[:, :, :70, :70]
        velocity_batch = velocity_batch.unsqueeze(1) if velocity_batch.ndim == 3 else velocity_batch
        velocity_batch = velocity_batch[:, :, :70, :70]

        preds = model(seismic_batch)

        preds_flat = preds.view(preds.size(0), -1).cpu().numpy()
        targets_flat = velocity_batch.view(velocity_batch.size(0), -1).cpu().numpy()

        for pred, target in zip(preds_flat, targets_flat):
            mae = mean_absolute_error(target, pred)
            val_mae.append(mae)

print("Validation MAE: {:.4f}".format(np.mean(val_mae)))


model.eval()

with torch.no_grad():
    for seismic_batch, velocity_batch in dataloader:
        seismic_batch = seismic_batch.to(device)
        velocity_batch = velocity_batch.to(device)

        if seismic_batch.ndim == 4:
            seismic_batch = seismic_batch.permute(0, 1, 3, 2)

        seismic_batch = seismic_batch[:, :, :70, :70]
        velocity_batch = velocity_batch.unsqueeze(1) if velocity_batch.ndim == 3 else velocity_batch
        velocity_batch = velocity_batch[:, :, :70, :70]

        preds = model(seismic_batch)

        pred_sample = preds[0].squeeze().cpu().numpy()
        target_sample = velocity_batch[0].squeeze().cpu().numpy()

        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.imshow(target_sample, cmap='viridis')
        plt.title("Ground Truth Velocity")
        plt.colorbar()

        plt.subplot(1, 2, 2)
        plt.imshow(pred_sample, cmap='viridis')
        plt.title("Predicted Velocity")
        plt.colorbar()

        plt.suptitle("Prediction vs. Ground Truth")
        plt.tight_layout()

        break


import csv

test_dir = '/kaggle/input/waveform-inversion/train_samples'

test_subfolders = [
    'CurveFault_A', 'CurveFault_B', 'CurveVel_A', 'CurveVel_B',
    'FlatFault_A', 'FlatFault_B', 'FlatVel_A', 'FlatVel_B',
    'Style_A', 'Style_B'
]

all_test_npy_file_paths = []
all_test_oids = []

print(f"\n--- Starting Inference: Iterating through subfolders in: {test_dir} ---")

for subfolder_name in test_subfolders:
    current_subfolder_path = os.path.join(test_dir, subfolder_name)

    path_to_scan = os.path.join(current_subfolder_path, 'data')

    if not os.path.isdir(path_to_scan):
        print(f"Warning: Path to scan '{path_to_scan}' not found or is not a directory for subfolder '{subfolder_name}'. Skipping.")
        continue
    print(f"  Scanning for .npy files in: {path_to_scan}")
    found_in_subfolder_count = 0  
    try:
        for f_path in glob(os.path.join(path_to_scan, '*.npy')):
            all_test_npy_file_paths.append(f_path)
            oid = os.path.basename(f_path).replace('.npy', '')
            all_test_oids.append(oid)
            found_in_subfolder_count += 1
        if found_in_subfolder_count > 0:
            print(f"    Found {found_in_subfolder_count} .npy files in {path_to_scan}.")
        else:
            print(f"    No .npy files found in {path_to_scan}")
    except Exception as e:
        print(f"Error accessing or listing files in {path_to_scan}: {e}")

if not all_test_npy_file_paths: 
    print("CRITICAL ERROR: No .npy test files found in any of the specified subfolders after scanning. Cannot perform inference.")
    with open('submission.csv', mode='w', newline='') as f_csv:
        writer = csv.writer(f_csv)
        writer.writerow(['id'] + [f"x_{i}" for i in range(1, 70, 2)])
    exit()

print(f"Total .npy files found across all subfolders: {len(all_test_npy_file_paths)}")

test_seismic_list = []

final_test_oids_for_dataset = []

for i, file_path_to_load in enumerate(all_test_npy_file_paths): 
    try:
        test_seismic_list.append(np.load(file_path_to_load))
        final_test_oids_for_dataset.append(all_test_oids[i]) 
    except Exception as e:
        print(f"Warning: Could not load test file {file_path_to_load}: {e}. Skipping this file.")


if not test_seismic_list:
    print("ERROR: No test data successfully loaded after attempting all found files. Cannot create submission.")
    exit()

print(f"Successfully loaded {len(test_seismic_list)} seismic samples for testing.")

class TestDataset(Dataset):
    def __init__(self, seismic_list, oid_list):
        self.seismic_list = seismic_list
        self.oid_list = oid_list
        print(f"TestDataset initialized with {len(self.seismic_list)} samples.")

    def __len__(self):
        return len(self.seismic_list)

    def __getitem__(self, idx):
        seismic = self.seismic_list[idx].astype(np.float32)
        oid = self.oid_list[idx]

        loaded_seismic_data_for_oid = self.seismic_list[idx]
        oid = self.oid_list[idx]
        single_seismic_sample = None

        if loaded_seismic_data_for_oid.ndim == 4:
            if loaded_seismic_data_for_oid.shape[0] > 1 and loaded_seismic_data_for_oid.shape[1] == 5:
                print(f"    Note: Test file for OID {oid} contains {loaded_seismic_data_for_oid.shape[0]} samples. Processing the first one.")
                single_seismic_sample = loaded_seismic_data_for_oid[0]

            elif loaded_seismic_data_for_oid.shape[0] == 1 and loaded_seismic_data_for_oid.shape[
                1] == 5:  
                single_seismic_sample = loaded_seismic_data_for_oid.squeeze(
                    0)  
            else:
                raise ValueError(
                    f"Unexpected 4D seismic data shape for OID {oid}: {loaded_seismic_data_for_oid.shape}. Expected (N, 5, 1500, 70) or (1, 5, 1500, 70).")

        elif loaded_seismic_data_for_oid.ndim == 3 and loaded_seismic_data_for_oid.shape[0] == 5:  
            single_seismic_sample = loaded_seismic_data_for_oid
        else:
            raise ValueError(
                f"Unexpected seismic data shape for OID {oid}: {loaded_seismic_data_for_oid.shape}. Expected 3D (5, 1500, 70) or 4D.")

        seismic = single_seismic_sample.astype(np.float32)

        mean = seismic.mean()
        std = seismic.std()
        if std > 0:
            seismic = (seismic - mean) / std
        else:
            seismic = seismic - mean 

        seismic_tensor = torch.tensor(seismic, dtype=torch.float32)
        return seismic_tensor, oid

test_dataset = TestDataset(test_seismic_list, final_test_oids_for_dataset)
test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False) 
print(f"Test Dataloader created with {len(test_dataloader)} batches.")

model.eval()

submission_file_path = 'submission.csv' 
print(f"Writing submission to: {submission_file_path}")
rows_written_count = 0

with open(submission_file_path, mode='w', newline='') as f_csv: 
    writer = csv.writer(f_csv)
    header = ['id'] + ["x_{}".format(i) for i in range(1, 70, 2)] 
    writer.writerow(header)
    print("CSV Header written.")

    if len(test_dataloader) == 0:
        print("Test Dataloader is empty. No predictions to write.")
    else:
        with torch.no_grad(): 
            for i, data_batch in enumerate(test_dataloader):
                seismic_batch_test, oid_batch_test_tuple = data_batch  
                current_oid = oid_batch_test_tuple[0] 

                print(f"  Processing OID: {current_oid} (Batch {i + 1}/{len(test_dataloader)})")
                seismic_batch_test = seismic_batch_test.to(device) 

                if seismic_batch_test.ndim == 4:
 
                    input_to_model_test = seismic_batch_test.permute(0, 1, 3, 2)
                else:
                    input_to_model_test = seismic_batch_test  

                input_to_model_test = input_to_model_test[:, :, :70, :70]

                output_pred = model(input_to_model_test) 

 
                pred_map_normalized = output_pred[0].squeeze(0).cpu().numpy()

                final_pred_map = pred_map_normalized

                odd_columns_submission = final_pred_map[:, 1::2] 

                for y_idx, row_values in enumerate(odd_columns_submission):
                    row_label = "{}_y_{}".format(current_oid, y_idx)
                    writer.writerow([row_label] + list(row_values))
                    rows_written_count += 1
    print(f"Finished. Total data rows written to CSV: {rows_written_count}")
    if rows_written_count == 0 and len(test_dataloader) > 0:
        print("WARNING: Test Dataloader was not empty, but no rows were written. Check loops or OID processing.")


