import os
import yaml
from tqdm import tqdm
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl
from scipy.spatial.transform import Rotation as R

import torch
import torch.nn as nn
import torch.nn.functional as F

import kaggle_evaluation.cmi_inference_server


competition_dataset_directory = Path('/kaggle/input/cmi-detect-behavior-with-sensor-data')
external_dataset_directory = Path('/kaggle/input/cmi-dbwsd-dataset')

pd.set_option('display.max_rows', 1000)
pd.set_option('display.max_columns', 1000)


df_test = pd.read_csv(competition_dataset_directory / 'test.csv')
print(f'Test Set Shape {df_test.shape}')

is_submission = df_test.shape[0] != 107
print(f'Submission: {is_submission}')


def load_torch_model(model_directory, device):

    config_path = model_directory / 'config.yaml'
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    models = {}

    for model_path in tqdm(sorted(list(model_directory.glob('model*')))):
        model = eval(config['model']['model_class'])(**config['model']['model_args'])
        model_path = str(model_path)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model.to(device)
        model_file_name = model_path.split('/')[-1].split('.')[0]
        models[model_file_name] = model
        print(f'Loaded {model.__class__.__name__} Model from {model_path}')

    return config, models



def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):

    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor

class DropPath(nn.Module):

    def __init__(self, drop_prob=None, scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)



class EnhancedSEBlock(nn.Module):

    def __init__(self, channels, reduction=8):

        super(EnhancedSEBlock, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels * 2, channels // reduction, bias=False),
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        b, c, _ = x.size()
        avg_y = self.avg_pool(x).view(b, c)
        max_y = self.max_pool(x).view(b, c)
        y = torch.cat([avg_y, max_y], dim=1)
        y = self.excitation(y).view(b, c, 1)

        return x * y.expand_as(x)


class MultiScaleConv1d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_sizes):

        super(MultiScaleConv1d, self).__init__()

        self.conv_blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.conv_blocks.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', bias=False),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=True)
            ))

    def forward(self, x):
        conv_block_outputs = []
        for conv_block in self.conv_blocks:
            conv_block_outputs.append(conv_block(x))
        conv_block_outputs = torch.cat(conv_block_outputs, dim=1)
        return conv_block_outputs


class ResidualSEBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, pool_size=2, dropout_probability=0., drop_path_probability=0.):

        super(ResidualSEBlock, self).__init__()

        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding='same', bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = EnhancedSEBlock(out_channels, reduction=8)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm1d(out_channels)
            )

        self.pool = nn.MaxPool1d(pool_size)
        self.dropout = nn.Dropout(dropout_probability)
        self.drop_path = DropPath(drop_prob=drop_path_probability)

    def forward(self, x):

        shortcut = self.shortcut(x)

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        out = self.se(out)
        out = self.drop_path(out)
        out += shortcut
        out = F.relu(out)

        out = self.pool(out)
        out = self.dropout(out)

        return out


class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super(TemporalAttention, self).__init__()

        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        scores = torch.tanh(self.attention(x))
        weights = F.softmax(scores.squeeze(-1), dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)

        return context


class TOFTHMEncoder(nn.Module):

    def __init__(self, in_channels=10):

        super(TOFTHMEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x):

        batch_size, sequence_length, channel, height, width = x.shape
        x = x.view(batch_size * sequence_length, channel, height, width)
        x = self.encoder(x).view(batch_size, sequence_length, -1)
        x = F.avg_pool1d(x.transpose(1, 2), kernel_size=4, stride=4).transpose(1, 2)

        return x


class IMUHybridConvRNN(nn.Module):

    def __init__(self, imu_input_channels, rnn_dimensions, n_classes):

        super(IMUHybridConvRNN, self).__init__()

        self.imu_branches = nn.ModuleList(
            [
                nn.Sequential(
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7, 9]),
                    ResidualSEBlock(48, 64, 3, dropout_probability=0.1, drop_path_probability=0.1),
                    ResidualSEBlock(64, 48, 3, dropout_probability=0.1, drop_path_probability=0.1),
                )
                for _ in range(imu_input_channels)
            ]
        )
        self.gru = nn.GRU(
            input_size=48 * imu_input_channels,
            hidden_size=rnn_dimensions,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.,
        )

        self.gesture_head = nn.Sequential(
            nn.Linear(rnn_dimensions * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )
        self.phase_head = nn.Conv1d(in_channels=rnn_dimensions * 2, out_channels=1, kernel_size=(1,), padding='same')

    def forward(self, imu, metadata):

        batch_size, sequence_length, = imu.shape[:2]

        imu_branch_outputs = []
        for i in range(imu.shape[2]):
            channel_input = imu[:, :, i].unsqueeze(1)
            processed = self.imu_branches[i](channel_input)
            imu_branch_outputs.append(processed.transpose(1, 2))

        imu_branch_outputs = torch.cat(imu_branch_outputs, dim=2)

        gru_output, _ = self.gru(imu_branch_outputs)
        gru_output_sequence_length = gru_output.size(1)
        pooled_features = gru_output.mean(dim=1)

        phase_output = F.interpolate(
            gru_output.permute(0, 2, 1),
            scale_factor=sequence_length / gru_output_sequence_length,
            mode='nearest'
        )
        phase_output = self.phase_head(phase_output).squeeze(dim=1)
        gesture_output = self.gesture_head(pooled_features)

        return gesture_output, phase_output


class IMUTHMTOFHybridConvRNN(nn.Module):

    def __init__(self, imu_input_channels, rnn_dimensions, n_classes):

        super(IMUTHMTOFHybridConvRNN, self).__init__()

        self.imu_branches = nn.ModuleList(
            [
                nn.Sequential(
                    MultiScaleConv1d(1, 12, kernel_sizes=[3, 5, 7, 9]),
                    ResidualSEBlock(48, 64, 3, dropout_probability=0.1, drop_path_probability=0.1),
                    ResidualSEBlock(64, 48, 3, dropout_probability=0.1, drop_path_probability=0.1),
                )
                for _ in range(imu_input_channels)
            ]
        )
        self.tof_thm_encoder = TOFTHMEncoder(in_channels=10)
        self.gru = nn.GRU(
            input_size=48 * imu_input_channels + 128,
            hidden_size=rnn_dimensions,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
            dropout=0.,
        )

        self.gesture_head = nn.Sequential(
            nn.Linear(rnn_dimensions * 2, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, n_classes),
        )
        self.phase_head = nn.Conv1d(in_channels=rnn_dimensions * 2, out_channels=1, kernel_size=(1,), padding='same')

    def forward(self, imu, thm, tof, metadata):

        batch_size, sequence_length, = imu.shape[:2]

        imu_branch_outputs = []
        for i in range(imu.shape[2]):
            channel_input = imu[:, :, i].unsqueeze(1)
            processed = self.imu_branches[i](channel_input)
            imu_branch_outputs.append(processed.transpose(1, 2))

        imu_branch_outputs = torch.cat(imu_branch_outputs, dim=2)

        thm = thm.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 8, 8)
        tof = tof.view(batch_size, sequence_length, 5, 8, 8)
        tof_thm = torch.cat([tof, thm], dim=2)
        tof_thm_features = self.tof_thm_encoder(tof_thm)
        gru_input = torch.cat([imu_branch_outputs, tof_thm_features], dim=-1)
        gru_output, _ = self.gru(gru_input)
        gru_output_sequence_length = gru_output.size(1)

        phase_output = F.interpolate(
            gru_output.permute(0, 2, 1),
            scale_factor=sequence_length / gru_output_sequence_length,
            mode='nearest'
        )
        phase_output = self.phase_head(phase_output).squeeze(dim=1)

        pooled_output = gru_output.mean(dim=1)
        gesture_output = self.gesture_head(pooled_output)

        return gesture_output, phase_output



device = torch.device('cuda')

imu_model_config, imu_models = load_torch_model(external_dataset_directory / 'imu_model', device=device)
imu_thm_tof_model_config, imu_thm_tof_models = load_torch_model(external_dataset_directory / 'imu_thm_tof_model', device=device)


def create_imu_features(df):

    df['acc_mag'] = np.sqrt(df['acc_x'] ** 2 + df['acc_y'] ** 2 + df['acc_z'] ** 2)
    df['rot_mag'] = np.sqrt(df['rot_w'] ** 2 + df['rot_x'] ** 2 + df['rot_y'] ** 2 + df['rot_z'] ** 2)
    df['rot_angle'] = 2 * np.arccos(df['rot_w'].clip(-1, 1))

    acceleration = df[['acc_x', 'acc_y', 'acc_z']].values
    quaternion = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].values
    quaternion_shift = df[['rot_x', 'rot_y', 'rot_z', 'rot_w']].shift(1).bfill().values

    linear_acceleration = np.zeros((df.shape[0], 3))
    angular_velocity = np.zeros((df.shape[0], 3))
    
    mask = np.all(np.isnan(quaternion), axis=1) | np.all(np.isclose(quaternion, 0), axis=1)
    mask_shift = np.all(np.isnan(quaternion_shift), axis=1) | np.all(np.isclose(quaternion_shift, 0), axis=1)
    mask = mask & mask_shift

    if np.sum(~mask) > 0:
        
        rotation = R.from_quat(quaternion[~mask])
        rotation_shift = R.from_quat(quaternion_shift[~mask])
        
        delta_rotation = (rotation.inv() * rotation_shift).as_rotvec()
        angular_velocity[~mask] = delta_rotation
        df[['angular_vel_x', 'angular_vel_y', 'angular_vel_z']] = angular_velocity
        df['angular_vel_mag'] = np.sqrt(df['angular_vel_x'] ** 2 + df['angular_vel_y'] ** 2 + df['angular_vel_z'] ** 2)
        
        linear_acceleration[mask] = acceleration[mask]
        gravity_world = np.array([0, 0, 9.81])
        gravity_sensor_frame = rotation.apply(gravity_world, inverse=True)
        linear_acceleration[~mask] = gravity_sensor_frame
        df[['linear_acc_x', 'linear_acc_y', 'linear_acc_z']] = linear_acceleration
        df['linear_acc_mag'] = np.sqrt(df['linear_acc_x'] ** 2 + df['linear_acc_y'] ** 2 + df['linear_acc_z'] ** 2)

    else:
        df[['angular_vel_x', 'angular_vel_y', 'angular_vel_z', 'angular_vel_mag']] = 0.
        df[['linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag']] = 0.

    df['acc_x_diff_1'] = df['acc_x'].diff(1).fillna(0)
    df['acc_y_diff_1'] = df['acc_y'].diff(1).fillna(0)
    df['acc_z_diff_1'] = df['acc_z'].diff(1).fillna(0)
    df['rot_x_diff_1'] = df['rot_x'].diff(1).fillna(0)
    df['rot_y_diff_1'] = df['rot_y'].diff(1).fillna(0)
    df['rot_z_diff_1'] = df['rot_z'].diff(1).fillna(0)
    df['rot_w_diff_1'] = df['rot_w'].diff(1).fillna(0)
    df['acc_mag_diff_1'] = df['acc_mag'].diff(1).fillna(0)
    df['rot_mag_diff_1'] = df['rot_mag'].diff(1).fillna(0)
    df['linear_acc_x_diff_1'] = df['linear_acc_x'].diff(1).fillna(0)
    df['linear_acc_y_diff_1'] = df['linear_acc_y'].diff(1).fillna(0)
    df['linear_acc_z_diff_1'] = df['linear_acc_z'].diff(1).fillna(0)
    df['linear_acc_mag_diff_1'] = df['linear_acc_mag'].diff(1).fillna(0)
    df['angular_vel_x_diff_1'] = df['angular_vel_x'].diff(1).fillna(0)
    df['angular_vel_y_diff_1'] = df['angular_vel_y'].diff(1).fillna(0)
    df['angular_vel_z_diff_1'] = df['angular_vel_z'].diff(1).fillna(0)
    df['angular_vel_mag_diff_1'] = df['angular_vel_mag'].diff(1).fillna(0)

    return df



def crop_and_pad(
    imu, thm, tof,
    sequence_length,
    min_gesture_crop_ratio=0., max_gesture_crop_ratio=0.,
    min_transition_crop_ratio=0., max_transition_crop_ratio=0.,
    random_padding=False
):
    
    current_sequence_length = imu.shape[0]

    indices = np.arange(current_sequence_length)
    transition_indices = indices[:len(indices) // 2]
    transition_length = len(transition_indices)
    transition_ratio = 0.5
    gesture_indices = indices[len(indices) // 2:]
    gesture_length = len(gesture_indices)
    gesture_ratio = 0.5    

    # Estimate minimum required crop to match desired sequence length
    min_required_crop = np.maximum(0, current_sequence_length - sequence_length)

    crop_start = 0
    crop_end = current_sequence_length
    gesture_crop_deficit = 0

    if min_required_crop >= transition_length:
        crop_start = imu.shape[0] - sequence_length
        crop_end = imu.shape[0]
    else:
        if gesture_ratio > 0:
    
            # Compute minimum and maximum gesture crop values based on configuration ratios
            min_config_gesture_crop = int(gesture_length * min_gesture_crop_ratio)
            max_config_gesture_crop = int(gesture_length * max_gesture_crop_ratio)
            # Estimate the minimum gesture crop required to meet the total crop demand proportionally
            min_required_gesture_crop = int(np.floor(min_required_crop * gesture_ratio))
    
            # Final gesture crop range is constrained between:
            #   - At least the greater of config minimum or required crop
            #   - At most the smaller of config maximum and actual gesture length
            min_gesture_crop = np.maximum(min_config_gesture_crop, min_required_gesture_crop)
            min_gesture_crop = np.minimum(min_gesture_crop, max_config_gesture_crop)
            max_gesture_crop = np.minimum(max_config_gesture_crop, len(gesture_indices) - 1)
    
            gesture_crop_offset = np.random.randint(min_gesture_crop, max_gesture_crop + 1)
            gesture_crop_deficit = np.maximum(0, (min_required_gesture_crop - gesture_crop_offset))
            if gesture_crop_offset > 0:
                crop_end = gesture_indices[len(gesture_indices) - gesture_crop_offset]
    
        if transition_ratio > 0:
    
            # Compute minimum and maximum transition crop values based on configuration ratios.
            min_config_transition_crop = int(transition_length * min_transition_crop_ratio)
            max_config_transition_crop = int(transition_length * max_transition_crop_ratio)
            # Estimate the minimum transition crop required, adjusted by any gesture crop deficit
            min_required_transition_crop = int(np.ceil(min_required_crop * transition_ratio)) + gesture_crop_deficit
    
            # Final transition crop range is constrained to:
            #   - At least the greater of config minimum or required crop plus gesture deficit
            #   - At most the maximum allowed by config, ensuring valid sampling range
            min_transition_crop = np.maximum(min_config_transition_crop, min_required_transition_crop)
            max_transition_crop = np.maximum(min_transition_crop, max_config_transition_crop)
    
            transition_crop_offset = np.random.randint(min_transition_crop, max_transition_crop + 1)
            if transition_crop_offset > 0:
                crop_start = transition_indices[transition_crop_offset]

    imu = imu[crop_start:crop_end, :]
    thm = thm[crop_start:crop_end, :]
    tof = tof[crop_start:crop_end, :]

    total_pad = sequence_length - imu.shape[0]
    
    if total_pad > 0:
        
        if random_padding:
            transition_pad = np.random.randint(0, total_pad + 1)
            gesture_pad = total_pad - transition_pad
        else:
            transition_pad = total_pad
            gesture_pad = 0

        imu = np.vstack((
            np.zeros((transition_pad, imu.shape[1])),
            imu,
            np.zeros((gesture_pad, imu.shape[1])),
        ))

        thm = np.vstack((
            np.zeros((transition_pad, thm.shape[1])),
            thm,
            np.zeros((gesture_pad, thm.shape[1])),
        ))

        tof = np.vstack((
            np.zeros((transition_pad, tof.shape[1])),
            tof,
            np.zeros((gesture_pad, tof.shape[1])),
        ))

    return imu, thm, tof



def nn_predict(imu, thm, tof, models, verbose=False):

    if thm is not None and tof is not None:
        inputs = {'imu': imu, 'thm': thm, 'tof': tof, 'metadata': None}
    else:
        inputs = {'imu': imu, 'metadata': None}
    
    predictions = torch.zeros(1, 18, device=device)

    for model_file_name, model in models.items():
        
        with torch.no_grad():
            gesture_outputs, phase_outputs = model(**inputs)

        predictions += gesture_outputs / len(models)

    predictions = torch.softmax(predictions, dim=-1).squeeze(dim=0).cpu().numpy()
    
    return predictions



def predict(sequence, demographics):

    df_sequence = sequence.to_pandas()

    raw_imu_columns = [
        'acc_x', 'acc_y', 'acc_z',
        'rot_x', 'rot_y', 'rot_z', 'rot_w'
    ]
    thm_columns = [column for column in df_sequence.columns if column.startswith('thm')]
    tof_columns = [column for column in df_sequence.columns if column.startswith('tof')]
    
    df_sequence = df_sequence.loc[:, raw_imu_columns + thm_columns + tof_columns]
    df_sequence = df_sequence.interpolate(method='linear', limit_area='inside').fillna(0.)
    
    df_sequence = create_imu_features(df_sequence)

    imu_columns = [
        'acc_x', 'acc_y', 'acc_z',
        'rot_x', 'rot_y', 'rot_z', 'rot_w',

        'acc_mag', 'rot_mag', 'rot_angle',

        'linear_acc_x', 'linear_acc_y', 'linear_acc_z', 'linear_acc_mag',

        'acc_x_diff_1', 'acc_y_diff_1', 'acc_z_diff_1',
        'rot_x_diff_1', 'rot_y_diff_1', 'rot_z_diff_1', 'rot_w_diff_1',

        'acc_mag_diff_1', 'rot_mag_diff_1',

        'linear_acc_x_diff_1', 'linear_acc_y_diff_1', 'linear_acc_z_diff_1', 'linear_acc_mag_diff_1',

        'angular_vel_x', 'angular_vel_y', 'angular_vel_z',
        'angular_vel_mag',

        'angular_vel_x_diff_1', 'angular_vel_y_diff_1', 'angular_vel_z_diff_1',
        'angular_vel_mag_diff_1'
    ]
    
    imu = df_sequence.loc[:, imu_columns].values
    thm = df_sequence.loc[:, thm_columns].values
    tof = df_sequence.loc[:, tof_columns].values
    
    imu, thm, tof = crop_and_pad(imu=imu, thm=thm, tof=tof, sequence_length=96)
    print(f'IMU {imu.shape} - THM {thm.shape} - TOF {tof.shape}')

    imu = torch.as_tensor(imu.copy(), dtype=torch.float)
    imu = torch.unsqueeze(imu, dim=0)
    imu = imu.to(device)

    thm = torch.as_tensor(thm.copy(), dtype=torch.float)
    thm = torch.unsqueeze(thm, dim=0)
    thm = thm.to(device)

    tof = torch.as_tensor(tof.copy(), dtype=torch.float)
    tof = torch.unsqueeze(tof, dim=0)
    tof = tof.to(device)
    print(f'IMU {imu.shape} - THM {thm.shape} - TOF {tof.shape}')

    if torch.sum(thm) == 0 and torch.sum(tof) == 0:
        print('predict with 1')
        model_predictions = nn_predict(imu=imu, thm=None, tof=None, models=imu_models)
    else:
        print('predict with 2')
        model_predictions = nn_predict(imu=imu, thm=thm, tof=tof, models=imu_thm_tof_models)
    
    model_predictions = int(np.argmax(model_predictions))
    
    print(model_predictions)

    gesture_mapping = {
        # Target gestures
        0: 'Above ear - pull hair',
        1: 'Cheek - pinch skin',
        2: 'Eyebrow - pull hair',
        3: 'Eyelash - pull hair',
        4: 'Forehead - pull hairline',
        5: 'Forehead - scratch',
        6: 'Neck - pinch skin',
        7: 'Neck - scratch',
        # Non-target gestures
        8: 'Drink from bottle/cup',
        9: 'Feel around in tray and pull out an object',
        10: 'Glasses on/off',
        11: 'Pinch knee/leg skin',
        12: 'Pull air toward your face',
        13: 'Scratch knee/leg skin',
        14: 'Text on phone',
        15: 'Wave hello',
        16: 'Write name in air',
        17: 'Write name on leg'
    }

    class_prediction = gesture_mapping[model_predictions]
    print(class_prediction)
    
    return class_prediction


inference_server = kaggle_evaluation.cmi_inference_server.CMIInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        data_paths=(
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test.csv',
            '/kaggle/input/cmi-detect-behavior-with-sensor-data/test_demographics.csv',
        )
    )





