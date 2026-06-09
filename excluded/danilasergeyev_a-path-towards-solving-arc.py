import math
import os
import random
from random import randint
import time
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import IPython.display as display
from google.colab import output


# Cell types
NO_CELL = 0.0
EMPTY_CELL = 1.0
COLORS = [0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.]

# Transformation classes
C_IDENTITY = 0
C_TRANSLATE = 1
C_MIRROR = 2
C_ROTATE = 3
C_DUPLICATE = 4
C_SCALE = 5
C_FRACTALIZE = 6
C_CROP = 7
C_CHANGE_COLOR = 8
C_CREATE_CELL = 9
C_FILL = 10

# Transformation ids
T_IDENTITY = 0 # field size range: [1, 30]
T_TRANSLATE_PATTERN = 1 # field size range: [2, 30]
T_MIRROR_PATTERN = 2 # field size range: [4, 30]
T_MIRROR_FIELD = 3 # field size range: [4, 30]
T_ROTATE_PATTERN = 4 # field size range: [, 30]
T_ROTATE_FIELD = 5 # field size range: [3, 30]
T_DUPLICATE_PATTERN = 6 # field size range: [2, 30]
T_DUPLICATE_FIELD = 7 # field size range: [2, 15]
T_SCALE_PATTERN = 8 # field size range: [2, 30]
T_SCALE_FIELD = 9 # field size range: [1, 15]
T_FRACTALIZE = 10 # field size range: [2, 5]
T_CROP = 11 # field size range: [2, 30]
T_CHANGE_COLOR = 12 # field size range: [1, 30]
T_CREATE_CELL = 13 # field size range: [3, 30]
T_FILL = 14 # field size range: [3, 30]

TRANSFORMATION_CLASSES = [C_IDENTITY, C_TRANSLATE, C_DUPLICATE, C_SCALE, C_CROP, C_CHANGE_COLOR]


def base_field():
    return np.zeros((32, 32))

def pick_field_size(transformation_id):
    MAX_SIZE = 30
    sizeX = None
    sizeY = None
    if transformation_id in set((T_IDENTITY, T_CHANGE_COLOR)):
        sizeX = randint(1, 30)
        sizeY = randint(1, 30)
    elif transformation_id in set((T_TRANSLATE_PATTERN, T_DUPLICATE_PATTERN, T_SCALE_PATTERN, T_CROP)):
        sizeX = randint(2, 30)
        sizeY = randint(2, 30)
    elif transformation_id in set((T_ROTATE_PATTERN, T_ROTATE_FIELD, T_CREATE_CELL, T_FILL)):
        sizeX = randint(3, 30)
        sizeY = randint(3, 30)
    elif transformation_id in set((T_MIRROR_PATTERN, T_MIRROR_FIELD)):
        sizeX = randint(4, 30)
        sizeY = randint(4, 30)
    elif transformation_id == T_SCALE_FIELD:
        sizeX = randint(1, 15)
        sizeY = randint(1, 15)
    elif transformation_id == T_DUPLICATE_FIELD:
        sizeX = randint(2, 15)
        sizeY = randint(2, 15)
    elif transformation_id == T_FRACTALIZE:
        sizeX = randint(2, 5)
        sizeY = randint(2, 5)
    else:
        sizeX = randint(5, 15)
        sizeY = randint(5, 15)
    return sizeX, sizeY

def pick_pixel(x_range, y_range):
    posX = randint(x_range[0], x_range[1])
    posY = randint(y_range[0], y_range[1])
    color = float(randint(2, 10))
    return posX, posY, color

def generate_puzzle(transformation_id = None, transformation_class = None):
    field_width, field_heights = pick_field_size(transformation_id)
    if transformation_id == T_SCALE_PATTERN:
        pixel_pos_X, pixel_pos_Y, pixel_color = pick_pixel((1, field_width - 1), (1, field_heights - 1))
    else:
        pixel_pos_X, pixel_pos_Y, pixel_color = pick_pixel((1, field_width), (1, field_heights))

    start_field = base_field()
    for i in range(1, field_width + 1):
        for j in range(1, field_heights + 1):
            start_field[j][i] = 1.0
    start_field[pixel_pos_Y][pixel_pos_X] = pixel_color

    end_field = start_field.copy()
    pixel = (pixel_pos_X, pixel_pos_Y, pixel_color)
    end_field, params = apply_transformation(start_field, (field_width, field_heights), transformation_class, pixel)

    return start_field, end_field, params


def one_hot(index, num_classes=32):
    vec = np.zeros(num_classes, dtype=np.float32)
    vec[index] = 1.0
    return vec


def generate_transformation_vector(transformation_id, num_classes=32, dims=1024, **params):
    vector = np.zeros(dims, dtype=np.float32)
    vector[0:num_classes] = one_hot(transformation_id, num_classes)
    if 'apply_to' in params:
        vector[32] = 1 if params['apply_to'] == 'field' else 0 # apply transformation to the whole field or a pattern
    if 'x_offset' in params:
        vector[33] = params['x_offset']
    if 'y_offset' in params:
        vector[34] = params['y_offset']
    if 'rotation_sin' in params:
        vector[35] = params['rotation_sin']
    if 'rotation_cos' in params:
        vector[36] = params['rotation_cos']
    if 'mirror_direction' in params:
        vector[37] = params['mirror_direction']
    if 'scale_factor' in params:
        vector[38] = params['scale_factor']
    if 'axis' in params:
        vector[39] = 0 if params['axis'] == 'x' else 1
    if 'source_color' in params:
        vector[40] = params['source_color']
    if 'target_color' in params:
        vector[41] = params['target_color']
    if 'neighbors' in params:
        neighbors = np.array(params['neighbors'], dtype=np.float32)
        vector[42:50] = neighbors[:8]
    return vector


def apply_identity_transformation(start_field, field_shape):
    return start_field.copy(), generate_transformation_vector(C_IDENTITY)


def apply_translate_transformation(start_field, field_shape, pixel):
    width, height = field_shape
    x, y, color = pixel

    directions = []
    if x > 1:  # can move left
        directions.append((-1, 0))
    if x < width:  # can move right
        directions.append((1, 0))
    if y > 1:  # can move up
        directions.append((0, -1))
    if y < height:  # can move down
        directions.append((0, 1))

    if not directions:
        # No valid move — fall back to identity
        return start_field.copy(), generate_transformation_vector(C_IDENTITY)

    dx, dy = random.choice(directions)
    new_x, new_y = x + dx, y + dy

    end_field = start_field.copy()
    end_field[y][x] = 1.0  # restore to empty cell
    end_field[new_y][new_x] = color

    transformation = generate_transformation_vector(C_TRANSLATE, x_offset=dx, y_offset=dy)
    return end_field, transformation


def apply_duplicate_transformation(start_field, field_shape, pixel):
    width, height = field_shape
    x, y, color = pixel

    directions = []
    if x > 1:  # can copy left
        directions.append((-1, 0))
    if x < width:  # can copy right
        directions.append((1, 0))
    if y > 1:  # can copy up
        directions.append((0, -1))
    if y < height:  # can copy down
        directions.append((0, 1))

    if not directions:
        # No valid move — fall back to identity
        return start_field.copy(), generate_transformation_vector(C_IDENTITY)

    dx, dy = random.choice(directions)
    new_x, new_y = x + dx, y + dy

    x_start = min(x, new_x)
    x_end = max(x, new_x)
    y_start = min(y, new_y)
    y_end = max(y, new_y)

    end_field = start_field.copy()
    for i in range(x_start, x_end + 1):
        for j in range(y_start, y_end + 1):
            end_field[j][i] = color

    transformation = generate_transformation_vector(C_DUPLICATE, x_offset=dx, y_offset=dy, apply_to='pattern')
    return end_field, transformation


def apply_scale_transformation(start_field, field_shape, pixel):
    width, height = field_shape
    x, y, color = pixel

    end_field = start_field.copy()
    for i in range(x, x + 2):
        for j in range(y, y + 2):
            end_field[j][i] = color

    transformation = generate_transformation_vector(C_SCALE, scale_factor=2, apply_to='pattern')
    return end_field, transformation


def apply_crop_transformation(start_field, field_shape, pixel):
    width, height = field_shape
    x, y, color = pixel

    end_field = base_field()
    end_field[1][1] = color


    transformation = generate_transformation_vector(C_CROP)
    return end_field, transformation


def apply_change_color_transformation(start_field, field_shape, pixel):
    width, height = field_shape
    x, y, color = pixel

    colors = [2, 3, 4, 5, 6, 7, 8, 9, 10]
    colors.remove(color)
    new_color = random.choice(colors)

    end_field = start_field.copy()
    end_field[y][x] = new_color

    transformation = generate_transformation_vector(C_CHANGE_COLOR, source_color=color, target_color=new_color)
    return end_field, transformation


def apply_duplicate_field_transformation(start_field, field_shape):
    field = start_field.copy()
    puzzle_width, puzzle_height = field_shape
    axis = random.choice(['x', 'y'])

    success = False
    x_offset = 0
    y_offset = 0
    if axis == 'x':
        x_offset = 1
        target_start_x = puzzle_width
        target_end_x = puzzle_width * 2
        if target_end_x <= 31:
            field[1:puzzle_height+1, target_start_x+1:target_end_x+1] = field[1:puzzle_height+1, 1:puzzle_width+1]
            success = True
    else:
        y_offset = 1
        target_start_y = puzzle_height
        target_end_y = puzzle_height * 2
        if target_end_y <= 31:
            field[target_start_y+1:target_end_y+1, 1:puzzle_width+1] = field[1:puzzle_height+1, 1:puzzle_width+1]
            success = True

    if success:
        return field, generate_transformation_vector(C_DUPLICATE, x_offset=x_offset, y_offset=y_offset, apply_to='field')
    else:
        return None, None


def apply_transformation(start_field, field_shape, transformation_id, pixel):
    if transformation_id == C_IDENTITY:
        return apply_identity_transformation(start_field, field_shape)
    elif transformation_id == C_TRANSLATE:
        return apply_translate_transformation(start_field, field_shape, pixel)
    elif transformation_id == C_DUPLICATE:
        return apply_duplicate_transformation(start_field, field_shape, pixel)
    elif transformation_id == C_SCALE:
        return apply_scale_transformation(start_field, field_shape, pixel)
    elif transformation_id == C_CROP:
        return apply_crop_transformation(start_field, field_shape, pixel)
    elif transformation_id == C_CHANGE_COLOR:
        return apply_change_color_transformation(start_field, field_shape, pixel)


# Custom colormap
newcmp = colors.ListedColormap(
    ['#ffffff', '#000000', '#ff81c0', '#8f1402', '#ff474c', '#f97306', '#faee66', '#75fd63', '#3d7afd', '#d7fffe', '#6f828a'],
    name='arc',
)


def draw_puzzle(start, end=None):
    fig, axs = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(20, 8)
    axs[0].imshow(start, cmap=newcmp, vmin=0, vmax=10)
    if end is None:
        end = [[0]]
    axs[1].imshow(end, cmap=newcmp, vmin=0, vmax=10)


transformation_id = 12
transformation_class = 8
start, end, latent_vector = generate_puzzle(transformation_id, transformation_class)
draw_puzzle(start, end)


# LayerNorm that works with (B, C, H, W)
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels, eps=1e-5):
        super().__init__()
        self.norm = nn.LayerNorm(num_channels, eps=eps)

    def forward(self, x):
        # (B, C, H, W) → (B, H, W, C) → normalize → (B, C, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        return x.permute(0, 3, 1, 2)


# Depthwise Separable Conv Block
class DepthwiseSeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=in_channels
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = LayerNorm2d(out_channels)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        return x


# Residual version of Depthwise Separable Conv Block
class DSConvResBlock(nn.Module):
    def __init__(self, in_channels, out_channels=None, kernel_size=3, padding=1, dropout=0.2):
        super().__init__()
        out_channels = out_channels or in_channels
        self.main = nn.Sequential(
            DepthwiseSeparableConv2d(in_channels, out_channels, kernel_size, padding, dropout),
            DepthwiseSeparableConv2d(out_channels, out_channels, kernel_size, padding, dropout),
        )
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        return self.main(x) + self.proj(x)


# Linear Block
class LinearBlock(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.GELU(),
            nn.Linear(in_features, out_features),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.block(x)


# Fully connected residual block
class ResBlock(nn.Module):
    def __init__(self, num_features, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(num_features),
            nn.GELU(),
            nn.Linear(num_features, num_features),
            nn.LayerNorm(num_features),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(num_features, num_features),
        )

    def forward(self, x):
        return x + self.block(x)


class Add2DCoords(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        y_pos = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_pos = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y_pos, x_pos], dim=1)


class TransformationInferer(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()
        self.dropout = dropout

        self.add_coords = Add2DCoords()

        self.shared_encoder = nn.Sequential(
            DSConvResBlock(3, 64, dropout=self.dropout),
            DSConvResBlock(64, dropout=self.dropout),
            nn.AvgPool2d(kernel_size=2),
            DSConvResBlock(64, 96, dropout=self.dropout),
            DSConvResBlock(96, dropout=self.dropout),
            nn.AvgPool2d(kernel_size=2),
            DSConvResBlock(96, 128, dropout=self.dropout),
            DSConvResBlock(128, 128, dropout=self.dropout),
        )

        flattened_size = 128 * 8 * 8

        self.final_layers = nn.Sequential(
            LinearBlock(flattened_size * 2, 2048, self.dropout),
            LinearBlock(2048, 1024, self.dropout),
            nn.Dropout(self.dropout),
            nn.Linear(1024, 1024),
        )

    def forward(self, in_grid, out_grid):
        in_grid = self.add_coords(in_grid)
        in_feat = self.shared_encoder(in_grid)

        out_grid = self.add_coords(out_grid)
        out_feat = self.shared_encoder(out_grid)

        in_feat = in_feat.reshape(in_feat.size(0), -1)
        out_feat = out_feat.reshape(out_feat.size(0), -1)
        combined = torch.cat([in_feat, out_feat], dim=1)

        latent = self.final_layers(combined)
        return latent


class FiLM(nn.Module):
    def __init__(self, feature_dim, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, feature_dim)
        self.beta = nn.Linear(cond_dim, feature_dim)

    def forward(self, x, cond):
        # x: [B, C, H, W], cond: [B, cond_dim]
        gamma = torch.tanh(self.gamma(cond)).unsqueeze(-1).unsqueeze(-1)  # [B, C, 1, 1]
        beta = torch.tanh(self.beta(cond)).unsqueeze(-1).unsqueeze(-1)
        return x * gamma + beta


class CnnResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, groups=1, dropout=0.2):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, groups=groups),
        )
        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.block(x) + self.skip(x)


class ResBlock(nn.Module):
    def __init__(self, dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Linear(dim, dim),
            nn.Dropout(dropout),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.block(x) + x


class LinearBlock(nn.Module):
    def __init__(self, in_dim, out_dim, dropout=0.2):
        super().__init__()
        self.block = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.block(x)


class Add2DCoords(nn.Module):
    def forward(self, x):
        B, C, H, W = x.shape
        y_pos = torch.linspace(-1, 1, H, device=x.device).view(1, 1, H, 1).expand(B, 1, H, W)
        x_pos = torch.linspace(-1, 1, W, device=x.device).view(1, 1, 1, W).expand(B, 1, H, W)
        return torch.cat([x, y_pos, x_pos], dim=1)


class TransformationApplier(nn.Module):
    def __init__(self, image_size=32, transformation_dim=1024, dropout=0.2):
        super().__init__()
        self.num_filters = 128
        self.grid_size = image_size // 2  # after AvgPool(2)

        self.add_coords = Add2DCoords()

        self.encoder = nn.Sequential(
            CnnResBlock(3, self.num_filters, dropout=dropout),
            CnnResBlock(self.num_filters, self.num_filters, groups=self.num_filters, dropout=dropout),
            CnnResBlock(self.num_filters, self.num_filters, groups=self.num_filters, dropout=dropout),
            CnnResBlock(self.num_filters, self.num_filters, groups=self.num_filters, dropout=dropout),
            nn.AvgPool2d(kernel_size=2),
        )

        self.transform_net = nn.Sequential(
            nn.Linear(transformation_dim, 512),
            ResBlock(512, dropout),
            nn.Linear(512, transformation_dim),
        )

        self.norm_pre = nn.LayerNorm(self.num_filters)
        self.attn_pre = nn.MultiheadAttention(embed_dim=self.num_filters, num_heads=4, batch_first=True, dropout=dropout)

        self.film = FiLM(feature_dim=self.num_filters, cond_dim=transformation_dim)

        self.norm_post = nn.LayerNorm(self.num_filters)
        self.attn_post = nn.MultiheadAttention(embed_dim=self.num_filters, num_heads=4, batch_first=True, dropout=dropout)

        self.encoder_fc = nn.Sequential(
            LinearBlock(self.num_filters * self.grid_size * self.grid_size, 2048, dropout),
            ResBlock(2048, dropout),
            ResBlock(2048, dropout),
        )

        self.final = nn.Sequential(
            ResBlock(2048, dropout),
            ResBlock(2048, dropout),
            nn.Linear(2048, 32 * 32 * 11),
        )

    def forward(self, image, transformation):
        # Transform transformation vector
        cond = self.transform_net(transformation)

        # Encode image with FiLM modulation
        x = self.add_coords(image)
        x = self.encoder(x) # [B, C, H, W]
        
        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1) # [B, S, C]
        x = self.norm_pre(x)
        x_attn, _ = self.attn_pre(x, x, x)
        x = x + x_attn
        x = x.view(B, self.num_filters, H, W) # [B, C, H, W]
        
        x = self.film(x, cond)

        B, C, H, W = x.shape
        x = x.view(B, C, H * W).permute(0, 2, 1)  # [B, S, C]
        x = self.norm_post(x)
        x_attn, _ = self.attn_post(x, x, x)
        x = x + x_attn
        
        x = x.view(x.size(0), -1)
        x = self.encoder_fc(x)

        # Predict output grid
        out = self.final(x)
        out = out.view(-1, 11, 32, 32)
        return out


train_puzzles = [generate_puzzle(0, 0) for _ in range(750)] + \
    [generate_puzzle(1, 1) for _ in range(4000)] + \
    [generate_puzzle(6, 4) for _ in range(4000)] + \
    [generate_puzzle(8, 5) for _ in range(1000)] + \
    [generate_puzzle(11, 7) for _ in range(1000)] + \
    [generate_puzzle(12, 8) for _ in range(3000)]

val_puzzles = [generate_puzzle(0, 0) for _ in range(150)] + \
    [generate_puzzle(1, 1) for _ in range(800)] + \
    [generate_puzzle(6, 4) for _ in range(800)] + \
    [generate_puzzle(8, 5) for _ in range(200)] + \
    [generate_puzzle(11, 7) for _ in range(200)] + \
    [generate_puzzle(12, 8) for _ in range(600)]

class PuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, puzzles):
        self.puzzles = puzzles

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, index):
        start, end, vector = self.puzzles[index]
        if random.random() < 0.1:
            transformation_class = np.argmax(vector[:32])
            if transformation_class == 0:
                transformation_id = 0
            elif transformation_class == 1:
                transformation_id = 1
            elif transformation_class == 4:
                transformation_id = 6
            elif transformation_class == 5:
                transformation_id = 8
            elif transformation_class == 7:
                transformation_id = 11
            elif transformation_class == 8:
                transformation_id = 12
            start, end, vector = generate_puzzle(transformation_id, transformation_class)

        start = torch.from_numpy(start).float().unsqueeze(0)
        end = torch.from_numpy(end).float().unsqueeze(0)
        v = torch.from_numpy(vector).float()
        return start, end, v

train_dataset = PuzzleDataset(train_puzzles)
validation_dataset = PuzzleDataset(val_puzzles)


class TransformationLoss(nn.Module):
    def __init__(self, one_hot_weight=1.0, param_weight=1.0, zero_weight=0.01):
        super().__init__()
        self.one_hot_weight = one_hot_weight
        self.param_weight = param_weight
        self.zero_weight = zero_weight


    def forward(self, output, target):
        # Element-wise MSE
        loss_one_hot = F.mse_loss(output[:, :32], target[:, :32])
        loss_param = F.mse_loss(output[:, 32:50], target[:, 32:50])
        loss_rest = F.mse_loss(output[:, 50:], target[:, 50:])

        return self.one_hot_weight * loss_one_hot + self.param_weight * loss_param + self.zero_weight * loss_rest


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Trainer:
    def from_checkpoint(checkpoint, model, loss_fn, optimizer, scheduler=None, device=torch.device('cpu')):
        batch_size = checkpoint['batch_size']
        model = model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        trainer = Trainer(model, loss_fn, optimizer, scheduler=scheduler, batch_size=batch_size, device=device)
        trainer.training_losses = checkpoint['training_losses']
        trainer.validation_losses = checkpoint['validation_losses']
        return trainer

    def __init__(
        self, model, loss_fn, optimizer, scheduler=None, batch_size=64, max_norm=None, device=torch.device('cpu')
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.training_losses = []
        self.validation_losses = []
        self.scaler = None

    def train(
        self, train_dataset, validation_dataset, epochs=50, starting_epoch=0,
        save_model=True, save_period=5, save_after_first=0, early_stop=None,
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')

        last_saved_at = None
        last_epoch = starting_epoch + epochs
        self.scaler = torch.cuda.amp.GradScaler()
        for epoch in range(starting_epoch, last_epoch):
            start_time = time.time()
            self.model.train()
            batch_losses = []
            counter = 0
            for x_batch, y_batch, tf in train_loader:
                counter += 1
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                tf = tf.to(self.device)
                batch_loss = self._train_step(x_batch, y_batch, tf)
                print(f'\rbatch {counter} loss: {batch_loss}', end='')
                batch_losses.append(batch_loss)
            training_loss = np.mean(batch_losses)
            self.training_losses.append(training_loss)

            self.model.eval()
            with torch.no_grad():
                batch_val_losses = []
                counter = 0
                for x_val, y_val, t_val in validation_loader:
                    counter += 1
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    t_val = t_val.to(self.device)
                    that = self.model(x_val, y_val)
                    val_loss = self.loss_fn(that, t_val).item()
                    print(f'\rbatch {counter} validation loss: {val_loss}', end='')
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.validation_losses.append(validation_loss)

            exec_time = timeSince(start_time)
            ax.set_title(f'{epoch + 1}/{last_epoch} - {exec_time}')
            ax.plot(range(0, len(self.training_losses)), self.training_losses, color='blue')
            ax.plot(range(0, len(self.validation_losses)), self.validation_losses, color='red')
            display.clear_output(wait=True)
            display.display(plt.gcf())
            print(f'Training loss: {training_loss:.6f}\t Validation loss: {validation_loss:.6f}')
            if self._should_save(save_model, epoch, starting_epoch, last_epoch, save_after_first, save_period):
                # output.clear()
                if self.validation_losses[-1] < self.validation_losses[last_saved_at or -save_period]:
                    self._save_model(epoch)
                    last_saved_at = epoch
            if early_stop is not None and len(self.training_losses) > early_stop:
                test_list = self.validation_losses[-early_stop:]
                if all(i < j for i, j in zip(test_list, test_list[1:])):
                    return

    def _train_step(self, x, y, t):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            that = self.model(x, y)
            loss = self.loss_fn(that, t)
        self.scaler.scale(loss).backward()
        if self.max_norm is not None and self.max_norm > 0.0:
            total_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
            if random.random() < 0.05:
                print(f"Grad norm: {total_norm:.4f}")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def _should_save(self, save_model, epoch, starting_epoch, last_epoch, save_after_first, save_period):
        if save_model and epoch == last_epoch-1:
            return True
        else:
            return save_model and epoch >= (starting_epoch+save_after_first) and (epoch % save_period == (save_period-1))

    def _save_model(self, epoch):
        model_path = f'/kaggle/working/{self.model.__class__.__name__}.pt'
        training_state = {
            'epoch': epoch,
            'batch_size': self.batch_size,
            'state_dict': self.model.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(training_state, model_path)


num_epoches = 500
batch_size = 128

model = TransformationInferer()
loss_fn = TransformationLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoches)

trainer = Trainer(model, loss_fn, optimizer, scheduler, batch_size=batch_size, device=torch.device('cuda'))
trainer.train(
    train_dataset,
    validation_dataset,
    epochs=num_epoches,
    save_model=True,
    save_period=5,
    save_after_first=49,
    early_stop=7,
)


train_puzzles = [generate_puzzle(0, 0) for _ in range(1000)] + \
    [generate_puzzle(1, 1) for _ in range(4000)] + \
    [generate_puzzle(6, 4) for _ in range(4000)] + \
    [generate_puzzle(8, 5) for _ in range(4500)] + \
    [generate_puzzle(11, 7) for _ in range(3500)] + \
    [generate_puzzle(12, 8) for _ in range(4000)]

val_puzzles = [generate_puzzle(0, 0) for _ in range(200)] + \
    [generate_puzzle(1, 1) for _ in range(800)] + \
    [generate_puzzle(6, 4) for _ in range(800)] + \
    [generate_puzzle(8, 5) for _ in range(900)] + \
    [generate_puzzle(11, 7) for _ in range(700)] + \
    [generate_puzzle(12, 8) for _ in range(800)]

class PuzzleDataset(torch.utils.data.Dataset):
    def __init__(self, puzzles):
        self.puzzles = puzzles

    def __len__(self):
        return len(self.puzzles)

    def __getitem__(self, index):
        start, end, vector = self.puzzles[index]
        if random.random() < 0.2:
            transformation_class = np.argmax(vector[:32])
            if transformation_class == 0:
                transformation_id = 0
            elif transformation_class == 1:
                transformation_id = 1
            elif transformation_class == 4:
                transformation_id = 6
            elif transformation_class == 5:
                transformation_id = 8
            elif transformation_class == 7:
                transformation_id = 11
            elif transformation_class == 8:
                transformation_id = 12
            start, end, vector = generate_puzzle(transformation_id, transformation_class)

        start = torch.from_numpy(start).float().unsqueeze(0)
        end = torch.tensor(end, dtype=torch.long)
        v = torch.from_numpy(vector).float()
        return start, end, v

train_dataset = PuzzleDataset(train_puzzles)
validation_dataset = PuzzleDataset(val_puzzles)


class ApplierLoss(nn.Module):
    def __init__(self, field_weight=1.0, padding_weight=0.1):
        super().__init__()
        self.field_weight = field_weight
        self.padding_weight = padding_weight
        self.field_class_weights = torch.tensor([1.0, 0.25, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    def forward(self, output, target):
        self.field_class_weights = self.field_class_weights.to(output.device)
        loss_field = F.cross_entropy(output, target, ignore_index=0, weight=self.field_class_weights, label_smoothing=0.02)
        pad_target = target.clone()
        pad_target[pad_target != 0] = -1
        loss_pad = F.cross_entropy(output, pad_target, ignore_index=-1)

        return self.field_weight * loss_field + self.padding_weight * loss_pad


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class Trainer:
    def from_checkpoint(checkpoint, model, loss_fn, optimizer, scheduler=None, device=torch.device('cpu')):
        batch_size = checkpoint['batch_size']
        model = model.to(device)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])
        trainer = Trainer(model, loss_fn, optimizer, scheduler=scheduler, batch_size=batch_size, device=device)
        trainer.training_losses = checkpoint['training_losses']
        trainer.validation_losses = checkpoint['validation_losses']
        return trainer

    def __init__(
        self, model, loss_fn, optimizer, scheduler=None, batch_size=64, max_norm=None, device=torch.device('cpu')
    ):
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.batch_size = batch_size
        self.max_norm = max_norm
        self.training_losses = []
        self.validation_losses = []
        self.scaler = None

    def train(
        self, train_dataset, validation_dataset, epochs=50, starting_epoch=0,
        save_model=True, save_period=5, save_after_first=0, early_stop=None,
    ):
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=True,
            drop_last=False,
            pin_memory=True,
        )
        validation_loader = DataLoader(
            validation_dataset,
            batch_size=self.batch_size,
            num_workers=os.cpu_count(),
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        )

        fig, ax = plt.subplots()
        fig.set_size_inches(12, 6)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Value')

        last_saved_at = None
        last_epoch = starting_epoch + epochs
        self.scaler = torch.cuda.amp.GradScaler()
        for epoch in range(starting_epoch, last_epoch):
            start_time = time.time()
            self.model.train()
            batch_losses = []
            counter = 0
            for x_batch, y_batch, tf in train_loader:
                counter += 1
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                tf = tf.to(self.device)
                batch_loss = self._train_step(x_batch, y_batch, tf)
                print(f'\rbatch {counter} loss: {batch_loss}', end='')
                batch_losses.append(batch_loss)
            training_loss = np.mean(batch_losses)
            self.training_losses.append(training_loss)

            self.model.eval()
            with torch.no_grad():
                batch_val_losses = []
                counter = 0
                for x_val, y_val, t_val in validation_loader:
                    counter += 1
                    x_val, y_val = x_val.to(self.device), y_val.to(self.device)
                    t_val = t_val.to(self.device)
                    yhat = self.model(x_val, t_val)
                    val_loss = self.loss_fn(yhat, y_val).item()
                    print(f'\rbatch {counter} validation loss: {val_loss}', end='')
                    batch_val_losses.append(val_loss)
                validation_loss = np.mean(batch_val_losses)
                self.validation_losses.append(validation_loss)

            exec_time = timeSince(start_time)
            ax.set_title(f'{epoch + 1}/{last_epoch} - {exec_time}')
            ax.plot(range(0, len(self.training_losses)), self.training_losses, color='blue')
            ax.plot(range(0, len(self.validation_losses)), self.validation_losses, color='red')
            display.clear_output(wait=True)
            display.display(plt.gcf())
            print(f'Training loss: {training_loss:.6f}\t Validation loss: {validation_loss:.6f}')
            if self._should_save(save_model, epoch, starting_epoch, last_epoch, save_after_first, save_period):
                if self.validation_losses[-1] < self.validation_losses[last_saved_at or -save_period]:
                    self._save_model(epoch)
                    last_saved_at = epoch
            if early_stop is not None and len(self.training_losses) > early_stop:
                test_list = self.validation_losses[-early_stop:]
                if all(i < j for i, j in zip(test_list, test_list[1:])):
                    return

    def _train_step(self, x, y, t):
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            yhat = self.model(x, t)
            loss = self.loss_fn(yhat, y)
        self.scaler.scale(loss).backward()
        if self.max_norm is not None and self.max_norm > 0.0:
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10.0)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        if self.scheduler is not None:
            self.scheduler.step()
        return loss.item()

    def _should_save(self, save_model, epoch, starting_epoch, last_epoch, save_after_first, save_period):
        if save_model and epoch == last_epoch-1:
            return True
        else:
            return save_model and epoch >= (starting_epoch+save_after_first) and (epoch % save_period == (save_period-1))

    def _save_model(self, epoch):
        model_path = f'/kaggle/working/{self.model.__class__.__name__}.pt'
        training_state = {
            'epoch': epoch,
            'batch_size': self.batch_size,
            'state_dict': self.model.state_dict(),
            'training_losses': self.training_losses,
            'validation_losses': self.validation_losses,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict() if self.scheduler is not None else None,
        }
        torch.save(training_state, model_path)


num_epoches = 300
batch_size = 256

model = TransformationApplier()
loss_fn = ApplierLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epoches)

trainer = Trainer(model, loss_fn, optimizer, scheduler, batch_size=batch_size, device=torch.device('cuda'))
trainer.train(
    train_dataset,
    validation_dataset,
    epochs=num_epoches,
    save_model=True,
    save_period=5,
    save_after_first=49,
)

