from pathlib import Path

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, data_root, erase_last_row=False):
        self.data_root = Path(data_root)
        self.seis_files = sorted(self.data_root.glob("*.npy"))
        self.erase_last_row = erase_last_row

    def __len__(self):
        return len(self.seis_files)

    def __getitem__(self, idx):
        seis_file = self.seis_files[idx]
        seis = np.load(seis_file)
        seis = torch.from_numpy(seis)
        if self.erase_last_row:
            seis[:, -1, :] = 0
        oid = Path(seis_file).stem
        return {"seis": seis, "oid": oid}


test_ds = TestDataset("/kaggle/input/waveform-inversion/test", erase_last_row=True)
test_loader = torch.utils.data.DataLoader(test_ds, batch_size=32, shuffle=False)


def process_input(x):
    # x: (N, 5, 1000, 70)
    N = x.size(0)
    x = x.reshape(N, 5, 250, 4, 70)
    x = x.permute(0, 1, 2, 4, 3)
    x = x.reshape(N, 5, 250, 280)
    x = F.interpolate(x, size=(490, 490), mode="bilinear", align_corners=False)
    return x


class FWIModel(nn.Module):
    def __init__(self, pretrained=True, split_at=4):
        super().__init__()
        backbone = timm.create_model(
            "eva02_large_patch14_clip_224",
            pretrained=pretrained,
            dynamic_img_size=True,
            in_chans=1,
        )
        self.backbone = backbone
        self.head = nn.Linear(1024, 4)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.split_at = split_at

    def forward(self, x):
        x = process_input(x)
        N, C, H, W = x.size()
        # x = self.backbone.forward_features(x)
        x = self.backbone.patch_embed(x.reshape(N * C, 1, H, W))
        x, rot_pos_embed = self.backbone._pos_embed(x)
        for blk in self.backbone.blocks[: self.split_at]:
            x = blk(x, rope=rot_pos_embed)
        x = x.reshape(N, 5, *x.size()[1:])
        x = x.mean(1)
        for blk in self.backbone.blocks[self.split_at :]:
            x = blk(x, rope=rot_pos_embed)
        x = self.backbone.norm(x)

        x = x[:, 1:]
        x = self.head(x)
        x = x.permute(0, 2, 1).reshape(N, 4, 35, 35)
        x = self.pixel_shuffle(x)
        # x = x[:, :, 1:-1, 1:-1]

        x = F.sigmoid(x.float()) * 6000
        return x


model = FWIModel(pretrained=False, split_at=8)
model.load_state_dict(torch.load("/kaggle/input/fwi-checkpoints/release03_epoch1.pth", "cpu"))


model = model.cuda().eval()


data = next(iter(test_loader))
with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.float16):
    seis = data["seis"].cuda()
    pred_vel = model(seis)
pred_vel = pred_vel.cpu().float()


idx = 20
print(data["oid"][idx])
plt.imshow(pred_vel[idx, 0].numpy())




