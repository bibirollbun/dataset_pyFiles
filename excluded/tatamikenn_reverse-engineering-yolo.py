!pip install ultralytics


from ultralytics import YOLO


model = YOLO("/kaggle/input/train-yolo/yolo_weights/motor_detector/weights/best.pt")


from ultralytics.nn.tasks import DetectionModel

DetectionModel(cfg="yolov8n.yaml", ch=3, nc=1)
pass


import polars as pl

label_df = pl.read_csv("/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train_labels.csv")
label_df = label_df.filter(pl.col("Number of motors") > 0)
label_df.head(1)


row = label_df.to_dicts()[0]
tomo_id = row["tomo_id"]
z0 = int(row["Motor axis 0"])
y0 = row["Motor axis 1"]
x0 = row["Motor axis 2"]
image_path = (
    f"/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/train/{tomo_id}/slice_{z0:04d}.jpg"
)


model.predictor = model._smart_load("predictor")(overrides=dict(imgsz=960), _callbacks=model.callbacks)
model.predictor.setup_model(model=model.model, verbose=True)


predictor = model.predictor
predictor.setup_source(image_path)


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

for predictor.batch in predictor.dataset:
    original_img = model.predictor.batch[1][0]
    transformed_img = (
        model.predictor.preprocess(original_img[None, ...])
        .squeeze(0)
        .permute(1, 2, 0)
        .detach()
        .cpu()
        .numpy()
    )
    print(original_img.shape)
    print(transformed_img.shape)
    
    _, axes = plt.subplots(1, 2, figsize=(10, 5))
    ax = axes[0]
    ax.imshow(original_img, cmap="gray")
    H0, W0, _ = original_img.shape
    ax.set_title(f"Original Image ({H0}x{W0})")
    s = 24
    ax.add_patch(
        Rectangle(
            (x0 - s / 2, y0 - s / 2),
            s,
            s,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            alpha=0.5,
        )
    )
    
    ax = axes[1]
    ax.imshow(transformed_img, cmap="gray")
    H, W, _ = transformed_img.shape
    ax.set_title(f"Transformed Image ({H}x{W})")
    x = (x0 + 0.5) * W / W0 - 0.5
    y = (y0 + 0.5) * H / H0 - 0.5
    s = 24
    ax.add_patch(
        Rectangle(
            (x - s / 2, y - s / 2),
            s,
            s,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            alpha=0.5,
        )
    )
    plt.show()


from typing import Callable


def replace_method(obj, method_name: str, func: Callable):
    """
    Replace a method of an object with a new function.

    Args:
        obj: The object whose method is to be replaced.
        method_name (str): The name of the method to replace.
        func (FunctionType): The new function to set as the method.
    """
    if not hasattr(obj, method_name):
        raise AttributeError(
            f"{obj.__class__.__name__} has no method '{method_name}' to replace."
        )
    bound = func.__get__(obj, obj.__class__)
    setattr(obj, method_name, bound)


def _predict_once(self, x, profile=False, visualize=False, embed=None):
    """
    MODIFICATION:
        - return entire feature map instead of pooled embedding when embed specified

    License: AGPL-3.0
    Source:
        - https://github.com/ultralytics/ultralytics
    """
    print("new predict once")
    y, dt, embeddings = [], [], []  # outputs
    for m in self.model:
        if m.f != -1:  # if not from previous layer
            x = (
                y[m.f]
                if isinstance(m.f, int)
                else [x if j == -1 else y[j] for j in m.f]
            )  # from earlier layers
        if profile:
            self._profile_one_layer(m, x, dt)
        x = m(x)  # run
        y.append(x if m.i in self.save else None)  # save output
        if embed and m.i in embed:
            embeddings.append(x.detach())
            if m.i == max(embed):
                return embeddings
    return x


replace_method(model.model, "_predict_once", _predict_once)


embeddings = model(image_path, conf=0.005, imgsz=960, verbose=False, embed=[15, 18, 21])


for i, emb in enumerate(embeddings):
    print(i, emb.shape)


import numpy as np
from skimage.transform import resize

for j, emb in enumerate(embeddings):
    for i, fmap in enumerate(emb[0, :5]):
        _, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 3))
        ax0.imshow(transformed_img, extent=(0, transformed_img.shape[1], transformed_img.shape[0], 0))
    
        resized_fmap = resize(
            fmap, (transformed_img.shape[0], transformed_img.shape[1]), anti_aliasing=True
        )
        ax1.imshow(transformed_img, extent=(0, transformed_img.shape[1], transformed_img.shape[0], 0))
        g = ax1.imshow(resized_fmap, cmap="jet", alpha=0.5, extent=(0, transformed_img.shape[1], transformed_img.shape[0], 0))
        ax1.set_title(f"Feature Map {j} (dim={i})")
        plt.colorbar(g, ax=ax1)
        plt.show()
    print("-" * 80)


def calc_num_anchors(embedding):
    _, _, h, w = embedding.shape
    return h * w

anchor_sizes = []
total_anchors = 0
for i, emb in enumerate(embeddings):
    num_anchors = calc_num_anchors(emb)
    print(f"Embedding {i}: {emb.shape}")
    print(f"#anchors: {num_anchors}")
    total_anchors += num_anchors
    anchor_sizes.append(num_anchors)
print(f"Total #anchors: {total_anchors}")
print(anchor_sizes)


from ultralytics.utils.tal import make_anchors


head = model.model.model[-1]
anchors, strides = (x.transpose(0, 1) for x in make_anchors(embeddings, head.stride, 0.5))
anchors.shape, strides.shape


for i, emb in enumerate(embeddings):
    anchor, stride = [
        x.transpose(0, 1) for x in make_anchors([emb], head.stride[i : i + 1], 0.5)
    ]
    print(f"Embedding {i}: {emb.shape}")
    print(f"Anchor {i}: {anchor.shape}")
    print(f"Stride {i}: {stride.shape}")

    _, ax  = plt.subplots(figsize=(8, 8))
    ax.scatter(anchor[0].detach().cpu().numpy(), anchor[1].detach().cpu().numpy(), marker=".", alpha=0.5, s=1)
    ax.set(
        title=f"Anchor Positions {i}",
        xlabel="X",
        ylabel="Y",
        aspect="equal",
    )
    plt.show()


import torch


def new_forward(self, x):
    """
    MODIFICATION:
        - stop before _inference

    License: AGPL-3.0
    Source:
        - https://github.com/ultralytics/ultralytics
    """
    print("new forward")
    if self.end2end:
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat(
                (self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i])), 1
            )
            for i in range(self.nl)
        ]
        return dict(one2many=[], one2one=one2one)

    x_detach = [xi.detach() for xi in x]
    one2one = [
        torch.cat(
            (self.cv2[i](x_detach[i]), self.cv3[i](x_detach[i])), 1
        )
        for i in range(self.nl)
    ]
    return dict(one2many=[], one2one=one2one)


replace_method(head, "forward", new_forward)


import torch
from ultralytics.utils.tal import make_anchors


def new_inference(self, x):
    """
    MODIFICATION:
        - return the result of dfl head

    License: AGPL-3.0
    Source:
        - https://github.com/ultralytics/ultralytics
    """
    # Inference path
    shape = x[0].shape  # BCHW
    x_cat = torch.cat([xi.view(shape[0], self.no, -1) for xi in x], 2)

    if self.dynamic or shape != self.shape:
        self.anchors, self.strides = (
            x.transpose(0, 1) for x in make_anchors(x, self.stride, 0.5)
        )
        self.shape = shape
    box, cls = x_cat.split((self.reg_max * 4, self.nc), 1)
    dfl = self.dfl(box)
    dbox = self.decode_bboxes(dfl, self.anchors.unsqueeze(0), xywh=False) * self.strides

    return torch.cat((dbox, cls.sigmoid()), 1), dfl


replace_method(head, "_inference", new_inference)


output = head(embeddings)
results, dfl = head._inference(output["one2one"])
print(results.shape)
print(dfl.shape)


lt, rb = dfl.chunk(2, dim=1)
lt = lt[0].detach().cpu().numpy()
rb = rb[0].detach().cpu().numpy()
wh = rb + lt

offset = 0
for i, size in enumerate(anchor_sizes):
    li = offset
    ui = li + size
    offset += size
    s = head.stride[i].item()

    _, ax = plt.subplots(figsize=(4, 3))
    for label, value in zip(
        ["-lbx", "-lby", "+ubx", "+uby"],
        [-lt[0], -lt[1], rb[0], rb[1]],
    ):
        v = value[li:ui] * s

        ax.hist(v, label=f"{label}", bins=100, alpha=0.5)
    ax.set(
        title=f"Distribution of DFL {i}",
        xlabel="Value",
        ylabel="Frequency",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.axvline(-12, linestyle="--", color="gray", alpha=0.5, label="-12")
    ax.axvline(12, linestyle="--", color="gray", alpha=0.5, label="+12")
    plt.show()


lt, rb = dfl.chunk(2, dim=1)
lt = lt[0].detach().cpu().numpy()
rb = rb[0].detach().cpu().numpy()
wh = rb + lt

offset = 0
for i, size in enumerate(anchor_sizes):
    li = offset
    ui = li + size
    offset += size
    s = head.stride[i].item()

    _, ax = plt.subplots(figsize=(4, 3))
    for label, value in zip(["width", "height"], [wh[0], wh[1]]):
        v = value[li:ui] * s

        ax.hist(v, label=f"{label}", bins=100, alpha=0.5)
    ax.set(
        title=f"Distribution of DFL {i}",
        xlabel="Value",
        ylabel="Frequency",
    )
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.show()


_, ax = plt.subplots(figsize=(4, 3))

offset = 0
for i, size in enumerate(anchor_sizes):
    li = offset
    ui = li + size
    offset += size

    conf = results[0, 4, li:ui].detach().cpu().numpy()
    sorted_conf = np.sort(conf)[::-1]

    ax.plot(
        sorted_conf[:20],
        label=f"feature map {i}",
        alpha=0.5,
    )
    ax.set(
        title="Confidence",
        xlabel="Top-K",
        ylabel="Confidence",
    )

ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
plt.show()


final_results = head.postprocess(results.permute(0, 2, 1), max_det=head.max_det, nc=head.nc)
print(final_results.shape)


xyxys = final_results[0, :, :4].detach().cpu().numpy()
confs = final_results[0, :, 4].detach().cpu().numpy()


max_conf = confs.max()
_, ax = plt.subplots(figsize=(8, 8))
ax.imshow(transformed_img, cmap="gray")
for (x1, y1, x2, y2), c in zip(xyxys, confs):
    ax.add_patch(
        Rectangle(
            (x1, y1),
            x2 - x1,
            y2 - y1,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
            alpha=c**0.5 / max_conf**0.5,
        )
    )
ax.set(
    title="Predicted Boxes",
    xlabel="X",
    ylabel="Y",
    aspect="equal",
)
plt.show()

