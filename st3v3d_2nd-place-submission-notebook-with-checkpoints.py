import os
os.listdir('/kaggle/input')


!pip install --no-index --find-links /kaggle/input/2nd-place-byu-challenge-packages/ /kaggle/input/2nd-place-byu-challenge-packages/*.whl


# write the entire inference script to a file
script = r"""
import os
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor

import torch
import numpy as np
import pandas as pd
from batchgenerators.utilities.file_and_folder_operations import subdirs, join
from torch.nn.functional import interpolate

from nnunetv2.dataset_conversion.kaggle_byu.official_data_to_nnunet import convert_coordinates, load_jpgs
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.utilities.helpers import empty_cache


@torch.inference_mode()
def resize_image(image: np.ndarray, edge_length: int, device: torch.device) -> torch.Tensor:
    zoom = edge_length / max(image.shape)
    new_shape = [round(s * zoom) for s in image.shape]
    t = torch.from_numpy(image).to(device).float()
    t = interpolate(t[None, None], new_shape, mode='area')[0, 0]
    t = torch.clip(torch.round(t), 0, 255).byte()
    empty_cache(device)
    return t


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input-dir',
                   default="/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test")
    p.add_argument('--output-file',
                   default="/kaggle/working/submission.csv",
                   help="output-file path")
    p.add_argument('--ckpt-dir')
    p.add_argument(
        '--fold',
        type=ast.literal_eval,
        help="tuple of fold identifiers, e.g. ('all',) or (0,1,2)"
    )
    p.add_argument('--threshold', type=float)
    p.add_argument('--min-dist', type=int, default=13)
    p.add_argument('--edge', type=int, default=512)
    p.add_argument('--gpu-id', type=int, default=0,
                   help="This process's GPU index (0 to num_gpus-1)")
    p.add_argument('--num-gpus', type=int, default=1,
                   help="Total number of GPUs being used")
    return p.parse_args()

def main():
    args = parse_args()
    DEVICE = torch.device(f"cuda:{args.gpu_id}")

    # ensure unique filename per GPU
    out_path = args.output_file

    all_tomos = sorted(subdirs(args.input_dir, join=False))
    tomos = all_tomos[args.gpu_id::args.num_gpus]

    pred = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=True,
        device=DEVICE,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True
    )
    pred.initialize_from_trained_model_folder(args.ckpt_dir, args.fold)
    pred.label_manager._all_labels = [0]

    results = []
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(load_jpgs, join(args.input_dir, tomos[0]))
        for i, tomo in enumerate(tomos):
            img_np = future.result()
            if i + 1 < len(tomos):
                future = executor.submit(load_jpgs, join(args.input_dir, tomos[i+1]))

            orig_shape = img_np.shape
            img = resize_image(img_np, args.edge, DEVICE).float()
            img = (img - img.mean()) / img.std()

            out = pred.predict_logits_from_preprocessed_data(img[None], out_device=DEVICE).float()[None]
            out = torch.sigmoid(out)[0, 0]
            coords = torch.argwhere((out == torch.max(out)) & (out > args.threshold))
            ps = [out[tuple(c)].item() for c in coords]

            if len(ps) == 0:
                results.append({'tomo_id': tomo,
                                'Motor axis 0': -1,
                                'Motor axis 1': -1,
                                'Motor axis 2': -1})
            else:
                # all motors equally likely, pick first
                best = coords[0].tolist()
                xyz = convert_coordinates([best], img.shape, orig_shape)[0]
                results.append({'tomo_id': tomo,
                                'Motor axis 0': xyz[0],
                                'Motor axis 1': xyz[1],
                                'Motor axis 2': xyz[2]})

            # free up memory
            del img, out, coords, ps, img_np
            empty_cache(DEVICE)

    # write out clean CSV
    df = pd.DataFrame(results, columns=['tomo_id','Motor axis 0','Motor axis 1','Motor axis 2'])
    df.to_csv(out_path, index=False)
    print(f"Saved predictions to {out_path}")

if __name__ == "__main__":
    main()
"""
script_path = '/kaggle/working/inference.py'
with open(script_path, 'w') as f:
    f.write(script)
print(f"Saved {script_path}")



import os, sys, subprocess

os.environ['nnUNet_compile'] = 'True'
os.environ['torch.backends.cudnn.benchmark'] = 'True'

base = [
    sys.executable,
    "/kaggle/working/inference.py",
    "--num-gpus", "2",
    "--input-dir", "/kaggle/input/byu-locating-bacterial-flagellar-motors-2025/test",
    "--ckpt-dir", "/kaggle/input/2nd-place-byu-challenge-checkpoints/MotorRegressionTrainer_BCEtopK20Loss_moreDA_3_5kep_EDT25__nnUNetResEncUNetMPlans__3d_fullres_bs16_ps128_256_256",
    "--fold", "('all', )", 
    "--threshold", "0.15",
]

p0 = subprocess.Popen(base + ["--gpu-id", "0", "--output-file", "/kaggle/working/submission_gpu0.csv"])
p1 = subprocess.Popen(base + ["--gpu-id", "1", "--output-file", "/kaggle/working/submission_gpu1.csv"])
p0.wait(); p1.wait()


# write header from first file, then append data (skipping headers) from both
!(head -n 1 submission_gpu0.csv && tail -n +2 -q submission_gpu0.csv submission_gpu1.csv) > submission.csv


# print the final prediction file
!head submission.csv

