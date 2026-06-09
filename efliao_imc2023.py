!pip install --no-index --find-links=/kaggle/input/imc2023-dependencies /kaggle/input/imc2023-dependencies/*.whl


!pip install --no-index --find-links=/kaggle/input/imc2023-dependencies /kaggle/input/imc2023-dependencies/*.whl


!pip install --no-index --find-links=/kaggle/input/imc2023-dependencies /kaggle/input/imc2023-dependencies/*.whl


!ls /kaggle/input/image-matching-challenge-2023/test


SUBMISSION_FILE_PATH = "/kaggle/working/submission.csv"


# Copyright (c), ETH Zurich and UNC Chapel Hill.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of ETH Zurich and UNC Chapel Hill nor the names of
#       its contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.


# This script is based on an original implementation by True Price.

import sqlite3
import sys

import numpy as np

IS_PYTHON3 = sys.version_info[0] >= 3

MAX_IMAGE_ID = 2**31 - 1

CREATE_CAMERAS_TABLE = """CREATE TABLE IF NOT EXISTS cameras (
    camera_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    model INTEGER NOT NULL,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    params BLOB,
    prior_focal_length INTEGER NOT NULL)"""

CREATE_DESCRIPTORS_TABLE = """CREATE TABLE IF NOT EXISTS descriptors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_IMAGES_TABLE = """CREATE TABLE IF NOT EXISTS images (
    image_id INTEGER PRIMARY KEY AUTOINCREMENT NOT NULL,
    name TEXT NOT NULL UNIQUE,
    camera_id INTEGER NOT NULL,
    CONSTRAINT image_id_check CHECK(image_id >= 0 and image_id < {}),
    FOREIGN KEY(camera_id) REFERENCES cameras(camera_id))
""".format(
    MAX_IMAGE_ID
)

CREATE_POSE_PRIORS_TABLE = """CREATE TABLE IF NOT EXISTS pose_priors (
    image_id INTEGER PRIMARY KEY NOT NULL,
    position BLOB,
    coordinate_system INTEGER NOT NULL,
    position_covariance BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)"""

CREATE_TWO_VIEW_GEOMETRIES_TABLE = """
CREATE TABLE IF NOT EXISTS two_view_geometries (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    config INTEGER NOT NULL,
    F BLOB,
    E BLOB,
    H BLOB,
    qvec BLOB,
    tvec BLOB)
"""

CREATE_KEYPOINTS_TABLE = """CREATE TABLE IF NOT EXISTS keypoints (
    image_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB,
    FOREIGN KEY(image_id) REFERENCES images(image_id) ON DELETE CASCADE)
"""

CREATE_MATCHES_TABLE = """CREATE TABLE IF NOT EXISTS matches (
    pair_id INTEGER PRIMARY KEY NOT NULL,
    rows INTEGER NOT NULL,
    cols INTEGER NOT NULL,
    data BLOB)"""

CREATE_NAME_INDEX = (
    "CREATE UNIQUE INDEX IF NOT EXISTS index_name ON images(name)"
)

CREATE_ALL = "; ".join(
    [
        CREATE_CAMERAS_TABLE,
        CREATE_IMAGES_TABLE,
        CREATE_POSE_PRIORS_TABLE,
        CREATE_KEYPOINTS_TABLE,
        CREATE_DESCRIPTORS_TABLE,
        CREATE_MATCHES_TABLE,
        CREATE_TWO_VIEW_GEOMETRIES_TABLE,
        CREATE_NAME_INDEX,
    ]
)


def image_ids_to_pair_id(image_id1, image_id2):
    if image_id1 > image_id2:
        image_id1, image_id2 = image_id2, image_id1
    return image_id1 * MAX_IMAGE_ID + image_id2


def pair_id_to_image_ids(pair_id):
    image_id2 = pair_id % MAX_IMAGE_ID
    image_id1 = (pair_id - image_id2) / MAX_IMAGE_ID
    return image_id1, image_id2


def array_to_blob(array):
    if IS_PYTHON3:
        return array.tostring()
    else:
        return np.getbuffer(array)


def blob_to_array(blob, dtype, shape=(-1,)):
    if IS_PYTHON3:
        return np.fromstring(blob, dtype=dtype).reshape(*shape)
    else:
        return np.frombuffer(blob, dtype=dtype).reshape(*shape)


class COLMAPDatabase(sqlite3.Connection):
    @staticmethod
    def connect(database_path):
        return sqlite3.connect(database_path, factory=COLMAPDatabase)

    def __init__(self, *args, **kwargs):
        super(COLMAPDatabase, self).__init__(*args, **kwargs)

        self.create_tables = lambda: self.executescript(CREATE_ALL)
        self.create_cameras_table = lambda: self.executescript(
            CREATE_CAMERAS_TABLE
        )
        self.create_descriptors_table = lambda: self.executescript(
            CREATE_DESCRIPTORS_TABLE
        )
        self.create_images_table = lambda: self.executescript(
            CREATE_IMAGES_TABLE
        )
        self.create_pose_priors_table = lambda: self.executescript(
            CREATE_POSE_PRIORS_TABLE
        )
        self.create_two_view_geometries_table = lambda: self.executescript(
            CREATE_TWO_VIEW_GEOMETRIES_TABLE
        )
        self.create_keypoints_table = lambda: self.executescript(
            CREATE_KEYPOINTS_TABLE
        )
        self.create_matches_table = lambda: self.executescript(
            CREATE_MATCHES_TABLE
        )
        self.create_name_index = lambda: self.executescript(CREATE_NAME_INDEX)

    def add_camera(
        self,
        model,
        width,
        height,
        params,
        prior_focal_length=False,
        camera_id=None,
    ):
        params = np.asarray(params, np.float64)
        cursor = self.execute(
            "INSERT INTO cameras VALUES (?, ?, ?, ?, ?, ?)",
            (
                camera_id,
                model,
                width,
                height,
                array_to_blob(params),
                prior_focal_length,
            ),
        )
        return cursor.lastrowid

    def add_image(
        self,
        name,
        camera_id,
        image_id=None,
    ):
        cursor = self.execute(
            "INSERT INTO images VALUES (?, ?, ?)", (image_id, name, camera_id)
        )
        return cursor.lastrowid

    def add_pose_prior(
        self,
        image_id,
        position,
        coordinate_system=-1,
        position_covariance=None,
    ):
        position = np.asarray(position, dtype=np.float64)
        if position_covariance is None:
            position_covariance = np.full((3, 3), np.nan, dtype=np.float64)
        self.execute(
            "INSERT INTO pose_priors VALUES (?, ?, ?, ?)",
            (
                image_id,
                array_to_blob(position),
                coordinate_system,
                array_to_blob(position_covariance),
            ),
        )

    def add_keypoints(self, image_id, keypoints):
        assert len(keypoints.shape) == 2
        assert keypoints.shape[1] in [2, 4, 6]

        keypoints = np.asarray(keypoints, np.float32)
        self.execute(
            "INSERT INTO keypoints VALUES (?, ?, ?, ?)",
            (image_id,) + keypoints.shape + (array_to_blob(keypoints),),
        )

    def add_descriptors(self, image_id, descriptors):
        descriptors = np.ascontiguousarray(descriptors, np.uint8)
        self.execute(
            "INSERT INTO descriptors VALUES (?, ?, ?, ?)",
            (image_id,) + descriptors.shape + (array_to_blob(descriptors),),
        )

    def add_matches(self, image_id1, image_id2, matches):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        self.execute(
            "INSERT INTO matches VALUES (?, ?, ?, ?)",
            (pair_id,) + matches.shape + (array_to_blob(matches),),
        )

    def add_two_view_geometry(
        self,
        image_id1,
        image_id2,
        matches,
        F=np.eye(3),
        E=np.eye(3),
        H=np.eye(3),
        qvec=np.array([1.0, 0.0, 0.0, 0.0]),
        tvec=np.zeros(3),
        config=2,
    ):
        assert len(matches.shape) == 2
        assert matches.shape[1] == 2

        if image_id1 > image_id2:
            matches = matches[:, ::-1]

        pair_id = image_ids_to_pair_id(image_id1, image_id2)
        matches = np.asarray(matches, np.uint32)
        F = np.asarray(F, dtype=np.float64)
        E = np.asarray(E, dtype=np.float64)
        H = np.asarray(H, dtype=np.float64)
        qvec = np.asarray(qvec, dtype=np.float64)
        tvec = np.asarray(tvec, dtype=np.float64)
        self.execute(
            "INSERT INTO two_view_geometries VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (pair_id,)
            + matches.shape
            + (
                array_to_blob(matches),
                config,
                array_to_blob(F),
                array_to_blob(E),
                array_to_blob(H),
                array_to_blob(qvec),
                array_to_blob(tvec),
            ),
        )



import pycolmap


import sys
sys.path.append("/kaggle/input/lightglue")

from lightglue import ALIKED, DISK, SIFT, DoGHardNet, LightGlue, SuperPoint
from lightglue.utils import load_image, rbd, read_image



import csv
import logging
import pathlib
import pprint
import sys
import time
from collections import Counter, defaultdict
import csv
import pathlib
import sys
from collections import Counter, defaultdict



import numpy as np
import pycolmap
import torch
from loguru import logger
from sklearn.cluster import DBSCAN
from tqdm import tqdm

import cv2
import numpy as np
import pycolmap
import torch
from loguru import logger
from PIL import ExifTags, Image
from tqdm import tqdm


!mkdir -p /root/.cache/torch/hub/checkpoints/

!cp /kaggle/input/imc2023-models/pytorch/default/1/aliked-n16rot.pth /root/.cache/torch/hub/checkpoints/aliked-n16rot.pth
!cp /kaggle/input/imc2023-models/pytorch/default/1/aliked_lightglue_v0-1_arxiv.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth
!cp /kaggle/input/imc2023-models/pytorch/default/1/depth-save.pth /root/.cache/torch/hub/checkpoints/depth-save.pth
!cp /kaggle/input/imc2023-models/pytorch/default/1/disk_lightglue_v0-1_arxiv.pth /root/.cache/torch/hub/checkpoints/disk_lightglue_v0-1_arxiv.pth
!cp /kaggle/input/imc2023-models/pytorch/default/1/superpoint_lightglue_v0-1_arxiv.pth /root/.cache/torch/hub/checkpoints/superpoint_lightglue_v0-1_arxiv.pth
!cp /kaggle/input/imc2023-models/pytorch/default/1/superpoint_v1.pth /root/.cache/torch/hub/checkpoints/superpoint_v1.pth
# !cp /kaggle/input/aliked-n16rot/pytorch/default/1/aliked-n16rot.pth /root/.cache/torch/hub/checkpoints/aliked-n16rot.pth
# !cp /kaggle/input/aliked_lightglue_v0-1_arxiv/pytorch/default/1/aliked_lightglue_v0-1_arxiv.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv.pth


extractors = {
    "aliked": ALIKED(
        model_name="aliked-n16rot",
        device="cuda",
        top_k=-1,
        scores_th=0.2,
        n_limit=5000,
        nms_radius=2,
    )
    .eval()
    .cuda(),
    "disk": DISK(device="cuda").eval().cuda(),
    "superpoint": SuperPoint(device="cuda").eval().cuda(),
}

matchers = {
    "aliked": LightGlue(features="aliked", filter_threshold=0.1).eval().cuda(),
    "disk": LightGlue(features="disk", filter_threshold=0.1).eval().cuda(),
    "superpoint": LightGlue(features="superpoint", filter_threshold=0.1)
    .eval()
    .cuda(),
}



CAMERA_MODEL_MAP = {
    "SIMPLE_RADIAL": 2,
    "OPENCV": 4,
}


DATA_DIR_PATH = pathlib.Path("/kaggle/input/image-matching-challenge-2023/test")


scene_set = set()
for path in DATA_DIR_PATH.iterdir():
    if path.is_dir():  # dataset
        for path_ in path.iterdir():  # scene
            if path_.is_dir():
                scene_set.add((path.stem, path_.stem))


with open(SUBMISSION_FILE_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "image_path",
            "dataset",
            "scene",
            "rotation_matrix",
            "translation_vector",
        ]
    )


for dataset_name, scene_name in scene_set:
    logger.debug(f"{dataset_name=}, {scene_name=}")
    image_dir = DATA_DIR_PATH / dataset_name / scene_name / "images"

    image_paths = []
    for ext in ("*.jpg", "*.JPG", "*.jpeg", "*.png", "*.PNG"):
        image_paths.extend(image_dir.rglob(ext))

    feature_dir = pathlib.Path("feature") / f"{dataset_name}_{scene_name}"
    feature_dir.mkdir(parents=True, exist_ok=True)

    database_path = feature_dir / "colmap.db"
    db = COLMAPDatabase.connect(database_path)
    db.create_tables()

    output_dir = feature_dir / "sfm"
    output_dir.mkdir(parents=True, exist_ok=True)

    image_id_map = {}
    for image_path in tqdm(image_paths, desc="Add cameras and images"):
        image = read_image(image_path)
        h, w, c = image.shape
        focal = 1.2 * max(w, h)
        camera_id = db.add_camera(
            CAMERA_MODEL_MAP["SIMPLE_RADIAL"],
            width=w,
            height=h,
            params=np.array([focal, w / 2, h / 2, 0.1]),
            prior_focal_length=False,
        )
        image_id = db.add_image(image_path.name, camera_id)
        image_id_map[str(image_path)] = image_id

    feature_map_all = {k: {} for k in extractors}
    keypoint_freq_map_all = {k: {} for k in extractors}

    for image_path in tqdm(image_paths, desc="Extracting features (no crop)"):
        image_tensor = load_image(image_path).cuda()
        for name, extractor in extractors.items():
            features = extractor.extract(image_tensor)
            feature_map_all[name][image_path] = features
            keypoint_freq_map_all[name][image_path] = Counter()

    pool = []
    for i, p0 in enumerate(image_paths):
        for j, p1 in enumerate(image_paths):
            if j <= i:
                continue
            for name in extractors:
                matches = rbd(
                    matchers[name](
                        {
                            "image0": feature_map_all[name][p0],
                            "image1": feature_map_all[name][p1],
                        }
                    )
                )
                matches_arr = matches["matches"].cpu().numpy()
                for idx0, idx1 in matches_arr:
                    keypoint_freq_map_all[name][p0][idx0] += 1
                    keypoint_freq_map_all[name][p1][idx1] += 1
                pool.append(
                    {
                        "cropped": False,
                        "p0": p0,
                        "p1": p1,
                        "bbox0": None,
                        "bbox1": None,
                        "kps0": rbd(feature_map_all[name][p0])["keypoints"]
                        .cpu()
                        .numpy(),
                        "kps1": rbd(feature_map_all[name][p1])["keypoints"]
                        .cpu()
                        .numpy(),
                        "matches": matches_arr,
                    }
                )

    cropped_bbox_list_map = {}
    for image_path in tqdm(image_paths, desc="Cropping"):
        cropped_bbox_list_map[image_path] = []
        for name in extractors:
            freq_counter = keypoint_freq_map_all[name][image_path]
            if not freq_counter:
                continue
            max_count = max(freq_counter.values())
            high_freq_keypoint_ids = [
                k for k, v in freq_counter.items() if v / max_count > 0.15
            ]
            image_tensor = load_image(image_path)
            c, h, w = image_tensor.shape
            features = rbd(feature_map_all[name][image_path])
            keypoints = (
                features["keypoints"]
                .cpu()
                .numpy()
                .astype(np.float32)[high_freq_keypoint_ids]
            )
            num_keypoints = len(keypoints)
            normalized_keypoints = keypoints / np.array([w, h])
            clustering = DBSCAN(eps=0.05, min_samples=16).fit(
                normalized_keypoints
            )
            groups, counts = np.unique(clustering.labels_, return_counts=True)
            max_group, max_count = None, 0
            no_crop = False
            for group, count in zip(groups, counts):
                if group == -1:
                    if count / num_keypoints > 0.2:
                        no_crop = True
                    continue
                if count > max_count:
                    max_count = count
                    max_group = group
            if no_crop:
                continue
            candidate_groups = set()
            for group, count in zip(groups, counts):
                if group == -1:
                    continue
                if group == max_group or count / num_keypoints > 0.05:
                    candidate_groups.add(group)
            for group in candidate_groups:
                points = keypoints[clustering.labels_ == group]
                x_min, y_min = np.min(points, axis=0).astype(int)
                x_max, y_max = np.max(points, axis=0).astype(int)
                x_min = max(x_min - 10, 0)
                y_min = max(y_min - 10, 0)
                x_max = min(x_max + 10, w)
                y_max = min(y_max + 10, h)
                cropped_bbox_list_map[image_path].append(
                    (x_min, y_min, x_max, y_max)
                )

    cropped_feature_list_map = {k: defaultdict(list) for k in extractors}
    for image_path in tqdm(image_paths, desc="Extracting features (cropped)"):
        image_tensor = load_image(image_path).cuda()
        for name, extractor in extractors.items():
            for x_min, y_min, x_max, y_max in cropped_bbox_list_map[
                image_path
            ]:
                cropped_image_tensor = image_tensor[
                    :, y_min:y_max, x_min:x_max
                ]
                cropped_features = extractor.extract(cropped_image_tensor)
                cropped_feature_list_map[name][image_path].append(
                    {
                        "bbox": (x_min, y_min, x_max, y_max),
                        "features": cropped_features,
                    }
                )

    for i, p0 in enumerate(image_paths):
        for j, p1 in enumerate(image_paths):
            if j <= i:
                continue
            for name in extractors:
                for e0 in cropped_feature_list_map[name][p0]:
                    for e1 in cropped_feature_list_map[name][p1]:
                        cropped_matches = matchers[name](
                            {
                                "image0": e0["features"],
                                "image1": e1["features"],
                            }
                        )
                        pool.append(
                            {
                                "cropped": True,
                                "p0": p0,
                                "p1": p1,
                                "bbox0": e0["bbox"],
                                "bbox1": e1["bbox"],
                                "kps0": rbd(e0["features"])["keypoints"]
                                .cpu()
                                .numpy(),
                                "kps1": rbd(e1["features"])["keypoints"]
                                .cpu()
                                .numpy(),
                                "matches": rbd(cropped_matches)["matches"]
                                .cpu()
                                .numpy(),
                            }
                        )

    image_local_to_global = {}
    global_kp_counter = defaultdict(int)
    image_kp_map = defaultdict(list)
    global_matches_map = defaultdict(list)

    for entry in pool:
        p0, p1 = str(entry["p0"]), str(entry["p1"])
        kps0, kps1 = entry["kps0"], entry["kps1"]
        matches = entry["matches"]
        bbox0, bbox1 = entry["bbox0"], entry["bbox1"]
        if entry["cropped"]:
            kps0 += np.array([bbox0[0], bbox0[1]])
            kps1 += np.array([bbox1[0], bbox1[1]])
        for local_idx, kp in enumerate(kps0):
            key = (p0, local_idx)
            if key not in image_local_to_global:
                image_local_to_global[key] = global_kp_counter[p0]
                image_kp_map[p0].append(kp)
                global_kp_counter[p0] += 1
        for local_idx, kp in enumerate(kps1):
            key = (p1, local_idx)
            if key not in image_local_to_global:
                image_local_to_global[key] = global_kp_counter[p1]
                image_kp_map[p1].append(kp)
                global_kp_counter[p1] += 1
        for m in matches:
            gid0 = image_local_to_global[(p0, m[0])]
            gid1 = image_local_to_global[(p1, m[1])]
            global_matches_map[(p0, p1)].append((gid0, gid1))

    for image_path, keypoints in image_kp_map.items():
        db.add_keypoints(
            image_id_map[image_path], np.array(keypoints, dtype=np.float32)
        )

    for (p0, p1), matches in global_matches_map.items():
        db.add_matches(
            image_id_map[p0],
            image_id_map[p1],
            np.array(matches, dtype=np.uint32),
        )

    db.commit()
    db.close()

    pycolmap.match_exhaustive(str(database_path))
    logger.debug(f"incremental_mapping...")

    mapper_options = pycolmap.IncrementalPipelineOptions(
        min_model_size=3,
        max_num_models=5,
    )

    recons = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=image_dir,
        output_path=str(output_dir),
        options=mapper_options,
    )

    best_recon_num_images = 0
    best_recon_idx = None
    for idx, recon in recons.items():
        num_images = len(recon.images)
        if num_images > best_recon_num_images:
            best_recon_num_images = num_images
            best_recon_idx = idx

    recon_results = defaultdict(lambda: {"R": np.eye(3), "t": np.zeros(3)})
    if best_recon_idx is not None:
        recon = recons[best_recon_idx]
        for image_id in recon.reg_image_ids():
            image = recon.images[image_id]
            cam_from_world = image.cam_from_world.matrix()
            recon_results[image.name] = {
                "R": cam_from_world[:3, :3],
                "t": cam_from_world[:3, 3],
            }

    for image_path in tqdm(image_paths, desc="Extracting features"):
        image_name = image_path.name
        with open(SUBMISSION_FILE_PATH, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    f"{dataset_name}/{scene_name}/images/{image_name}",
                    dataset_name,
                    scene_name,
                    ";".join(
                        map(str, recon_results[image_name]["R"].flatten())
                    ),
                    ";".join(map(str, recon_results[image_name]["t"])),
                ]
            )


