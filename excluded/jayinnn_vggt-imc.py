!pip install -q /kaggle/input/einops-v0-8-0/einops-0.8.0-py3-none-any.whl --no-index --find-links /kaggle/input/einops-v0-8-0
!pip install /kaggle/input/imc2023-vggt-whl/* --no-deps --no-index --find-links /kaggle/input/imc2023-vggt-whl

!mkdir -p /root/.cache/torch/hub/checkpoints
!cp /kaggle/input/aliked/pytorch/aliked-n16/1/aliked-n16.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/
!cp /kaggle/input/lightglue/pytorch/aliked/1/aliked_lightglue.pth /root/.cache/torch/hub/checkpoints/aliked_lightglue_v0-1_arxiv-pth
!cp /kaggle/input/vggt-object-tracker/pytorch/v1/4/superpoint_v1.pth /root/.cache/torch/hub/checkpoints/


!cd /kaggle/input/pkg-colmap/colmap_offline/ && dpkg -i *.deb


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]


!cp -r /kaggle/input/vggt/pytorch/default/1/vggt/vggt ./


import argparse
import random
import numpy as np
import glob
import os
import copy
import torch
import torch.nn.functional as F
import gc
import subprocess
import shutil

# Configure CUDA settings
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

import argparse
from pathlib import Path
import trimesh
import pycolmap
from transformers import AutoImageProcessor, AutoModel


from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
# from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap, batch_np_matrix_to_pycolmap_wo_track,
from vggt.dependency.np_to_pycolmap import _build_pycolmap_intri
from vggt.dependency.projection import project_3D_points_np

from vggt.dependency.vggsfm_utils import *
from vggt.dependency.track_predict import _forward_on_query, _augment_non_visible_frames


def run_VGGT(model, images, dtype, resolution=518):
    # images: [B, 3, H, W]

    assert len(images.shape) == 4
    assert images.shape[1] == 3

    # hard-coded to use 518 for VGGT
    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    extrinsic = extrinsic.squeeze(0).cpu().numpy()
    intrinsic = intrinsic.squeeze(0).cpu().numpy()
    depth_map = depth_map.squeeze(0).cpu().numpy()
    depth_conf = depth_conf.squeeze(0).cpu().numpy()
    return extrinsic, intrinsic, depth_map, depth_conf


def batch_np_matrix_to_pycolmap(
    points3d,
    extrinsics,
    intrinsics,
    tracks,
    image_size,
    image_basenames,
    name_to_image_id,
    masks=None,
    max_reproj_error=None,
    max_points3D_val=3000,
    shared_camera=False,
    camera_type="SIMPLE_PINHOLE",
    extra_params=None,
    min_inlier_per_frame=64,
    points_rgb=None,
):
    """
    Convert Batched NumPy Arrays to PyCOLMAP

    Check https://github.com/colmap/pycolmap for more details about its format

    NOTE that colmap expects images/cameras/points3D to be 1-indexed
    so there is a +1 offset between colmap index and batch index


    NOTE: different from VGGSfM, this function:
    1. Use np instead of torch
    2. Frame index and camera id starts from 1 rather than 0 (to fit the format of PyCOLMAP)
    """
    # points3d: Px3
    # extrinsics: Nx3x4
    # intrinsics: Nx3x3
    # tracks: NxPx2
    # masks: NxP
    # image_size: 2, assume all the frames have been padded to the same size
    # where N is the number of frames and P is the number of tracks

    N, P, _ = tracks.shape
    assert len(extrinsics) == N
    assert len(intrinsics) == N
    assert len(points3d) == P
    assert image_size.shape[0] == 2

    if max_reproj_error is not None:
        projected_points_2d, projected_points_cam = project_3D_points_np(points3d, extrinsics, intrinsics)
        projected_diff = np.linalg.norm(projected_points_2d - tracks, axis=-1)
        projected_points_2d[projected_points_cam[:, -1] <= 0] = 1e6
        reproj_mask = projected_diff < max_reproj_error

    if masks is not None and reproj_mask is not None:
        masks = np.logical_and(masks, reproj_mask)
    elif masks is not None:
        masks = masks
    else:
        masks = reproj_mask

    assert masks is not None

    if masks.sum(1).min() < min_inlier_per_frame:
        print(f"Not enough inliers per frame, skip BA.")
        return None, None

    # Reconstruction object, following the format of PyCOLMAP/COLMAP
    reconstruction = pycolmap.Reconstruction()

    inlier_num = masks.sum(0)
    valid_mask = inlier_num >= 2  # a track is invalid if without two inliers
    valid_idx = np.nonzero(valid_mask)[0]

    # Only add 3D points that have sufficient 2D points
    for vidx in valid_idx:
        # Use RGB colors if provided, otherwise use zeros
        rgb = points_rgb[vidx] if points_rgb is not None else np.zeros(3)
        reconstruction.add_point3D(points3d[vidx], pycolmap.Track(), rgb)

    num_points3D = len(valid_idx)
    camera = None
    # frame idx
    for fidx in range(N):
        # set camera
        image_id = name_to_image_id[image_basenames[fidx]]
        if camera is None or (not shared_camera):
            pycolmap_intri = _build_pycolmap_intri(fidx, intrinsics, camera_type, extra_params)

            camera = pycolmap.Camera(
                model=camera_type, width=image_size[0], height=image_size[1], params=pycolmap_intri, camera_id=fidx + 1
            )

            # add camera
            reconstruction.add_camera(camera)

        # set image
        cam_from_world = pycolmap.Rigid3d(
            pycolmap.Rotation3d(extrinsics[fidx][:3, :3]), extrinsics[fidx][:3, 3]
        )  # Rot and Trans

        image = pycolmap.Image(
            id=image_id, name=f"image_{image_id}", camera_id=camera.camera_id, cam_from_world=cam_from_world
        )

        points2D_list = []

        point2D_idx = 0

        # NOTE point3D_id start by 1
        for point3D_id in range(1, num_points3D + 1):
            original_track_idx = valid_idx[point3D_id - 1]

            if (reconstruction.points3D[point3D_id].xyz < max_points3D_val).all():
                if masks[fidx][original_track_idx]:
                    # It seems we don't need +0.5 for BA
                    point2D_xy = tracks[fidx][original_track_idx]
                    # Please note when adding the Point2D object
                    # It not only requires the 2D xy location, but also the id to 3D point
                    points2D_list.append(pycolmap.Point2D(point2D_xy, point3D_id))

                    # add element
                    track = reconstruction.points3D[point3D_id].track
                    track.add_element(image_id, point2D_idx)
                    point2D_idx += 1

        assert point2D_idx == len(points2D_list)

        try:
            image.points2D = pycolmap.ListPoint2D(points2D_list)
            image.registered = True
        except:
            print(f"frame {fidx + 1} is out of BA")
            image.registered = False

        # add image
        reconstruction.add_image(image)

    return reconstruction, valid_mask


def rename_colmap_recons_and_rescale_camera(
    reconstruction, 
    image_paths_in_batch,
    original_coords_in_batch,
    img_size,
    image_id_to_name,
    image_id_to_batch_id,
    shift_point2d_to_original_res=False, 
    shared_camera=False
):
    rescale_camera = True

    # Image paths in batch should be the base names, not full paths for pyimage.name
    base_image_paths_in_batch = [os.path.basename(p) for p in image_paths_in_batch]

    for pyimageid in list(reconstruction.images.keys()): # Iterate over a copy of keys if modifying
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        batch_image_id = image_id_to_batch_id[pyimageid]
        # pyimageid is 1-indexed. Ensure image_paths_in_batch is correctly indexed.
        if (batch_image_id - 1) < len(base_image_paths_in_batch):
            pyimage.name = image_id_to_name[pyimageid]
        else:
            print(f"Warning: batch_image_id {batch_image_id} out of range for image_paths_in_batch (len: {len(base_image_paths_in_batch)})")
            continue # Or handle error appropriately

        if rescale_camera:            
            if (batch_image_id - 1) < len(original_coords_in_batch):
                real_image_size = original_coords_in_batch[batch_image_id - 1, -2:]
                resize_ratio = max(real_image_size) / img_size # img_size is the reconstruction_resolution
                pred_params = pred_params * resize_ratio
                real_pp = real_image_size / 2
                pred_params[-2:] = real_pp
                pycamera.params = pred_params
                pycamera.width = int(real_image_size[0])
                pycamera.height = int(real_image_size[1])
            else:
                print(f"Warning: batch_image_id {batch_image_id} out of range for original_coords_in_batch (len: {len(original_coords_in_batch)})")
                # Decide how to handle this: skip rescaling for this camera, or raise error
                continue


        if shift_point2d_to_original_res:
            if (batch_image_id - 1) < len(original_coords_in_batch):
                top_left = original_coords_in_batch[batch_image_id - 1, :2]
                # resize_ratio needs to be defined here as well if not falling through from above
                real_image_size_for_ratio = original_coords_in_batch[batch_image_id - 1, -2:]
                resize_ratio_for_points = max(real_image_size_for_ratio) / img_size

                for point_idx in range(len(pyimage.points2D)):
                    point2D = pyimage.points2D[point_idx]
                    # Ensure point2D.xy is a numpy array for subtraction
                    xy_np = np.array(point2D.xy)
                    top_left_np = np.array(top_left)
                    point2D.xy = (xy_np - top_left_np) * resize_ratio_for_points
            else:
                 print(f"Warning: batch_image_id {batch_image_id} out of range for original_coords_in_batch during point2D shift.")


        if shared_camera:
            rescale_camera = False
    return reconstruction


def predict_tracks(
    images,
    conf=None,
    points_3d=None,
    masks=None,
    max_query_pts=2048,
    query_frame_num=5,
    keypoint_extractor="aliked+sp",
    max_points_num=163840,
    fine_tracking=True,
    complete_non_vis=True,
):
    """
    Predict tracks for the given images and masks.

    TODO: support non-square images
    TODO: support masks


    This function predicts the tracks for the given images and masks using the specified query method
    and track predictor. It finds query points, and predicts the tracks, visibility, and scores for the query frames.

    Args:
        images: Tensor of shape [S, 3, H, W] containing the input images.
        conf: Tensor of shape [S, 1, H, W] containing the confidence scores. Default is None.
        points_3d: Tensor containing 3D points. Default is None.
        masks: Optional tensor of shape [S, 1, H, W] containing masks. Default is None.
        max_query_pts: Maximum number of query points. Default is 2048.
        query_frame_num: Number of query frames to use. Default is 5.
        keypoint_extractor: Method for keypoint extraction. Default is "aliked+sp".
        max_points_num: Maximum number of points to process at once. Default is 163840.
        fine_tracking: Whether to use fine tracking. Default is True.
        complete_non_vis: Whether to augment non-visible frames. Default is True.

    Returns:
        pred_tracks: Numpy array containing the predicted tracks.
        pred_vis_scores: Numpy array containing the visibility scores for the tracks.
        pred_confs: Numpy array containing the confidence scores for the tracks.
        pred_points_3d: Numpy array containing the 3D points for the tracks.
        pred_colors: Numpy array containing the point colors for the tracks. (0, 255)
    """

    device = images.device
    dtype = images.dtype
    model_path = "/kaggle/input/vggt-object-tracker/pytorch/v1/4/vggsfm_v2_tracker.pt"
    tracker = build_vggsfm_tracker(model_path).to(device, dtype)

    # Find query frames
    query_frame_indexes = generate_rank_by_dino_(images, query_frame_num=query_frame_num, device=device)

    # Add the first image to the front if not already present
    if 0 in query_frame_indexes:
        query_frame_indexes.remove(0)
    query_frame_indexes = [0, *query_frame_indexes]

    # TODO: add the functionality to handle the masks
    keypoint_extractors = initialize_feature_extractors(
        max_query_pts, extractor_method=keypoint_extractor, device=device
    )

    pred_tracks = []
    pred_vis_scores = []
    pred_confs = []
    pred_points_3d = []
    pred_colors = []

    fmaps_for_tracker = tracker.process_images_to_fmaps(images)

    if fine_tracking:
        print("For faster inference, consider disabling fine_tracking")

    for query_index in query_frame_indexes:
        print(f"Predicting tracks for query frame {query_index}")
        pred_track, pred_vis, pred_conf, pred_point_3d, pred_color = _forward_on_query(
            query_index,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            device,
        )

        pred_tracks.append(pred_track)
        pred_vis_scores.append(pred_vis)
        pred_confs.append(pred_conf)
        pred_points_3d.append(pred_point_3d)
        pred_colors.append(pred_color)

    if complete_non_vis:
        pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors = _augment_non_visible_frames(
            pred_tracks,
            pred_vis_scores,
            pred_confs,
            pred_points_3d,
            pred_colors,
            images,
            conf,
            points_3d,
            fmaps_for_tracker,
            keypoint_extractors,
            tracker,
            max_points_num,
            fine_tracking,
            min_vis=500,
            non_vis_thresh=0.1,
            device=device,
        )

    pred_tracks = np.concatenate(pred_tracks, axis=1)
    pred_vis_scores = np.concatenate(pred_vis_scores, axis=1)
    pred_confs = np.concatenate(pred_confs, axis=0) if pred_confs else None
    pred_points_3d = np.concatenate(pred_points_3d, axis=0) if pred_points_3d else None
    pred_colors = np.concatenate(pred_colors, axis=0) if pred_colors else None

    # from vggt.utils.visual_track import visualize_tracks_on_images
    # visualize_tracks_on_images(images[None], torch.from_numpy(pred_tracks[None]), torch.from_numpy(pred_vis_scores[None])>0.2, out_dir="track_visuals")

    return pred_tracks, pred_vis_scores, pred_confs, pred_points_3d, pred_colors


def generate_rank_by_dino_(
    images, query_frame_num, image_size=336, model_name="dinov2_vitb14_reg", device="cuda", spatial_similarity=False
):
    images_resized = F.interpolate(images, (image_size, image_size), mode="bilinear", align_corners=False)
    
    dino_model_path = '/kaggle/input/dinov2/pytorch/base/1/' # Ensure this is correct
    # Load DINO model only when needed and ensure it's on the correct device
    dino_v2_model = AutoModel.from_pretrained(dino_model_path).eval().to(device)
    
    resnet_mean = torch.tensor(_RESNET_MEAN, device=device).view(1, 3, 1, 1)
    resnet_std = torch.tensor(_RESNET_STD, device=device).view(1, 3, 1, 1)
    images_resnet_norm = (images_resized - resnet_mean) / resnet_std
    
    with torch.no_grad():
        outputs = dino_v2_model(images_resnet_norm)
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
            cls_token = hidden_states[:, 0, :]
            patch_tokens = hidden_states[:, 1:, :]
            frame_feat = {
                "x_norm_clstoken": F.normalize(cls_token, p=2, dim=1),
                "x_norm_patchtokens": F.normalize(patch_tokens, p=2, dim=2)
            }
        else:
            # This part might need adjustment based on the exact DINOv2 output for your model version
            # The original notebook code had a more direct way to get x_norm_clstoken and x_norm_patchtokens
            # If the AutoModel output is already a dict with these keys, use that directly.
            # For now, proceeding with HuggingFace typical output structure.
            print("Warning: DINOv2 output format might differ from original implementation. Check `frame_feat` structure.")
            if isinstance(outputs, dict) and "x_norm_clstoken" in outputs and "x_norm_patchtokens" in outputs:
                 frame_feat = outputs
            else:
                raise ValueError("Unexpected DINO model output format from AutoModel")

    if spatial_similarity:
        frame_feat_data = frame_feat["x_norm_patchtokens"]
        frame_feat_norm = frame_feat_data.permute(1, 0, 2) # Check dimensions for bmm
        similarity_matrix = torch.bmm(frame_feat_norm, frame_feat_norm.transpose(-1, -2))
        similarity_matrix = similarity_matrix.mean(dim=0)
    else:
        frame_feat_data = frame_feat["x_norm_clstoken"]
        similarity_matrix = torch.mm(frame_feat_data, frame_feat_data.transpose(-1, -2))
    
    distance_matrix = 100 - similarity_matrix.clone()
    similarity_matrix.fill_diagonal_(-100)
    similarity_sum = similarity_matrix.sum(dim=1)
    most_common_frame_index = torch.argmax(similarity_sum).item()
    
    # Ensure farthest_point_sampling is defined and accessible
    fps_idx = farthest_point_sampling(distance_matrix, query_frame_num, most_common_frame_index)
    
    del frame_feat, frame_feat_data, similarity_matrix, distance_matrix, similarity_sum
    del dino_v2_model, images_resized, images_resnet_norm, outputs, cls_token, patch_tokens, hidden_states
    gc.collect()
    torch.cuda.empty_cache()
    
    return fps_idx


# --- Main Modified Function ---
def demo_fn_batched(args):
    print("Arguments:", vars(args))

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    print(f"Setting seed as: {args.seed}")

    scene_dir = f"/kaggle/input/image-matching-challenge-2023/{args.variant}/{args.dataset}/{args.scene}"
    image_dir = os.path.join(scene_dir, "images")
    all_image_path_list = sorted(glob.glob(os.path.join(image_dir, "*"))) # Sort for consistent batching

    if not all_image_path_list:
        raise ValueError(f"No images found in {image_dir}")

    # Shuffle once before batching if desired (as per discussion "random shuffle then sequential")
    # However, for SfM, a somewhat coherent sequence is often better for merging.
    # If random shuffle is critical, uncomment below. Otherwise, sequential slicing is often preferred.
    # random.shuffle(all_image_path_list)

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}, Using dtype: {dtype}")

    model = VGGT()
    model_path = "/kaggle/input/vggt-object-tracker/pytorch/v1/4/model.pt" # Ensure this path is correct
    model.load_state_dict(torch.load(model_path))
    model.eval().to(device)
    print("VGGT Model loaded")

    vggt_fixed_resolution = 518
    img_load_resolution = 1024 # Resolution for initial loading and track prediction

    # --- Batching Logic ---
    batch_size = args.batch_size
    overlap = args.batch_overlap
    num_images = len(all_image_path_list)
    print(f"Total number of images: {num_images}")
    all_image_basenames = [os.path.basename(p) for p in all_image_path_list]
    # Ensure unique basenames if not already guaranteed; if duplicates exist, this strategy needs refinement.
    # For simplicity, assuming basenames are unique identifiers for unique images.
    name_to_image_id = {name: image_id+1 for image_id, name in enumerate(all_image_basenames)}
    image_id_to_name = {image_id: name for name, image_id in name_to_image_id.items()}
    
    batch_reconstruction_paths = []
    
    # Create a main output directory for this scene
    scene_output_dir = f"/kaggle/working/{args.variant}/{args.dataset}/{args.scene}"
    os.makedirs(scene_output_dir, exist_ok=True)

    for i in range(0, num_images, batch_size - overlap):
        start_idx = i
        end_idx = min(i + batch_size, num_images)
        if end_idx - start_idx < batch_size: # Need at least 2 images for reconstruction
            current_batch_image_paths = all_image_path_list[start_idx:end_idx]
            num_remain = batch_size - (end_idx - start_idx)
            current_batch_image_paths = all_image_path_list[0:num_remain] + current_batch_image_paths
        else:
            current_batch_image_paths = all_image_path_list[start_idx:end_idx]
        
        if len(current_batch_image_paths) < 2: # Check again if slicing resulted in too few images
            print(f"Skipping batch {i//(batch_size-overlap) +1} after slicing: Not enough images ({len(current_batch_image_paths)})")
            continue

        assert len(current_batch_image_paths) == batch_size, f"Num of image not matched, {len(current_batch_image_paths)} != batch size {batch_size}"
        image_basenames = [os.path.basename(p) for p in current_batch_image_paths]
        image_id_to_batch_id = {name_to_image_id[name]: batch_id for batch_id, name in enumerate(image_basenames)}

        batch_num = i // (batch_size - overlap) + 1
        print(f"\n--- Processing Batch {batch_num} ({len(current_batch_image_paths)} images from index {start_idx} to {end_idx-1}) ---")
        
        batch_output_dir = os.path.join(scene_output_dir, f"batch_{batch_num}")
        os.makedirs(batch_output_dir, exist_ok=True)
        
        images, original_coords_batch = load_and_preprocess_images_square(current_batch_image_paths, img_load_resolution)
        images = images.to(device) # Shape: [S, 3, H, W] where S is num images in batch
        original_coords_batch = original_coords_batch.to(device) # Shape: [S, 4]
        print(f"Loaded {len(images)} images for batch {batch_num}")

        # Run VGGT for this batch
        extrinsic_batch, intrinsic_batch, depth_map_batch, depth_conf_batch = run_VGGT(model, images, dtype, vggt_fixed_resolution)
        points_3d_vggt_batch = unproject_depth_map_to_point_map(depth_map_batch, extrinsic_batch, intrinsic_batch)

        reconstruction_batch = None
        batch_reconstruction_resolution = vggt_fixed_resolution # Default for no-BA path

        if args.use_ba_per_batch:
            print(f"Attempting BA for batch {batch_num}")
            image_size_ba = np.array(images.shape[-2:]) # This is img_load_resolution
            scale_ba = img_load_resolution / vggt_fixed_resolution
            shared_camera_ba = args.shared_camera

            with torch.cuda.amp.autocast(dtype=dtype):
                pred_tracks_batch, pred_vis_scores_batch, _, pred_points_3d_colmap_batch, points_rgb_batch = predict_tracks(
                    images,
                    conf=depth_conf_batch,
                    points_3d=points_3d_vggt_batch,
                    masks=None,
                    max_query_pts=args.max_query_pts,
                    query_frame_num=args.query_frame_num,
                    keypoint_extractor="aliked+sp",
                    fine_tracking=args.fine_tracking,
                )
                # pred_tracks_batch, pred_vis_scores_batch, _, pred_points_3d_colmap_batch, points_rgb_batch = predict_tracks(
                #     images, # These are at img_load_resolution
                #     conf=torch.from_numpy(depth_conf_batch).to(device).unsqueeze(1), # depth_conf_batch needs to be [S,1,H,W] and on device
                #     points_3d_vggt=torch.from_numpy(points_3d_vggt_batch).to(device), # points_3d_vggt_batch needs to be tensor and on device
                #     max_query_pts=args.max_query_pts,
                #     query_frame_num=args.query_frame_num,
                #     fine_tracking=args.fine_tracking,
                # )
            torch.cuda.empty_cache()
            gc.collect()

            if pred_tracks_batch.size == 0 or pred_points_3d_colmap_batch is None or pred_points_3d_colmap_batch.size == 0:
                print(f"Warning: No tracks or 3D points from predict_tracks for batch {batch_num}. Skipping BA for this batch.")
                # Fallback to no-BA reconstruction for this batch
            else:
                intrinsic_ba_batch = intrinsic_batch.copy() # Intrinsic from VGGT is for vggt_fixed_resolution
                intrinsic_ba_batch[:, :2, :] *= scale_ba # Rescale for img_load_resolution (where tracks are)
                
                track_mask_batch = pred_vis_scores_batch > args.vis_thresh

                reconstruction_batch, _ = batch_np_matrix_to_pycolmap(
                    pred_points_3d_colmap_batch, # From predict_tracks
                    extrinsic_batch, # From VGGT (poses)
                    intrinsic_ba_batch, # Rescaled intrinsics
                    pred_tracks_batch,
                    image_size_ba, # img_load_resolution
                    image_basenames,
                    name_to_image_id,
                    masks=track_mask_batch,
                    max_reproj_error=args.max_reproj_error,
                    shared_camera=shared_camera_ba,
                    camera_type=args.camera_type,
                    points_rgb=points_rgb_batch,
                )

                if reconstruction_batch and len(reconstruction_batch.images)>0 :
                    print(f"Performing per-batch BA for batch {batch_num}")
                    ba_options = pycolmap.BundleAdjustmentOptions()
                    # Configure BA options if needed, e.g., ba_options.solver_options.max_num_iterations = 20
                    pycolmap.bundle_adjustment(reconstruction_batch, ba_options)
                    batch_reconstruction_resolution = img_load_resolution
                else:
                    print(f"Failed to create initial reconstruction for BA for batch {batch_num}. Falling back.")
                    # reconstruction_batch = None # Ensure it's None to trigger no-BA path
                    batch_reconstruction_resolution = img_load_resolution

        # if reconstruction_batch is None: # Fallback if BA failed or not used
        #     print(f"Using feedforward reconstruction (no BA) for batch {batch_num}")
        #     conf_thres_value = 5
        #     max_points_for_colmap = 100000
            
        #     # image_size for no-BA path is vggt_fixed_resolution
        #     image_size_no_ba = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
        #     num_frames_batch, height_batch, width_batch, _ = points_3d_vggt_batch.shape # From VGGT output

        #     # Resize source images to vggt_fixed_resolution for RGB color sampling
        #     points_rgb_no_ba_src_images = F.interpolate(
        #         images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
        #     )
        #     points_rgb_no_ba = (points_rgb_no_ba_src_images.cpu().numpy() * 255).astype(np.uint8)
        #     points_rgb_no_ba = points_rgb_no_ba.transpose(0, 2, 3, 1) # S, H, W, 3

        #     points_xyf_batch = create_pixel_coordinate_grid(num_frames_batch, height_batch, width_batch)
            
        #     # depth_conf_batch is (S, H, W) from VGGT
        #     conf_mask_batch = depth_conf_batch >= conf_thres_value 
        #     conf_mask_batch = randomly_limit_trues(conf_mask_batch, max_points_for_colmap)

        #     points_3d_filtered = points_3d_vggt_batch[conf_mask_batch]
        #     points_xyf_filtered = points_xyf_batch[conf_mask_batch]
        #     points_rgb_filtered = points_rgb_no_ba[conf_mask_batch]

        #     if points_3d_filtered.shape[0] > 0:
        #          reconstruction_batch = batch_np_matrix_to_pycolmap_wo_track(
        #             points_3d_filtered,
        #             points_xyf_filtered,
        #             points_rgb_filtered,
        #             extrinsic_batch, # from VGGT
        #             intrinsic_batch, # from VGGT (for vggt_fixed_resolution)
        #             image_size_no_ba, # vggt_fixed_resolution
        #             shared_camera=False, # Typically False for no-BA from VGGT
        #             camera_type="PINHOLE", # Typically PINHOLE for no-BA from VGGT
        #         )
        #     else:
        #         print(f"Error: No 3D points left after filtering for batch {batch_num}. Skipping this batch.")
        #         del images, original_coords_batch, extrinsic_batch, intrinsic_batch, depth_map_batch, depth_conf_batch, points_3d_vggt_batch
        #         gc.collect()
        #         torch.cuda.empty_cache()
        #         continue


        #     batch_reconstruction_resolution = vggt_fixed_resolution
        
        if reconstruction_batch and len(reconstruction_batch.images) > 0 :
            reconstruction_batch = rename_colmap_recons_and_rescale_camera(
                reconstruction_batch,
                current_batch_image_paths, # Full paths for the current batch
                original_coords_batch.cpu().numpy(), # Original coords for the current batch
                img_size=batch_reconstruction_resolution, # Resolution at which reconstruction_batch was made
                image_id_to_name=image_id_to_name,
                image_id_to_batch_id=image_id_to_batch_id,
                shift_point2d_to_original_res=True, # Important for consistency
                shared_camera=args.shared_camera if args.use_ba_per_batch else False,
            )
            
            batch_sparse_dir = os.path.join(batch_output_dir, "sparse")
            os.makedirs(batch_sparse_dir, exist_ok=True)
            reconstruction_batch.write(batch_sparse_dir)
            batch_reconstruction_paths.append(batch_sparse_dir)
            print(f"Saved reconstruction for batch {batch_num} to {batch_sparse_dir}")
        else:
            print(f"Error: Reconstruction for batch {batch_num} is empty or invalid. Skipping.")

        del images, original_coords_batch, extrinsic_batch, intrinsic_batch, depth_map_batch, depth_conf_batch, points_3d_vggt_batch, reconstruction_batch
        if 'pred_tracks_batch' in locals(): del pred_tracks_batch # Clean up BA specific variables
        gc.collect()
        torch.cuda.empty_cache()
    
    del model # Free VGGT model memory
    gc.collect()
    torch.cuda.empty_cache()

    # --- Merging Reconstructions ---
    if not batch_reconstruction_paths:
        print("No batch reconstructions were generated. Exiting.")
        return False
    
    if len(batch_reconstruction_paths) == 1:
        print("Only one batch was processed. Using its reconstruction as the merged result.")
        merged_model_path = batch_reconstruction_paths[0]
    else:
        print("\n--- Merging Batch Reconstructions ---")
        merged_dir = os.path.join(scene_output_dir, "merged_stages")
        os.makedirs(merged_dir, exist_ok=True)
        
        current_merged_path = batch_reconstruction_paths[0]
        
        for i in range(1, len(batch_reconstruction_paths)):
            input_path1 = current_merged_path
            input_path2 = batch_reconstruction_paths[i]
            output_path_temp = os.path.join(merged_dir, f"merged_step_{i-1}_to_{i}")
            if os.path.exists(output_path_temp): # Clean up previous attempt for this step
                shutil.rmtree(output_path_temp)
            os.makedirs(output_path_temp, exist_ok=True)

            print(f"Merging: '{os.path.basename(os.path.dirname(input_path1))}' and '{os.path.basename(os.path.dirname(input_path2))}' into '{os.path.basename(output_path_temp)}'")
            
            colmap_executable = "colmap" # Assumes it's in PATH after dpkg -i
            
            merge_cmd = [
                colmap_executable, "model_merger",
                "--input_path1", input_path1,
                "--input_path2", input_path2,
                "--output_path", output_path_temp,
                # Add other model_merger options if needed, e.g.:
                # "--robust_merge_max_reproj_error", "16", # Default is 8
                # "--robust_merge_min_inlier_ratio", "0.05", # Default is 0.1
            ]
            print(f"Executing: {' '.join(merge_cmd)}")
            try:
                completed_process = subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
                print("Merge stdout:", completed_process.stdout)
                print("Merge successful for this step.")
                current_merged_path = output_path_temp
            except subprocess.CalledProcessError as e:
                print(f"Error during model merging step {i}:")
                print("Command:", e.cmd)
                print("Return code:", e.returncode)
                print("Stdout:", e.stdout)
                print("Stderr:", e.stderr)
                print(f"Failed to merge {input_path2}. Trying to continue with previous merged model if possible, or stopping.")
                # Decide on error strategy: stop, or try to merge next available into current_merged_path
                # For now, if a merge fails, the chain breaks.
                # A more robust strategy might try to merge batch_reconstruction_paths[i+1] into current_merged_path
                return False # Stop if any merge fails
        merged_model_path = current_merged_path

    print(f"\nFinal merged model (before global BA) is at: {merged_model_path}")

    # --- Final Bundle Adjustment ---
    if args.use_global_ba:
        print("\n--- Performing Final Global Bundle Adjustment ---")
        try:
            final_reconstruction = pycolmap.read_model(merged_model_path, format=".bin") # or .txt if saved as text
            print(f"Loaded final merged reconstruction with {len(final_reconstruction.images)} images and {len(final_reconstruction.points3D)} points.")
            
            if len(final_reconstruction.images) > 0:
                ba_options_global = pycolmap.BundleAdjustmentOptions()
                # Configure global BA options if needed (e.g., more iterations)
                # ba_options_global.solver_options.max_num_iterations = 50
                pycolmap.bundle_adjustment(final_reconstruction, ba_options_global)
                print("Global BA successful.")

                final_sparse_dir = os.path.join(scene_output_dir, "sparse_global_ba")
                os.makedirs(final_sparse_dir, exist_ok=True)
                final_reconstruction.write(final_sparse_dir)
                print(f"Saved final BA reconstruction to: {final_sparse_dir}")

                # Optional: Save point cloud of the final model
                if len(final_reconstruction.points3D) > 0:
                    points3D_final = np.array([p.xyz for p in final_reconstruction.points3D.values()])
                    colors_final = np.array([p.color for p in final_reconstruction.points3D.values()])
                    if points3D_final.size > 0:
                         trimesh.PointCloud(points3D_final, colors=colors_final).export(os.path.join(final_sparse_dir, "points_final_ba.ply"))
                         print(f"Saved final point cloud to {final_sparse_dir}/points_final_ba.ply")

            else:
                print("Merged reconstruction is empty. Skipping final BA.")
        except Exception as e:
            print(f"Error during final BA or saving: {e}")
            print("The merged model without final BA is available at:", merged_model_path)
    else:
        print("Skipping final global BA as per arguments.")
        final_output_dir_no_ba = os.path.join(scene_output_dir, "sparse_merged_no_global_ba")
        if os.path.exists(merged_model_path) and merged_model_path != final_output_dir_no_ba :
             if os.path.exists(final_output_dir_no_ba): # shutil.copytree needs dst to not exist
                 shutil.rmtree(final_output_dir_no_ba)
             shutil.copytree(merged_model_path, final_output_dir_no_ba) # Copy to a final named directory
             print(f"Final merged reconstruction (no global BA) saved to: {final_output_dir_no_ba}")


    # Clean up intermediate batch directories if desired
    if args.cleanup_batch_dirs and len(batch_reconstruction_paths) > 1: # Only if merging happened
        print("\nCleaning up intermediate batch reconstruction directories...")
        for batch_path in batch_reconstruction_paths:
            parent_dir = os.path.dirname(batch_path) # Gets to batch_X
            if os.path.exists(parent_dir):
                shutil.rmtree(parent_dir)
                print(f"Removed: {parent_dir}")
    # Clean up merged_stages if desired
    if args.cleanup_merge_stages and 'merged_dir' in locals() and os.path.exists(merged_dir):
        print("Cleaning up intermediate merge stage directories...")
        shutil.rmtree(merged_dir)
        print(f"Removed: {merged_dir}")


    print("\nProcessing finished.")
    return True


args = argparse.Namespace(
    variant="train",
    dataset="phototourism", # Example, change as needed
    scene="st_pauls_cathedral",     # Example, change as needed
    seed=42,
    
    batch_size=10, # New: Number of images per batch
    batch_overlap=5, # New: Number of overlapping images between consecutive batches
    
    use_ba_per_batch=True, # Original 'use_ba', now for per-batch BA
    max_reproj_error=8.0,
    shared_camera=False, # For per-batch BA
    camera_type="SIMPLE_PINHOLE", # For per-batch BA. COLMAP merger works best if cameras are consistent or can be resolved.
    vis_thresh=0.2,
    query_frame_num=5, # Per batch
    max_query_pts=2048, # Per batch
    fine_tracking=False, # Per batch
    
    use_global_ba=True, # New: Whether to perform BA after merging all batches
    
    cleanup_batch_dirs=False, # New: Remove individual batch reconstruction folders after merging
    cleanup_merge_stages=False # New: Remove intermediate merged_step_* folders
)
args.dataset = 'haiper'
args.scene = 'bike'


with torch.no_grad():
    demo_fn_batched(args)


# print("\n--- Merging Batch Reconstructions ---")
# scene_output_dir = "/kaggle/working/train/phototourism/st_pauls_cathedral"
# merged_dir = os.path.join(scene_output_dir, "merged_stages")
# os.makedirs(merged_dir, exist_ok=True)
# batch_reconstruction_paths = [f"/kaggle/working/train/phototourism/st_pauls_cathedral/batch_{i}/sparse" for i in range(1, 16)]
# current_merged_path = batch_reconstruction_paths[0]

# for i in range(1, len(batch_reconstruction_paths)):
#     input_path1 = current_merged_path
#     input_path2 = batch_reconstruction_paths[i]
#     output_path_temp = os.path.join(merged_dir, f"merged_step_{i-1}_to_{i}")
#     if os.path.exists(output_path_temp): # Clean up previous attempt for this step
#         shutil.rmtree(output_path_temp)
#     os.makedirs(output_path_temp, exist_ok=True)

#     print(f"Merging: '{os.path.basename(os.path.dirname(input_path1))}' and '{os.path.basename(os.path.dirname(input_path2))}' into '{os.path.basename(output_path_temp)}'")
    
#     colmap_executable = "colmap" # Assumes it's in PATH after dpkg -i
    
#     merge_cmd = [
#         colmap_executable, "model_merger",
#         "--input_path1", input_path1,
#         "--input_path2", input_path2,
#         "--output_path", output_path_temp,
#         # Add other model_merger options if needed, e.g.:
#         # "--robust_merge_max_reproj_error", "16", # Default is 8
#         # "--robust_merge_min_inlier_ratio", "0.05", # Default is 0.1
#     ]
#     print(f"Executing: {' '.join(merge_cmd)}")
#     try:
#         completed_process = subprocess.run(merge_cmd, check=True, capture_output=True, text=True)
#         print("Merge stdout:", completed_process.stdout)
#         print("Merge successful for this step.")
#         current_merged_path = output_path_temp
#     except subprocess.CalledProcessError as e:
#         print(f"Error during model merging step {i}:")
#         print("Command:", e.cmd)
#         print("Return code:", e.returncode)
#         print("Stdout:", e.stdout)
#         print("Stderr:", e.stderr)
#         print(f"Failed to merge {input_path2}. Trying to continue with previous merged model if possible, or stopping.")
#         # Decide on error strategy: stop, or try to merge next available into current_merged_path
#         # For now, if a merge fails, the chain breaks.
#         # A more robust strategy might try to merge batch_reconstruction_paths[i+1] into current_merged_path

# merged_model_path = current_merged_path


# Cell to run after your main script/function call

# 1. Delete any global variables that might still hold GPU tensors
#    (Adapt this list based on what might be left in the global scope)
print("Deleting global variables that might hold GPU tensors...")
vars_to_delete = ['model', 'images_global_ref', 'optimizer_global', 'dino_v2_model', 'tracker'] # Example variable names
for var_name in vars_to_delete:
    if var_name in globals():
        del globals()[var_name]
        print(f"Deleted global variable: {var_name}")

# 2. Run garbage collection
import gc
print("Running garbage collection...")
gc.collect()

# 3. Empty PyTorch CUDA cache
import torch
if torch.cuda.is_available():
    print("Emptying CUDA cache...")
    torch.cuda.empty_cache()
    print("CUDA cache emptied.")
else:
    print("CUDA not available, skipping cache empty.")

print("GPU memory release steps attempted.")

# You can check memory usage now using:
# !nvidia-smi




