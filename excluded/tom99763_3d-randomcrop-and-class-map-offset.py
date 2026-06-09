import torch
import random


def random_crop_patch_around_target(volume, target_coord, patch_size):
    """
    Randomly crop a 3D patch ensuring the target is inside.

    Args:
        volume (torch.Tensor): Input volume of shape (C, D_full, H_full, W_full).
        target_coord (tuple): (z, y, x) coordinates of the object in the full volume.
        patch_size (tuple): (D, H, W) size of the patch.

    Returns:
        cropped_patch (torch.Tensor): Cropped volume of shape (C, D, H, W).
        crop_start (tuple): (z_start, y_start, x_start) of the cropped patch.
    """
    C, D_full, H_full, W_full = volume.shape
    D, H, W = patch_size
    z_target, y_target, x_target = target_coord

    # Define valid range for cropping
    z_min = max(0, z_target - D)  
    z_max = min(D_full - D, z_target)  
    y_min = max(0, y_target - H)  
    y_max = min(H_full - H, y_target)  
    x_min = max(0, x_target - W)  
    x_max = min(W_full - W, x_target)  

    # Randomly select a valid crop start position
    z_start = random.randint(z_min, z_max)
    y_start = random.randint(y_min, y_max)
    x_start = random.randint(x_min, x_max)

    # Crop the patch
    cropped_patch = volume[:, z_start:z_start + D, y_start:y_start + H, x_start:x_start + W]

    return cropped_patch, (z_start, y_start, x_start)
    

def generate_3d_labels(global_target, roi_start, roi_size, stride=2):
    """
    Generate class_map and offset_map for 3D object detection.

    Args:
        global_target (tuple): (z, y, x) coordinate in the full volume.
        roi_start (tuple): (z_start, y_start, x_start) of the extracted patch in full volume.
        roi_size (tuple): (depth, height, width) of the extracted patch.
        class_map_size (tuple): (depth, height, width) of the class_map output.
        stride (int): The stride factor, default is 2.

    Returns:
        class_map (torch.Tensor): Binary tensor of shape class_map_size.
        offset_map (torch.Tensor): Offset tensor of shape (3, *class_map_size).
    """
    #roi coordinate space
    z_local = global_target[0] - roi_start[0]
    y_local = global_target[1] - roi_start[1]
    x_local = global_target[2] - roi_start[2]

    #shrinkage coordinate
    z_shrink = z_local//stride
    y_shrink = y_local//stride
    x_shrink = x_local//stride
    
    #relative coordinate 
    z_ratio = z_local/roi_size[0]
    y_ratio = y_local/roi_size[1]
    x_ratio = x_local/roi_size[2]

    #
    D, H, W = roi_size[0]//2, roi_size[1]//2, roi_size[2]//2
    class_map = torch.zeros(1, D, H, W)
    offset_map = torch.zeros(3, D, H, W)

    for dz in range(2):
        for dy in range(2):
            for dx in range(2):
                z_idx = z_shrink + dz
                y_idx = y_shrink + dy
                x_idx = x_shrink + dx
                if z_idx<D and y_idx<H and x_idx<W:
                    class_map[:, z_idx, y_idx, x_idx] = 1
                    offset_map[0, z_idx, y_idx, x_idx] = z_ratio
                    offset_map[1, z_idx, y_idx, x_idx] = y_ratio
                    offset_map[2, z_idx, y_idx, x_idx] = x_ratio
    return class_map, offset_map



# Example usage
volume = torch.randn(1, 184, 630, 630)  # Simulated 3D volume (C, D, H, W)
target_coord = (162, 23, 224)  # Object position in full volume
roi_size = (96, 128, 128)  # Desired patch size

# Step 1: Randomly crop a patch ensuring the object is inside
cropped_patch, roi_start = random_crop_patch_around_target(volume, target_coord, roi_size)

class_map, offset_map = generate_3d_labels(target_coord, roi_start, roi_size)


print("Class Map Shape:", class_map.shape)
print("Offset Map Shape:", offset_map.shape)
print("Class Map Non-Zero Indices:", torch.nonzero(class_map))
print("Offset Values at Class Map Locations:", offset_map[class_map.bool().repeat(3, 1, 1, 1)])


class_map.shape, offset_map.shape


def recover_original_coordinate(class_map, offset_map, roi_start, roi_size, stride=2):
    """
    Recover the original global target coordinate from class_map and offset_map.

    Args:
        class_map (torch.Tensor): Binary tensor of shape (1, D, H, W) indicating object presence.
        offset_map (torch.Tensor): Offset tensor of shape (3, D, H, W).
        roi_start (tuple): (z_start, y_start, x_start) of the extracted patch in full volume.
        roi_size (tuple): (depth, height, width) of the extracted patch.
        stride (int): The stride factor, default is 2.

    Returns:
        recovered_target (tuple): (z, y, x) coordinate in the full volume.
    """
    # Get indices where class_map == 1
    indices = torch.nonzero(class_map.squeeze(), as_tuple=True)
    coords = []
    # Take the first detected point (assuming 1 target object)
    for z_idx, y_idx, x_idx in zip(*indices):
        # Retrieve corresponding offsets
        z_offset = offset_map[0, z_idx, y_idx, x_idx].item()
        y_offset = offset_map[1, z_idx, y_idx, x_idx].item()
        x_offset = offset_map[2, z_idx, y_idx, x_idx].item()
        # Convert back to local coordinates within the ROI
        z_local = (z_idx + z_offset) * stride
        y_local = (y_idx + y_offset) * stride
        x_local = (x_idx + x_offset) * stride
    
        # Convert back to global coordinates
        z_global = int(z_local + roi_start[0])
        y_global = int(y_local + roi_start[1])
        x_global = int(x_local + roi_start[2])
        coords.append((z_global, y_global, x_global))
    return coords


# Recover the target coordinate
recovered_coord = recover_original_coordinate(class_map, offset_map, roi_start, roi_size)
print("Recovered Target Coordinate:", recovered_coord)




