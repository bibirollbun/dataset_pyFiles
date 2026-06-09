import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# Configure font settings for Matplotlib
rcParams['font.sans-serif'] = ['SimHei']  
rcParams['axes.unicode_minus'] = False

"""Create standard color checker template with physical dimensions"""

# Set physical dimensions (60mm x 15mm)
physical_width_mm = 60
physical_height_mm = 15

# Set resolution (pixels per mm)
pixels_per_mm = 10  # 10 pixels/mm for sufficient resolution

# Calculate pixel dimensions
template_width = int(physical_width_mm * pixels_per_mm)  # 600 pixels
template_height = int(physical_height_mm * pixels_per_mm)  # 150 pixels

# Create canvas
template = np.zeros((template_height, template_width, 3), dtype=np.uint8)

# First row (0-5mm): ruler scale
first_row_height = int(5 * pixels_per_mm)  # 50 pixels

# Draw ruler background (white)
template[:first_row_height, :] = [255, 255, 255]

# Draw scale marks (main mark every 10mm, minor mark every 1mm)
for mm in range(0, physical_width_mm + 1):
    x_pixel = int(mm * pixels_per_mm)
    if x_pixel <= template_width:
        if mm % 10 == 0 and mm > 0:  # Main mark (every 10mm, skip 0)
            # Long mark line
            cv2.line(template, (x_pixel, 0), (x_pixel, first_row_height), (0, 0, 0), 2)
            # Add number label (1-6)
            label_num = mm // 10
            if label_num <= 6:
                cv2.putText(template, str(label_num), (x_pixel-10, first_row_height-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        else:  # Minor mark (every 1mm)
            # Short mark line
            cv2.line(template, (x_pixel, 0), (x_pixel, first_row_height//2), (0, 0, 0), 1)

# Second row (5-10mm): black and white squares
second_row_start = first_row_height
second_row_height = int(5 * pixels_per_mm)  # 50 pixels
second_row_end = second_row_start + second_row_height

# Each square width 10mm
square_width_mm = 10
square_width_pixels = int(square_width_mm * pixels_per_mm)  # 100 pixels

for i in range(6):  # 6 squares
    x1 = i * square_width_pixels
    x2 = (i + 1) * square_width_pixels
    if x2 <= template_width:
        if i % 2 == 0:
            template[second_row_start:second_row_end, x1:x2] = [0, 0, 0]  # Black
        else:
            template[second_row_start:second_row_end, x1:x2] = [255, 255, 255]  # White
        
        # Add thin border
        cv2.rectangle(template, (x1, second_row_start), (x2, second_row_end), (0, 0, 0), 1)

# Third row (10-15mm): grayscale and color patches
third_row_start = second_row_end
third_row_height = int(5 * pixels_per_mm)  # 50 pixels

# Grayscale: 6 steps, each 5mm x 5mm
gray_patch_width_mm = 5
gray_patch_width_pixels = int(gray_patch_width_mm * pixels_per_mm)  # 50 pixels

# 6-step grayscale: from white to black
gray_values = [255, 204, 153, 102, 51, 0]

for i in range(6):
    x1 = i * gray_patch_width_pixels
    x2 = (i + 1) * gray_patch_width_pixels
    if x2 <= template_width:
        gray_value = gray_values[i]
        template[third_row_start:third_row_start+third_row_height, x1:x2] = [gray_value, gray_value, gray_value]
        
        # Add thin border
        cv2.rectangle(template, (x1, third_row_start), (x2, third_row_start+third_row_height), (0, 0, 0), 1)

# Color patches: 6 colors, each 5mm x 5mm
color_patch_width_mm = 5
color_patch_width_pixels = int(color_patch_width_mm * pixels_per_mm)  # 50 pixels

# 6 colors: Cyan, Magenta, Yellow, Red, Green, Blue
colors = [
    [0, 180, 230],    # Cyan (C)
    [240, 0, 120],    # Magenta (M)
    [240, 230, 0],    # Yellow (Y)
    [230, 0, 50],     # Red (R)
    [0, 200, 150],    # Green (G)
    [80, 60, 180]     # Blue (B)
]

for i in range(6):
    x1 = (i + 6) * color_patch_width_pixels  # Start from 7th position
    x2 = (i + 7) * color_patch_width_pixels
    if x2 <= template_width:
        template[third_row_start:third_row_start+third_row_height, x1:x2] = colors[i]
        
        # Add thin border
        cv2.rectangle(template, (x1, third_row_start), (x2, third_row_start+third_row_height), (0, 0, 0), 1)

# Save templates in different formats
# PNG format (recommended)
cv2.imwrite('standard_color_checker_60x15mm.png', cv2.cvtColor(template, cv2.COLOR_RGB2BGR))

# High resolution version (2x resolution)
high_res_template = cv2.resize(template, (template_width*2, template_height*2))
cv2.imwrite('standard_color_checker_60x15mm_high_res.png', cv2.cvtColor(high_res_template, cv2.COLOR_RGB2BGR))

# Display template
#plt.figure(figsize=(15, 8))
plt.imshow(template)
plt.title('Standard Color Checker Template (60mm x 15mm 10 pixels/mm)')
plt.axis('off')
plt.show()



import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def rotate_and_crop_image(image_path, rotation_angle=-0.4, crop_coords=None):
    """
    Rotate and crop image / å¯¹å›¾ç‰‡è¿›è¡Œæ—‹è½¬å’Œè£�å‰ª
    
    Parameters / å�‚æ•°:
    image_path: image path / å›¾ç‰‡è·¯å¾„
    rotation_angle: rotation angle in degrees / æ—‹è½¬è§’åº¦ï¼ˆåº¦ï¼‰
    crop_coords: crop coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y) / è£�å‰ªå��æ ‡ (å·¦ä¸Šx, å·¦ä¸Šy, å�³ä¸‹x, å�³ä¸‹y)
    """
    # Use PIL to read image to handle Chinese path / ä½¿ç”¨PILè¯»å�–å›¾ç‰‡ä»¥å¤„ç�†ä¸­æ–‡è·¯å¾„
    try:
        pil_image = Image.open(image_path)
        # Convert to OpenCV format / è½¬æ�¢ä¸ºOpenCVæ ¼å¼�
        image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    except Exception as e:
        print(f"Cannot read image / æ— æ³•è¯»å�–å›¾ç‰‡: {image_path}")
        print(f"Error message / é”™è¯¯ä¿¡æ�¯: {e}")
        return None
    
    # Get image dimensions / è�·å�–å›¾ç‰‡å°ºå¯¸
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate rotation matrix / è®¡ç®—æ—‹è½¬çŸ©é˜µ
    rotation_matrix = cv2.getRotationMatrix2D(center, rotation_angle, 1.0)
    
    # Perform rotation / æ‰§è¡Œæ—‹è½¬
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                  flags=cv2.INTER_LINEAR, 
                                  borderMode=cv2.BORDER_CONSTANT, 
                                  borderValue=(255, 255, 255))
    
    # If crop coordinates are specified, perform cropping / å¦‚æ�œæŒ‡å®šäº†è£�å‰ªå��æ ‡ï¼Œè¿›è¡Œè£�å‰ª
    if crop_coords:
        x1, y1, x2, y2 = crop_coords
        cropped_image = rotated_image[y1:y2, x1:x2]
        return cropped_image
    
    return rotated_image

def save_image_cv2(image, output_path):
    """Save image using OpenCV / ä½¿ç”¨OpenCVä¿�å­˜å›¾ç‰‡"""
    try:
        success = cv2.imwrite(output_path, image)
        if success:
            print(f"Image saved to / å›¾ç‰‡å·²ä¿�å­˜åˆ°: {output_path}")
            return True
        else:
            print(f"Save failed / ä¿�å­˜å¤±è´¥: {output_path}")
            return False
    except Exception as e:
        print(f"Error during saving / ä¿�å­˜æ—¶å‡ºé”™: {e}")
        return False

def save_image_pil(image, output_path):
    """Save image using PIL / ä½¿ç”¨PILä¿�å­˜å›¾ç‰‡"""
    try:
        # Convert back to PIL format / è½¬æ�¢å›�PILæ ¼å¼�
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        pil_image.save(output_path)
        print(f"Image saved to / å›¾ç‰‡å·²ä¿�å­˜åˆ°: {output_path}")
        return True
    except Exception as e:
        print(f"Error during saving / ä¿�å­˜æ—¶å‡ºé”™: {e}")
        return False





# Image path / å›¾ç‰‡è·¯å¾„
image_path = "/kaggle/input/h690/h690/sherd_images/JD00001_exterior.jpg"

# Crop coordinates (top_left_x, top_left_y, bottom_right_x, bottom_right_y) / è£�å‰ªå��æ ‡ (å·¦ä¸Šx, å·¦ä¸Šy, å�³ä¸‹x, å�³ä¸‹y)
crop_coords = (280, 780, 742, 908)

# Perform rotation and cropping / æ‰§è¡Œæ—‹è½¬å’Œè£�å‰ª
result = rotate_and_crop_image(image_path, rotation_angle=-0.4, crop_coords=crop_coords)

if result is not None:
    # Save result / ä¿�å­˜ç»“æ�œ
    output_path = f"/kaggle/working/cropped_checker_simple_{crop_coords[0]}_{crop_coords[1]}_{crop_coords[2]}_{crop_coords[3]}.jpg"
    
    # Try to save with PIL first / å°�è¯•ä½¿ç”¨PILä¿�å­˜
    if not save_image_pil(result, output_path):
        # If PIL save fails, try OpenCV / å¦‚æ�œPILä¿�å­˜å¤±è´¥ï¼Œå°�è¯•ä½¿ç”¨OpenCV
        save_image_cv2(result, output_path)
    
    print(f"Crop region / è£�å‰ªåŒºåŸŸ: top_left({crop_coords[0]}, {crop_coords[1]}) bottom_right({crop_coords[2]}, {crop_coords[3]})")
    print(f"Rotation angle / æ—‹è½¬è§’åº¦: {-0.4} degrees / åº¦")
    print(f"Result image size / ç»“æ�œå›¾ç‰‡å°ºå¯¸: {result.shape[1]} x {result.shape[0]}")
    
    # Display result / æ˜¾ç¤ºç»“æ�œ
    plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
    plt.title('Cropped Color Checker')
    plt.axis('off')
    plt.show()

else:
    print("Processing failed! / å¤„ç�†å¤±è´¥ï¼�")




