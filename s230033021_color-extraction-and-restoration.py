!apt-get -qq install fonts-wqy-zenhei  # å®‰è£…æ–‡æ³‰é©¿å­—ä½“ï¼ˆKaggleçš„Linuxç�¯å¢ƒï¼‰


import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib import rcParams
from scipy import ndimage
from sklearn.linear_model import LinearRegression
import matplotlib.font_manager as fm
import shutil
import os

# 1. Kaggleä¸­ï¼šç¡®ä¿�å­—ä½“æ–‡ä»¶å­˜åœ¨
font_path = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
assert os.path.exists(font_path), "å­—ä½“æ–‡ä»¶ä¸�å­˜åœ¨ï¼�"
# 2. Kaggleä¸­ï¼šæ‰‹åŠ¨æ³¨å†Œå­—ä½“ï¼ˆé�¿å…�ç¼“å­˜é—®é¢˜ï¼‰
fm.fontManager.addfont(font_path)  # å¼ºåˆ¶æ·»åŠ åˆ°å­—ä½“ç®¡ç�†å™¨
font_name = fm.FontProperties(fname=font_path).get_name()
# 3. Kaggleä¸­ï¼šå…¨å±€è®¾ç½®å­—ä½“
plt.rcParams['font.family'] = font_name
plt.rcParams['axes.unicode_minus'] = False  # ä¿®å¤�è´Ÿå�·æ˜¾ç¤º

# æœ¬åœ°è¿�è¡Œï¼šé…�ç½®ä¸­æ–‡å­—ä½“ï¼ˆWindows ä½¿ç”¨ SimHeiï¼ŒMac å�¯æ”¹ä¸º STHeitiï¼ŒLinux å�¯æ”¹ä¸º Noto Sans CJKï¼‰
# rcParams['font.sans-serif'] = ['Noto Sans CJK']  
# rcParams['axes.unicode_minus'] = False


def load_standard_template():
    """Load standard color checker template / åŠ è½½æ ‡å‡†è‰²å�¡æ¨¡æ�¿"""
    template_path = '/kaggle/input/standard-color-checker-generator/cropped_checker_simple_280_780_742_908.jpg'
    try:
        template = cv2.imread(template_path)
        if template is None:
            print(f"â�Œ Failed to load standard color checker template: File not found or corrupted / æ— æ³•åŠ è½½æ ‡å‡†è‰²å�¡æ¨¡æ�¿: æ–‡ä»¶ä¸�å­˜åœ¨æˆ–æ�Ÿå��")
            return None
        template_rgb = cv2.cvtColor(template, cv2.COLOR_BGR2RGB)
        # print(f"âœ… Successfully loaded standard color checker template: {template_rgb.shape} / æˆ�åŠŸåŠ è½½æ ‡å‡†è‰²å�¡æ¨¡æ�¿: {template_rgb.shape}")
        return template_rgb
    except Exception as e:
        print(f"â�Œ Failed to load standard color checker template: {e} / æ— æ³•åŠ è½½æ ‡å‡†è‰²å�¡æ¨¡æ�¿: {e}")
        return None

def detect_color_checker_by_template(image_rgb, template_rgb):
    """Detect color checker region using template matching / ä½¿ç”¨æ¨¡æ�¿åŒ¹é…�æ£€æµ‹è‰²å�¡åŒºåŸŸ"""
    # Convert to grayscale for template matching / è½¬æ�¢ä¸ºç�°åº¦å›¾è¿›è¡Œæ¨¡æ�¿åŒ¹é…�
    image_gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    template_gray = cv2.cvtColor(template_rgb, cv2.COLOR_RGB2GRAY)
    
    # Multi-scale template matching / å¤šå°ºåº¦æ¨¡æ�¿åŒ¹é…�
    scales = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]
    best_match = None
    best_score = 0
    
    for scale in scales:
        # Scale template / ç¼©æ”¾æ¨¡æ�¿
        width = int(template_gray.shape[1] * scale)
        height = int(template_gray.shape[0] * scale)
        
        # Ensure scaled template isn't larger than image / ç¡®ä¿�ç¼©æ”¾å��çš„æ¨¡æ�¿ä¸�å¤§äº�å›¾åƒ�
        if width > image_gray.shape[1] or height > image_gray.shape[0]:
            continue
            
        resized_template = cv2.resize(template_gray, (width, height))
        
        # Template matching / æ¨¡æ�¿åŒ¹é…�
        result = cv2.matchTemplate(image_gray, resized_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        if max_val > best_score:
            best_score = max_val
            best_match = {
                'location': max_loc,
                'scale': scale,
                'template_size': (width, height),
                'score': max_val
            }
    
    if best_match and best_match['score'] > 0.4:  # Increased threshold / æ��é«˜é˜ˆå€¼
        x, y = best_match['location']
        w, h = best_match['template_size']
        
        # Extend by 15 pixels in all directions / å�‘å››å‘¨æ‰©å±•15åƒ�ç´ 
        extend_px = 15
        x_extended = max(0, x - extend_px)
        y_extended = max(0, y - extend_px)
        w_extended = min(image_rgb.shape[1] - x_extended, w + 2 * extend_px)
        h_extended = min(image_rgb.shape[0] - y_extended, h + 2 * extend_px)
        
        return image_rgb[y_extended:y_extended+h_extended, x_extended:x_extended+w_extended], (x_extended, y_extended, w_extended, h_extended), best_match['score']
    
    return None, None, 0

def extract_checker_contour_detailed(checker_region):
    """
    Extract precise outer contour from color checker region - enhanced method / ä»�è‰²å�¡åŒºåŸŸæ��å�–ç²¾ç¡®çš„å¤–è¾¹æ¡†è½®å»“ - å¼ºåŒ–æ–¹æ³•
    """
    # print("ğŸ”� Starting detailed contour extraction (enhanced method)... / å¼€å§‹è¯¦ç»†è½®å»“æ��å�–ï¼ˆå¼ºåŒ–æ–¹æ³•ï¼‰...")
    
    # Step 1: Convert to grayscale / æ­¥éª¤1: è½¬æ�¢ä¸ºç�°åº¦å›¾
    gray = cv2.cvtColor(checker_region, cv2.COLOR_RGB2GRAY)
    
    # Step 2: Gaussian blur for noise reduction / æ­¥éª¤2: é«˜æ–¯æ¨¡ç³Šå�»å™ª
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Step 3: Edge detection (Canny) / æ­¥éª¤3: è¾¹ç¼˜æ£€æµ‹ï¼ˆCannyï¼‰
    edges = cv2.Canny(blurred, 50, 150)
    
    # Step 4: Morphological operations to enhance edges / æ­¥éª¤4: å½¢æ€�å­¦æ“�ä½œå¢�å¼ºè¾¹ç¼˜
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)) 
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Step 5: Create protected border region (expanded) / æ­¥éª¤5: åˆ›å»ºè¾¹ç•Œä¿�æŠ¤åŒºåŸŸï¼ˆæ‰©å¤§ä¿�æŠ¤åŒºåŸŸï¼‰
    protected = dilated.copy()
    border_width = 5  # Increased border width / å¢�åŠ ä¿�æŠ¤è¾¹ç•Œå®½åº¦
    height, width = protected.shape
    
    # Create color version for visualizing border region / åˆ›å»ºå½©è‰²ç‰ˆæœ¬ç”¨äº�å�¯è§†åŒ–è¾¹ç•ŒåŒºåŸŸ
    protected_color = cv2.cvtColor(protected, cv2.COLOR_GRAY2RGB)
    
    # Mark border protection area in blue / ç”¨è“�è‰²æ ‡è®°è¾¹ç•Œä¿�æŠ¤åŒºåŸŸ
    protected_color[0:border_width, :] = [0, 0, 255]  # Top border - blue / ä¸Šè¾¹ç•Œ - è“�è‰²
    protected_color[height-border_width:height, :] = [0, 0, 255]  # Bottom border - blue / ä¸‹è¾¹ç•Œ - è“�è‰²
    protected_color[:, 0:border_width] = [0, 0, 255]  # Left border - blue / å·¦è¾¹ç•Œ - è“�è‰²
    protected_color[:, width-border_width:width] = [0, 0, 255]  # Right border - blue / å�³è¾¹ç•Œ - è“�è‰²
    
    # Original binary version for contour detection / å�Ÿå§‹çš„äºŒå€¼åŒ–ç‰ˆæœ¬ç”¨äº�è½®å»“æ£€æµ‹
    protected[0:border_width, :] = 0  # Top border / ä¸Šè¾¹ç•Œ
    protected[height-border_width:height, :] = 0  # Bottom border / ä¸‹è¾¹ç•Œ
    protected[:, 0:border_width] = 0  # Left border / å·¦è¾¹ç•Œ
    protected[:, width-border_width:width] = 0  # Right border / å�³è¾¹ç•Œ
    
    # Step 6: Find contours - focus on internal contours only / æ­¥éª¤6: æŸ¥æ‰¾è½®å»“ - å�ªå…³æ³¨å†…éƒ¨è½®å»“
    contours, _ = cv2.findContours(protected, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        print("â�Œ No contours detected / æœªæ£€æµ‹åˆ°è½®å»“")
        return None, None, {
            'edges': edges,
            'dilated': dilated,
            'protected': protected
        }
    
    # Find largest internal contour / æ‰¾åˆ°æœ€å¤§çš„å†…éƒ¨è½®å»“
    valid_contours = []
    min_area = (height * width) * 0.05  # Contour must cover at least 5% area / è½®å»“å¿…é¡»è‡³å°‘å� 5%çš„é�¢ç§¯
    max_area = (height * width) * 0.95  # Maximum 95% / æœ€å¤§95%
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_area < area < max_area:
            valid_contours.append(contour)
    
    if not valid_contours:
        print("â�Œ No qualified internal contours found / æœªæ‰¾åˆ°ç¬¦å�ˆæ�¡ä»¶çš„å†…éƒ¨è½®å»“")
        return None, None, {
            'edges': edges,
            'dilated': dilated,
            'protected': protected,
            'contours': contours
        }
    
    # Select contour with largest area / é€‰æ‹©é�¢ç§¯æœ€å¤§çš„è½®å»“
    largest_contour = max(valid_contours, key=cv2.contourArea)
    
    # Approximate contour to get four corners / è½®å»“è¿‘ä¼¼ï¼Œè�·å�–å››ä¸ªè§’ç‚¹
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    # Adjust approximation until getting 4 points / è°ƒæ•´è½®å»“è¿‘ä¼¼ç›´åˆ°è�·å¾—4ä¸ªç‚¹
    if len(approx) != 4:
        print(f"âš ï¸� Incorrect number of contour corners: {len(approx)} != 4, trying to adjust... / è½®å»“è§’ç‚¹æ•°ä¸�æ­£ç¡®: {len(approx)} != 4ï¼Œå°�è¯•è°ƒæ•´...")
        for factor in [0.01, 0.03, 0.05, 0.08, 0.1]:
            epsilon = factor * cv2.arcLength(largest_contour, True)
            approx = cv2.approxPolyDP(largest_contour, epsilon, True)
            if len(approx) == 4:
                print(f"âœ… Successfully obtained 4 corners using epsilon factor {factor} / ä½¿ç”¨epsilonå› å­� {factor} æˆ�åŠŸè�·å¾—4ä¸ªè§’ç‚¹")
                break
            elif len(approx) < 4:
                break
    
    # Try minimum area rectangle / å°�è¯•æœ€å°�å¤–æ�¥çŸ©å½¢
    if len(approx) != 4:
        print("ğŸ”„ Trying minimum area rectangle... / å°�è¯•ä½¿ç”¨æœ€å°�å¤–æ�¥çŸ©å½¢...")
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        approx = box.reshape(-1, 1, 2).astype(np.int32)
        if len(approx) == 4:
            print("âœ… Obtained 4 corners using minimum area rectangle / ä½¿ç”¨æœ€å°�å¤–æ�¥çŸ©å½¢è�·å¾—4ä¸ªè§’ç‚¹")
    
    if len(approx) != 4:
        print(f"â�Œ Failed to get 4 corners: {len(approx)} / æ— æ³•è�·å¾—4ä¸ªè§’ç‚¹: {len(approx)}")
        return None, None, {
            'edges': edges,
            'dilated': dilated,
            'protected': protected,
            'contours': contours,
            'largest_contour': largest_contour,
            'approx': approx
        }
    
    # Order points as top-left, top-right, bottom-right, bottom-left / æŒ‰å·¦ä¸Šã€�å�³ä¸Šã€�å�³ä¸‹ã€�å·¦ä¸‹çš„é¡ºåº�æ�’åˆ—è§’ç‚¹
    corners = order_points(approx.reshape(4, 2))
    
    # Create contour visualization image / åˆ›å»ºè½®å»“å�¯è§†åŒ–å›¾åƒ�
    contour_vis = checker_region.copy()
    cv2.drawContours(contour_vis, [largest_contour], -1, (0, 255, 0), 2)
    
    # Draw corner points / ç»˜åˆ¶è§’ç‚¹
    for i, corner in enumerate(corners):
        cv2.circle(contour_vis, tuple(corner.astype(int)), 5, (255, 0, 0), -1)
        cv2.putText(contour_vis, str(i+1), tuple(corner.astype(int) + [10, 10]), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return largest_contour, corners, {
        'edges': edges,
        'dilated': dilated,
        'protected': protected,
        'protected_color': protected_color,
        'contours': contours,
        'largest_contour': largest_contour,
        'approx': approx,
        'contour_vis': contour_vis
    }

def order_points(pts):
    """
    Order four corner points as top-left, top-right, bottom-right, bottom-left / å°†å››ä¸ªè§’ç‚¹æŒ‰å·¦ä¸Šã€�å�³ä¸Šã€�å�³ä¸‹ã€�å·¦ä¸‹çš„é¡ºåº�æ�’åˆ—
    """
    # Initialize coordinate list / åˆ�å§‹åŒ–å��æ ‡åˆ—è¡¨
    rect = np.zeros((4, 2), dtype="float32")
    
    # Top-left point will have smallest sum, bottom-right will have largest / å·¦ä¸Šè§’ç‚¹å°†å…·æœ‰æœ€å°�çš„å’Œï¼Œå�³ä¸‹è§’ç‚¹å°†å…·æœ‰æœ€å¤§çš„å’Œ
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # Top-left / å·¦ä¸Š
    rect[2] = pts[np.argmax(s)]  # Bottom-right / å�³ä¸‹
    
    # Calculate difference between points, top-right will have smallest difference, bottom-left will have largest / è®¡ç®—ç‚¹ä¹‹é—´çš„å·®å€¼ï¼Œå�³ä¸Šè§’ç‚¹å°†å…·æœ‰æœ€å°�çš„å·®å€¼ï¼Œå·¦ä¸‹è§’ç‚¹å°†å…·æœ‰æœ€å¤§çš„å·®å€¼
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # Top-right / å�³ä¸Š
    rect[3] = pts[np.argmax(diff)]  # Bottom-left / å·¦ä¸‹
    
    return rect

def calculate_rotation_angle(corners):
    """
    Calculate rotation angle based on color checker contour / æ ¹æ�®è‰²å�¡è½®å»“è®¡ç®—æ—‹è½¬è§’åº¦
    
    Parameters / å�‚æ•°:
    corners: Four corner coordinates / å››ä¸ªè§’ç‚¹å��æ ‡
    
    Returns / è¿”å›�:
    angle: Rotation angle in degrees / æ—‹è½¬è§’åº¦ï¼ˆåº¦ï¼‰
    """
    # Calculate angle of top edge / è®¡ç®—ä¸Šè¾¹çº¿çš„è§’åº¦
    top_left = corners[0]
    top_right = corners[1]
    
    # Calculate vector of top edge / è®¡ç®—ä¸Šè¾¹çº¿çš„å�‘é‡�
    dx = top_right[0] - top_left[0]
    dy = top_right[1] - top_left[1]
    
    # Calculate angle (radians) / è®¡ç®—è§’åº¦ï¼ˆå¼§åº¦ï¼‰
    angle_rad = np.arctan2(dy, dx)
    
    # Convert to degrees / è½¬æ�¢ä¸ºåº¦
    angle_deg = np.degrees(angle_rad)
    
    # Limit angle between -90 and 90 degrees / å°†è§’åº¦é™�åˆ¶åœ¨-90åˆ°90åº¦ä¹‹é—´
    if angle_deg > 90:
        angle_deg -= 180
    elif angle_deg < -90:
        angle_deg += 180
    
    print(f"ğŸ“� Calculated rotation angle: {angle_deg:.2f}Â° / è®¡ç®—å¾—åˆ°æ—‹è½¬è§’åº¦: {angle_deg:.2f}Â°")
    return angle_deg

def get_line_equation(corners):
    """
    Get equation of top edge of color checker y = kx + b / è�·å�–è‰²å�¡ä¸Šè¾¹çº¿çš„æ–¹ç¨‹ y = kx + b
    
    Parameters / å�‚æ•°:
    corners: Four corner coordinates / å››ä¸ªè§’ç‚¹å��æ ‡
    
    Returns / è¿”å›�:
    k: Slope / æ–œç�‡
    b: Intercept / æˆªè·�
    """
    top_left = corners[0]
    top_right = corners[1]
    
    # Calculate slope / è®¡ç®—æ–œç�‡
    if top_right[0] - top_left[0] != 0:
        k = (top_right[1] - top_left[1]) / (top_right[0] - top_left[0])
    else:
        k = float('inf')  # Vertical line / å�‚ç›´çº¿
    
    # Calculate intercept / è®¡ç®—æˆªè·�
    b = top_left[1] - k * top_left[0]
    
    return k, b



def extract_color_patches_precise(checker_region, template_rgb):
    """
    Precisely extract color patches using 3x12 grid layout with center alignment
    ç²¾ç¡®æ��å�–é¢œè‰²å�— - ä½¿ç”¨3è¡Œ12åˆ—ç½‘æ ¼å¸ƒå±€ï¼Œä¸­å¿ƒå¯¹é½�
    
    Grid layout description:
    1~12 cells are rulers (not processed)
    13,14,17,18,21,22 cells are black
    15,16,19,20,23,24 cells are white
    25~30 cells are 6-step grayscale (white, light gray, medium gray, dark gray, darker gray, black)
    31~36 cells are cyan, magenta, yellow, red, green, purple
    
    ç½‘æ ¼å¸ƒå±€è¯´æ˜�:
    1~12æ ¼ä¸ºæ ‡å°ºï¼Œä¸�å¤„ç�†
    13ã€�14ã€�17ã€�18ã€�21ã€�22æ ¼å­�æ˜¯é»‘è‰²
    15ã€�16ã€�19ã€�20ã€�23ã€�24æ ¼å­�æ˜¯ç™½è‰²
    25~30æ ¼å­�æ˜¯6é˜¶ç�°é˜¶ï¼ˆç™½è‰²ã€�æµ…ç�°ã€�ä¸­ç�°ã€�æ·±ç�°ã€�æš—ç�°ã€�é»‘è‰²ï¼‰
    31~36æ ¼å­�æ˜¯é�’è‰²ã€�å“�çº¢ã€�é»„è‰²ã€�çº¢è‰²ã€�ç»¿è‰²ã€�ç´«è‰²
    """
    h, w = checker_region.shape[:2]
    
    # 3 rows x 12 columns grid with center alignment / 3è¡Œ12åˆ—ç½‘æ ¼ï¼Œä¸­å¿ƒå¯¹é½�
    rows = 3
    cols = 12
    
    # Calculate total grid dimensions (minus margins) / è®¡ç®—æ ¼å­�åŒºåŸŸçš„æ€»å°ºå¯¸ï¼ˆå‡�å�»è¾¹è·�ï¼‰
    total_grid_height = h - 0
    total_grid_width = w - 0
    
    # Calculate individual cell size / è®¡ç®—æ¯�ä¸ªæ ¼å­�çš„å°ºå¯¸
    cell_height = total_grid_height // rows
    cell_width = total_grid_width // cols
    
    # Calculate grid starting position (center aligned) / è®¡ç®—æ ¼å­�åŒºåŸŸçš„èµ·å§‹ä½�ç½®ï¼ˆä¸­å¿ƒå¯¹é½�ï¼‰
    grid_start_x = (w - total_grid_width) // 2 
    grid_start_y = (h - total_grid_height) // 2
    
    
    # Store extracted colors / å­˜å‚¨æ��å�–çš„é¢œè‰²
    black_squares = []  # 13,14,17,18,21,22
    white_squares = []  # 15,16,19,20,23,24
    gray_patches = []   # 25-30
    color_patches = []  # 31-36
    
    # Define cell numbers / å®šä¹‰æ ¼å­�ç¼–å�·
    black_cells = [13, 14, 17, 18, 21, 22]
    white_cells = [15, 16, 19, 20, 23, 24]
    gray_cells = [25, 26, 27, 28, 29, 30]
    color_cells = [31, 32, 33, 34, 35, 36]
    
    # Data for table output / ç”¨äº�è¡¨æ ¼è¾“å‡ºçš„æ•°æ�®
    left_side_data = []  # Stores 13-24 cell data / å­˜å‚¨13-24æ ¼æ•°æ�®
    right_side_data = []  # Stores 25-36 cell data / å­˜å‚¨25-36æ ¼æ•°æ�®
    
    # Iterate through all cells / é��å�†æ‰€æœ‰æ ¼å­�
    for row in range(rows):
        for col in range(cols):
            cell_num = row * cols + col + 1  # Cell numbering starts at 1 / æ ¼å­�ç¼–å�·ä»�1å¼€å§‹
            
            # Calculate cell position (based on grid start position) / è®¡ç®—æ ¼å­�ä½�ç½®ï¼ˆåŸºäº�æ ¼å­�åŒºåŸŸèµ·å§‹ä½�ç½®ï¼‰
            x1 = int(grid_start_x + col * cell_width)
            y1 = int(grid_start_y + row * cell_height)
            x2 = int(grid_start_x + (col + 1) * cell_width)
            y2 = int(grid_start_y + (row + 1) * cell_height)
            
            # Extract cell region / æ��å�–æ ¼å­�åŒºåŸŸ
            cell = checker_region[y1:y2, x1:x2]
            
            # Skip ruler area (cells 1-12) / è·³è¿‡æ ‡å°ºåŒºåŸŸï¼ˆ1-12æ ¼ï¼‰
            if cell_num <= 12:
                continue
            
            # Remove edges, only take center region / å�»é™¤è¾¹ç¼˜ï¼Œå�ªå�–ä¸­å¿ƒåŒºåŸŸ
            margin = max(2, min(cell_width, cell_height) // 3)
            # margin: edge width, larger value removes more edges, smaller center region
            # 1. min(cell_width, cell_height) // 4: calculate 1/4 of smallest dimension
            # 2. max(2, ...): ensure minimum edge width of 2 pixels
            center_cell = cell[margin:-margin, margin:-margin]
            
            # Use median filter to remove noise (enhanced effect) / ä½¿ç”¨ä¸­å€¼æ»¤æ³¢å�»é™¤å™ªç‚¹ï¼ˆå¢�å¼ºæ•ˆæ�œï¼‰
            denoised_cell = cv2.medianBlur(center_cell, 5)  # Increased kernel size / å¢�å¤§æ ¸å¤§å°�
            median_color = np.median(denoised_cell, axis=(0, 1))
            
            # Classify and build table data / æ ¹æ�®æ ¼å­�ç¼–å�·åˆ†ç±»å¤„ç�†å¹¶æ�„å»ºè¡¨æ ¼æ•°æ�®
            cell_info = []
            if cell_num in black_cells:
                black_squares.append(median_color)
                cell_info = [cell_num, 'BK', median_color.astype(int)]
            elif cell_num in white_cells:
                white_squares.append(median_color)
                cell_info = [cell_num, 'WH', median_color.astype(int)]
            elif cell_num in gray_cells:
                gray_patches.append(median_color)
                cell_info = [cell_num, 'GY', median_color.astype(int)]
            elif cell_num in color_cells:
                color_patches.append(median_color)
                # Add type abbreviation for color cells / ä¸ºå½©è‰²æ ¼å­�æ·»åŠ ç±»å�‹ç¼©å†™
                color_types = {31: 'C', 32: 'M', 33: 'Y', 34: 'R', 35: 'G', 36: 'B'}
                cell_info = [cell_num, color_types[cell_num], median_color.astype(int)]
            
            # Add to table data / æ·»åŠ åˆ°è¡¨æ ¼æ•°æ�®ä¸­
            if cell_num <= 24 or cell_num >= 25:  # Only process cells 13-36 / å�ªå¤„ç�†13-36çš„æ ¼å­�
                if cell_num <= 24:
                    left_side_data.append(cell_info)
                else:
                    right_side_data.append(cell_info)

    print(f"ğŸ“� Grid area size / æ ¼å­�åŒºåŸŸå°ºå¯¸: {total_grid_width} x {total_grid_height} pixels")
    print(f"ğŸ“� Center Sampling Size / ä¸­å¿ƒå�–æ ·åŒºåŸŸå°ºå¯¸: {cell_width - 2 * margin} x {cell_height - 2 * margin} pixels")


    # Print table / æ‰“å�°è¡¨æ ¼
    print("ğŸ“Š Color Extraction Results / é¢œè‰²æ��å�–ç»“æ�œè¡¨æ ¼:")
    print("{:<8} {:<4} {:<15} {:<8} {:<4} {:<15}".format(
        "Cell No.", "Type", "Extracted RGB", "Cell No.", "Type", "Extracted RGB"))
    print("-"*60)
    
    # Merge left and right side data (left 13-24, right 25-36) / å�ˆå¹¶å·¦å�³ä¸¤ä¾§æ•°æ�®ï¼ˆå·¦ä¾§13-24ï¼Œå�³ä¾§25-36ï¼‰
    for left, right in zip(left_side_data, right_side_data):
        # Convert numpy array to normal list and remove np.int64 markers / è½¬æ�¢numpyæ•°ç»„ä¸ºæ™®é€šåˆ—è¡¨å¹¶å�»é™¤np.int64æ ‡è®°
        left_rgb = [int(x) for x in left[2]]
        right_rgb = [int(x) for x in right[2]]
        
        print("{:<8} {:<4} {:<15} {:<8} {:<4} {:<15}".format(
            left[0], left[1], str(left_rgb),
            right[0], right[1], str(right_rgb)
        ))
    
    # Combine black and white squares / å�ˆå¹¶é»‘ç™½æ–¹æ ¼
    squares = np.array(black_squares + white_squares)
    # Combine grayscale and color patches / å�ˆå¹¶ç�°é˜¶å’Œå½©è‰²æ�¡
    patches = np.array(gray_patches + color_patches)
    
    return squares, patches


# ====================== 1. Gray Balance Correction / 1. ç�°å¹³è¡¡çŸ«æ­£ ======================
def create_calibration_curve(original_values, target_values, degree=2):
    """Create polynomial gray balance correction curve / åˆ›å»ºå¤šé¡¹å¼�ç�°å¹³è¡¡çŸ«æ­£æ›²çº¿"""
    coeffs = np.polyfit(original_values, target_values, deg=degree)
    return lambda x: np.polyval(coeffs, x)

def gray_balance_correction(img, gray_patches, target_gray_values):
    """
    Main function for gray balance correction / ç�°å¹³è¡¡çŸ«æ­£ä¸»å‡½æ•°
    :param img: Input image (BGR format) / è¾“å…¥å›¾åƒ�(BGRæ ¼å¼�)
    :param gray_patches: List of grayscale patch samples / ç�°é˜¶è‰²å�¡é‡‡æ ·å€¼åˆ—è¡¨
    :param target_gray_values: List of grayscale target values / ç�°é˜¶ç›®æ ‡å€¼åˆ—è¡¨
    :return: Gray-balanced corrected image / ç�°å¹³è¡¡çŸ«æ­£å��çš„å›¾åƒ�
    """
    # Calculate ideal neutral gray target values for each grayscale point / è®¡ç®—æ¯�ä¸ªç�°é˜¶ç‚¹çš„ç�†æƒ³ä¸­æ€§ç�°ç›®æ ‡å€¼
    target_values = [np.mean(patch) for patch in target_gray_values]
    
    # Separate channel original values / åˆ†ç¦»é€šé�“å�Ÿå§‹å€¼
    r_orig = [p[0] for p in gray_patches]
    g_orig = [p[1] for p in gray_patches]
    b_orig = [p[2] for p in gray_patches]
    
    # Create correction functions for each channel / åˆ›å»ºå�„é€šé�“çš„çŸ«æ­£å‡½æ•°
    r_corrector = create_calibration_curve(r_orig, target_values)
    g_corrector = create_calibration_curve(g_orig, target_values)
    b_corrector = create_calibration_curve(b_orig, target_values)
    
    # Convert to float for calculation / è½¬æ�¢ä¸ºæµ®ç‚¹æ•°ä¾¿äº�è®¡ç®—
    img_float = img.astype(np.float32)
    
    # Separate channels / åˆ†ç¦»é€šé�“
    b, g, r = cv2.split(img_float)
    
    # Apply correction curves / åº”ç”¨çŸ«æ­£æ›²çº¿
    r_corrected = r_corrector(r)
    g_corrected = g_corrector(g)
    b_corrected = b_corrector(b)
    
    # Merge channels and clip range / å�ˆå¹¶é€šé�“å¹¶é™�åˆ¶èŒƒå›´
    corrected = cv2.merge([b_corrected, g_corrected, r_corrected])
    return np.clip(corrected, 0, 255).astype(np.uint8)

# ====================== 2. Color Correction / 2. å½©è‰²çŸ«æ­£ ======================
def create_correction_matrix(src_points, dst_points):
    """Create color correction matrix (3x3 transformation matrix) / åˆ›å»ºè‰²å½©çŸ«æ­£çŸ©é˜µ(3x3å�˜æ�¢çŸ©é˜µ)"""
    A = []
    b = []
    for src, dst in zip(src_points, dst_points):
        A.append([src[0], src[1], src[2], 0, 0, 0, 0, 0, 0])
        A.append([0, 0, 0, src[0], src[1], src[2], 0, 0, 0])
        A.append([0, 0, 0, 0, 0, 0, src[0], src[1], src[2]])
        b.extend(dst)
    
    A = np.array(A, dtype=np.float32)
    b = np.array(b, dtype=np.float32)
    
    # Solve linear equations / è§£çº¿æ€§æ–¹ç¨‹ç»„
    X = np.linalg.lstsq(A, b, rcond=None)[0]
    return X.reshape((3, 3))

def calculate_deltaE(rgb1, rgb2):
    """Calculate simple color difference between two RGB colors / è®¡ç®—ä¸¤ä¸ªRGBé¢œè‰²ä¹‹é—´çš„ç®€å�•è‰²å·®"""
    # Simplified color difference calculation / ç®€åŒ–çš„è‰²å·®è®¡ç®—
    return np.sqrt(np.sum((np.array(rgb1) - np.array(rgb2))**2))

def adjust_matrix_by_deltaE(src_colors, dst_colors, matrix):
    """
    Automatically enhance correction weights for high deltaE regions / æ ¹æ�®è‰²å·®Î”Eè‡ªåŠ¨å¢�å¼ºé«˜è‰²å·®åŒºåŸŸçš„å¤„ç�†æ�ƒé‡�
    :return: Adjusted matrix / è°ƒæ•´å��çš„çŸ©é˜µ
    """
    # Calculate deltaE for each color / è®¡ç®—æ¯�ä¸ªé¢œè‰²çš„è‰²å·®Î”E
    deltaEs = []
    for i in range(len(src_colors)):
        src_color = src_colors[i]
        dst_color = dst_colors[i]
        dE = calculate_deltaE(src_color, dst_color)
        deltaEs.append(dE)
    
    # Find regions with deltaE > 15 / æ‰¾å‡ºÎ”E>15çš„é«˜è‰²å·®åŒºåŸŸ
    high_delta_indices = [i for i, dE in enumerate(deltaEs) if dE > 15]
    
    if not high_delta_indices:
        return matrix
    
    # Enhance correction strength for high deltaE regions / å¢�å¼ºé«˜è‰²å·®åŒºåŸŸçš„çŸ«æ­£åŠ›åº¦
    adjust_factor = 1.2  # Increase by 20% / å¢�å¼º20%
    for i in high_delta_indices:
        # Get corresponding color position in matrix / è�·å�–å¯¹åº”é¢œè‰²åœ¨çŸ©é˜µä¸­çš„ä½�ç½®
        row_start = i * 3
        for j in range(3):
            matrix[j] = matrix[j] * adjust_factor
    
    return matrix

def color_correction(img, src_colors, dst_colors, strength=0.8):
    """
    Apply color correction matrix / åº”ç”¨è‰²å½©çŸ«æ­£çŸ©é˜µ
    :param img: Gray-balanced image / ç�°å¹³è¡¡çŸ«æ­£å��çš„å›¾åƒ�
    :param src_colors: Actual sampled colors (after gray balance) / å®�é™…é‡‡æ ·é¢œè‰²(å·²ç�°å¹³è¡¡çŸ«æ­£)
    :param dst_colors: Target colors / ç›®æ ‡é¢œè‰²
    :param strength: Correction strength (0.0~1.0) / çŸ«æ­£å¼ºåº¦(0.0~1.0)
    :return: (Final corrected image, correction matrix) / (æœ€ç»ˆçŸ«æ­£å›¾åƒ�, çŸ«æ­£çŸ©é˜µ)
    """
    # Create base correction matrix / åˆ›å»ºåŸºç¡€çŸ«æ­£çŸ©é˜µ
    base_matrix = create_correction_matrix(src_colors, dst_colors)
    
    # Automatically adjust matrix based on deltaE / æ ¹æ�®Î”Eè‡ªåŠ¨è°ƒæ•´çŸ©é˜µ
    final_matrix = adjust_matrix_by_deltaE(src_colors, dst_colors, base_matrix)
    
    # Convert image to float [0,1] / å°†å›¾åƒ�è½¬æ�¢ä¸ºæµ®ç‚¹æ•°[0,1]
    img_f = img.astype(np.float32) / 255.0
    
    # Separate channels (RGB order) / åˆ†ç¦»é€šé�“(RGBé¡ºåº�)
    r, g, b = cv2.split(img_f)
    
    # Apply transformation matrix / åº”ç”¨å�˜æ�¢çŸ©é˜µ
    corrected = np.zeros_like(img_f)
    for i in range(3):
        corrected[..., i] = (
            final_matrix[i, 0] * r + 
            final_matrix[i, 1] * g + 
            final_matrix[i, 2] * b
        )
    
    # Blend original and corrected images / æ··å�ˆå�Ÿå§‹å›¾åƒ�å’ŒçŸ«æ­£ç»“æ�œ
    result = (1 - strength) * img_f + strength * corrected
    
    # Clip range and return / é™�åˆ¶èŒƒå›´å¹¶è¿”å›�
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    
    # Saturation adjustment (prevent oversaturation) / é¥±å’Œåº¦å¾®è°ƒ(é˜²æ­¢è¿‡é¥±å’Œ)
    hsv = cv2.cvtColor(result, cv2.COLOR_RGB2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * 0.95, 0, 255)  # Reduce by 5% / é™�ä½�5%é¥±å’Œåº¦
    final_result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    
    return final_result, final_matrix

# ====================== 3. Main Processing Pipeline / 3. ä¸»å¤„ç�†æµ�ç¨‹ ======================
def restore_colors(image_rgb, squares, patches, detection_method):
    """
    Color restoration based on color checker data / åŸºäº�è‰²å�¡æ•°æ�®è¿›è¡Œé¢œè‰²è¿˜å�Ÿ
    
    Parameters / å�‚æ•°:
    image_rgb: Original RGB image / å�Ÿå§‹RGBå›¾åƒ�
    squares: Black and white square color data / é»‘ç™½æ–¹æ ¼é¢œè‰²æ•°æ�®
    patches: Grayscale and color bar color data / ç�°é˜¶å’Œå½©è‰²æ�¡é¢œè‰²æ•°æ�®
    detection_method: Detection method description / æ£€æµ‹æ–¹æ³•æ��è¿°
    
    Returns / è¿”å›�:
    Dictionary containing color restoration results / åŒ…å�«é¢œè‰²è¿˜å�Ÿç»“æ�œçš„å­—å…¸
    """
    
    # 1. Define color checker target values / å®šä¹‰è‰²å�¡ç›®æ ‡å€¼
    # Grayscale target values (6 steps) / ç�°é˜¶ç›®æ ‡å€¼(6é˜¶)
    target_gray_values = [
        [255, 255, 255],  # White / ç™½
        [204, 204, 204],  # Light gray / äº®ç�°
        [153, 153, 153],  # Medium gray / ä¸­ç�°
        [102, 102, 102],  # Dark gray / æš—ç�°
        [51, 51, 51],     # Deep gray / æ·±ç�°
        [0, 0, 0]         # Black / é»‘
    ]
    
    # Color target values (6 colors) / å½©è‰²ç›®æ ‡å€¼(6è‰²)
    target_color_values = [
        [0, 180, 230],   # Cyan (C) / é�’è‰²(C)
        [240, 0, 120],   # Magenta (M) / å“�çº¢(M)
        [240, 230, 0],   # Yellow (Y) / é»„è‰²(Y)
        [230, 0, 50],    # Red (R) / çº¢è‰²(R)
        [0, 200, 150],   # Green (G) / ç»¿è‰²(G)
        [80, 60, 180]    # Blue (B) / è“�è‰²(B)
    ]
    
    # 2. Extract actual sampled values / æ��å�–å®�é™…é‡‡æ ·å€¼
    # Extract detected grayscale data (first 6 are grayscale) / æ��å�–å®�é™…æ£€æµ‹åˆ°çš„ç�°é˜¶æ•°æ�®ï¼ˆå‰�6ä¸ªæ˜¯ç�°é˜¶ï¼‰
    if len(patches) >= 6:
        gray_sampled = patches[:6]
    else:
        print("âš ï¸� Insufficient grayscale data, using defaults / ç�°é˜¶æ•°æ�®ä¸�è¶³ï¼Œä½¿ç”¨é»˜è®¤å€¼")
        gray_sampled = [[128, 128, 128]] * 6
    
    # Extract detected color data (last 6 are colors) / æ��å�–å®�é™…æ£€æµ‹åˆ°çš„å½©è‰²æ•°æ�®ï¼ˆå��6ä¸ªæ˜¯å½©è‰²ï¼‰
    if len(patches) >= 12:
        color_sampled = patches[6:12]
    else:
        print("âš ï¸� Insufficient color data, using defaults / å½©è‰²æ•°æ�®ä¸�è¶³ï¼Œä½¿ç”¨é»˜è®¤å€¼")
        color_sampled = [[128, 128, 128]] * 6
    
    # 3. Gray balance correction (must be before color correction) / ç�°å¹³è¡¡çŸ«æ­£(å¿…é¡»åœ¨å½©è‰²çŸ«æ­£ä¹‹å‰�)
    gray_balanced_img = gray_balance_correction(image_rgb, gray_sampled, target_gray_values)
    
    # 4. Correct color samples after gray balance / å¯¹ç�°å¹³è¡¡å��çš„å½©è‰²é‡‡æ ·å€¼è¿›è¡ŒçŸ«æ­£
    corrected_color_sampled = []
    for color in color_sampled:
        # Apply gray balance correction to single color patch / å¯¹å�•ä¸ªé¢œè‰²å�—åº”ç”¨ç�°å¹³è¡¡çŸ«æ­£
        color_array = np.array([color], dtype=np.float32)
        corrected_color = gray_balance_correction(color_array.reshape(1, 1, 3), gray_sampled, target_gray_values)
        corrected_color_sampled.append(corrected_color[0, 0])
    
    # 5. Apply color correction / åº”ç”¨å½©è‰²çŸ«æ­£
    final_img, final_matrix = color_correction(gray_balanced_img, corrected_color_sampled, target_color_values, strength=0.8)
    
    # 6. Calculate correction evaluation metrics / è®¡ç®—çŸ«æ­£æ•ˆæ�œè¯„ä¼°
    # Verify effect using corrected grayscale data / ä½¿ç”¨çŸ«æ­£å��çš„ç�°é˜¶æ•°æ�®éªŒè¯�æ•ˆæ�œ
    corrected_gray_patches = []
    for patch in gray_sampled:
        # Apply correction to single patch / å¯¹å�•ä¸ªé¢œè‰²å�—åº”ç”¨çŸ«æ­£
        patch_array = np.array([patch], dtype=np.float32)
        corrected_patch = gray_balance_correction(patch_array.reshape(1, 1, 3), gray_sampled, target_gray_values)
        corrected_gray_patches.append(corrected_patch[0, 0])
    
    # Calculate errors before/after correction / è®¡ç®—çŸ«æ­£å‰�å��çš„è¯¯å·®
    target_gray_means = [np.mean(patch) for patch in target_gray_values]
    original_errors = [abs(np.mean(patch) - target) for patch, target in zip(gray_sampled, target_gray_means)]
    corrected_errors = [abs(np.mean(patch) - target) for patch, target in zip(corrected_gray_patches, target_gray_means)]
    
    mean_original_error = np.mean(original_errors)
    mean_corrected_error = np.mean(corrected_errors)
    max_original_error = np.max(original_errors)
    max_corrected_error = np.max(corrected_errors)
    
    # Calculate grayscale linearity correlation coefficient / è®¡ç®—ç�°åº¦çº¿æ€§åº¦ç›¸å…³ç³»æ•°
    original_means = [np.mean(patch) for patch in gray_sampled]
    correlation = np.corrcoef(original_means, target_gray_means)[0, 1]
    
    print(f"ğŸ“Š Pre-correction mean error / çŸ«æ­£å‰�å¹³å�‡è¯¯å·®: {mean_original_error:.2f}\tPost-correction mean error / çŸ«æ­£å��å¹³å�‡è¯¯å·®: {mean_corrected_error:.2f}")
    print(f"ğŸ“Š Pre-correction max error / çŸ«æ­£å‰�æœ€å¤§è¯¯å·®: {max_original_error:.2f}\tPost-correction max error / çŸ«æ­£å��æœ€å¤§è¯¯å·®: {max_corrected_error:.2f}")
    print(f"ğŸ“Š Grayscale linearity correlation / ç�°åº¦çº¿æ€§åº¦ç›¸å…³ç³»æ•°: {correlation:.4f}")
    

    return {
        'gray_balanced_image': gray_balanced_img,  # Gray balance only / ä»…ç�°å¹³è¡¡çŸ«æ­£
        'corrected_image': final_img,  # Gray balance + color correction / å…ˆç�°å¹³è¡¡çŸ«æ­£å��å†�å½©è‰²çŸ«æ­£
        'correction_matrix': final_matrix,  # Correction matrix / çŸ«æ­£çŸ©é˜µ
        'gray_correlation': correlation,
        'color_accuracy': {
            'mean_error': mean_corrected_error, 
            'max_error': max_corrected_error, 
            'original_errors': original_errors,
            'corrected_errors': corrected_errors
        },
        'theoretical_gray_colors': target_gray_values,
        'theoretical_color_colors': target_color_values,
        'observed_gray_colors': gray_sampled,
        'observed_color_colors': color_sampled,
        'corrected_gray_colors': corrected_gray_patches,
        'corrected_color_colors': corrected_color_sampled
    }


def visualize(step_images, contour, corners, rotation_angle, line_equation, warped_image,
              image_rgb, checker_region, squares, patches, analysis_results, 
                     detection_method, rotation_info, template_rgb):
    """
    Visualize processing results: Display 6 subplots / å�¯è§†åŒ–å¤„ç�†ç»“æ�œï¼š6ä¸ªå­�å›¾æ˜¾ç¤º
    
    Parameters / å�‚æ•°:
    image_rgb: Original image / å�Ÿå§‹å›¾åƒ�
    checker_region: Detected color checker region / æ£€æµ‹åˆ°çš„è‰²å�¡åŒºåŸŸ
    squares: Black and white square color data / é»‘ç™½æ–¹æ ¼é¢œè‰²æ•°æ�®
    patches: Grayscale and color bar color data / ç�°é˜¶å’Œå½©è‰²æ�¡é¢œè‰²æ•°æ�®
    analysis_results: Color restoration results / é¢œè‰²è¿˜å�Ÿç»“æ�œ
    detection_method: Detection method description / æ£€æµ‹æ–¹æ³•æ��è¿°
    rotation_info: Rotation matching information / æ—‹è½¬åŒ¹é…�ä¿¡æ�¯
    template_rgb: Standard template / æ ‡å‡†æ¨¡æ�¿
    """
    
    # Create 3x3 subplots / åˆ›å»º3è¡Œ3åˆ—çš„å­�å›¾
    fig, axes = plt.subplots(3, 3, figsize=(18, 12))
                
    # First row: Processing steps / ç¬¬ä¸€è¡Œï¼šå¤„ç�†æ­¥éª¤
    # First plot / ç¬¬ä¸€ä¸ªå›¾ï¼š
    axes[0, 0].imshow(image_rgb)
    axes[0, 0].set_title('1. Original Image / å�Ÿå§‹å›¾åƒ�')
    axes[0, 0].axis('off')

    if 'edges' in step_images:
        axes[0, 1].imshow(step_images['edges'], cmap='gray')
        axes[0, 1].set_title('2. Canny Edge Detection / Canny è¾¹ç¼˜æ£€æµ‹')
        axes[0, 1].axis('off')
    
    if 'protected_color' in step_images:
        axes[0, 2].imshow(step_images['protected_color'])
        axes[0, 2].set_title('3. Morphological Dilation Enhancement with Border Protection (blue marked) \nå½¢æ€�å­¦è†¨èƒ€å¢�å¼ºï¼ŒåŠ è¾¹ç•ŒæŠ—å¹²æ‰°ä¿�æŠ¤ï¼ˆè“�è‰²æ ‡è®°ï¼‰')
        axes[0, 2].axis('off')

    
    # Second row / ç¬¬äºŒè¡Œï¼š
    # 4th plot / ç¬¬4ä¸ªå›¾ï¼š
    if 'contour_vis' in step_images:
        axes[1, 0].imshow(step_images['contour_vis'])
        axes[1, 0].set_title('4. Detected Color Checker Contour / æ£€æµ‹åˆ°çš„è‰²å�¡è½®å»“')
        axes[1, 0].axis('off')
    

    # 5th plot: Visualization of sampling regions / ç¬¬5ä¸ªå›¾ï¼šè‰²å�¡åŒºåŸŸçš„å�–æ ·åŒºåŸŸå�¯è§†åŒ–
    # Create sampling region visualization / åˆ›å»ºå�–æ ·åŒºåŸŸå�¯è§†åŒ–
    h, w = checker_region.shape[:2]
    rows = 3
    cols = 12
    
    # Calculate total grid dimensions (minus margins) / è®¡ç®—æ ¼å­�åŒºåŸŸçš„æ€»å°ºå¯¸ï¼ˆå‡�å�»è¾¹è·�ï¼‰
    total_grid_height = h - 0
    total_grid_width = w - 0
    
    # Calculate each cell size / è®¡ç®—æ¯�ä¸ªæ ¼å­�çš„å°ºå¯¸
    cell_height = total_grid_height // rows
    cell_width = total_grid_width // cols
    
    # Calculate grid starting position (center aligned) / è®¡ç®—æ ¼å­�åŒºåŸŸçš„èµ·å§‹ä½�ç½®ï¼ˆä¸­å¿ƒå¯¹é½�ï¼‰
    grid_start_x = (w - total_grid_width) // 2 
    grid_start_y = (h - total_grid_height) // 2
    
    # Create visualization image / åˆ›å»ºå�¯è§†åŒ–å›¾åƒ�
    vis_checker = checker_region.copy()
    vis_checker = cv2.medianBlur(vis_checker, 5)
    
    # Define cell types / å®šä¹‰æ ¼å­�ç±»å�‹
    black_cells = [13, 14, 17, 18, 21, 22]
    white_cells = [15, 16, 19, 20, 23, 24]
    gray_cells = [25, 26, 27, 28, 29, 30]
    color_cells = [31, 32, 33, 34, 35, 36]
    
    
    # Draw grid and numbering / ç»˜åˆ¶ç½‘æ ¼å’Œç¼–å�·
    for row in range(rows):
        for col in range(cols):
            cell_num = row * cols + col + 1  # Cell numbering starts from 1 / æ ¼å­�ç¼–å�·ä»�1å¼€å§‹
            
            # Calculate cell position (based on grid starting position) / è®¡ç®—æ ¼å­�ä½�ç½®ï¼ˆåŸºäº�æ ¼å­�åŒºåŸŸèµ·å§‹ä½�ç½®ï¼‰
            x1 = grid_start_x + col * cell_width
            y1 = grid_start_y + row * cell_height
            x2 = grid_start_x + (col + 1) * cell_width
            y2 = grid_start_y + (row + 1) * cell_height
            
            # Skip ruler area (cells 1-12) / è·³è¿‡æ ‡å°ºåŒºåŸŸï¼ˆ1-12æ ¼ï¼‰
            if cell_num <= 12:
                # Draw ruler area (thin black lines) / ç»˜åˆ¶æ ‡å°ºåŒºåŸŸï¼ˆé»‘è‰²ç»†çº¿ï¼‰
                cv2.rectangle(vis_checker, (x1, y1), (x2, y2), (0, 0, 0), 1)
                # Add ruler labels / æ·»åŠ æ ‡å°ºæ ‡ç­¾
                cv2.putText(vis_checker, f'R{cell_num}', (x1+2, y1+15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
                continue
            
            # Calculate center sampling region (increased margin width) / è®¡ç®—ä¸­å¿ƒå�–æ ·åŒºåŸŸï¼ˆå¢�å¤§è¾¹ç¼˜å®½åº¦ï¼‰
            margin = max(2, min(cell_width, cell_height) // 3)
            center_x1 = x1 + margin
            center_y1 = y1 + margin
            center_x2 = x2 - margin
            center_y2 = y2 - margin
            
            # Draw full area (thin black lines) / ç»˜åˆ¶å®Œæ•´åŒºåŸŸï¼ˆé»‘è‰²ç»†çº¿ï¼‰
            cv2.rectangle(vis_checker, (x1, y1), (x2, y2), (0, 0, 0), 1)
            # Draw center sampling area (thin black lines) / ç»˜åˆ¶ä¸­å¿ƒå�–æ ·åŒºåŸŸï¼ˆé»‘è‰²ç»†çº¿ï¼‰
            cv2.rectangle(vis_checker, (center_x1, center_y1), (center_x2, center_y2), (0, 0, 0), 1)
            # Add cell number / æ·»åŠ æ ¼å­�åº�å�·
            cv2.putText(vis_checker, f'{cell_num}', (x1+2, y1+15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    axes[1, 1].imshow(vis_checker)
    axes[1, 1].set_title(f'5. Color Checker Region After Transformation Correction and Median Filter Denoising\nå�˜æ�¢çŸ«æ­£ä¹‹å��ä¸”ä¸­å€¼æ»¤æ³¢å�»é™¤å™ªç‚¹å��çš„è‰²å�¡åŒºåŸŸ\nCenter Sampling Size / ä¸­å¿ƒå�–æ ·åŒºåŸŸå°ºå¯¸: {cell_width - 2 * margin} x {cell_height - 2 * margin} pixels')
    axes[1, 1].axis('off')
    
    # Third row / ç¬¬ä¸‰è¡Œï¼š
    # 6th plot: Color checker before color correction (4x12 comparison) / ç¬¬6ä¸ªå›¾ï¼šè¯†åˆ«å�–è‰²çŸ«æ­£å‰�çš„è‰²å�¡ï¼ˆ4è¡Œ12åˆ—å¯¹æ¯”å›¾ï¼‰
    # Create a 4x12 color checker image / åˆ›å»ºä¸€ä¸ª4è¡Œ12åˆ—çš„è‰²å�¡å›¾åƒ�
    reconstructed_checker = np.zeros((4 * 50, 12 * 50, 3), dtype=np.uint8)  # Each cell 50x50 pixels / æ¯�ä¸ªæ ¼å­�50x50åƒ�ç´ 
    
    # Define theoretical color values / å®šä¹‰ç�†è®ºé¢œè‰²å€¼
    theoretical_colors = {
        # Ruler area (cells 1-12) - gray / æ ‡å°ºåŒºåŸŸï¼ˆ1-12æ ¼ï¼‰- ç�°è‰²
        1: [128, 128, 128], 2: [128, 128, 128], 3: [128, 128, 128], 4: [128, 128, 128],
        5: [128, 128, 128], 6: [128, 128, 128], 7: [128, 128, 128], 8: [128, 128, 128],
        9: [128, 128, 128], 10: [128, 128, 128], 11: [128, 128, 128], 12: [128, 128, 128],
        
        # Black and white squares (cells 13-24) / é»‘ç™½æ–¹æ ¼ï¼ˆ13-24æ ¼ï¼‰
        13: [0, 0, 0], 14: [0, 0, 0], 15: [255, 255, 255], 16: [255, 255, 255],
        17: [0, 0, 0], 18: [0, 0, 0], 19: [255, 255, 255], 20: [255, 255, 255],
        21: [0, 0, 0], 22: [0, 0, 0], 23: [255, 255, 255], 24: [255, 255, 255],
        
        # Grayscale (cells 25-30) / ç�°é˜¶ï¼ˆ25-30æ ¼ï¼‰
        25: [255, 255, 255], 26: [204, 204, 204], 27: [153, 153, 153],
        28: [102, 102, 102], 29: [51, 51, 51], 30: [0, 0, 0],
        
        # Color (cells 31-36) / å½©è‰²ï¼ˆ31-36æ ¼ï¼‰
        31: [0, 180, 230], 32: [240, 0, 120], 33: [240, 230, 0],
        34: [230, 0, 50], 35: [0, 200, 150], 36: [80, 60, 180]
    }
    
    # Define mapping of actually detected colors / å®šä¹‰å®�é™…æ£€æµ‹åˆ°çš„é¢œè‰²æ˜ å°„
    actual_colors = {}
    
    # Ruler area (cells 1-12) - gray / æ ‡å°ºåŒºåŸŸï¼ˆ1-12æ ¼ï¼‰- ç�°è‰²
    for i in range(1, 13):
        actual_colors[i] = [128, 128, 128]  # Gray / ç�°è‰²
    
    # Black and white squares (cells 13-24) - use actually detected colors / é»‘ç™½æ–¹æ ¼ï¼ˆ13-24æ ¼ï¼‰- ä½¿ç”¨å®�é™…æ£€æµ‹åˆ°çš„é¢œè‰²
    for i, cell_num in enumerate(black_cells):
        if i < len(squares):
            actual_colors[cell_num] = squares[i].astype(int)
        else:
            actual_colors[cell_num] = [0, 0, 0]
    
    for i, cell_num in enumerate(white_cells):
        if i + 6 < len(squares):
            actual_colors[cell_num] = squares[i + 6].astype(int)
        else:
            actual_colors[cell_num] = [255, 255, 255]
    
    # Grayscale (cells 25-30) - use actually detected colors / ç�°é˜¶ï¼ˆ25-30æ ¼ï¼‰- ä½¿ç”¨å®�é™…æ£€æµ‹åˆ°çš„é¢œè‰²
    for i, cell_num in enumerate(gray_cells):
        if i < len(patches) - 6:
            actual_colors[cell_num] = patches[i].astype(int)
        else:
            actual_colors[cell_num] = [128, 128, 128]
    
    # Color (cells 31-36) - use actually detected colors / å½©è‰²ï¼ˆ31-36æ ¼ï¼‰- ä½¿ç”¨å®�é™…æ£€æµ‹åˆ°çš„é¢œè‰²
    for i, cell_num in enumerate(color_cells):
        if i + 6 < len(patches):
            actual_colors[cell_num] = patches[i + 6].astype(int)
        else:
            actual_colors[cell_num] = [128, 128, 128]
    
    # Fill color checker / å¡«å……è‰²å�¡
    for row in range(4):
        for col in range(12):
            # Determine cell number range based on row / æ ¹æ�®è¡Œæ•°ç¡®å®šæ ¼å­�ç¼–å�·èŒƒå›´
            if row == 0:  # First row: Actually sampled cells 13-24 / ç¬¬ä¸€è¡Œï¼šå®�é™…å�–è‰²çš„13-24æ ¼å­�
                cell_num = col + 13  # 13, 14, ..., 24
            elif row == 1:  # Second row: Theoretical cells 13-24 / ç¬¬äºŒè¡Œï¼šç�†è®ºè‰²å�¡çš„13-24æ ¼å­�
                cell_num = col + 13  # 13, 14, ..., 24
            elif row == 2:  # Third row: Actually sampled cells 25-36 / ç¬¬ä¸‰è¡Œï¼šå®�é™…å�–è‰²çš„25-36æ ¼å­�
                cell_num = col + 25  # 25, 26, ..., 36
            else:  # Fourth row: Theoretical cells 25-36 / ç¬¬å››è¡Œï¼šç�†è®ºè‰²å�¡çš„25-36æ ¼å­�
                cell_num = col + 25  # 25, 26, ..., 36
            
            # Calculate cell position / è®¡ç®—æ ¼å­�ä½�ç½®
            y1 = row * 50
            y2 = (row + 1) * 50
            x1 = col * 50
            x2 = (col + 1) * 50
            
            # Select color based on row / æ ¹æ�®è¡Œæ•°é€‰æ‹©é¢œè‰²
            if row == 0:  # First row: Actually sampled cells 13-24 / ç¬¬ä¸€è¡Œï¼šå®�é™…å�–è‰²çš„13-24æ ¼å­�
                color = actual_colors.get(cell_num, [128, 128, 128])
            elif row == 1:  # Second row: Theoretical cells 13-24 / ç¬¬äºŒè¡Œï¼šç�†è®ºè‰²å�¡çš„13-24æ ¼å­�
                color = theoretical_colors.get(cell_num, [128, 128, 128])
            elif row == 2:  # Third row: Actually sampled cells 25-36 / ç¬¬ä¸‰è¡Œï¼šå®�é™…å�–è‰²çš„25-36æ ¼å­�
                color = actual_colors.get(cell_num, [128, 128, 128])
            else:  # Fourth row: Theoretical cells 25-36 / ç¬¬å››è¡Œï¼šç�†è®ºè‰²å�¡çš„25-36æ ¼å­�
                color = theoretical_colors.get(cell_num, [128, 128, 128])
            
            # Fill color / å¡«å……é¢œè‰²
            reconstructed_checker[y1:y2, x1:x2] = color
            
            # Add black border / æ·»åŠ é»‘è‰²è¾¹æ¡†
            cv2.rectangle(reconstructed_checker, (x1, y1), (x2, y2), (0, 0, 0), 1)
            
            # Add cell number / æ·»åŠ æ ¼å­�ç¼–å�·
            cv2.putText(reconstructed_checker, f'{cell_num}', (x1+5, y1+20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    axes[1, 2].imshow(reconstructed_checker)
    axes[1, 2].set_title('6. Rows 1 & 3: Actual Colors | Rows 2 & 4: Theoretical Colors \nç¬¬1è¡Œå’Œç¬¬3è¡Œ: å®�é™…å�–è‰² | ç¬¬2è¡Œå’Œç¬¬4è¡Œ: ç�†è®ºè‰²å�¡')
    axes[1, 2].axis('off')
    
    # 7th plot: Image after gray balance correction / ç¬¬7ä¸ªå›¾ï¼šå…ˆç�°å¹³è¡¡çŸ«æ­£å��çš„å›¾åƒ�
    axes[2, 0].imshow(analysis_results['gray_balanced_image'])
    axes[2, 0].set_title('7. After Gray Balance Correction / å…ˆç�°å¹³è¡¡çŸ«æ­£')
    axes[2, 0].axis('off')
    
    # 8th plot: Image after color correction / ç¬¬8ä¸ªå›¾ï¼šå†�å½©è‰²çŸ«æ­£å��çš„å›¾åƒ�
    axes[2, 1].imshow(analysis_results['corrected_image'])
    axes[2, 1].set_title('8. After Color Correction / å†�å½©è‰²çŸ«æ­£')
    axes[2, 1].axis('off')

    # 9th plot
    axes[2, 2].set_title('9. ')
    axes[2, 2].axis('off')
    
    plt.tight_layout()
    plt.show()


def process_color_checker(image_path, use_rotation_match=True):
    """
    Advanced color checker processing function focusing on geometric detection and correction, with detailed visualization steps
    é«˜çº§è‰²å�¡å¤„ç�†å‡½æ•°ï¼Œä¸“æ³¨äº�å‡ ä½•æ£€æµ‹å’ŒçŸ«æ­£ï¼ŒåŒ…å�«è¯¦ç»†æ­¥éª¤å�¯è§†åŒ–
    
    Parameters / å�‚æ•°:
    image_path: Path to the image / å›¾åƒ�è·¯å¾„
    use_rotation_match: Whether to use rotation matching / æ˜¯å�¦ä½¿ç”¨æ—‹è½¬åŒ¹é…�
    """
    
    # 1. Load image and standard template / åŠ è½½å›¾åƒ�å’Œæ ‡å‡†æ¨¡æ�¿
    print(f"ğŸ“‚ Processing image / å¤„ç�†å›¾åƒ�: {image_path}")
    image = cv2.imread(image_path)
    if image is None:
        print(f"â�Œ Failed to load image / æ— æ³•åŠ è½½å›¾åƒ�: {image_path}")
        return None
    
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    template_rgb = load_standard_template()
    
    # 2. Detect color checker region / æ£€æµ‹è‰²å�¡åŒºåŸŸ
    checker_region = None
    detection_method = ""
    
    # Try template matching / å°�è¯•æ¨¡æ�¿åŒ¹é…�
    if template_rgb is not None:
        checker_region, coords, score = detect_color_checker_by_template(image_rgb, template_rgb)
        if checker_region is not None:
            detection_method = f"Template matching (confidence: {score:.3f}) / æ¨¡æ�¿åŒ¹é…� (ç½®ä¿¡åº¦: {score:.3f})"
            print(f"âœ… Template matching successful / æ¨¡æ�¿åŒ¹é…�æˆ�åŠŸ: {checker_region.shape}, confidence / ç½®ä¿¡åº¦: {score:.3f}")
        else:
            print("âš ï¸� Template matching failed... / æ¨¡æ�¿åŒ¹é…�å¤±è´¥...")
    else:
        print(f"â�Œ Failed to load color checker template... / æ— æ³•åŠ è½½è‰²å�¡æ¨¡æ�¿...")

    # 3. Extract color checker contour (detailed steps) / æ��å�–è‰²å�¡è½®å»“ï¼ˆè¯¦ç»†æ­¥éª¤ï¼‰
    contour, corners, step_images = extract_checker_contour_detailed(checker_region)
    if contour is None or corners is None:
        print("â�Œâ�Œâš ï¸�âš ï¸� Failed to extract color checker contour âš ï¸�âš ï¸�â�Œâ�Œ / æ— æ³•æ��å�–è‰²å�¡è½®å»“ âš ï¸�âš ï¸�â�Œâ�Œ")
        return None
    
    # 4. Calculate rotation angle and line equation / è®¡ç®—æ—‹è½¬è§’åº¦å’Œè¾¹çº¿æ–¹ç¨‹
    rotation_angle = calculate_rotation_angle(corners)
    line_equation = get_line_equation(corners)
    
    # 5. Perspective transform correction / é€�è§†å�˜æ�¢çŸ«æ­£è‰²å�¡
    # Calculate target rectangle dimensions / è®¡ç®—ç›®æ ‡çŸ©å½¢çš„å°ºå¯¸
    width = int(max(
        float(np.linalg.norm(corners[1] - corners[0])),
        float(np.linalg.norm(corners[2] - corners[3]))
    ))
    height = int(max(
        float(np.linalg.norm(corners[3] - corners[0])),
        float(np.linalg.norm(corners[2] - corners[1]))
    ))
    
    # Target rectangle's four corners / ç›®æ ‡çŸ©å½¢çš„å››ä¸ªè§’ç‚¹
    dst_corners = np.array([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ], dtype="float32")
    
    # Calculate perspective transform matrix / è®¡ç®—é€�è§†å�˜æ�¢çŸ©é˜µ
    perspective_matrix = cv2.getPerspectiveTransform(corners, dst_corners)
    
    # Apply perspective transform / åº”ç”¨é€�è§†å�˜æ�¢
    warped_image = cv2.warpPerspective(checker_region, perspective_matrix, (width, height))
    
    # 6. Precise color patch extraction (using median method) / ç²¾ç¡®æ��å�–é¢œè‰²å�—ï¼ˆä½¿ç”¨medianæ–¹æ³•ï¼‰
    print(f"ğŸ�¯ Using color sampling method: median / ä½¿ç”¨é¢œè‰²å�–æ ·æ–¹æ³•: median")
    squares, patches = extract_color_patches_precise(warped_image, template_rgb)
    
    # 7. Color restoration / é¢œè‰²è¿˜å�Ÿ
    analysis_results = restore_colors(image_rgb, squares, patches, detection_method)

    # 8. Detailed visualization / è¯¦ç»†å�¯è§†åŒ–ç»“æ�œ
    visualize(step_images, contour, corners, rotation_angle, line_equation, warped_image, 
              image_rgb, warped_image, squares, patches, analysis_results, detection_method, None, template_rgb)
    
    return {
        'squares': squares,
        'patches': patches,
        'correction_matrix': analysis_results['correction_matrix'],
        'corrected_image': analysis_results['corrected_image'],
        'gray_correlation': analysis_results['gray_correlation'],
        'color_accuracy': analysis_results['color_accuracy'],
        'detection_method': detection_method,
        'sampling_method': 'median',
        'rotation_info': None
    }


if __name__ == "__main__":
    import os
    import random
    
    # Get all image files in sherd_images directory / è�·å�–sherd_imagesç›®å½•ä¸­çš„æ‰€æœ‰å›¾åƒ�æ–‡ä»¶
    sherd_dir = "/kaggle/input/h690/h690/sherd_images"
    if os.path.exists(sherd_dir):
        # Supported image formats / æ”¯æŒ�çš„å›¾åƒ�æ ¼å¼�
        image_extensions = ['.jpg', '.jpeg', '.png']
        image_files = []
        
        for file in os.listdir(sherd_dir):
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_files.append(file)
        
        # Randomly select 3 images / éš�æœºé€‰æ‹©3ä¸ªå›¾åƒ�
        selected_images = random.sample(image_files, 3)
        print(f"ğŸ�² Randomly selected 3 images: {selected_images} \nğŸ�² éš�æœºé€‰æ‹©çš„3ä¸ªå›¾åƒ�: {selected_images}")
        
        # Process each selected image / å¤„ç�†æ¯�ä¸ªé€‰ä¸­çš„å›¾åƒ�
        for i, image_file in enumerate(selected_images, 1):
            image_path = os.path.join(sherd_dir, image_file)
    
            print(f"{'='*60}")
            print(f"ğŸ“¸ Processing image {i}: {image_file} / å¤„ç�†ç¬¬ {i} ä¸ªå›¾åƒ�: {image_file}")
            print(f"{'='*60}")
            
            try:
                # Use advanced color checker detection / ä½¿ç”¨é«˜çº§è‰²å�¡æ£€æµ‹
                results = process_color_checker(image_path)
                    
                if results:
                    continue
                    # print(f"\nğŸ�‰ Image {i} processing completed! / ç¬¬ {i} ä¸ªå›¾åƒ�å¤„ç�†å®Œæˆ�ï¼�")
                    # print(f"Detection method: {results['detection_method']} / æ£€æµ‹æ–¹æ³•: {results['detection_method']}")
                    # print(f"Gray linearity correlation coefficient: {results['gray_correlation']:.4f} / ç�°åº¦çº¿æ€§åº¦ç›¸å…³ç³»æ•°: {results['gray_correlation']:.4f}")
                    # print(f"Corrected mean error: {results['color_accuracy']['mean_error']:.2f} / çŸ«æ­£å��å¹³å�‡è¯¯å·®: {results['color_accuracy']['mean_error']:.2f}")
                else:
                    print(f"â�Œ Failed to process image {i} / ç¬¬ {i} ä¸ªå›¾åƒ�å¤„ç�†å¤±è´¥")
                    
            except Exception as e:
                print(f"â�Œ Error processing image {i}: {e} / å¤„ç�†ç¬¬ {i} ä¸ªå›¾åƒ�æ—¶å‡ºé”™: {e}")

