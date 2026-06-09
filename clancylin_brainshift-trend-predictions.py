# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# Cell 1: å»ºç«‹è³‡æ–™å¤¾
!mkdir -p brainshift_agent
print("âœ… è³‡æ–™å¤¾ brainshift_agent å·²å»ºç«‹")


%%writefile brainshift_agent/requirements.txt
google-adk
opencv-python
numpy
scikit-learn


%%writefile brainshift_agent/agent.py

import os
import cv2
import numpy as np
from google.adk.agents import Agent
import vertexai

# --- 1. çµ„æ…‹èˆ‡è·¯å¾‘ç®¡ç�† (Configuration) ---
# ç‚ºäº†è®“ Tool èƒ½æ‰¾åˆ°æª”æ¡ˆï¼Œæˆ‘å€‘éœ€è¦�ç¡¬ç·¨ç¢¼æˆ–å‹•æ…‹ç�²å�–è·¯å¾‘
class Config:
    BASE_DIR = "/kaggle/input/brainshift-data/case0"
    POSE_FILE = os.path.join(BASE_DIR, "poses.toon")
    IMAGE_DIR = os.path.join(BASE_DIR, "proj")
    BASELINE_IMG = os.path.join(IMAGE_DIR, "0000_style01.png")

# --- 2. å®šç¾©å·¥å…· (Tools) ---

def analyze_visual_shift(target_filename: str) -> dict:
    """
    Tool: ä½¿ç”¨ OpenCV è¨ˆç®—ç›®æ¨™åœ–ç‰‡èˆ‡åŸºæº–åœ–ç‰‡ (0000) çš„è¦–è¦ºä½�ç§»ã€‚
    
    Args:
        target_filename: ç›®æ¨™åœ–ç‰‡çš„æª”å�� (ä¾‹å¦‚ '0738_style01.png')
        
    Returns:
        dict: åŒ…å�«åƒ�ç´ ä½�ç§»å�‘é‡� (shift_vector) å’Œç‰¹å¾µåŒ¹é…�æ•¸ (matches)
    """
    target_path = os.path.join(Config.IMAGE_DIR, target_filename)
    
    # æª¢æŸ¥æª”æ¡ˆæ˜¯å�¦å­˜åœ¨
    if not os.path.exists(target_path):
        return {"error": f"File {target_filename} not found."}
    if not os.path.exists(Config.BASELINE_IMG):
        return {"error": "Baseline image not found."}

    # è®€å�–åœ–ç‰‡
    img1 = cv2.imread(Config.BASELINE_IMG, cv2.IMREAD_GRAYSCALE) # åŸºæº–
    img2 = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)         # ç›®æ¨™
    
    if img1 is None or img2 is None:
        return {"error": "Failed to load images."}

    # ORB ç‰¹å¾µè¨ˆç®—
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    
    if des1 is None or des2 is None:
        return {"visual_shift": [0, 0], "matches": 0, "status": "low_features"}

    # ç‰¹å¾µåŒ¹é…�
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    
    # å�–å‰� 15 å€‹æœ€ä½³é»�
    good_matches = matches[:15]
    if len(good_matches) < 5:
        return {"visual_shift": [0, 0], "matches": len(good_matches), "status": "low_confidence"}

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # è¨ˆç®—ä½�ç§»: Target - Baseline
    shift = np.mean(pts2 - pts1, axis=0)
    
    return {
        "visual_shift_x": round(float(shift[0]), 2),
        "visual_shift_y": round(float(shift[1]), 2),
        "matches": len(good_matches),
        "status": "success"
    }

def get_ground_truth_data(target_filename: str) -> dict:
    """
    Tool: è®€å�– pose æª”æ¡ˆä»¥ç�²å�–çœŸå¯¦çš„ç‰©ç�†åº§æ¨™ (ç”¨æ–¼é©—è­‰)ã€‚
    
    Args:
        target_filename: åœ–ç‰‡æª”å��ï¼Œç”¨æ–¼è§£æ�� IDã€‚
    """
    try:
        # è§£æ�� ID: 0738_style01.png -> 738
        target_id = int(target_filename.split('_')[0])
    except:
        return {"error": "Invalid filename format"}

    # è®€å�–åŸºæº– (ID 0) å’Œ ç›®æ¨™ ID çš„åº§æ¨™
    poses = {}
    if os.path.exists(Config.POSE_FILE):
        with open(Config.POSE_FILE, 'r') as f:
            for line in f:
                if ',' in line and '{' not in line:
                    parts = line.strip().split(',')
                    pid = int(parts[0])
                    # è®€å�– tx, ty
                    poses[pid] = [float(parts[1]), float(parts[2])]
    
    baseline_pose = poses.get(0)
    target_pose = poses.get(target_id)
    
    if baseline_pose and target_pose:
        physical_shift = [target_pose[0] - baseline_pose[0], target_pose[1] - baseline_pose[1]]
        return {
            "target_id": target_id,
            "true_physical_pose": target_pose,
            "true_physical_shift": [round(x, 4) for x in physical_shift]
        }
    
    return {"error": "Pose ID not found in database"}

# --- 3. å®šç¾© Agent (The Brain) ---

# System Prompt: å®šç¾© Agent çš„äººè¨­èˆ‡å·¥ä½œæµ�ç¨‹
instruction = """
You are a BrainShift Diagnostic Agent. Your goal is to predict physical brain displacement based on 2D visual shifts.

Your Workflow:
1. Receive a target image filename from the user.
2. Call the tool `analyze_visual_shift` to measure the pixel displacement (x, y) compared to the baseline.
3. Call the tool `get_ground_truth_data` to retrieve the actual physical coordinates (Ground Truth).
4. Compare the Visual Shift (pixels) with the Physical Shift (mm).
5. Output a structured report summarizing the correlation.

Rules:
- If visual shift is detected, there SHOULD be a physical shift.
- Note that visual X/Y axes might correlate differently to physical X/Y axes (e.g., inverted or scaled).
- Be concise and professional.
"""

# åˆ�å§‹åŒ– Agent
root_agent = Agent(
    name="brainshift_agent",
    model="gemini-1.5-flash",
    description="Analyzes brainshift using computer vision and pose data.",
    instruction=instruction,
    tools=[analyze_visual_shift, get_ground_truth_data] # è¨»å†Šå·¥å…·
)


import sys
import os
import google.generativeai as genai
from kaggle_secrets import UserSecretsClient

# 1. ç¢ºä¿�è·¯å¾‘æ­£ç¢º
sys.path.append(os.getcwd())


# 2. è¼‰å…¥æˆ‘å€‘å®šç¾©å¥½çš„ Agent è—�åœ–
from brainshift_agent.agent import root_agent

print(f"ğŸ“‹ è®€å�– Agent è¨­å®š: {root_agent.name}")
print(f"   - Tools æ•¸é‡�: {len(root_agent.tools)}")
print(f"   - æŒ‡ä»¤é•·åº¦: {len(root_agent.instruction)}")

# è¨­å®š API Key
try:
    user_secrets = UserSecretsClient()
    genai.configure(api_key=user_secrets.get_secret("GOOGLE_API_KEY"))
except:
    print("âš ï¸� API Key è¨­å®šå�¯èƒ½æœ‰èª¤ï¼Œè«‹æª¢æŸ¥ Secrets")


# 3. å•Ÿå‹• Runtime (é€™æ‰�æ˜¯çœŸæ­£çš„å¤§è…¦)
model = genai.GenerativeModel(
    model_name='gemini-2.5-flash', # ä½¿ç”¨ Agent å®šç¾©çš„æ¨¡å�‹
    tools=root_agent.tools,        # è¼‰å…¥ Agent çš„æ‰‹è…³ (OpenCV Tool)
    system_instruction=root_agent.instruction # è¼‰å…¥ Agent çš„äººè¨­
)

# 4. å»ºç«‹å°�è©± Session (å•Ÿç”¨è‡ªå‹•å‡½æ•¸å‘¼å�«åŠŸèƒ½)
chat = model.start_chat(enable_automatic_function_calling=True)

# 5. ç™¼é€�æ¸¬è©¦æŒ‡ä»¤
test_image = "0738_style01.png" 
user_query = f"Analyze the brainshift for image: {test_image}"

print(f"\n>>> User Input: {user_query}")
print("-" * 40)

try:
    response = chat.send_message(user_query)
    print("\n>>> Agent Report:")
    print(response.text)
except Exception as e:
    print(f"\nâ�Œ åŸ·è¡ŒéŒ¯èª¤: {e}")
    # é¡¯ç¤ºæ›´è©³ç´°çš„éŒ¯èª¤ï¼Œé€šå¸¸æ˜¯ Tool åŸ·è¡Œå›�å‚³äº†é��é �æœŸçš„æ ¼å¼�

