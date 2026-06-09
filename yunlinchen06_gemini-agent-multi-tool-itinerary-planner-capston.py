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


# Code Cell 1: å®‰è£� Google GenAI SDK
!pip install google-genai


# Code Cell 2: ä½¿ç”¨ UserSecretsClient è®€å�– Key ä¸¦åˆ�å§‹åŒ–å®¢æˆ¶ç«¯
import os
import json
from google import genai
from google.genai.errors import APIError 
# å°�å…¥ç”¨æ–¼è®€å�– Kaggle Secret çš„å°ˆç”¨æ¨¡çµ„
from kaggle_secrets import UserSecretsClient 

try:
    # 1. ä½¿ç”¨ UserSecretsClient è®€å�– Secret
    user_secrets = UserSecretsClient()
    API_KEY = user_secrets.get_secret("Kaggle-Capstone-Key") # ğŸ‘ˆ ç›´æ�¥å‘¼å�«æ‚¨è¨­å®šçš„é‡‘é‘°å��ç¨±
    
    # æª¢æŸ¥æ˜¯å�¦è®€åˆ° Key
    if not API_KEY:
        raise ValueError("API Key æœªè¨­å®šã€‚è«‹ç¢ºèª� Secret å��ç¨±å’Œå€¼æ­£ç¢ºã€‚")
    
    # 2. åˆ�å§‹åŒ– Gemini å®¢æˆ¶ç«¯
    client = genai.Client(api_key=API_KEY)
    
    print("âœ… Gemini å®¢æˆ¶ç«¯åˆ�å§‹åŒ–æˆ�åŠŸï¼�")

except Exception as e:
    print(f"â�Œ åˆ�å§‹åŒ–å¤±æ•—ï¼š{e}")
    print("è«‹æª¢æŸ¥æ‚¨çš„ Kaggle Secret å��ç¨±æ˜¯å�¦ç‚º 'Kaggle-Capstone-Key'ï¼Œä¸¦ç¢ºä¿�é‡‘é‘°æœ¬èº«æœ‰æ•ˆã€‚")


# Code Cell 3: å®šç¾© Agent çš„æ‰€æœ‰å·¥å…·å‡½å¼� (åŒ…å�«æ–°çš„å¯¦ç”¨å·¥å…·)
import json # ç¢ºä¿� json å‡½å¼�åº«å·²ç¶“å°�å…¥

# =======================================================
# å·¥å…· A: æ�œå°‹ä½�å®¿è³‡è¨Š (search_accommodation)
# =======================================================
def search_accommodation(city: str, date_range: str) -> str:
    """
    ç”¨ä¾†æ�œå°‹ç‰¹å®šåŸ�å¸‚å’Œæ—¥æœŸç¯„åœ�å…§çš„ä½�å®¿é�¸é …ï¼Œä¸¦æ��ä¾›ç°¡è¦�è³‡è¨Šã€‚
    Agent æœƒæ ¹æ“šç”¨æˆ¶é �ç®—ï¼ˆä¾‹å¦‚ï¼šä¸­ç­‰ï¼‰ä¾†æ±ºå®šå¾�åˆ—è¡¨ä¸­ç¯©é�¸å“ªä¸€é …ã€‚
    Args:
        city: æ�œå°‹ä½�å®¿çš„ç›®æ¨™åŸ�å¸‚å��ç¨± (e.g., "å�°åŒ—", "é«˜é›„")ã€‚
        date_range: ä½�å®¿çš„æ—¥æœŸç¯„åœ� (e.g., "2025/12/10-2025/12/12")ã€‚
    Returns:
        JSON æ ¼å¼�çš„ä½�å®¿æ¸…å–®ï¼ŒåŒ…å�«å��ç¨±ã€�é¡�å�‹å’Œæ¨¡æ“¬åƒ¹æ ¼ã€‚
    """
    if "å�°åŒ—" in city:
        data = [
            {"name": "å�°åŒ—æ™¶è�¯é…’åº—", "type": "é«˜ç´šé£¯åº—", "price_per_night": 6500, "rating": 4.5},
            {"name": "è¥¿é–€ç”ºèƒŒåŒ…å®¢æ£§", "type": "é�’å¹´æ—…é¤¨", "price_per_night": 1200, "rating": 4.2},
            {"name": "å¤§å®‰å�€é¢¨æ ¼æ°‘å®¿", "type": "ç²¾å“�æ°‘å®¿", "price_per_night": 3800, "rating": 4.7},
        ]
    elif "é«˜é›„" in city:
        data = [
            {"name": "é«˜é›„è�¬è±ªé…’åº—", "type": "é«˜ç´šé£¯åº—", "price_per_night": 5200, "rating": 4.6},
            {"name": "é§�äºŒè—�è¡“ç‰¹å�€æ°‘å®¿", "type": "ç‰¹è‰²ä½�å®¿", "price_per_night": 2500, "rating": 4.0},
        ]
    else:
        return "æ‰¾ä¸�åˆ°è©²åŸ�å¸‚çš„ä½�å®¿è³‡è¨Šã€‚"
        
    return json.dumps(data, ensure_ascii=False)


# =======================================================
# å·¥å…· B: ç�²å�–æ™¯é»�è³‡è¨Š (get_attraction_details)
# =======================================================
def get_attraction_details(attraction_name: str) -> str:
    """
    ç�²å�–æŒ‡å®šå�°ç�£ç†±é–€æ™¯é»�çš„è©³ç´°è³‡è¨Šã€�é–‹æ”¾æ™‚é–“å’Œå»ºè­°é�Šç�©æ™‚é–“ã€‚
    Args:
        attraction_name: æ™¯é»�çš„å��ç¨± (e.g., "å�°åŒ—101", "è¥¿å­�ç�£").
    Returns:
        æ™¯é»�çš„è©³ç´°æ��è¿°ã€�é–‹æ”¾æ™‚é–“å’Œå»ºè­°é�Šç�©æ™‚é–“ (JSON æ ¼å¼�)ã€‚
    """
    details = {
        "å�°åŒ—101": {"description": "æ›¾æ˜¯ä¸–ç•Œç¬¬ä¸€é«˜æ¨“ï¼Œå�¯ä¿¯ç�°å�°åŒ—å¸‚æ™¯ï¼Œæœ‰è³¼ç‰©ä¸­å¿ƒã€‚", "hours": "09:00-22:00", "duration": "2h", "location_type": "å®¤å…§è§€æ™¯"},
        "æ•…å®®å�šç‰©é™¢": {"description": "æ”¶è—�äº†å¤§é‡�ä¸­è�¯æ–‡ç‰©ï¼Œç‚ºä¸–ç•Œäº”å¤§å�šç‰©é¤¨ä¹‹ä¸€ã€‚", "hours": "08:30-18:30", "duration": "3-4h", "location_type": "æ–‡åŒ–è—�è¡“"},
        "è¥¿å­�ç�£": {"description": "é«˜é›„çš„å¤©ç„¶æµ·ç�£ï¼Œä»¥å¤•é™½ç¾�æ™¯å’Œè‹±åœ‹é ˜äº‹é¤¨è��å��ã€‚", "hours": "å…¨å¤©é–‹æ”¾", "duration": "2h", "location_type": "æˆ¶å¤–è‡ªç„¶"},
        "é§�äºŒè—�è¡“ç‰¹å�€": {"description": "ç”±èˆŠå€‰åº«æ”¹å»ºçš„æ–‡å‰µåœ’å�€ï¼Œå……æ»¿å�„ç¨®è—�è¡“è£�ç½®å’Œç‰¹è‰²å°�åº—ã€‚", "hours": "10:00-20:00", "duration": "2-3h", "location_type": "æ–‡å‰µåœ’å�€"}
    }
    
    if attraction_name in details:
        return json.dumps(details[attraction_name], ensure_ascii=False)
    else:
        return f"æ‰¾ä¸�åˆ°æ™¯é»� '{attraction_name}' çš„è©³ç´°è³‡è¨Šã€‚"


# =======================================================
# å·¥å…· C: æª¢æŸ¥æ—…é�Šå¯¦ç”¨è³‡è¨Š (check_travel_utility)
# =======================================================
def check_travel_utility(city: str, date: str) -> str:
    """
    æ��ä¾›é—œæ–¼ç‰¹å®šåŸ�å¸‚å’Œæ—¥æœŸçš„æ—…é�Šå¯¦ç”¨è³‡è¨Šï¼ŒåŒ…æ‹¬å»ºè­°äº¤é€šæ–¹å¼�å’Œç·Šæ€¥è�¯çµ¡é›»è©±ã€‚
    Args:
        city: æŸ¥è©¢å¯¦ç”¨è³‡è¨Šçš„åŸ�å¸‚å��ç¨± (e.g., "å�°åŒ—", "é«˜é›„")ã€‚
        date: æ—…é�Šçš„æ—¥æœŸ (e.g., "2025/12/10")ï¼Œç”¨æ–¼æ¨¡æ“¬å¤©æ°£æŸ¥è©¢ã€‚
    Returns:
        JSON æ ¼å¼�çš„å¯¦ç”¨è³‡è¨Šï¼ŒåŒ…å�«å»ºè­°äº¤é€šã€�ç•¶æ—¥å¤©æ°£å’Œç·Šæ€¥é›»è©±ã€‚
    """
    # é€™è£¡çš„é‚�è¼¯æ˜¯æ ¹æ“šæ—¥æœŸä¸­æ˜¯å�¦åŒ…å�« '10' ä¾†æ¨¡æ“¬ä¸�å�Œçš„å¤©æ°£
    mock_weather = "æ™´æœ—ï¼Œæ°£æº« 18-24Â°C" if '10' in date else "å¤šé›²ï¼Œå�¶æœ‰çŸ­æš«é™£é›¨"
    
    if "å�°åŒ—" in city:
        data = {
            "main_transport": "æ�·é�‹ (MRT) ç³»çµ±ç™¼é�”ï¼Œå»ºè­°è³¼è²·æ‚ é�Šå�¡ã€‚",
            "weather_forecast": mock_weather,
            "emergency_contact": "ç·Šæ€¥å ±è­¦: 110 / ç·Šæ€¥é†«ç™‚: 119"
        }
    elif "é«˜é›„" in city:
        data = {
            "main_transport": "æ�·é�‹å’Œè¼•è»Œç³»çµ±ï¼Œæ™¯é»�é–“å�¯åˆ©ç”¨YouBikeã€‚",
            "weather_forecast": mock_weather,
            "emergency_contact": "ç·Šæ€¥å ±è­¦: 110 / ç·Šæ€¥é†«ç™‚: 119"
        }
    else:
        return "æ‰¾ä¸�åˆ°è©²åŸ�å¸‚çš„å¯¦ç”¨è³‡è¨Šã€‚"
        
    return json.dumps(data, ensure_ascii=False)


# Code Cell 4: æœ€çµ‚ç‰ˆæœ¬ - æ¸¬è©¦ Agent æ±ºç­–ä¸¦åŸ·è¡Œå¤šè¼ªç·¨æ�’ (åŒ…å�«ä¸‰å€‹å·¥å…·)

# 1. è¨­å®šæ¨¡å�‹å’Œå·¥å…·
model = "gemini-2.5-flash" 
# ç¢ºä¿�é€™è£¡åŒ…å�«äº†æ‰€æœ‰ä¸‰å€‹å·¥å…·
available_tools = [search_accommodation, get_attraction_details, check_travel_utility] 

# 2. ç”¨æˆ¶çš„è¤‡é›œè«‹æ±‚ (æ›´æ–°æ��ç¤ºï¼Œè¦�æ±‚äº¤é€šå’Œå¤©æ°£è³‡è¨Š)
prompt_1 = "æˆ‘è¨ˆåŠƒåœ¨ 2025 å¹´ 12 æœˆ 10 è™Ÿåˆ° 12 è™Ÿå�»å�°åŒ—ç�©ä¸‰å¤©ï¼Œé �ç®—ä¸­ç­‰ã€‚è«‹å¹«æˆ‘æ�¨è–¦ä¸€å€‹ä½�å®¿é�¸é …ï¼Œç„¶å¾Œè¦�åŠƒæˆ‘ç¬¬ä¸€å¤©å�»æ•…å®®å�šç‰©é™¢çš„è¡Œç¨‹ï¼Œä¸¦å‘Šè¨´æˆ‘ç•¶åœ°çš„å»ºè­°äº¤é€šæ–¹å¼�å’Œå¤©æ°£ã€‚"

print("--- Agent é€²è¡Œç¬¬ä¸€æ¬¡æ�¨ç�† (æ±ºç­–èª¿ç”¨å·¥å…·) ---")
# ç¬¬ä¸€æ¬¡å‘¼å�« (ä½¿ç”¨ config å�ƒæ•¸ä¸¦å¼·åˆ¶èª¿ç”¨å·¥å…·)
response = client.models.generate_content(
    model=model,
    contents=prompt_1,
    config={
        "tools": available_tools,
        "tool_config": {"function_calling_config": {"mode": "ANY"}} 
    }
)

# 3. æª¢æŸ¥ä¸¦åŸ·è¡Œ Agent è«‹æ±‚çš„å·¥å…· (Function Calling Loop)
tool_calls = response.function_calls
tool_responses = []

if tool_calls:
    print("\n--- å¯¦éš›åŸ·è¡Œ Agent è«‹æ±‚çš„å·¥å…· (Function Calling) ---")
    
    for call in tool_calls:
        func_name = call.name
        func_args = dict(call.args)
        
        # åŸ·è¡Œå°�æ‡‰çš„ Python å‡½å¼�ï¼Œä¸¦åŒ…å�«æ–°çš„ check_travel_utility å·¥å…·
        if func_name == "search_accommodation":
            result = search_accommodation(**func_args)
        elif func_name == "get_attraction_details":
            result = get_attraction_details(**func_args)
        elif func_name == "check_travel_utility":
            result = check_travel_utility(**func_args) # ğŸ‘ˆ åŸ·è¡Œæ–°çš„äº¤é€šå¯¦ç”¨å·¥å…·
        else:
            result = f"éŒ¯èª¤ï¼šAgent è«‹æ±‚äº†æœªå®šç¾©çš„å·¥å…·: {func_name}"

        print(f"  -> åŸ·è¡Œ {func_name}({func_args}) æˆ�åŠŸã€‚")

        # æº–å‚™å·¥å…·å›�å‚³çµ¦ Agent çš„æ ¼å¼� (FunctionResponse)
        tool_responses.append(
            genai.types.FunctionResponse(
                name=func_name,
                response={"result": result} 
            )
        )
    
    # 4. ç¬¬äºŒæ¬¡æ�¨ç�†ï¼šå°‡æ‰€æœ‰å…§å®¹æ‰“åŒ…æˆ�æ¨™æº–çš„ 'Content' æ ¼å¼� (æœ€çµ‚ç©©å®šæ ¼å¼�)
    print("\n--- Agent æ­£åœ¨é€²è¡Œç¬¬äºŒæ¬¡æ�¨ç�†ï¼Œç”Ÿæˆ�æœ€çµ‚è¡Œç¨‹ ---")

    # æ§‹é€ å®Œæ•´çš„å¤šè¼ªå°�è©±æ­·å�² (è§£æ±º SDK é¡�å�‹éŒ¯èª¤çš„æœ€çµ‚ç©©å®šæ ¼å¼�)
    contents_for_final_call = [
        # 1. å�Ÿå§‹ç”¨æˆ¶æ��ç¤º (User Role) - ä½¿ç”¨å­—å…¸æ ¼å¼�ï¼Œé�¿å…� SDK æ§‹é€ å™¨éŒ¯èª¤
        {"role": "user", "parts": [{"text": prompt_1}]},
        
        # 2. æ¨¡å�‹è«‹æ±‚èª¿ç”¨å·¥å…·çš„å…§å®¹ (Model Role)
        response.candidates[0].content, 
        
        # 3. å·¥å…·å›�è¦† (Tool Role)
        *tool_responses
    ]
    
    final_response = client.models.generate_content(
        model=model,
        contents=contents_for_final_call,
        config={"tools": available_tools} 
    )

    # 5. è¼¸å‡ºæœ€çµ‚çš„è¡Œç¨‹è¦�åŠƒ
    print("\n=============================================")
    print("âœ… æœ€çµ‚ Agent è¡Œç¨‹è¦�åŠƒè¼¸å‡º:")
    print("=============================================")
    print(final_response.text)

else:
    print("\nâ�Œ Agent æœªè«‹æ±‚èª¿ç”¨ä»»ä½•å·¥å…·ï¼Œç„¡æ³•ç”Ÿæˆ�è¡Œç¨‹ã€‚")
    print(f"Agent å�Ÿå§‹å›�æ‡‰: {response.text}")

# ==========================================================
# 6. é—œé�µä¿®æ­£ï¼šè¼¸å‡ºä¸€å€‹å�‡çš„æª”æ¡ˆä»¥æ»¿è¶³ Kaggle æ��äº¤éœ€æ±‚ 
# ç”±æ–¼èª²ç¨‹è¦�æ±‚ Notebook å¿…é ˆè¼¸å‡ºä¸€å€‹æª”æ¡ˆæ‰�èƒ½æ��äº¤ï¼Œæˆ‘å€‘ç”Ÿæˆ�ä¸€å€‹ç©ºçš„ submission.csv
# ==========================================================
import pandas as pd
pd.DataFrame({"Id": [1], "Prediction": ["success"]}).to_csv("submission.csv", index=False)
print("\n[æ��äº¤ä¿®æ­£] å·²æˆ�åŠŸç”Ÿæˆ� submission.csv æª”æ¡ˆï¼Œç‰ˆæœ¬å�¯æ��äº¤ï¼�")

