!pip install fastapi uvicorn nest-asyncio requests



from fastapi import FastAPI
import nest_asyncio
import uvicorn
import threading
import time
import json
import requests



# Create API App
app = FastAPI()

# Mock Database: User Profile
USER_DATABASE = {
    "u001": {"name": "Arun", "phone_os": "Android", "security_level": "Medium"},
    "u002": {"name": "Priya", "phone_os": "iOS", "security_level": "High"},
}

# Mock AI Malware Detection Model
def ai_security_scan(user_id: str):
    # Example Static Output â€” Replace with real model later
    return {
        "user_id": user_id,
        "device_health_score": 88,
        "malware_detected": False,
        "network_risk": "Low",
        "data_leak_risk": "Very Low"
    }

# User Profile Fetcher Tool
def fetch_user_profile(user_id: str):
    return USER_DATABASE.get(user_id, {
        "name": "Unknown User",
        "phone_os": "Unknown",
        "security_level": "Low"
    })

# Multi-Agent AI + Tools Coordination (Orchestrator)
@app.get("/orchestrate/{user_id}")
def orchestrate(user_id: str):

    profile = fetch_user_profile(user_id)
    scan_result = ai_security_scan(user_id)

    if scan_result["malware_detected"]:
        decision = "ALERT â€” Device NOT safe â�Œ"
    else:
        decision = "Device SAFE âœ” All Good"

    return {
        "profile": profile,
        "security_report": scan_result,
        "ai_final_decision": decision
    }





nest_asyncio.apply()

def start_server():
    uvicorn.run(app, host="0.0.0.0", port=8000)

thread = threading.Thread(target=start_server, daemon=True)
thread.start()

time.sleep(2)  # Wait for server to start
print("ğŸš€ FastAPI backend running successfully!")



user_id = "u001"  # You can change this ID (u001 / u002)

url = f"http://127.0.0.1:8000/orchestrate/{user_id}"
response = requests.get(url)

print("\nğŸ”¹ Final Cybersecurity Assessment Report:\n")
print(json.dumps(response.json(), indent=4))

