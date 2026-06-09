# AI Support Agent for Troubleshooting Phone and Laptop Issues

def classify_issue(user_message: str) -> str:
    """Very simple keyword-based classifier."""
    msg = user_message.lower()

    if any(word in msg for word in ["battery", "charge", "charging", "power"]):
        return "battery"
    if any(word in msg for word in ["hot", "heat", "overheat", "over heating", "temperature"]):
        return "overheating"
    if any(word in msg for word in ["crash", "crashes", "stopped working", "force close", "stops working"]):
        return "app_crash"
    if any(word in msg for word in ["slow", "lag", "lagging", "freezing", "freeze", "hang", "hanging"]):
        return "performance"
    return "general"


def battery_troubleshooting(device_type: str) -> str:
    return f"""
[{device_type.upper()}] Battery Troubleshooting:
1. Reduce screen brightness and turn off auto-brightness if not needed.
2. Close apps running in the background that you are not using.
3. Turn off Bluetooth, GPS, and Wi-Fi when you don’t need them.
4. Check which apps are using the most battery and uninstall / limit them.
5. If the device is old (2–3+ years), the battery may be degraded and may need replacement.
"""


def overheating_troubleshooting(device_type: str) -> str:
    return f"""
[{device_type.upper()}] Overheating Troubleshooting:
1. Avoid using the device on soft surfaces (bed, pillow) that block ventilation.
2. Close heavy apps like games, video editors, or many browser tabs.
3. For laptops: clean the vents/fans and consider using a cooling pad.
4. Avoid charging and gaming at the same time for long periods.
5. If it shuts down often from heat, it may need internal cleaning or service.
"""


def app_crash_troubleshooting(device_type: str) -> str:
    return f"""
[{device_type.upper()}] App Crash Troubleshooting:
1. Force stop the app and reopen it.
2. Clear the app cache (and data, if you can log back in easily).
3. Check for app updates in the app store / software center.
4. Restart the device to clear temporary issues.
5. If the app still crashes, uninstall and reinstall it.
6. If only one specific app crashes, it may be a bug in that app.
"""


def performance_troubleshooting(device_type: str) -> str:
    return f"""
[{device_type.upper()}] Slow Performance Troubleshooting:
1. Restart the device to clear temporary files and memory.
2. Close unused apps and browser tabs.
3. Check available storage; keep at least 10–20% space free.
4. Uninstall apps you don’t use anymore.
5. Turn off fancy animations or visual effects if possible.
6. For laptops: consider upgrading RAM or switching to SSD if still slow.
"""


def general_troubleshooting(device_type: str) -> str:
    return f"""
[{device_type.upper()}] General Troubleshooting:
1. Restart the device once if you haven’t already.
2. Check for system / OS updates and install them.
3. Make sure you have enough free storage space.
4. If the issue continues, note exact error messages and search or contact support.
"""


def troubleshoot_issue(user_message: str, device_type: str = "phone/laptop") -> str:
    """Main agent logic: classify the issue and call the right troubleshooting function."""
    issue_type = classify_issue(user_message)

    if issue_type == "battery":
        advice = battery_troubleshooting(device_type)
    elif issue_type == "overheating":
        advice = overheating_troubleshooting(device_type)
    elif issue_type == "app_crash":
        advice = app_crash_troubleshooting(device_type)
    elif issue_type == "performance":
        advice = performance_troubleshooting(device_type)
    else:
        advice = general_troubleshooting(device_type)

    response = f"""
User issue: {user_message}

Detected issue type: {issue_type}

Suggested steps:
{advice}
"""
    return response


# ---- Demo section ----

example_issues = [
    ("My phone battery is draining very fast even when I am not using it.", "phone"),
    ("My laptop gets very hot and sometimes shuts down while I am working.", "laptop"),
    ("An app on my phone keeps crashing every time I open it.", "phone"),
    ("My laptop has become very slow when opening files and browsers.", "laptop"),
    ("My device is acting weird, I don't know what is wrong.", "phone/laptop"),
]

for text, dev_type in example_issues:
    print("=" * 80)
    print(troubleshoot_issue(text, dev_type))


