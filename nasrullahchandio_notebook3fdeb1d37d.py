import pandas as pd

# Load both datasets
df = pd.read_csv("/kaggle/input/telco-qa-data/video_stream_qoe.csv")
df_ext = pd.read_csv("/kaggle/input/telco-qa-data/video_stream_qoe_extended.csv")

# Normalize column names so both match (df uses user_id, df_ext uses customer_id)
df_ext = df_ext.rename(columns={"customer_id": "user_id"})

# Combine into one master dataset
df_all = pd.concat([df, df_ext], ignore_index=True)

# Quick checks
df_all.shape, df_all.columns



import numpy as np

# 1) Tool: get full details of a single session
def get_session_details(session_id: str):
    row = df_all[df_all["session_id"] == session_id]
    if row.empty:
        return {"message": f"No session found for id {session_id}"}
    return row.to_dict(orient="records")[0]


# 2) Tool: aggregate QoE stats for a user across df + df_ext
def get_user_qoe_summary(user_id: str):
    user_df = df_all[df_all["user_id"] == user_id]
    if user_df.empty:
        return {"message": f"No sessions found for user {user_id}"}
    
    return {
        "user_id": user_id,
        "sessions_count": int(len(user_df)),
        "avg_stalls": float(user_df["stall_count"].mean()),
        "avg_mos": float(user_df["mos"].mean()),
        "avg_bitrate_mbps": float(user_df["avg_bitrate_mbps"].mean()),
        "max_latency_ms": float(user_df["latency_ms"].max()),
        "min_latency_ms": float(user_df["latency_ms"].min()),
        "rebuffer_time_sec_avg": float(user_df["rebuffer_time_sec"].mean())
    }


# 3) Tool: cell-level network health
def get_cell_health(cell_id: str):
    cell_df = df_all[df_all["cell_id"] == cell_id]
    if cell_df.empty:
        return {"message": f"No sessions found for cell {cell_id}"}
    
    poor_sessions = cell_df[cell_df["mos"] < 3.0]
    
    return {
        "cell_id": cell_id,
        "sessions_count": int(len(cell_df)),
        "poor_quality_sessions": int(len(poor_sessions)),
        "poor_quality_ratio": float(len(poor_sessions) / len(cell_df)),
        "avg_latency_ms": float(cell_df["latency_ms"].mean()),
        "avg_stalls": float(cell_df["stall_count"].mean())
    }


# 4) Tool: session-level health score
def get_session_health_score(session_id: str):
    row = df_all[df_all["session_id"] == session_id]
    if row.empty:
        return {"message": f"No session found for id {session_id}"}
    
    r = row.iloc[0]
    score = (r["avg_bitrate_mbps"] / (r["stall_count"] + 1)) - (r["latency_ms"] / 200)
    
    return {
        "session_id": session_id,
        "health_score": round(float(score), 2),
        "mos": float(r["mos"]),
        "stall_count": int(r["stall_count"]),
        "latency_ms": int(r["latency_ms"]),
        "avg_bitrate_mbps": float(r["avg_bitrate_mbps"])
    }



def router_agent(message: str) -> str:
    msg = message.lower()
    
    if "session" in msg or "stall" in msg or "mos" in msg:
        return "qoe_agent"
    if "latency" in msg or "slow" in msg or "buffer" in msg or "freeze" in msg:
        return "network_agent"
    if "what is" in msg or "explain" in msg:
        return "faq_agent"
    return "report_agent"



def qoe_agent(message: str):
    msg = message.lower().split()
    
    # Try to detect session/user/cell ID from user message
    session_id = None
    user_id = None
    cell_id = None
    
    for token in msg:
        if token.startswith("s"):
            session_id = token.upper()
        if token.startswith("u"):
            user_id = token.upper()
        if "cell_" in token:
            cell_id = token.upper()
    
    if "user" in msg and user_id:
        return get_user_qoe_summary(user_id)
    if "cell" in msg and cell_id:
        return get_cell_health(cell_id)
    if session_id:
        if "health" in msg or "score" in msg:
            return get_session_health_score(session_id)
        else:
            return get_session_details(session_id)
    
    return "I can analyze sessions, users, and cells. Try: 'Show session S001' or 'User U1001 QoE summary'"



def network_agent(message: str):
    return {
        "possible_causes": [
            "High latency between user and cell/site",
            "Wi-Fi congestion or weak signal strength",
            "Insufficient bitrate during peak hours"
        ],
        "recommended_actions": [
            "Ask user to move closer to Wi-Fi router or window.",
            "Suggest rebooting router / modem.",
            "If latency > 150 ms for many sessions in same cell â†’ escalate to RF/Network team.",
            "Consider CDN / caching optimization for popular content."
        ]
    }



def faq_agent(message: str):
    msg = message.lower()
    
    if "mos" in msg:
        return "MOS (Mean Opinion Score) ranges from 1 (bad) to 5 (excellent) and measures perceived video quality."
    if "latency" in msg:
        return "Latency is network delay. High latency causes buffering and slow video start."
    if "stall" in msg or "rebuffer" in msg:
        return "Stalls occur when the video buffer empties due to insufficient throughput."
    if "bitrate" in msg:
        return "Bitrate is data per second used by video. Higher bitrate = clearer video."
    
    return "I can answer questions about MOS, stalls, bitrate, latency, and QoE concepts."



def report_agent(message: str = ""):
    return {
        "overall_summary": {
            "total_sessions": len(df_all),
            "avg_mos": float(df_all["mos"].mean()),
            "avg_stalls": float(df_all["stall_count"].mean()),
            "avg_latency_ms": float(df_all["latency_ms"].mean())
        },
        "note": "This report combines both base and extended QoE datasets."
    }



conversation_memory = {}

def add_to_memory(conv_id: str, role: str, content):
    if conv_id not in conversation_memory:
        conversation_memory[conv_id] = []
    conversation_memory[conv_id].append({"role": role, "content": content})



def ask_system(message: str, conv_id: str = "conv_1"):
    add_to_memory(conv_id, "user", message)
    
    agent = router_agent(message)
    print(f"[Router] Routed to: {agent}")
    
    if agent == "qoe_agent":
        response = qoe_agent(message)
    elif agent == "network_agent":
        response = network_agent(message)
    elif agent == "faq_agent":
        response = faq_agent(message)
    else:
        response = report_agent(message)
    
    add_to_memory(conv_id, agent, response)
    return response



ask_system("Show details for session S001")
ask_system("Give me health score for session S00002")
ask_system("User C001 QoE summary")  # from df_ext
ask_system("Why is my video slow?")
ask_system("What is MOS?")
ask_system("Give overall QoE report")


