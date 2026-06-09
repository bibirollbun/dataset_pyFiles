import re, json, time
from datetime import datetime

# Simple observability collector (NO circular references)
OBS = {"logs": [], "traces": []}

def log_event(kind, msg):
    OBS["logs"].append({
        "time": datetime.utcnow().isoformat(),
        "kind": kind,
        "msg": msg
    })

def trace(step, detail):
    OBS["traces"].append({
        "time": datetime.utcnow().isoformat(),
        "step": step,
        "detail": detail
    })

print("ThreatVision Pro â€” Core observability system ready.")


# ThreatVision Log Parser Tool

def threatvision_parse_logs(raw_text: str):
    log_event("parser_start", "Parsing raw log input.")
    
    # Extract IPs
    ips = re.findall(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", raw_text)
    
    # Count suspicious keywords
    suspicious_hits = len(re.findall(r"failed|denied|unauthorized|invalid", raw_text, re.I))
    
    parsed_output = {
        "unique_ips": list(set(ips)),
        "suspicious_count": suspicious_hits,
        "characters": len(raw_text)
    }
    
    trace("parser_output", parsed_output)
    log_event("parser_done", "Parsing completed.")
    
    return parsed_output

print("ğŸ”§ ThreatVision Log Parser Ready.")



# Threat Detection Agent

def threatvision_detect(parsed):
    log_event("detection_start", "Evaluating parsed indicators.")
    
    alerts = []

    if parsed["suspicious_count"] > 5:
        alerts.append("High volume of suspicious authentication failures.")
    
    if len(parsed["unique_ips"]) > 3:
        alerts.append("Unusual number of unique source IPs detected.")
    
    detection_output = {
        "alerts": alerts,
        "ip_count": len(parsed["unique_ips"]),
        "failure_events": parsed["suspicious_count"]
    }

    trace("detection_output", detection_output)
    log_event("detection_done", "Threat evaluation complete.")
    
    return detection_output

print("ğŸ›¡ï¸� ThreatVision Detection Agent Ready.")



# Analyst Intelligence Agent

def threatvision_analyst(detection):
    log_event("analyst_start", "Generating analyst-friendly summary.")
    
    lines = []

    if detection["alerts"]:
        lines.append("âš ï¸� Potential threat indicators detected.")
        for a in detection["alerts"]:
            lines.append(f"- {a}")
    else:
        lines.append("âœ… No significant threat indicators found.")
    
    lines.append(f"Total suspicious events: {detection['failure_events']}")
    lines.append(f"Unique IP sources observed: {detection['ip_count']}")

    analyst_note = "\n".join(lines)

    trace("analyst_summary", analyst_note)
    log_event("analyst_done", "Analyst report produced.")
    
    return {"analyst_summary": analyst_note}

print("ğŸ“� ThreatVision Analyst Agent Ready.")



#ThreatVision SOC Orchestrator

def threatvision_pipeline(log_text: str):
    log_event("pipeline_start", "ThreatVision Pipeline initiated.")
    
    parsed = threatvision_parse_logs(log_text)
    detected = threatvision_detect(parsed)
    analyst = threatvision_analyst(detected)

    final_output = {
        "parsed_output": parsed,
        "detection_output": detected,
        "analyst_output": analyst,
        "observability": OBS
    }

    trace("pipeline_complete", final_output)
    log_event("pipeline_done", "ThreatVision Pipeline completed.")
    
    return final_output

print("ğŸš€ ThreatVision Pipeline Ready.")



# Example Logs for Testing 

sample_logs = """
Jan 21 12:01:22 server sshd[1204]: Failed password for root from 192.168.1.10 port 55421 ssh2
Jan 21 12:01:25 server sshd[1205]: Failed password for invalid user admin from 10.0.0.17 port 54412 ssh2
Jan 21 12:03:45 server sshd[1210]: Unauthorized access attempt from 44.17.90.22
Jan 21 12:05:10 server sshd[1304]: Failed password from 192.168.1.11
"""

print("ğŸ“„ Sample logs loaded.")



# Execute ThreatVision SOC Pipeline 

final_report = threatvision_pipeline(sample_logs)

print("âœ”ï¸� ThreatVision Pipeline Executed Successfully!")



print("\n===== Analyst Output Raw =====")
for k, v in final_report["analyst_output"].items():
    print(f"{k}: {v}")


