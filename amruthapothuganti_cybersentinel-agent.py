# Cell 1: Requirements & imports
# Run this cell first.

import os
import json
import random
import datetime
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

# reproducibility
RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

print("Python packages loaded. Pandas:", pd.__version__, "NumPy:", np.__version__)
# Cell 2: Synthetic log generator
def generate_synthetic_logs(n=3000, seed=RANDOM_SEED, inject_bruteforce=True, inject_exfil=True):
    random.seed(seed)
    np.random.seed(seed)
    logs = []
    base_time = datetime.datetime.now().replace(microsecond=0)
    for i in range(n):
        t = base_time + datetime.timedelta(seconds=i * random.randint(1, 5))
        host = f"host{random.randint(1,40)}"
        src_ip = f"192.168.{random.randint(0,5)}.{random.randint(1,250)}"
        dst_ip = f"10.0.{random.randint(0,3)}.{random.randint(1,250)}"
        user = random.choice(['alice', 'bob', 'carol', 'dave', 'eve', 'service'])
        action = random.choices(['login', 'download', 'upload', 'exec'],
                                weights=[0.6, 0.2, 0.15, 0.05])[0]
        status = random.choices(['success', 'fail'], weights=[0.9, 0.1])[0]
        bytes_t = 0
        if action in ['download', 'upload'] and status == 'success':
            bytes_t = int(abs(np.random.normal(loc=5000, scale=2000)))
        logs.append({
            'timestamp': t.isoformat(),
            'host': host,
            'src_ip': src_ip,
            'dst_ip': dst_ip,
            'user': user,
            'action': action,
            'status': status,
            'bytes': bytes_t
        })

    injected_ground_truth = {'bruteforce_ips': set(), 'exfil_ips': set()}

    if inject_bruteforce:
        # Inject burst of failed logins from an attacker IP (simulate brute-force)
        attacker_ip = '203.0.113.55'
        for j in range(60):  # 60 failed logins
            logs.append({
                'timestamp': (base_time + datetime.timedelta(seconds=n + j)).isoformat(),
                'host': 'host1',
                'src_ip': attacker_ip,
                'dst_ip': '10.0.1.3',
                'user': 'unknown',
                'action': 'login',
                'status': 'fail',
                'bytes': 0
            })
        injected_ground_truth['bruteforce_ips'].add(attacker_ip)

    if inject_exfil:
        # Inject a large data upload to an external host (data exfiltration)
        exfil_ip = '192.168.2.50'
        logs.append({
            'timestamp': (base_time + datetime.timedelta(seconds=n + 100)).isoformat(),
            'host': 'host7',
            'src_ip': exfil_ip,
            'dst_ip': '198.51.100.10',
            'user': 'service',
            'action': 'upload',
            'status': 'success',
            'bytes': 50_000_000
        })
        injected_ground_truth['exfil_ips'].add(exfil_ip)

    df = pd.DataFrame(logs)
    # Shuffle to avoid accidental ordering artifacts
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df, injected_ground_truth

logs_df, ground_truth = generate_synthetic_logs(n=3000)
print("Generated logs:", len(logs_df))
logs_df.head()
# Save sample CSV for Kaggle submission
logs_df.to_csv("sample_logs.csv", index=False)
print("Saved sample_logs.csv")

# Cell 3: Parse timestamps and compute aggregated features per src_ip
logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp'])
logs_df['date'] = logs_df['timestamp'].dt.date
logs_df['hour'] = logs_df['timestamp'].dt.floor('H')

# We'll build per-src_ip aggregated features (count, fails, total_bytes, unique_dst_count, unique_users)
agg_funcs = {
    'timestamp': 'count',
    'status': lambda s: (s == 'fail').sum(),
    'bytes': 'sum',
    'dst_ip': pd.Series.nunique,
    'user': pd.Series.nunique
}
agg = logs_df.groupby('src_ip').agg({
    'timestamp': 'count',
    'status': lambda s: (s == 'fail').sum(),
    'bytes': 'sum',
    'dst_ip': pd.Series.nunique,
    'user': pd.Series.nunique
}).rename(columns={
    'timestamp': 'event_count',
    'status': 'fail_count',
    'bytes': 'total_bytes',
    'dst_ip': 'unique_dst_count',
    'user': 'unique_user_count'
}).reset_index()

# Add simple ratios
agg['fail_rate'] = agg['fail_count'] / agg['event_count']
agg = agg.fillna(0)
agg.head(8)
# Cell 4: Rule-based detections
BLACKLIST = {'203.0.113.55', '198.51.100.77'}  # example blacklist

def detect_blacklist_ips(df, blacklist=BLACKLIST):
    return set(df[df['src_ip'].isin(blacklist)]['src_ip'].unique())

def detect_bruteforce(df, fail_threshold=10, window_minutes=60):
    # simple approach: count failed events per src_ip overall (for synthetic dataset)
    fails = df[df['status'] == 'fail'].groupby('src_ip').size()
    return set(fails[fails > fail_threshold].index)

blacklist_hits = detect_blacklist_ips(logs_df)
brute_ips = detect_bruteforce(logs_df, fail_threshold=10)

print("Blacklist hits:", blacklist_hits)
print("Brute-force candidate IPs:", brute_ips)
# Cell 5: IsolationForest on aggregated features
feature_cols = ['event_count', 'fail_count', 'total_bytes', 'unique_dst_count', 'unique_user_count', 'fail_rate']
X = agg[feature_cols].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

iso = IsolationForest(n_estimators=200, contamination=0.01, random_state=RANDOM_SEED)
iso.fit(X_scaled)
agg['anomaly_score'] = iso.decision_function(X_scaled)  # higher -> less anomalous
agg['is_anomaly'] = iso.predict(X_scaled) == -1  # True for anomalies

# show most anomalous
agg.sort_values('anomaly_score').head(10)
# Cell 6: Triage logic - combine signals into incidents with a risk score
incidents = []
for _, row in agg.iterrows():
    src = row['src_ip']
    score = 0
    notes = []
    # rule signals
    if src in brute_ips:
        score += 40
        notes.append('brute-force failed login bursts')
    if src in blacklist_hits:
        score += 50
        notes.append('blacklisted IP')
    # ML signal
    if row['is_anomaly']:
        score += 30
        notes.append('ML anomaly (IsolationForest)')
    # high volume transfer
    if row['total_bytes'] > 10_000_000:
        score += 30
        notes.append('high total bytes transferred')
    # high fail rate
    if row['fail_rate'] > 0.4 and row['fail_count'] > 5:
        score += 15
        notes.append('high failure rate')

    if score > 0:
        incidents.append({
            'src_ip': src,
            'risk_score': score,
            'notes': notes,
            'event_count': int(row['event_count']),
            'fail_count': int(row['fail_count']),
            'total_bytes': int(row['total_bytes']),
            'anomaly_score': float(row['anomaly_score']),
            'is_anomaly': bool(row['is_anomaly'])
        })

# sort incidents by risk
incidents = sorted(incidents, key=lambda x: -x['risk_score'])
print("Top incidents:", len(incidents))
# show top 10
for inc in incidents[:10]:
    print(json.dumps(inc, indent=2))
# Cell 7: helper to query agent and generate textual report
def query_agent_by_ip(ip):
    for inc in incidents:
        if inc['src_ip'] == ip:
            report = {
                'summary': f"IP {ip} flagged (risk_score={inc['risk_score']})",
                'details': inc,
                'recommended_actions': [
                    "Isolate host / block IP at firewall",
                    "Force password reset for affected accounts",
                    "Investigate connected hosts and exfil destinations",
                    "Collect full packet/flow capture for timeframe"
                ]
            }
            return report
    return {'summary': f"No incidents for {ip}", 'details': None, 'recommended_actions': []}

# Example: check injected known bad IPs
for ip in sorted(list(ground_truth['bruteforce_ips'] | ground_truth['exfil_ips'])):
    print(json.dumps(query_agent_by_ip(ip), indent=2))
# Cell 8: Evaluate detection for injected anomalies
detected_ips = set([inc['src_ip'] for inc in incidents])
gt_brute = ground_truth['bruteforce_ips']
gt_exfil = ground_truth['exfil_ips']
gt_all = gt_brute.union(gt_exfil)

true_positives = detected_ips.intersection(gt_all)
false_negatives = gt_all - detected_ips
false_positives = detected_ips - gt_all

precision = len(true_positives) / max(1, len(detected_ips))
recall = len(true_positives) / max(1, len(gt_all))

print("Ground truth (injected):", gt_all)
print("Detected:", detected_ips)
print("TP:", true_positives)
print("FN:", false_negatives)
print("FP (sample):", list(false_positives)[:10])
print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
# Cell 9: Save incidents to JSON file for easy download / submission
with open("incidents_report.json", "w") as fh:
    json.dump({'incidents': incidents, 'generated_at': datetime.datetime.utcnow().isoformat()}, fh, indent=2)

print("Saved incidents_report.json with", len(incidents), "incidents")


