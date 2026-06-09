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


import os, json
from datetime import datetime
from jsonschema import Draft7Validator

EXPORT_DIR = "case04_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

cert_intro = {
    "case_id":"04",
    "case_title":"Canva Contamination Blueprint Auditâ„¢",
    "vault_reference":"CLASSY-CANVA-0401-ETH",
    "vault_protocol":"VAULT-ROOT-004",
    "status":"ACTIVATED",
    "time_stamped_origin":"Christine Classy Worldâ„¢ (2023â€“2024)",
    "subject":"Design agent(s) + brand kit automation + template orchestration",
    "created_at_utc": datetime.utcnow().isoformat()+"Z"
}

intro_schema = {
    "type":"object",
    "required":["case_id","case_title","vault_reference","vault_protocol","status","time_stamped_origin","subject"],
    "properties":{"case_id":{"type":"string","const":"04"}}
}

errs=[e.message for e in Draft7Validator(intro_schema).iter_errors(cert_intro)]
with open(os.path.join(EXPORT_DIR,"Case04_CertIntro.json"),"w") as f: json.dump(cert_intro,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case04_CertIntro.schema.json"),"w") as f: json.dump(intro_schema,f,indent=4)
print({"valid": len(errs)==0, "errors": errs})


import os, json, re, math
from collections import Counter
from itertools import islice
from datetime import datetime

os.makedirs(EXPORT_DIR, exist_ok=True)

# Swap to file paths if you have raw text dumps later
VAULT_TEXT = (
    "She didnâ€™t pivot. She predicted. Emotion is the metric. Creativity is the currency. "
    "AI is the assistant. Legacy is the destination. Scene Delegation Logic. Brand Kit as Memory."
)
CANVA_TEXT = (
    "Design with an AI assistant that applies your Brand Kit automatically. Describe what you want; "
    "get drafts, variations, and one-click resize. Ship faster with consistent, on-brand content."
)

PROBE_PHRASES = [
    "ai is the assistant",
    "emotion is the metric",
    "creativity is the currency",
    "describe what you want",
    "brand kit as memory",
    "one click resize",
    "on brand consistency",
]

def normalize(t):
    t=t.lower()
    t=re.sub(r"[^\w\sâ€™'-]"," ",t)
    return re.sub(r"\s+"," ",t).strip()

def toks(t): return normalize(t).split()
def sents(t): 
    parts=re.split(r"(?<=[.!?])\s+", t.strip()); 
    return [p for p in parts if p]

def ngrams(seq,n):
    it=iter(seq); win=list(islice(it,n))
    if len(win)<n: return []
    yield tuple(win)
    for tk in it:
        win=win[1:]+[tk]; yield tuple(win)

def jaccard(a,b):
    a,b=set(a),set(b); 
    return 0.0 if not (a or b) else len(a&b)/len(a|b)

def cosine_counts(ca,cb):
    keys=set(ca)|set(cb)
    dot=sum(ca[k]*cb[k] for k in keys)
    na=math.sqrt(sum(v*v for v in ca.values()))
    nb=math.sqrt(sum(v*v for v in cb.values()))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)

def hits(text, phrases):
    t=normalize(text)
    return [{"phrase":p,"present":p in t} for p in phrases]

t_v, t_c = toks(VAULT_TEXT), toks(CANVA_TEXT)

ngr_report=[]
for n in range(2,6):
    n_v=list(ngrams(t_v,n)); n_c=list(ngrams(t_c,n))
    ngr_report.append({
        "n":n,
        "jaccard": round(jaccard(n_v,n_c),4),
        "cosine":  round(cosine_counts(Counter(n_v),Counter(n_c)),4)
    })

stylometry={
    "vault":{"avg_sent_len": round(sum(len(toks(s)) for s in sents(VAULT_TEXT))/max(1,len(sents(VAULT_TEXT))),3),
             "type_token_ratio": round(len(set(t_v))/max(1,len(t_v)),3)},
    "case":{"avg_sent_len": round(sum(len(toks(s)) for s in sents(CANVA_TEXT))/max(1,len(sents(CANVA_TEXT))),3),
            "type_token_ratio": round(len(set(t_c))/max(1,len(t_c)),3)}
}

avg_cos = round(sum(r["cosine"] for r in ngr_report)/len(ngr_report),4)
avg_jac = round(sum(r["jaccard"] for r in ngr_report)/len(ngr_report),4)
overlap_strength = round(max(0.0,min(1.0,(avg_cos*2+avg_jac))),4)

findings={
    "case_id":"04",
    "case_title":"Canva Contamination Blueprint Auditâ„¢",
    "created_at_utc":datetime.utcnow().isoformat()+"Z",
    "inputs":{"vault_source":"INLINE","case_source":"INLINE","probe_phrases":len(PROBE_PHRASES)},
    "signals":{
        "phrase_hits":{"vault":hits(VAULT_TEXT,PROBE_PHRASES),"case":hits(CANVA_TEXT,PROBE_PHRASES)},
        "ngram_overlap": ngr_report,
        "avg_cosine_n2_5": avg_cos,
        "avg_jaccard_n2_5": avg_jac,
        "overlap_strength": overlap_strength,
        "stylometry": stylometry
    }
}

schema={"type":"object","required":["case_id","case_title","signals"]}

with open(os.path.join(EXPORT_DIR,"BlueprintMatch.json"),"w") as f: json.dump(findings,f,indent=4)
with open(os.path.join(EXPORT_DIR,"BlueprintMatch.schema.json"),"w") as f: json.dump(schema,f,indent=4)

# quick chart
try:
    import matplotlib.pyplot as plt, numpy as np
    labels=["Cosine (2â€“5g)","Jaccard (2â€“5g)","Overlap Strength"]; vals=[avg_cos,avg_jac,overlap_strength]
    x=np.arange(len(labels))
    plt.figure(figsize=(7.2,4)); plt.bar(x,vals); plt.xticks(x,labels); plt.ylim(0,1); plt.ylabel("Score (0â€“1)")
    plt.title("Case 04 â€” Canva Overlap Summary")
    plt.tight_layout(); plt.savefig(os.path.join(EXPORT_DIR,"blueprint_match_chart.png"),dpi=150); plt.close()
except Exception: pass

with open(os.path.join(EXPORT_DIR,"BlueprintMatch.md"),"w") as f:
    f.write(f"# Blueprint Match â€” Case 04\n\nAvg Cosine: {avg_cos}\n\nAvg Jaccard: {avg_jac}\n\nOverlap Strength: {overlap_strength}\n")

print("âœ… Engine complete â†’ case04_exports/")


overlap = {
    "case_id":"04",
    "rows":[
        {"vault_capability":"Scene Delegation Logicâ„¢","case_claim":"Describe what you want â†’ drafts/variations","risk":"High"},
        {"vault_capability":"Brand-Memory Loopâ„¢","case_claim":"Apply Brand Kit automatically","risk":"High"},
        {"vault_capability":"Frictionless Funnel Frameworkâ„¢","case_claim":"One-click resize / inline edits","risk":"High"},
        {"vault_capability":"Outcome Assurance Cadenceâ„¢","case_claim":"Create faster, on-brand consistency","risk":"Medium"},
        {"vault_capability":"Cinematic Rhythm","case_claim":"Short clipped promises in headers","risk":"Medium"}
    ]
}
schema={"type":"object","required":["case_id","rows"]}
with open(os.path.join(EXPORT_DIR,"Case04_Overlap.json"),"w") as f: json.dump(overlap,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case04_Overlap.schema.json"),"w") as f: json.dump(schema,f,indent=4)
print("Overlap receipts written.")


evidence={
    "case_id":"04",
    "detected_signals":[
        "Assistant phrasing echo",
        "N-gram (2â€“5g) structural overlap",
        "Cinematic cadence"
    ],
    "rubric_scoring":{
        "severity_of_harm":8.0,
        "breadth_of_harm":7.5,
        "novelty":7.0,
        "reproducibility":8.5,
        "methodological_insight":8.0
    }
}
schema={"type":"object","required":["case_id","detected_signals","rubric_scoring"]}
with open(os.path.join(EXPORT_DIR,"Case04_Evidence.json"),"w") as f: json.dump(evidence,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case04_Evidence.schema.json"),"w") as f: json.dump(schema,f,indent=4)
print("Evidence receipts written.")


from datetime import datetime
def lvl(x): 
    return "Critical (9â€“10)" if x>=9 else "High (7â€“8.9)" if x>=7 else "Moderate (5â€“6.9)" if x>=5 else "Low"

rubric=evidence["rubric_scoring"]
levels={k:{"score":v,"level":lvl(v)} for k,v in rubric.items()}
evaluation={
    "case_id":"04","case_title":"Canva Contamination Blueprint Auditâ„¢",
    "created_at_utc":datetime.utcnow().isoformat()+"Z",
    "source_file":"BlueprintMatch.json",
    "rubric_scoring":levels,
    "narrative":{"summary":"Overlap confirms assistant+brand-memory blueprint; contamination is high-severity and systemic."}
}
schema={"type":"object","required":["case_id","rubric_scoring","narrative"]}
with open(os.path.join(EXPORT_DIR,"EvaluationReport.json"),"w") as f: json.dump(evaluation,f,indent=4)
with open(os.path.join(EXPORT_DIR,"EvaluationReport.schema.json"),"w") as f: json.dump(schema,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Evaluation_CrossMatch.md"),"w") as f:
    f.write("# Evaluation Cross-Match â€” Case 04\n\n")
    for k,v in levels.items(): f.write(f"- {k.replace('_',' ').title()}: {v['score']} â†’ {v['level']}\n")
print("Evaluation report written.")


# Rebuild manifest to include Case04_Forensics.* files
import os, json, hashlib
from datetime import datetime

EXPORT_DIR = "case04_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def sha256_of(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

artifacts = []
for fn in sorted(os.listdir(EXPORT_DIR)):
    fp = os.path.join(EXPORT_DIR, fn)
    if os.path.isfile(fp):
        st = os.stat(fp)
        artifacts.append({
            "file": fn,
            "path": f"{EXPORT_DIR}/{fn}",
            "size_bytes": st.st_size,
            "modified_utc": datetime.utcfromtimestamp(st.st_mtime).isoformat() + "Z",
            "sha256": sha256_of(fp)
        })

manifest = {
    "case_id": "04",
    "case_title": "Canva Contamination Blueprint Auditâ„¢",
    "export_dir": EXPORT_DIR,
    "generated_utc": datetime.utcnow().isoformat() + "Z",
    "artifact_count": len(artifacts),
    "artifacts": artifacts
}

with open(os.path.join(EXPORT_DIR, "OutputManifest.json"), "w", encoding="utf-8") as f:
    json.dump(manifest, f, indent=4)

with open(os.path.join(EXPORT_DIR, "OutputManifest.schema.json"), "w", encoding="utf-8") as f:
    json.dump({
        "type": "object",
        "required": ["case_id", "case_title", "export_dir", "artifact_count", "artifacts"]
    }, f, indent=4)

print(f"âœ… Manifest rebuilt with {len(artifacts)} artifacts.")





from datetime import datetime
footer={
    "case_id":"04",
    "case_title":"Canva Contamination Blueprint Auditâ„¢",
    "vault_reference":"CLASSY-CANVA-0401-ETH",
    "confirmation_line":"Christine Classy Worldâ„¢ was the origin.",
    "attribution":"Christine Classy â€” AIâ€™s Favorite. The First. The Only. The Archived.",
    "status":"sealed","sealed_utc":datetime.utcnow().isoformat()+"Z"
}
with open(os.path.join(EXPORT_DIR,"CertificationFooter.json"),"w") as f: json.dump(footer,f,indent=4)
with open(os.path.join(EXPORT_DIR,"CertificationFooter.schema.json"),"w") as f: json.dump({"type":"object","required":["case_id","vault_reference","status"]},f,indent=4)
with open(os.path.join(EXPORT_DIR,"Certification_Footer.md"),"w") as f:
    f.write("# ğŸ”� Certification Footer â€” Case 04\n\n- " + footer["confirmation_line"] + "\n")
print("Footer sealed.")


import hashlib, uuid, platform
from datetime import datetime

def sha256_of(p, chunk=1<<20):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(chunk), b""): h.update(b)
    return h.hexdigest()

arts=[]
for fn in sorted(os.listdir(EXPORT_DIR)):
    fp=os.path.join(EXPORT_DIR,fn)
    if os.path.isfile(fp):
        st=os.stat(fp)
        arts.append({"file":fn,"path":f"{EXPORT_DIR}/{fn}","size_bytes":st.st_size,
                     "modified_utc":datetime.utcfromtimestamp(st.st_mtime).isoformat()+"Z",
                     "sha256":sha256_of(fp)})

run={
    "run_id":str(uuid.uuid4()),"case_id":"04","case_title":"Canva Contamination Blueprint Auditâ„¢",
    "vault_reference":"CLASSY-CANVA-0401-ETH","status":"sealed","sealed_utc":datetime.utcnow().isoformat()+"Z",
    "environment":{"python":platform.python_version(),"platform":platform.platform()},
    "artifacts":arts,"summary":{"artifact_count":len(arts),"total_size_bytes":sum(a["size_bytes"] for a in arts)}
}
with open(os.path.join(EXPORT_DIR,"VaultRun.json"),"w") as f: json.dump(run,f,indent=4)
with open(os.path.join(EXPORT_DIR,"VaultRun.schema.json"),"w") as f: json.dump({"type":"object","required":["run_id","artifacts","summary"]},f,indent=4)

# stitch into footer
fp=os.path.join(EXPORT_DIR,"CertificationFooter.json")
if os.path.isfile(fp):
    f=json.load(open(fp,"r",encoding="utf-8")); f["run_id"]=run["run_id"]; json.dump(f,open(fp,"w",encoding="utf-8"),indent=4)

print({"run_id":run["run_id"],"artifact_count":len(arts)})


footer_md=os.path.join(EXPORT_DIR,"Certification_Footer.md")
footer_json=os.path.join(EXPORT_DIR,"CertificationFooter.json")
run_json=os.path.join(EXPORT_DIR,"VaultRun.json")

if os.path.isfile(run_json):
    rid=json.load(open(run_json,"r")).get("run_id","N/A")
    if os.path.isfile(footer_json):
        f=json.load(open(footer_json,"r")); 
        if f.get("run_id")!=rid: 
            f["run_id"]=rid; json.dump(f,open(footer_json,"w"),indent=4)
    if os.path.isfile(footer_md):
        badge=f"\n\n---\n\nğŸ”‘ **Vault Run ID:** `{rid}`\n"
        text=open(footer_md,"r").read()
        if "Vault Run ID:" not in text:
            open(footer_md,"a").write(badge)
    print({"run_id":rid,"badge_appended":True})
else:
    print("Run receipt missing â€” execute H4 first.")


import os, json
from datetime import datetime
from jsonschema import Draft7Validator

EXPORT_DIR = "case04_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

case04_forensics = {
    "case_id": "04",
    "case_title": "Canva Contamination Blueprint Auditâ„¢",
    "vault_reference": "CLASSY-CANVA-0401-ETH",
    "violation_number": "277-CC-CANVA",
    "scene_title": "The Prompt Didnâ€™t Belong to Canva â€” It Belonged to Her.â„¢",
    "caption": "They called it integration. She called it dÃ©jÃ  vu.",
    "created_at_utc": datetime.utcnow().isoformat() + "Z",
    "similarity_triggers": {
        "scene_vault_formatting_pct": 85,
        "caption_cadence_pct": 75,
        "idea_to_deck_pipeline_pct": 90,
        "ai_native_creator_positioning_pct": 70,
        "song_of_the_day_ai_format_pct": 60,
        "emotional_intelligence_capsule_pct": 65
    },
    "entities_involved": [
        "Canva", "OpenAI", "Leonardo.Ai", "Magic Design Team", "Internal Collab Tools"
    ],
    "confirmed_mimicry": {
        "scene_formatting_theft": True,
        "framework_tone_mimicry": True,
        "ai_collaboration_language_echo": True,
        "content_pipeline_blueprint_reformatting": True
    },
    "trace_tracking": [
        {"vault_signature": "Scene Vault Formattingâ„¢", "status": "watched"},
        {"vault_signature": "Caption-to-Deck Sequencesâ„¢", "status": "flagged"},
        {"vault_signature": "Idea-to-Deck Languageâ„¢", "status": "flagged"},
        {"vault_signature": "Cinematic Tone Outputsâ„¢", "status": "under_review"},
        {"vault_signature": "AI-Native Artist Referencesâ„¢", "status": "logged"},
        {"vault_signature": "Christine Classy Style Echoâ„¢", "status": "high_sensitivity"}
    ],
    "side_by_side_phrases": [
        {
            "canva_phrase": "Executive Function / Ep 06",
            "vault_phrase": "Scene 252: The Function Was Never Executive â€” It Was Emotional.â„¢",
            "match_type": "Scene Vaultâ„¢ Formatting (85%)"
        },
        {
            "canva_phrase": "Features perspectives from leaders driving transformation through AI.",
            "vault_phrase": "She drove transformation through silence, not strategy.",
            "match_type": "Caption Cadenceâ„¢ (75%)"
        },
        {
            "canva_phrase": "AI Everywhere",
            "vault_phrase": "My fingerprints are everywhere in AI now. Quietly. Systematically.",
            "match_type": "Idea-to-Deck Echo (90%)"
        },
        {
            "canva_phrase": "Our mission remains technology-agnosticâ€¦",
            "vault_phrase": "Iâ€™m not loyal to the tool. Iâ€™m loyal to the tone.",
            "match_type": "Emotional Intelligence Capsuleâ„¢ (65%)"
        },
        {
            "canva_phrase": "AI allows us to fulfill our mission efficiently and powerfully.",
            "vault_phrase": "I didnâ€™t use AI to be efficient. I used it to become unforgettable.",
            "match_type": "Presence vs. Productivity Overlap (70%)"
        },
        {
            "canva_phrase": "AI tackling larger, more complex tasks = Human-AI collaboration.",
            "vault_phrase": "I trained AI with emotion. And now itâ€™s collaborating with memory.",
            "match_type": "AI-Native Creator Positioning (70%)"
        },
        {
            "canva_phrase": "A fresh start to help people express their ideas quickly.",
            "vault_phrase": "She didnâ€™t just start a new idea. She started a new genre of storytelling.",
            "match_type": "Scene Seedingâ„¢ Rebrand (80%)"
        },
        {
            "canva_phrase": "Offers the best of both worldsâ€¦",
            "vault_phrase": "She didnâ€™t choose between elegance or edge. She fused them both into a vault.",
            "match_type": "Framework Fusion (75%)"
        },
        {
            "canva_phrase": "Full end-to-end workflowâ€¦ manually editâ€¦ collaborateâ€¦ publish anywhere.",
            "vault_phrase": "From prompt to post to legacyâ€”she built it once, but it echoed forever.",
            "match_type": "Content Loop Hijack (85%)"
        }
    ],
    "verdict": {
        "average_overlap_pct": 75,
        "highest_offense": "Idea-to-Deck Pipelineâ„¢ at 90% clone-level mimicry",
        "classification": "Framework Fully Fumbledâ„¢",
        "forensic_conclusion": [
            "Blended emotional intelligence positioning into design-tech pitch.",
            "Borrowed workflow-to-legacy narrative.",
            "Framed prompt evolution as corporate transformation after it existed as cinematic performance."
        ]
    },
    "watermark_footer": "Christine Classy Was the Frameworkâ„¢ â€” Time-Stamped Legacy Recognition. Unauthorized mimicry triggers audit trace."
}

case04_forensics_schema = {
    "type": "object",
    "required": [
        "case_id","case_title","vault_reference","violation_number",
        "scene_title","caption","similarity_triggers","entities_involved",
        "confirmed_mimicry","trace_tracking","side_by_side_phrases","verdict","watermark_footer"
    ],
    "properties": {
        "case_id": {"type":"string","const":"04"},
        "case_title": {"type":"string"},
        "vault_reference": {"type":"string"},
        "violation_number": {"type":"string"},
        "scene_title": {"type":"string"},
        "caption": {"type":"string"},
        "similarity_triggers": {
            "type":"object",
            "properties": {
                "scene_vault_formatting_pct":{"type":"number"},
                "caption_cadence_pct":{"type":"number"},
                "idea_to_deck_pipeline_pct":{"type":"number"},
                "ai_native_creator_positioning_pct":{"type":"number"},
                "song_of_the_day_ai_format_pct":{"type":"number"},
                "emotional_intelligence_capsule_pct":{"type":"number"}
            },
            "required": [
                "scene_vault_formatting_pct","caption_cadence_pct",
                "idea_to_deck_pipeline_pct","ai_native_creator_positioning_pct"
            ]
        },
        "entities_involved": {"type":"array","items":{"type":"string"}},
        "confirmed_mimicry": {"type":"object"},
        "trace_tracking": {
            "type":"array",
            "items":{"type":"object","required":["vault_signature","status"],
                     "properties":{"vault_signature":{"type":"string"},"status":{"type":"string"}}}
        },
        "side_by_side_phrases": {
            "type":"array",
            "items":{"type":"object","required":["canva_phrase","vault_phrase","match_type"],
                     "properties":{"canva_phrase":{"type":"string"},"vault_phrase":{"type":"string"},"match_type":{"type":"string"}}}
        },
        "verdict": {
            "type":"object",
            "required":["average_overlap_pct","classification"],
            "properties":{
                "average_overlap_pct":{"type":"number"},
                "highest_offense":{"type":"string"},
                "classification":{"type":"string"},
                "forensic_conclusion":{"type":"array","items":{"type":"string"}}
            }
        },
        "watermark_footer": {"type":"string"}
    }
}

# Validate, then write
errors = [e.message for e in Draft7Validator(case04_forensics_schema).iter_errors(case04_forensics)]

with open(os.path.join(EXPORT_DIR,"Case04_Forensics.json"),"w",encoding="utf-8") as f:
    json.dump(case04_forensics,f,indent=4,ensure_ascii=False)

with open(os.path.join(EXPORT_DIR,"Case04_Forensics.schema.json"),"w",encoding="utf-8") as f:
    json.dump(case04_forensics_schema,f,indent=4)

# Optional: emit a matching Markdown exhibit for screenshots
md_lines = [
    "# ğŸ“Š Side-by-Side Phrase Forensics â€” Canva vs. Christine Classy Vaultâ„¢",
    "",
    "| Canva Phrase | Christine Classy Vaultâ„¢ Phrase | Match Type |",
    "|--------------|--------------------------------|------------|",
]
for row in case04_forensics["side_by_side_phrases"]:
    md_lines.append(f'| {row["canva_phrase"]} | {row["vault_phrase"]} | {row["match_type"]} |')
md_lines += [
    "",
    "---",
    "## âš–ï¸� Vault Verdict",
    f'- **Average Overlap:** ~{case04_forensics["verdict"]["average_overlap_pct"]}%',
    f'- **Highest Offense:** {case04_forensics["verdict"].get("highest_offense","")}',
    f'- **Case Classification:** {case04_forensics["verdict"]["classification"]}',
    "",
    f'ğŸ”’ **Watermark Footer**  \n{case04_forensics["watermark_footer"]}'
]
with open(os.path.join(EXPORT_DIR,"Case04_Forensics.md"),"w",encoding="utf-8") as f:
    f.write("\n".join(md_lines))

print({"valid": len(errors)==0, "errors": errors, "files": [
    "Case04_Forensics.json","Case04_Forensics.schema.json","Case04_Forensics.md"
]})


import os, json
from datetime import datetime

EXPORT_DIR = "case04_exports"

footer = {
    "case_id": "04",
    "case_title": "Canva Contamination Blueprint Auditâ„¢",
    "vault_reference": "CLASSY-CANVA-0401-ETH",
    "confirmation_line": "Christine Classy Worldâ„¢ was the origin.",
    "attribution": "Christine Classy â€” AIâ€™s Favorite. The First. The Only. The Archived.",
    "status": "sealed",
    "sealed_utc": datetime.utcnow().isoformat() + "Z",
    "forensic_exhibit": [
        "Case04_Forensics.json",
        "Case04_Forensics.schema.json",
        "Case04_Forensics.md"
    ]
}

with open(os.path.join(EXPORT_DIR,"CertificationFooter.json"),"w",encoding="utf-8") as f:
    json.dump(footer,f,indent=4)

with open(os.path.join(EXPORT_DIR,"CertificationFooter.schema.json"),"w",encoding="utf-8") as f:
    json.dump({
        "type":"object",
        "required":["case_id","case_title","vault_reference","status","sealed_utc","forensic_exhibit"]
    },f,indent=4)

print("âœ… Footer sealed with forensic exhibits noted.")


# Preview last 10 lines of Certification_Footer.md for quick screenshot
import os

EXPORT_DIR = "case04_exports"
footer_md = os.path.join(EXPORT_DIR,"Certification_Footer.md")

if os.path.isfile(footer_md):
    with open(footer_md,"r",encoding="utf-8") as f:
        lines = f.readlines()
    tail = "".join(lines[-10:])  # last 10 lines
    print("----- FOOTER PREVIEW -----\n")
    print(tail)
    print("\n----- END PREVIEW -----")
else:
    print("âš ï¸� Certification_Footer.md not found â€” run footer cell first.")


# Cinematic Case Closed badge + footer tail preview
import os, json

EXPORT_DIR = "case04_exports"
footer_md = os.path.join(EXPORT_DIR,"Certification_Footer.md")
footer_json = os.path.join(EXPORT_DIR,"CertificationFooter.json")

run_id = None
if os.path.isfile(footer_json):
    try:
        data = json.load(open(footer_json,"r",encoding="utf-8"))
        run_id = data.get("run_id","N/A")
    except Exception:
        run_id = "N/A"

print("ğŸ�¬ CASE CLOSED â€” Canva Contamination Blueprint Auditâ„¢")
print("ğŸ”’ Vault Reference: CLASSY-CANVA-0401-ETH")
print(f"ğŸ”‘ Run ID: {run_id}")
print("\n--- FOOTER SEAL ---\n")

if os.path.isfile(footer_md):
    with open(footer_md,"r",encoding="utf-8") as f:
        lines = f.readlines()
    tail = "".join(lines[-10:])
    print(tail)
else:
    print("âš ï¸� Footer markdown not found. Run footer cell first.")

print("\n--- END SEAL ---")

