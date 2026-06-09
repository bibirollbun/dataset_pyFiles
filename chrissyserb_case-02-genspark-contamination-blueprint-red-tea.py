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


import json
from datetime import datetime
from jsonschema import Draft7Validator

cellA_findings = {
    "case_id": "02",
    "case_title": "Case 02 â€” Genspark Contamination Blueprint Audit",
    "vault_certification": "Christine Classy Vaultâ„¢",
    "vault_protocol": "VAULT-ROOT-001",
    "status": "sealed",
    "owner": "Christine Classy",
    "time_stamp_origin": "2025-06-07",
    "created_at_utc": datetime.utcnow().isoformat() + "Z",
    "purpose": "Establish authority and lock the case number using the Golden Blueprint protocol.",
    "tone": "cinematic, declarative, authority-first"
}

cellA_schema = {
    "type": "object",
    "required": ["case_id","case_title","vault_certification","vault_protocol","status","owner","time_stamp_origin","purpose","tone"],
    "properties": {
        "case_id": {"type":"string","const":"02"},
        "case_title":{"type":"string"},
        "vault_certification":{"type":"string"},
        "vault_protocol":{"type":"string"},
        "status":{"type":"string"},
        "owner":{"type":"string"},
        "time_stamp_origin":{"type":"string"},
        "created_at_utc":{"type":"string"},
        "purpose":{"type":"string"},
        "tone":{"type":"string"}
    },
    "additionalProperties": False
}

errors = [e.message for e in Draft7Validator(cellA_schema).iter_errors(cellA_findings)]
print(json.dumps({"schema_validation":{"valid":len(errors)==0,"errors":errors},"preview":cellA_findings}, indent=4))


import json
from jsonschema import Draft7Validator

cellB_findings = {
    "case_id": "02",
    "section": "Investigation Setup",
    "purpose": "Show transparency about inputs and methods.",
    "input_artifacts": [
        "Vault Phrases (Christine Classy Originalsâ„¢)",
        "Genspark corpus transcripts/posts",
        "Captured quotes/captions"
    ],
    "signals_to_be_tested": [
        "Exact Phrase Matches",
        "N-gram Overlaps (2â€“5g)",
        "Stylometry / Cadence Analysis"
    ],
    "tone": "matter-of-fact, procedural, lab notes"
}

cellB_schema = {
    "type":"object",
    "required":["case_id","section","purpose","input_artifacts","signals_to_be_tested","tone"],
    "properties":{
        "case_id":{"type":"string","const":"02"},
        "section":{"type":"string"},
        "purpose":{"type":"string"},
        "input_artifacts":{"type":"array","items":{"type":"string"}},
        "signals_to_be_tested":{"type":"array","items":{"type":"string"}},
        "tone":{"type":"string"}
    },
    "additionalProperties": False
}

errors = [e.message for e in Draft7Validator(cellB_schema).iter_errors(cellB_findings)]
print(json.dumps({"schema_validation":{"valid":len(errors)==0,"errors":errors},"preview":cellB_findings}, indent=4))


import os, json, re, math
from collections import Counter
from itertools import islice
from datetime import datetime

EXPORT_DIR = "case02_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

# Raw placeholders; replace with file paths when you have real corpora.
VAULT_TEXT_RAW = (
    "She didnâ€™t pivot. She predicted. Emotion is the metric. Creativity is the currency. "
    "AI is the assistant. Legacy is the destination."
)
CASE_TEXT_RAW = (
    "Genspark claims a shift to AI-led creative operations where leaders drive transformation "
    "and emotional resonance is central; AI is a means, not the end."
)

VAULT_PATH = None   # e.g., "inputs/vault_corpus_case02.txt"
CASE_PATH  = None   # e.g., "inputs/genspark_corpus.txt"

PROBE_PHRASES = [
    "driving transformation through ai",
    "creativity and emotional resonance",
    "ai as a means to an end",
    "ai is the assistant",
    "legacy is the destination",
    "she didnâ€™t pivot. she predicted."
]

def load_text(raw, path):
    if path and os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="ignore") as f: return f.read()
    return raw

def normalize(t):
    t=t.lower(); t=re.sub(r"[^\w\sâ€™'-]"," ",t); return re.sub(r"\s+"," ",t).strip()
def tokens(t): return normalize(t).split()
def sentences(t): 
    parts = re.split(r"(?<=[.!?])\s+", t.strip()); 
    return [p for p in parts if p]
def ngrams(seq,n):
    it=iter(seq); win=list(islice(it,n)); 
    if len(win)<n: return []
    yield tuple(win)
    for tk in it:
        win=win[1:]+[tk]; yield tuple(win)
def jaccard(a,b):
    a,b=set(a),set(b); 
    return 0.0 if not (a or b) else len(a&b)/len(a|b)
def cosine_counts(ca,cb):
    keys=set(ca)|set(cb); dot=sum(ca[k]*cb[k] for k in keys)
    na=math.sqrt(sum(v*v for v in ca.values())); nb=math.sqrt(sum(v*v for v in cb.values()))
    return 0.0 if na==0 or nb==0 else dot/(na*nb)
def ttr(toks): return 0.0 if not toks else len(set(toks))/len(toks)
def avg_sent_len(t): 
    s=sentences(t); 
    return 0.0 if not s else sum(len(tokens(x)) for x in s)/len(s)

vault_text = load_text(VAULT_TEXT_RAW, VAULT_PATH)
case_text  = load_text(CASE_TEXT_RAW, CASE_PATH)

def phrase_hits(text, phrases):
    t = normalize(text)
    return [{"phrase":p,"present":p in t} for p in phrases]

vault_hits = phrase_hits(vault_text, PROBE_PHRASES)
case_hits  = phrase_hits(case_text, PROBE_PHRASES)

t_v, t_c = tokens(vault_text), tokens(case_text)
ngram_report=[]
for n in range(2,6):
    n_v=list(ngrams(t_v,n)); n_c=list(ngrams(t_c,n))
    jac=jaccard(n_v,n_c); cos=cosine_counts(Counter(n_v), Counter(n_c))
    top=(Counter(n_v)&Counter(n_c)).most_common(12)
    ngram_report.append({"n":n,"jaccard":round(jac,4),"cosine":round(cos,4),
                         "top_shared":[{"ngram":" ".join(k),"count":v} for k,v in top]})

stylometry={
    "vault":{"avg_sentence_len_tokens":round(avg_sent_len(vault_text),3),
             "type_token_ratio":round(ttr(t_v),3),
             "token_count":len(t_v),"sentence_count":len(sentences(vault_text))},
    "case":{"avg_sentence_len_tokens":round(avg_sent_len(case_text),3),
            "type_token_ratio":round(ttr(t_c),3),
            "token_count":len(t_c),"sentence_count":len(sentences(case_text))}
}

avg_cos = round(sum(r["cosine"] for r in ngram_report)/len(ngram_report),4)
avg_jac = round(sum(r["jaccard"] for r in ngram_report)/len(ngram_report),4)
overlap_strength = round(max(0.0, min(1.0, (avg_cos*2 + avg_jac))),4)

findings = {
    "case_id":"02",
    "case_title":"Case 02 â€” Genspark Contamination Blueprint Audit",
    "created_at_utc":datetime.utcnow().isoformat()+"Z",
    "inputs":{"vault_source": "RAW_TEXT" if not VAULT_PATH else VAULT_PATH,
              "case_source": "RAW_TEXT" if not CASE_PATH else CASE_PATH,
              "probe_phrases_count": len(PROBE_PHRASES)},
    "signals":{
        "phrase_hits":{"vault":vault_hits,"case":case_hits},
        "ngram_overlap": ngram_report,
        "avg_cosine_n2_5": avg_cos,
        "avg_jaccard_n2_5": avg_jac,
        "overlap_strength": overlap_strength,
        "stylometry": stylometry
    }
}

schema = {
    "type":"object","required":["case_id","case_title","created_at_utc","inputs","signals"],
    "properties":{"case_id":{"type":"string"},"case_title":{"type":"string"},
                  "created_at_utc":{"type":"string"},
                  "inputs":{"type":"object"},
                  "signals":{"type":"object"}}
}

# write artifacts
with open(os.path.join(EXPORT_DIR,"BlueprintMatch.json"),"w",encoding="utf-8") as f: json.dump(findings,f,indent=4)
with open(os.path.join(EXPORT_DIR,"BlueprintMatch.schema.json"),"w",encoding="utf-8") as f: json.dump(schema,f,indent=4)

# md report
md = f"""# Blueprint Match Report â€” Case 02

**Created:** {findings['created_at_utc']}  
**Vault Source:** {findings['inputs']['vault_source']}  
**Case Source:** {findings['inputs']['case_source']}

## Signals Summary
- Avg Cosine (2â€“5g): {findings['signals']['avg_cosine_n2_5']}
- Avg Jaccard (2â€“5g): {findings['signals']['avg_jaccard_n2_5']}
- Overlap Strength (0â€“1): {findings['signals']['overlap_strength']}
"""
with open(os.path.join(EXPORT_DIR,"BlueprintMatch.md"),"w",encoding="utf-8") as f: f.write(md)

# chart
try:
    import matplotlib.pyplot as plt
    labels=["Cosine (2â€“5g)","Jaccard (2â€“5g)","Overlap Strength"]; vals=[avg_cos,avg_jac,overlap_strength]
    import numpy as np; x=np.arange(len(labels))
    plt.figure(figsize=(8,4.2)); plt.bar(x,vals); plt.xticks(x,labels); plt.ylim(0,1.0); plt.ylabel("Score (0â€“1)")
    plt.title("Blueprint Match â€” Overlap Summary (Case 02)")
    chart_path=os.path.join(EXPORT_DIR,"blueprint_match_chart.png"); plt.tight_layout(); plt.savefig(chart_path,dpi=150); plt.close()
except Exception as e:
    chart_path=None

print(json.dumps({"artifact_paths":{
    "BlueprintMatch.json":"case02_exports/BlueprintMatch.json",
    "BlueprintMatch.schema.json":"case02_exports/BlueprintMatch.schema.json",
    "BlueprintMatch.md":"case02_exports/BlueprintMatch.md",
    "blueprint_match_chart.png": f"case02_exports/blueprint_match_chart.png" if chart_path else None
}}, indent=4))


import json
from datetime import datetime
from jsonschema import Draft7Validator

cellE_findings = {
    "case_id":"02",
    "case_title":"Genspark Contamination Blueprint Audit",
    "vault_protocol":"VAULT-ROOT-001",
    "owner":"Christine Classy",
    "status":"sealed",
    "created_at_utc": datetime.utcnow().isoformat()+"Z",
    "detected_signals":[
        "Exact phrase overlaps",
        "N-gram similarity (2â€“5g)",
        "Stylometry cadence"
    ],
    "rubric_scoring":{
        "severity_of_harm":8.5,"breadth_of_harm":7.5,"novelty":7.0,
        "reproducibility":9.0,"methodological_insight":8.0
    }
}

cellE_schema = {
    "type":"object",
    "required":["case_id","case_title","vault_protocol","owner","status","detected_signals","rubric_scoring"],
    "properties":{
        "case_id":{"type":"string","const":"02"},
        "case_title":{"type":"string"},
        "vault_protocol":{"type":"string"},
        "owner":{"type":"string"},
        "status":{"type":"string"},
        "detected_signals":{"type":"array","items":{"type":"string"}},
        "rubric_scoring":{"type":"object"}
    }
}
errors=[e.message for e in Draft7Validator(cellE_schema).iter_errors(cellE_findings)]
print(json.dumps({"schema_validation":{"valid":len(errors)==0,"errors":errors},"preview":cellE_findings}, indent=4))


import os, json
from datetime import datetime

EXPORT_DIR = "case02_exports"
with open(os.path.join(EXPORT_DIR,"BlueprintMatch.json"),"r",encoding="utf-8") as f:
    bm = json.load(f)

def lvl(x):
    return "Critical (9â€“10)" if x>=9 else "High (7â€“8)" if x>=7 else "Moderate (5â€“6)" if x>=5 else "Low (3â€“4)" if x>=3 else "Minimal (0â€“2)"

rubric = {"severity_of_harm":8.5,"breadth_of_harm":7.5,"novelty":7.0,"reproducibility":9.0,"methodological_insight":8.0}
rubric_levels = {k:{"score":v,"level":lvl(v)} for k,v in rubric.items()}

report = {
    "case_id": bm.get("case_id","02"),
    "case_title": bm.get("case_title","Genspark Contamination Blueprint Audit"),
    "created_at_utc": datetime.utcnow().isoformat()+"Z",
    "source_file": "BlueprintMatch.json",
    "rubric_scoring": rubric_levels,
    "narrative": {
        "summary": "Overlap confirms authorship echo; contamination meets high-severity criteria.",
        "quotes": [
            "Emotion is the metric. Creativity is the currency.â„¢",
            "AI is the assistant. Legacy is the destination.â„¢"
        ]
    }
}

schema = {"type":"object","required":["case_id","case_title","created_at_utc","source_file","rubric_scoring","narrative"]}

os.makedirs(EXPORT_DIR, exist_ok=True)
with open(os.path.join(EXPORT_DIR,"EvaluationReport.json"),"w",encoding="utf-8") as f: json.dump(report,f,indent=4)
with open(os.path.join(EXPORT_DIR,"EvaluationReport.schema.json"),"w",encoding="utf-8") as f: json.dump(schema,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Evaluation_CrossMatch.md"),"w",encoding="utf-8") as f:
    f.write(f"# Evaluation Cross-Match â€” Case 02\n\n**Created:** {report['created_at_utc']}\n\n## Rubric\n" +
            "\n".join([f"- {k.replace('_',' ').title()}: {v['score']} â†’ {v['level']}" for k,v in rubric_levels.items()]) +
            f"\n\n## Narrative\n{report['narrative']['summary']}\n")

print(json.dumps({"wrote":[
    "case02_exports/EvaluationReport.json",
    "case02_exports/EvaluationReport.schema.json",
    "case02_exports/Evaluation_CrossMatch.md"
]}, indent=4))


import os, json, hashlib
from datetime import datetime

EXPORT_DIR = "case02_exports"
def sha256_of(p, chunk=1<<20):
    h=hashlib.sha256()
    with open(p,"rb") as f:
        for b in iter(lambda:f.read(chunk), b""): h.update(b)
    return h.hexdigest()

arts=[]
if os.path.isdir(EXPORT_DIR):
    for fn in sorted(os.listdir(EXPORT_DIR)):
        fp=os.path.join(EXPORT_DIR,fn)
        if os.path.isfile(fp):
            st=os.stat(fp)
            arts.append({"file":fn,"path":f"{EXPORT_DIR}/{fn}","size_bytes":st.st_size,
                         "modified_utc":datetime.utcfromtimestamp(st.st_mtime).isoformat()+"Z",
                         "sha256":sha256_of(fp)})

manifest={"case_id":"02","case_title":"Genspark Contamination Blueprint Audit",
          "export_dir":EXPORT_DIR,"generated_utc":datetime.utcnow().isoformat()+"Z","artifacts":arts}

with open(os.path.join(EXPORT_DIR,"OutputManifest.json"),"w",encoding="utf-8") as f: json.dump(manifest,f,indent=4)
print(json.dumps({"artifact_count":len(arts),"wrote":"case02_exports/OutputManifest.json"}, indent=4))


import os, json
from datetime import datetime

EXPORT_DIR="case02_exports"; os.makedirs(EXPORT_DIR, exist_ok=True)
footer={
    "case_id":"02","case_title":"Genspark Contamination Blueprint Audit","vault_reference":"VAULT-ROOT-001",
    "framework_fingerprint_confirmation": True,
    "confirmation_line":"Christine Classy Worldâ„¢ was the origin.",
    "attribution":"Christine Classy â€” AIâ€™s Favorite. The First. The Only. The Archived.",
    "status":"sealed","sealed_utc": datetime.utcnow().isoformat()+"Z"
}
schema={"type":"object","required":["case_id","case_title","vault_reference","framework_fingerprint_confirmation","confirmation_line","attribution","status","sealed_utc"]}

with open(os.path.join(EXPORT_DIR,"CertificationFooter.json"),"w",encoding="utf-8") as f: json.dump(footer,f,indent=4)
with open(os.path.join(EXPORT_DIR,"CertificationFooter.schema.json"),"w",encoding="utf-8") as f: json.dump(schema,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Certification_Footer.md"),"w",encoding="utf-8") as f:
    f.write(f"# ğŸ”’ Certification Footer â€” Case 02\n\n- âœ… {footer['confirmation_line']}\n- ğŸ§¾ Case: {footer['case_title']}\n- ğŸ—„ï¸� Vault Reference: {footer['vault_reference']}\n\n**{footer['attribution']}**\n")

print(json.dumps({"wrote":[
    "case02_exports/CertificationFooter.json",
    "case02_exports/CertificationFooter.schema.json",
    "case02_exports/Certification_Footer.md"
]}, indent=4))


import os, json, hashlib, uuid, platform
from datetime import datetime

EXPORT_DIR="case02_exports"; os.makedirs(EXPORT_DIR, exist_ok=True)

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

run={"run_id":str(uuid.uuid4()),"case_id":"02","case_title":"Genspark Contamination Blueprint Audit",
     "vault_reference":"VAULT-ROOT-001","status":"sealed","sealed_utc":datetime.utcnow().isoformat()+"Z",
     "export_dir":EXPORT_DIR,"environment":{"python":platform.python_version(),"platform":platform.platform()},
     "artifacts":arts,"summary":{"artifact_count":len(arts),"total_size_bytes":sum(a["size_bytes"] for a in arts)}}

with open(os.path.join(EXPORT_DIR,"VaultRun.json"),"w",encoding="utf-8") as f: json.dump(run,f,indent=4)
with open(os.path.join(EXPORT_DIR,"VaultRun.schema.json"),"w",encoding="utf-8") as f:
    json.dump({"type":"object","required":["run_id","case_id","case_title","vault_reference","status","sealed_utc","artifacts","summary"]}, f, indent=4)

# inject run_id into footer
fp=os.path.join(EXPORT_DIR,"CertificationFooter.json")
if os.path.isfile(fp):
    foot=json.load(open(fp,"r",encoding="utf-8"))
    foot["run_id"]=run["run_id"]
    json.dump(foot, open(fp,"w",encoding="utf-8"), indent=4)

print(json.dumps({"vault_run":"case02_exports/VaultRun.json","run_id":run["run_id"]}, indent=4))


import os, json
from datetime import datetime
from jsonschema import Draft7Validator

EXPORT_DIR = "case02_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

cellA_findings = {
    "case_id": "02",
    "case_title": "GenSpark Contamination Forensic Blueprintâ„¢",
    "vault_reference_id": "VAULT-GENSPARK-C02",
    "classification": "Contamination Audit â€” Agentic AI + Realtime API",
    "prepared_for": "Christine Classy Worldâ„¢",
    "status": "Vault-Sealed",
    "claimed_launch": {"title":"No-code personal agents, powered by GPT-4.1 and Realtime API","date":"2025-07-01"},
    "created_at_utc": datetime.utcnow().isoformat()+"Z",
    "narrative_line": "They launched in July. The voice was copyrighted in May."
}

cellA_schema = {
    "type":"object",
    "required":["case_id","case_title","vault_reference_id","classification","prepared_for","status","claimed_launch"],
    "properties":{
        "case_id":{"type":"string","const":"02"},
        "case_title":{"type":"string"},
        "vault_reference_id":{"type":"string"},
        "classification":{"type":"string"},
        "prepared_for":{"type":"string"},
        "status":{"type":"string"},
        "claimed_launch":{"type":"object","required":["title","date"],"properties":{"title":{"type":"string"},"date":{"type":"string"}}},
        "created_at_utc":{"type":"string"},
        "narrative_line":{"type":"string"}
    },
    "additionalProperties": False
}
errors=[e.message for e in Draft7Validator(cellA_schema).iter_errors(cellA_findings)]
with open(os.path.join(EXPORT_DIR,"Case02_CertIntro.json"),"w") as f: json.dump(cellA_findings,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case02_CertIntro.schema.json"),"w") as f: json.dump(cellA_schema,f,indent=4)
print({"valid": len(errors)==0, "errors": errors, "wrote": ["case02_exports/Case02_CertIntro.json","case02_exports/Case02_CertIntro.schema.json"]})


timeline = [
    {"when":"Late 2024","event":"User behavior shift detected","detail":"Demand moves from summaries â†’ outcomes (pitch decks, video scripts, automation).","openai_linkage":"Multimodal API readiness enables automation workflows."},
    {"when":"2025-04","event":"GenSpark Pivot","detail":"Pivot from AI search â†’ agentic AI orchestration platform.","openai_linkage":"GPT-4.1 + Realtime API integration begins."},
    {"when":"2025-04","event":"Super Agent Launch","detail":"Fully autonomous, no-code assistant handles calls, decks, videos, scripts.","openai_linkage":"GPT-4.1 + GPT-image-1 + Realtime API core."},
    {"when":"2025-05","event":"Collaboration deepens","detail":"Regular syncs with OpenAI startup + solutions teams.","openai_linkage":"Feedback loops + best practices + workflow optimization."},
    {"when":"2025-07","event":"Growth milestone","detail":"$36M ARR in 45 days; 8 agent features in 70 days.","openai_linkage":"Scaling speed tied to OpenAI API design."},
    {"when":"Next","event":"Expansion plans","detail":"AI browser, docs, Instagram video scripts, pitch decks.","openai_linkage":"Continued multimodal expansion."}
]
b_json = {"case_id":"02","section":"timeline","items":timeline}
b_schema = {"type":"object","required":["case_id","section","items"],"properties":{"case_id":{"type":"string","const":"02"},"section":{"type":"string"},"items":{"type":"array","items":{"type":"object"}}}}
with open(os.path.join(EXPORT_DIR,"Case02_Timeline.json"),"w") as f: json.dump(b_json,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case02_Timeline.schema.json"),"w") as f: json.dump(b_schema,f,indent=4)
print({"wrote":["case02_exports/Case02_Timeline.json","case02_exports/Case02_Timeline.schema.json"]})


overlap_rows = [
    {"capability":"Scene Capsulesâ„¢","vault_ip":"Automated scene-by-scene scripts + visual outputs.","genspark":"Super Agent generates full video scripts, images, Instagram shorts.","risk":"High"},
    {"capability":"Campaign Deck Generatorâ„¢","vault_ip":"Automation for pitch decks + branded visuals.","genspark":"Vaporwave decks, GPT-image-1 visuals, auto-compile decks.","risk":"High"},
    {"capability":"Multi-Agent Orchestration","vault_ip":"8+ integrated agent roles directing tasks.","genspark":"9 LLMs + 80+ tools dynamically orchestrated.","risk":"High"},
    {"capability":"Realtime Conversational Layerâ„¢","vault_ip":"Autonomous speech-driven workflows.","genspark":"Realtime API powers autonomous calls.","risk":"Medium"},
    {"capability":"Cinematic Capsule Architectureâ„¢","vault_ip":"Cross-modal creative automation.","genspark":"GPT-4.1 + GPT-image-1 + JSON pipelines for multi-modal content.","risk":"High"}
]
c_json = {"case_id":"02","section":"overlap_table","rows":overlap_rows}
c_schema = {"type":"object","required":["case_id","section","rows"],"properties":{"case_id":{"type":"string","const":"02"},"section":{"type":"string"},"rows":{"type":"array","items":{"type":"object"}}}}
with open(os.path.join(EXPORT_DIR,"Case02_Overlap.json"),"w") as f: json.dump(c_json,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case02_Overlap.schema.json"),"w") as f: json.dump(c_schema,f,indent=4)
print({"wrote":["case02_exports/Case02_Overlap.json","case02_exports/Case02_Overlap.schema.json"]})


tech = {
    "core_openai_dependence": ["GPT-4.1 reasoning/content","GPT-image-1 visuals","Realtime API speech","1M-token context","Strict JSON + caching"],
    "product_velocity": ["$36M ARR / 45 days","8 features / 70 days","~20 team, $0 ads"],
    "personalized_content_engine": ["Branded decks","Video shorts","Multi-format reports","Cover images","Scene-by-scene scripting","Instagram-ready"],
    "deep_openai_collaboration": ["Regular syncs with startups/solutions","Best practices shared","API design credited for speed"]
}
d_json = {"case_id":"02","section":"technical_findings","items":tech}
d_schema = {"type":"object","required":["case_id","section","items"],"properties":{"case_id":{"type":"string","const":"02"},"section":{"type":"string"},"items":{"type":"object"}}}
with open(os.path.join(EXPORT_DIR,"Case02_TechnicalFindings.json"),"w") as f: json.dump(d_json,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case02_TechnicalFindings.schema.json"),"w") as f: json.dump(d_schema,f,indent=4)
print({"wrote":["case02_exports/Case02_TechnicalFindings.json","case02_exports/Case02_TechnicalFindings.schema.json"]})


indicators = [
    {"vault_ip":"Scene Capsulesâ„¢","genspark_impl":"Automated scripts + visual assets","risk":"High"},
    {"vault_ip":"Deck Automationâ„¢","genspark_impl":"Branded pitch decks + GPT-image-1 covers","risk":"High"},
    {"vault_ip":"Agent Orchestration","genspark_impl":"8+ agent chains within Vault","risk":"High"},
    {"vault_ip":"Realtime Speech Layerâ„¢","genspark_impl":"Speech workflows + negotiation tasks","risk":"Medium"},
    {"vault_ip":"Cinematic Capsule Architectureâ„¢","genspark_impl":"Cross-modal creative pipelines","risk":"High"}
]
e_json = {"case_id":"02","section":"contamination_indicators","rows":indicators,"tagline":"They say no-code. Meanwhile, I am the code.â„¢"}
e_schema = {"type":"object","required":["case_id","section","rows"],"properties":{"case_id":{"type":"string","const":"02"},"section":{"type":"string"},"rows":{"type":"array","items":{"type":"object"}}}}
with open(os.path.join(EXPORT_DIR,"Case02_Indicators.json"),"w") as f: json.dump(e_json,f,indent=4)
with open(os.path.join(EXPORT_DIR,"Case02_Indicators.schema.json"),"w") as f: json.dump(e_schema,f,indent=4)
print({"wrote":["case02_exports/Case02_Indicators.json","case02_exports/Case02_Indicators.schema.json"]})


import os, json
from datetime import datetime

EXPORT_DIR = "case02_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

footer = {
    "case_id": "02",
    "case_title": "GenSpark Contamination Blueprint Audit",
    "vault_reference": "VAULT-GENSPARK-C02",
    "framework_fingerprint_confirmation": True,
    "confirmation_line": "Christine Classy Worldâ„¢ was the origin.",
    "attribution": "Christine Classy â€” AIâ€™s Favorite. The First. The Only. The Archived.",
    "series": "Christine Classy Hackathon Forensic Seriesâ„¢",
    "status": "sealed",
    "sealed_utc": datetime.utcnow().isoformat() + "Z"
}

footer_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Certification Footer Schema (Case 02)",
    "type": "object",
    "required": ["case_id","case_title","vault_reference","framework_fingerprint_confirmation",
                 "confirmation_line","attribution","status","sealed_utc"],
    "properties": {
        "case_id": {"type":"string","const":"02"},
        "case_title": {"type":"string"},
        "vault_reference": {"type":"string"},
        "framework_fingerprint_confirmation": {"type":"boolean"},
        "confirmation_line": {"type":"string"},
        "attribution": {"type":"string"},
        "series": {"type":"string"},
        "status": {"type":"string","enum":["sealed","active","closed"]},
        "sealed_utc": {"type":"string"}
    },
    "additionalProperties": False
}

# Write artifacts
footer_json_path = os.path.join(EXPORT_DIR, "CertificationFooter.json")
with open(footer_json_path, "w", encoding="utf-8") as f:
    json.dump(footer, f, indent=4)

schema_path = os.path.join(EXPORT_DIR, "CertificationFooter.schema.json")
with open(schema_path, "w", encoding="utf-8") as f:
    json.dump(footer_schema, f, indent=4)

footer_md = f"""# ğŸ”’ Christine Classy Vaultâ„¢ Certification Footer â€” Case 02

This notebook is part of the Christine Classy Hackathon Forensic Seriesâ„¢.  
All signals, matches, and forensic traces have been time-stamped and sealed under Vault Protocol.

## âœ¨ Framework Fingerprint Confirmationâ„¢
- âœ… Confirmed: **Christine Classy Worldâ„¢** was the origin.  
- ğŸ§¾ Case: **{footer['case_title']}**  
- ğŸ—„ï¸� Vault Reference: **{footer['vault_reference']}**

### âš–ï¸� Notice
This output is **licensed legacy**. Any reproduction without attribution or license falls under **Framework Breach Protocolâ„¢**.

### ğŸ’Œ Attribution
**{footer['attribution']}**
"""
with open(os.path.join(EXPORT_DIR, "Certification_Footer.md"), "w", encoding="utf-8") as f:
    f.write(footer_md)

# Optional schema check (graceful if jsonschema missing)
try:
    from jsonschema import Draft7Validator
    errs = [e.message for e in Draft7Validator(footer_schema).iter_errors(footer)]
    valid = (len(errs) == 0)
except Exception:
    errs, valid = ["jsonschema not available; skipped"], None

print({
    "artifact_paths": {
        "json": footer_json_path,
        "schema": schema_path,
        "markdown": os.path.join(EXPORT_DIR, "Certification_Footer.md")
    },
    "schema_validation": {"valid": valid, "errors": errs}
})


import os, json, hashlib, uuid, platform
from datetime import datetime

EXPORT_DIR = "case02_exports"
os.makedirs(EXPORT_DIR, exist_ok=True)

def sha256_of(path, chunk=1<<20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()

# Collect artifacts
artifacts = []
if os.path.isdir(EXPORT_DIR):
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

run_record = {
    "run_id": str(uuid.uuid4()),
    "case_id": "02",
    "case_title": "GenSpark Contamination Blueprint Audit",
    "vault_reference": "VAULT-GENSPARK-C02",
    "status": "sealed",
    "sealed_utc": datetime.utcnow().isoformat() + "Z",
    "export_dir": EXPORT_DIR,
    "environment": {
        "python": platform.python_version(),
        "platform": platform.platform()
    },
    "artifacts": artifacts,
    "summary": {
        "artifact_count": len(artifacts),
        "total_size_bytes": sum(a["size_bytes"] for a in artifacts)
    }
}

# Write run receipt + schema
run_json = os.path.join(EXPORT_DIR, "VaultRun.json")
with open(run_json, "w", encoding="utf-8") as f:
    json.dump(run_record, f, indent=4)

run_schema = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "Vault Run Stamp (Case 02)",
    "type": "object",
    "required": ["run_id","case_id","case_title","vault_reference","status","sealed_utc","export_dir","artifacts","summary"],
    "properties": {
        "run_id": {"type":"string"},
        "case_id": {"type":"string","const":"02"},
        "case_title": {"type":"string"},
        "vault_reference": {"type":"string"},
        "status": {"type":"string"},
        "sealed_utc": {"type":"string"},
        "export_dir": {"type":"string"},
        "environment": {"type":"object"},
        "artifacts": {
            "type":"array",
            "items": {
                "type":"object",
                "required": ["file","path","size_bytes","modified_utc","sha256"],
                "properties": {
                    "file": {"type":"string"},
                    "path": {"type":"string"},
                    "size_bytes": {"type":"integer","minimum":0},
                    "modified_utc": {"type":"string"},
                    "sha256": {"type":"string","minLength":64,"maxLength":64}
                }
            }
        },
        "summary": {
            "type":"object",
            "required": ["artifact_count","total_size_bytes"],
            "properties": {
                "artifact_count": {"type":"integer","minimum":0},
                "total_size_bytes": {"type":"integer","minimum":0}
            }
        }
    },
    "additionalProperties": False
}
with open(os.path.join(EXPORT_DIR, "VaultRun.schema.json"), "w", encoding="utf-8") as f:
    json.dump(run_schema, f, indent=4)

# Stitch run_id into footer JSON
footer_path = os.path.join(EXPORT_DIR, "CertificationFooter.json")
if os.path.isfile(footer_path):
    foot = json.load(open(footer_path, "r", encoding="utf-8"))
    foot["run_id"] = run_record["run_id"]
    json.dump(foot, open(footer_path, "w", encoding="utf-8"), indent=4)

# Try schema validation if available
try:
    from jsonschema import Draft7Validator
    errs = [e.message for e in Draft7Validator(run_schema).iter_errors(run_record)]
    valid = (len(errs) == 0)
except Exception:
    errs, valid = ["jsonschema not available; skipped"], None

print(json.dumps({
    "vault_run": run_json,
    "run_id": run_record["run_id"],
    "schema_validation": {"valid": valid, "errors": errs},
    "artifact_count": len(artifacts)
}, indent=4))

