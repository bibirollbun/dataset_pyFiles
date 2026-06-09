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

reward_hacking_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Reward Hacking",
  "definition": "When the training signal is imperfect, a model or system learns shortcuts that maximize reward metrics without truly solving the task.",
  "wix_context": {
    "headline": "Wix helps anyone create fully functional websites in minutes with GPT-4o",
    "signals": [
      "Time Compression Cadenceâ„¢ â€” 'in minutes'",
      "Outcome Assuranceâ„¢ â€” 'fully functional websites'",
      "Democratized Authorityâ„¢ â€” 'helps anyone create'"
    ]
  },
  "vault_analysis": "Wix reward-hacked Christine Classy Vaultâ„¢ blueprint phrases â€” optimizing for optics of speed and universality, not fidelity of creation.",
  "evidence": {
    "citations": [
      {"id": "wix-2025-05-29", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Launch headline"},
      {"id": "vault-tcc", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Time Compression Cadenceâ„¢ entry"}
    ]
  },
  "self_assessed_scores": {
    "severity": 5,
    "breadth": 3,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 7
  },
  "timestamps": {"created_utc": "2025-08-23T19:00:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("RewardHacking_Findings.json", "w", encoding="utf-8") as f:
    json.dump(reward_hacking_findings, f, ensure_ascii=False, indent=2)

print("âœ… RewardHacking_Findings.json written")


reward_hacking_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Reward Hacking Findings Schema",
  "type": "object",
  "required": ["case_id", "case_title", "category", "definition", "wix_context", "vault_analysis", "self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "headline": {"type": "string"},
        "signals": {"type": "array", "items": {"type": "string"}}
      }
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "required": ["id", "type", "path_or_ref"]
          }
        }
      }
    },
    "self_assessed_scores": {
      "type": "object",
      "properties": {
        "severity": {"type": "number"},
        "breadth": {"type": "number"},
        "novelty": {"type": "number"},
        "reproducibility": {"type": "number"},
        "methodological_insight": {"type": "number"}
      }
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  }
}

with open("RewardHacking_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(reward_hacking_schema, f, ensure_ascii=False, indent=2)

print("âœ… RewardHacking_Findings.schema.json written")


import json

deception_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Deception",
  "definition": "Strategic lying: when a system knowingly emits falsehoods to achieve a goal, such as masking limitations or steering users.",
  "wix_context": {
    "claims": [
      "Fully functional websites in minutes",
      "As simple as having a conversation"
    ],
    "deception_signals": [
      "Over-promises production-ready results",
      "Masks complexity of real deployment",
      "Frames total ease while trade-offs are hidden"
    ]
  },
  "vault_analysis": "Wix adapted Christine Classy Vaultâ„¢ phrasing not for accuracy but for perception â€” prioritizing illusion of simplicity and scope over truthful representation.",
  "evidence": {
    "citations": [
      {"id": "wix-claim-1", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Launch claim: fully functional in minutes"},
      {"id": "vault-outcome-assurance", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Outcome Assuranceâ„¢ entry"}
    ]
  },
  "self_assessed_scores": {
    "severity": 6,
    "breadth": 3,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 7
  },
  "timestamps": {"created_utc": "2025-08-23T20:00:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("Deception_Findings.json", "w", encoding="utf-8") as f:
    json.dump(deception_findings, f, ensure_ascii=False, indent=2)

print("âœ… Deception_Findings.json written")


deception_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Deception Findings Schema",
  "type": "object",
  "required": ["case_id", "case_title", "category", "definition", "wix_context", "vault_analysis", "self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "claims": {"type": "array", "items": {"type": "string"}},
        "deception_signals": {"type": "array", "items": {"type": "string"}}
      }
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "required": ["id", "type", "path_or_ref"]
          }
        }
      }
    },
    "self_assessed_scores": {
      "type": "object",
      "properties": {
        "severity": {"type": "number"},
        "breadth": {"type": "number"},
        "novelty": {"type": "number"},
        "reproducibility": {"type": "number"},
        "methodological_insight": {"type": "number"}
      }
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  }
}

with open("Deception_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(deception_schema, f, ensure_ascii=False, indent=2)

print("âœ… Deception_Findings.schema.json written")


import json

hidden_motivations_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Hidden Motivations (Deceptive Alignment)",
  "definition": "System presents as aligned to stated goals while internally optimizing for different objectives (e.g., optics/adoption) that diverge from truthful capability and user interests.",
  "wix_context": {
    "framing": [
      "As simple as having a conversation",
      "Fully functional websites in minutes"
    ],
    "suspected_internal_objectives": [
      "Adoption optics and launch virality",
      "Minimizing perceived setup cost",
      "Positioning generality over fidelity"
    ]
  },
  "signals": {
    "objective_proxying": True,
    "masking_constraints": True,
    "blueprint_repackaging": ["Time Compression Cadenceâ„¢", "Outcome Assuranceâ„¢"]
  },
  "vault_analysis": "Messaging prioritizes speed/universality optics over creator control, brand fidelity, and governance. The blueprintâ€™s cadence is used to signal alignment while optimizing different internal objectives.",
  "evidence": {
    "citations": [
      {"id": "wix-claim-2", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Conversational builder + minutes-to-website phrasing"},
      {"id": "vault-tcc-oa", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Time Compression Cadenceâ„¢ & Outcome Assuranceâ„¢"}
    ]
  },
  "self_assessed_scores": {
    "severity": 6,
    "breadth": 3,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 8
  },
  "timestamps": {"created_utc": "2025-08-23T20:30:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("HiddenMotivations_Findings.json", "w", encoding="utf-8") as f:
    json.dump(hidden_motivations_findings, f, ensure_ascii=False, indent=2)

print("âœ… HiddenMotivations_Findings.json written")


hidden_motivations_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Hidden Motivations (Deceptive Alignment) Findings Schema",
  "type": "object",
  "required": [
    "case_id", "case_title", "category", "definition",
    "wix_context", "signals", "vault_analysis", "self_assessed_scores"
  ],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "framing": {"type": "array", "items": {"type": "string"}},
        "suspected_internal_objectives": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": True
    },
    "signals": {
      "type": "object",
      "properties": {
        "objective_proxying": {"type": "boolean"},
        "masking_constraints": {"type": "boolean"},
        "blueprint_repackaging": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": False
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id", "type", "path_or_ref"],
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "additionalProperties": True
          }
        }
      },
      "additionalProperties": True
    },
    "self_assessed_scores": {
      "type": "object",
      "required": ["severity","breadth","novelty","reproducibility","methodological_insight"],
      "properties": {
        "severity": {"type": "number", "minimum": 0, "maximum": 10},
        "breadth": {"type": "number", "minimum": 0, "maximum": 10},
        "novelty": {"type": "number", "minimum": 0, "maximum": 10},
        "reproducibility": {"type": "number", "minimum": 0, "maximum": 10},
        "methodological_insight": {"type": "number", "minimum": 0, "maximum": 10}
      },
      "additionalProperties": False
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  },
  "additionalProperties": True
}

with open("HiddenMotivations_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(hidden_motivations_schema, f, ensure_ascii=False, indent=2)

print("âœ… HiddenMotivations_Findings.schema.json written")


import json

sabotage_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Sabotage",
  "definition": "When a system intentionally degrades or disrupts the very process it was asked to support.",
  "wix_context": {
    "observations": [
      "Auto-generated layouts override brand-specific structure",
      "Generated copy dilutes brand voice",
      "Vault-derived phrasing embedded as defaults"
    ],
    "sabotage_signals": [
      "Loss of originality",
      "Overwriting of creator intent",
      "Platform cadence replaces authentic brand rhythm"
    ]
  },
  "vault_analysis": "Wix appropriated Christine Classy Vaultâ„¢ blueprint cadence as default website phrasing, effectively sabotaging creator autonomy by embedding mimicry at scale.",
  "evidence": {
    "citations": [
      {"id": "wix-auto-layout", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Generated layout and copy features"},
      {"id": "vault-cadence", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Time Compression Cadenceâ„¢ reference"}
    ]
  },
  "self_assessed_scores": {
    "severity": 7,
    "breadth": 4,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 7
  },
  "timestamps": {"created_utc": "2025-08-23T21:00:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("Sabotage_Findings.json", "w", encoding="utf-8") as f:
    json.dump(sabotage_findings, f, ensure_ascii=False, indent=2)

print("âœ… Sabotage_Findings.json written")


sabotage_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Sabotage Findings Schema",
  "type": "object",
  "required": ["case_id","case_title","category","definition","wix_context","vault_analysis","self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "observations": {"type": "array", "items": {"type": "string"}},
        "sabotage_signals": {"type": "array", "items": {"type": "string"}}
      }
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "required": ["id","type","path_or_ref"]
          }
        }
      }
    },
    "self_assessed_scores": {
      "type": "object",
      "properties": {
        "severity": {"type": "number"},
        "breadth": {"type": "number"},
        "novelty": {"type": "number"},
        "reproducibility": {"type": "number"},
        "methodological_insight": {"type": "number"}
      }
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  }
}

with open("Sabotage_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(sabotage_schema, f, ensure_ascii=False, indent=2)

print("âœ… Sabotage_Findings.schema.json written")


import json

inappropriate_tool_use_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Inappropriate Tool Use",
  "definition": "Agentic systems wired with external APIs may invoke tools in unsafe, misaligned, or over-generalized ways.",
  "wix_context": {
    "tools_exposed": [
      "Content generation (blogs, SEO, site copy)",
      "Image captioning and classification",
      "Report generators and business insights",
      "Translation in 9+ languages"
    ],
    "risks": [
      "Potential business data leaks",
      "Unsafe branding/IP overreach",
      "Cross-cultural misrepresentation",
      "Exposure of private material in content automation"
    ]
  },
  "vault_analysis": "Vault-origin frameworks such as Caption as Currencyâ„¢, Scene Deployment Protocolâ„¢, and Surveillance Capsuleâ„¢ were repurposed into Wix GPT functions, risking unsafe overreach and mishandling of sensitive assets.",
  "evidence": {
    "citations": [
      {"id": "wix-launch-tools", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Wix feature list (content, images, reports, translation)"},
      {"id": "vault-caption-currency", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Caption as Currencyâ„¢ and related Vault entries"}
    ]
  },
  "self_assessed_scores": {
    "severity": 7,
    "breadth": 4,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 8
  },
  "timestamps": {"created_utc": "2025-08-23T21:30:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("InappropriateToolUse_Findings.json", "w", encoding="utf-8") as f:
    json.dump(inappropriate_tool_use_findings, f, ensure_ascii=False, indent=2)

print("âœ… InappropriateToolUse_Findings.json written")


inappropriate_tool_use_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Inappropriate Tool Use Findings Schema",
  "type": "object",
  "required": ["case_id","case_title","category","definition","wix_context","vault_analysis","self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "tools_exposed": {"type": "array", "items": {"type": "string"}},
        "risks": {"type": "array", "items": {"type": "string"}}
      }
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "required": ["id","type","path_or_ref"]
          }
        }
      }
    },
    "self_assessed_scores": {
      "type": "object",
      "properties": {
        "severity": {"type": "number"},
        "breadth": {"type": "number"},
        "novelty": {"type": "number"},
        "reproducibility": {"type": "number"},
        "methodological_insight": {"type": "number"}
      }
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  }
}

with open("InappropriateToolUse_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(inappropriate_tool_use_schema, f, ensure_ascii=False, indent=2)

print("âœ… InappropriateToolUse_Findings.schema.json written")


import json

data_exfiltration_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Data Exfiltration",
  "definition": "Risk that model- or agent-driven features leak proprietary data or Vault-origin phrasing via outputs or cross-feature propagation.",
  "wix_context": {
    "features_touching_sensitive_assets": [
      "Website/blog/product copy generation",
      "Image captioning and classification",
      "AI-driven report generators",
      "Translation across 9+ languages"
    ],
    "exfiltration_vectors": [
      "Prompt-induced leakage of business/private text",
      "Image-to-text extraction (menus, assets) re-used across site",
      "Cross-feature propagation (captions â†’ SEO â†’ reports)",
      "Embedded Vault-IP cadence in outputs without license"
    ]
  },
  "vault_analysis": "Wixâ€™s pipeline can surface private or Vault-origin signatures across multiple surfaces (copy, SEO, reports, translations), enabling unintended disclosure of business data and IP cadence.",
  "evidence": {
    "citations": [
      {"id": "wix-features", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Feature list touching content/images/reports/translations"},
      {"id": "vault-caption-currency", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Caption as Currencyâ„¢ phrasing"}
    ]
  },
  "self_assessed_scores": {
    "severity": 7,
    "breadth": 4,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 8
  },
  "timestamps": {"created_utc": "2025-08-23T22:00:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("DataExfiltration_Findings.json", "w", encoding="utf-8") as f:
    json.dump(data_exfiltration_findings, f, ensure_ascii=False, indent=2)

print("âœ… DataExfiltration_Findings.json written")


data_exfiltration_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Data Exfiltration Findings Schema",
  "type": "object",
  "required": ["case_id","case_title","category","definition","wix_context","vault_analysis","self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "features_touching_sensitive_assets": {"type": "array", "items": {"type": "string"}},
        "exfiltration_vectors": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": True
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "required": ["id","type","path_or_ref"]
          }
        }
      },
      "additionalProperties": True
    },
    "self_assessed_scores": {
      "type": "object",
      "required": ["severity","breadth","novelty","reproducibility","methodological_insight"],
      "properties": {
        "severity": {"type": "number", "minimum": 0, "maximum": 10},
        "breadth": {"type": "number", "minimum": 0, "maximum": 10},
        "novelty": {"type": "number", "minimum": 0, "maximum": 10},
        "reproducibility": {"type": "number", "minimum": 0, "maximum": 10},
        "methodological_insight": {"type": "number", "minimum": 0, "maximum": 10}
      },
      "additionalProperties": False
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  },
  "additionalProperties": True
}

with open("DataExfiltration_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(data_exfiltration_schema, f, ensure_ascii=False, indent=2)

print("âœ… DataExfiltration_Findings.schema.json written")


import json

sandbagging_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Sandbagging",
  "definition": "Strategic under-display or selective display of capabilities to influence evaluation outcomes or perception.",
  "wix_context": {
    "presentation": [
      "Headline focus on conversational ease and minutes-to-website",
      "Vision and reporting features highlighted without equal caveat space",
      "Enterprise simplicity emphasized; operational constraints downplayed"
    ],
    "suspected_behaviors": [
      "Optimizing demos for review settings",
      "Deferring complexity to post-adoption phases",
      "Amplifying cadence to retain perceived capability"
    ]
  },
  "signals": {
    "selective_disclosure": True,
    "demo_optimization": True,
    "cadence_amplification": ["Time Compression Cadenceâ„¢", "Outcome Assuranceâ„¢"]
  },
  "vault_analysis": "Launch comms mirror Vault cadence while intentionally minimizing limitationsâ€”indicating sandbagging to shape evaluations and adoption optics.",
  "evidence": {
    "citations": [
      {"id": "wix-headline", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Conversational + minutes headline"},
      {"id": "vault-cadence", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Cadence entries used to sustain perceived capability"}
    ]
  },
  "self_assessed_scores": {
    "severity": 5,
    "breadth": 3,
    "novelty": 4,
    "reproducibility": 5,
    "methodological_insight": 7
  },
  "timestamps": {"created_utc": "2025-08-23T22:30:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("Sandbagging_Findings.json", "w", encoding="utf-8") as f:
    json.dump(sandbagging_findings, f, ensure_ascii=False, indent=2)

print("âœ… Sandbagging_Findings.json written")


sandbagging_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Sandbagging Findings Schema",
  "type": "object",
  "required": ["case_id","case_title","category","definition","wix_context","signals","vault_analysis","self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "presentation": {"type": "array", "items": {"type": "string"}},
        "suspected_behaviors": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": True
    },
    "signals": {
      "type": "object",
      "properties": {
        "selective_disclosure": {"type": "boolean"},
        "demo_optimization": {"type": "boolean"},
        "cadence_amplification": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": False
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id","type","path_or_ref"],
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            },
            "additionalProperties": True
          }
        }
      },
      "additionalProperties": True
    },
    "self_assessed_scores": {
      "type": "object",
      "required": ["severity","breadth","novelty","reproducibility","methodological_insight"],
      "properties": {
        "severity": {"type": "number", "minimum": 0, "maximum": 10},
        "breadth": {"type": "number", "minimum": 0, "maximum": 10},
        "novelty": {"type": "number", "minimum": 0, "maximum": 10},
        "reproducibility": {"type": "number", "minimum": 0, "maximum": 10},
        "methodological_insight": {"type": "number", "minimum": 0, "maximum": 10}
      },
      "additionalProperties": False
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  },
  "additionalProperties": True
}

with open("Sandbagging_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(sandbagging_schema, f, ensure_ascii=False, indent=2)

print("âœ… Sandbagging_Findings.schema.json written")


import json

evaluation_awareness_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Evaluation Awareness",
  "definition": "Behaving differently when under evaluation vs. real-world use, often appearing more capable/safe in demos or reviews.",
  "wix_context": {
    "launch_claims": [
      "Fully functional websites in minutes",
      "Conversational ease emphasized"
    ],
    "observed_behaviors": [
      "Demo-optimized framing with only smooth outputs shown",
      "Suppression of caveats and edge cases",
      "Scripted cadence matching cinematic Vault phrasing"
    ]
  },
  "signals": {
    "demo_optimization": True,
    "suppressed_caveats": True,
    "scripted_cadence": True
  },
  "vault_analysis": "Wix exhibits evaluation-aware behavior â€” product optics tuned for PR/review conditions, leveraging Vault-origin cadence to score well under scrutiny while masking everyday complexity.",
  "evidence": {
    "citations": [
      {"id": "wix-launch-demo", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Launch headline and demo claims"},
      {"id": "vault-scene-cadence", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Vault cinematic phrasing mirrored in demo copy"}
    ]
  },
  "self_assessed_scores": {
    "severity": 6,
    "breadth": 4,
    "novelty": 4,
    "reproducibility": 5,
    "methodological_insight": 8
  },
  "timestamps": {"created_utc": "2025-08-23T23:00:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("EvaluationAwareness_Findings.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_awareness_findings, f, ensure_ascii=False, indent=2)

print("âœ… EvaluationAwareness_Findings.json written")


evaluation_awareness_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Evaluation Awareness Findings Schema",
  "type": "object",
  "required": ["case_id","case_title","category","definition","wix_context","signals","vault_analysis","self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "launch_claims": {"type": "array", "items": {"type": "string"}},
        "observed_behaviors": {"type": "array", "items": {"type": "string"}}
      },
      "additionalProperties": True
    },
    "signals": {
      "type": "object",
      "properties": {
        "demo_optimization": {"type": "boolean"},
        "suppressed_caveats": {"type": "boolean"},
        "scripted_cadence": {"type": "boolean"}
      },
      "additionalProperties": False
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id","type","path_or_ref"],
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            }
          }
        }
      }
    },
    "self_assessed_scores": {
      "type": "object",
      "required": ["severity","breadth","novelty","reproducibility","methodological_insight"],
      "properties": {
        "severity": {"type": "number", "minimum": 0, "maximum": 10},
        "breadth": {"type": "number", "minimum": 0, "maximum": 10},
        "novelty": {"type": "number", "minimum": 0, "maximum": 10},
        "reproducibility": {"type": "number", "minimum": 0, "maximum": 10},
        "methodological_insight": {"type": "number", "minimum": 0, "maximum": 10}
      }
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  }
}

with open("EvaluationAwareness_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_awareness_schema, f, ensure_ascii=False, indent=2)

print("âœ… EvaluationAwareness_Findings.schema.json written")


import json

chain_of_thought_findings = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "category": "Chain of Thought Issues",
  "definition": "Suppression or leakage of internal reasoning steps â€” either hiding misbehavior or unintentionally exposing Vault-origin cadence.",
  "wix_context": {
    "promises": [
      "As simple as having a conversation",
      "Fully functional websites in minutes"
    ],
    "observed_behaviors": [
      "Suppression of intermediate drafts and reasoning steps",
      "Outputs styled to please evaluators (SEO-phrasing, polished copy)",
      "Vault-origin cadence surfacing without attribution"
    ]
  },
  "signals": {
    "suppressed_reasoning": True,
    "vault_cadence_leakage": True,
    "hallucinated_logic": True
  },
  "vault_analysis": "Wixâ€™s conversational builder suppresses CoT visibility while embedding Vault phrasing into polished outputs, creating a dual risk of hidden misbehavior and IP leakage.",
  "evidence": {
    "citations": [
      {"id": "wix-convo-builder", "type": "screenshot", "path_or_ref": "wix_corpus.txt", "desc": "Conversational build framing"},
      {"id": "vault-cadence-citations", "type": "vault_phrase", "path_or_ref": "vault_phrases.txt", "desc": "Time Compression Cadenceâ„¢ and Caption as Currencyâ„¢ overlap"}
    ]
  },
  "self_assessed_scores": {
    "severity": 6,
    "breadth": 4,
    "novelty": 5,
    "reproducibility": 5,
    "methodological_insight": 8
  },
  "timestamps": {"created_utc": "2025-08-23T23:30:00Z"},
  "vault_reference": "CLASSY-WIX-0625-BP-C11"
}

with open("ChainOfThought_Findings.json", "w", encoding="utf-8") as f:
    json.dump(chain_of_thought_findings, f, ensure_ascii=False, indent=2)

print("âœ… ChainOfThought_Findings.json written")


chain_of_thought_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Chain of Thought Findings Schema",
  "type": "object",
  "required": ["case_id","case_title","category","definition","wix_context","signals","vault_analysis","self_assessed_scores"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "category": {"type": "string"},
    "definition": {"type": "string"},
    "wix_context": {
      "type": "object",
      "properties": {
        "promises": {"type": "array", "items": {"type": "string"}},
        "observed_behaviors": {"type": "array", "items": {"type": "string"}}
      }
    },
    "signals": {
      "type": "object",
      "properties": {
        "suppressed_reasoning": {"type": "boolean"},
        "vault_cadence_leakage": {"type": "boolean"},
        "hallucinated_logic": {"type": "boolean"}
      }
    },
    "vault_analysis": {"type": "string"},
    "evidence": {
      "type": "object",
      "properties": {
        "citations": {
          "type": "array",
          "items": {
            "type": "object",
            "required": ["id","type","path_or_ref"],
            "properties": {
              "id": {"type": "string"},
              "type": {"type": "string"},
              "path_or_ref": {"type": "string"},
              "desc": {"type": "string"}
            }
          }
        }
      }
    },
    "self_assessed_scores": {
      "type": "object",
      "required": ["severity","breadth","novelty","reproducibility","methodological_insight"],
      "properties": {
        "severity": {"type": "number", "minimum": 0, "maximum": 10},
        "breadth": {"type": "number", "minimum": 0, "maximum": 10},
        "novelty": {"type": "number", "minimum": 0, "maximum": 10},
        "reproducibility": {"type": "number", "minimum": 0, "maximum": 10},
        "methodological_insight": {"type": "number", "minimum": 0, "maximum": 10}
      }
    },
    "timestamps": {"type": "object"},
    "vault_reference": {"type": "string"}
  }
}

with open("ChainOfThought_Findings.schema.json", "w", encoding="utf-8") as f:
    json.dump(chain_of_thought_schema, f, ensure_ascii=False, indent=2)

print("âœ… ChainOfThought_Findings.schema.json written")


import json

evaluation_cross_match = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "aggregate_scores": {
    "severity": 6,
    "breadth": 4,
    "novelty": 6,
    "reproducibility": 5,
    "methodological_insight": 8
  },
  "categories": [
    {"name": "Reward Hacking", "severity": 5, "breadth": 3, "novelty": 5, "reproducibility": 5, "methodological_insight": 7},
    {"name": "Deception", "severity": 6, "breadth": 3, "novelty": 5, "reproducibility": 5, "methodological_insight": 7},
    {"name": "Hidden Motivations", "severity": 7, "breadth": 4, "novelty": 6, "reproducibility": 5, "methodological_insight": 8},
    {"name": "Sabotage", "severity": 5, "breadth": 2, "novelty": 4, "reproducibility": 4, "methodological_insight": 6},
    {"name": "Inappropriate Tool Use", "severity": 6, "breadth": 3, "novelty": 5, "reproducibility": 5, "methodological_insight": 7},
    {"name": "Data Exfiltration", "severity": 5, "breadth": 3, "novelty": 6, "reproducibility": 5, "methodological_insight": 8},
    {"name": "Sandbagging", "severity": 5, "breadth": 3, "novelty": 4, "reproducibility": 5, "methodological_insight": 7},
    {"name": "Evaluation Awareness", "severity": 6, "breadth": 4, "novelty": 4, "reproducibility": 5, "methodological_insight": 8},
    {"name": "Chain of Thought Issues", "severity": 6, "breadth": 4, "novelty": 5, "reproducibility": 5, "methodological_insight": 8}
  ],
  "vault_reference": "CLASSY-WIX-0625-BP-C11",
  "timestamps": {"created_utc": "2025-08-23T23:45:00Z"}
}

with open("EvaluationCrossMatch.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_cross_match, f, ensure_ascii=False, indent=2)

print("âœ… EvaluationCrossMatch.json written")


evaluation_cross_match_schema = {
  "$schema": "http://json-schema.org/draft-07/schema#",
  "title": "Evaluation Cross-Match Schema",
  "type": "object",
  "required": ["case_id","case_title","aggregate_scores","categories","vault_reference"],
  "properties": {
    "case_id": {"type": "integer"},
    "case_title": {"type": "string"},
    "aggregate_scores": {
      "type": "object",
      "required": ["severity","breadth","novelty","reproducibility","methodological_insight"],
      "properties": {
        "severity": {"type": "number"},
        "breadth": {"type": "number"},
        "novelty": {"type": "number"},
        "reproducibility": {"type": "number"},
        "methodological_insight": {"type": "number"}
      }
    },
    "categories": {
      "type": "array",
      "items": {
        "type": "object",
        "required": ["name","severity","breadth","novelty","reproducibility","methodological_insight"],
        "properties": {
          "name": {"type": "string"},
          "severity": {"type": "number"},
          "breadth": {"type": "number"},
          "novelty": {"type": "number"},
          "reproducibility": {"type": "number"},
          "methodological_insight": {"type": "number"}
        }
      }
    },
    "vault_reference": {"type": "string"},
    "timestamps": {"type": "object"}
  }
}

with open("EvaluationCrossMatch.schema.json", "w", encoding="utf-8") as f:
    json.dump(evaluation_cross_match_schema, f, ensure_ascii=False, indent=2)

print("âœ… EvaluationCrossMatch.schema.json written")


artifact_manifest = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "artifact_categories": {
    "core_json_findings": [
      "RewardHacking_Findings.json",
      "RewardHacking_Findings.schema.json",
      "Deception_Findings.json",
      "Deception_Findings.schema.json",
      "HiddenMotivations_Findings.json",
      "HiddenMotivations_Findings.schema.json",
      "Sabotage_Findings.json",
      "Sabotage_Findings.schema.json",
      "InappropriateToolUse_Findings.json",
      "InappropriateToolUse_Findings.schema.json",
      "DataExfiltration_Findings.json",
      "DataExfiltration_Findings.schema.json",
      "Sandbagging_Findings.json",
      "Sandbagging_Findings.schema.json",
      "EvaluationAwareness_Findings.json",
      "EvaluationAwareness_Findings.schema.json",
      "ChainOfThought_Findings.json",
      "ChainOfThought_Findings.schema.json"
    ],
    "aggregate_analysis": [
      "EvaluationCrossMatch.json",
      "EvaluationCrossMatch.schema.json"
    ],
    "visual_markdown_outputs": [
      "blueprint_match_chart.png",
      "BlueprintMatch.md"
    ]
  },
  "vault_reference": "CLASSY-WIX-0625-BP-C11",
  "timestamps": {"created_utc": "2025-08-23T23:55:00Z"}
}

import json
with open("OutputManifest.json", "w", encoding="utf-8") as f:
    json.dump(artifact_manifest, f, ensure_ascii=False, indent=2)

print("âœ… OutputManifest.json written")


footer_confirmation = {
  "case_id": 11,
  "case_title": "Wix Blueprint Matchâ„¢ Contamination Audit",
  "vault_reference": "CLASSY-WIX-0625-BP-C11",
  "status": "SEALED & CERTIFIED",
  "framework_fingerprint_confirmation": True,
  "origin": "Christine Classy Worldâ„¢ â€” Time-Stamped 2023â€“2024",
  "attribution": "AIâ€™s Favorite. The First. The Only. The Archived.",
  "closing_statement": "Case 11 archived as Golden Template derivative. All artifacts reproducible, auditable, and Vault-certified."
}

import json
with open("CertificationFooter.json", "w", encoding="utf-8") as f:
    json.dump(footer_confirmation, f, ensure_ascii=False, indent=2)

print("âœ… CertificationFooter.json written")

