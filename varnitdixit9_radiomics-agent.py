# %% Imports & configuration

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

import SimpleITK as sitk  # for NIfTI IO and overlays

# ---------------------------
# Paths (EDIT THESE TO MATCH YOUR DATASETS)
# ---------------------------

# 1) Radiomics features CSV (precomputed using PyRadiomics offline)
#    Attach your radiomics dataset in Kaggle, then update this path:
RADIOMICS_CSV_PATH = "/kaggle/input/radiomicdataset/radiomics_features.csv"

# 2) NIfTI dataset root with patient_images.csv and nifti_images/... folders
NIFTI_ROOT = "/kaggle/input/radiomicdataset"
NIFTI_MAP_CSV = os.path.join(NIFTI_ROOT, "patient_images.csv")

# ---------------------------
# Optional: Gemini LLM integration (for local runs with internet)
# On Kaggle, keep USE_GEMINI = False.
# ---------------------------

USE_GEMINI = False  # set to True only when running locally with GEMINI_API_KEY + internet

gemini_client = None
if USE_GEMINI:
    try:
        from google import genai
        api_key = os.getenv("GEMINI_API_KEY")
        if api_key:
            gemini_client = genai.Client(api_key=api_key)
        else:
            print("âš ï¸� GEMINI_API_KEY not set; falling back to offline explanation.")
            USE_GEMINI = False
    except ImportError:
        print("âš ï¸� google-genai not installed; falling back to offline explanation.")
        USE_GEMINI = False



# %% Load radiomics features and NIfTI mapping

# Load radiomics feature table
df = pd.read_csv(RADIOMICS_CSV_PATH)
print("Radiomics table shape:", df.shape)
display(df.head())

# Load NIfTI mapping
if os.path.exists(NIFTI_MAP_CSV):
    patient_images_df = pd.read_csv(NIFTI_MAP_CSV)
    print("NIfTI mapping shape:", patient_images_df.shape)
    display(patient_images_df.head())
else:
    patient_images_df = None
    print("âš ï¸� patient_images.csv not found at", NIFTI_MAP_CSV)

print("Unique patient_ids in radiomics:", sorted(df["patient_id"].unique())[:10])
if patient_images_df is not None:
    print("Unique patient_ids in NIfTI mapping:", sorted(patient_images_df["patient_id"].unique())[:10])



# %% Prepare X, y

meta_cols = ["patient_id", "cancer_type", "modality"]
feature_cols = [c for c in df.columns if c not in meta_cols]

X = df[feature_cols].copy()
y_raw = df["cancer_type"].copy()

# Map cancer_type to numeric labels
label_map = {"BRCA": 0, "HNC": 1}  # adjust if your labels differ
y = y_raw.map(label_map)

print("Num patients:", X.shape[0])
print("Num radiomics features:", X.shape[1])
print("\nCancer type distribution:")
print(y_raw.value_counts())



# %% PCA visualization of radiomics feature space

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(6, 5))
for cancer in sorted(df["cancer_type"].unique()):
    idx = df["cancer_type"] == cancer
    plt.scatter(X_pca[idx, 0], X_pca[idx, 1], label=cancer)
plt.legend()
plt.title("PCA of Radiomics Features (BRCA vs HNC)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.grid(True)
plt.show()



# %% Train a classical ML model (RandomForest)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=200,
    max_depth=5,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))



# %% Global feature importance

importances = clf.feature_importances_
indices = np.argsort(importances)[::-1][:10]

print("Top 10 important features (model-level):")
for i in indices:
    print(f"{feature_cols[i]} â†’ {importances[i]:.4f}")



# %% NIfTI helpers and overlay visualization

def get_nifti_paths(patient_id: str):
    """
    Look up image and mask paths for a given patient_id from patient_images.csv.
    Paths in CSV may accidentally start with '/', so we normalize them
    to be relative paths under NIFTI_ROOT.
    """
    if patient_images_df is None:
        raise RuntimeError("patient_images.csv not loaded.")
    rows = patient_images_df.loc[patient_images_df["patient_id"] == patient_id]
    if rows.empty:
        raise ValueError(f"No NIfTI paths found for patient_id={patient_id}")
    row = rows.iloc[0]

    # Strip any leading "/" so os.path.join doesn't drop NIFTI_ROOT
    img_rel = str(row["image_path"]).lstrip("/")
    mask_rel = str(row["mask_path"]).lstrip("/")

    img_path = os.path.join(NIFTI_ROOT, img_rel)
    mask_path = os.path.join(NIFTI_ROOT, mask_rel)

    print("Using image path:", img_path)
    print("Using mask  path:", mask_path)

    return img_path, mask_path



def show_tumor_overlay(patient_id: str, slice_index: int | None = None):
    """
    Show a single slice of the MRI volume with tumor mask overlaid.
    Uses SimpleITK for IO and matplotlib for display.
    """
    img_path, mask_path = get_nifti_paths(patient_id)

    image = sitk.ReadImage(img_path)
    mask = sitk.ReadImage(mask_path)

    img_arr = sitk.GetArrayFromImage(image)  # (z, y, x)
    mask_arr = sitk.GetArrayFromImage(mask)

    # Choose a slice: center of tumor-containing slices if possible
    if slice_index is None:
        tumor_slices = np.where(mask_arr.sum(axis=(1, 2)) > 0)[0]
        if len(tumor_slices) == 0:
            slice_index = img_arr.shape[0] // 2
        else:
            slice_index = int(np.median(tumor_slices))

    img_slice = img_arr[slice_index]
    mask_slice = mask_arr[slice_index]

    plt.figure(figsize=(6, 6))
    plt.imshow(img_slice, cmap="gray")
    plt.imshow(np.ma.masked_where(mask_slice == 0, mask_slice),
               alpha=0.4)
    plt.title(f"Patient {patient_id} â€” slice {slice_index} (tumor overlay)")
    plt.axis("off")
    plt.show()



# Example: visualize first patient (if NIfTI available)
example_pid = df["patient_id"].iloc[2]
show_tumor_overlay(example_pid)



# %% Per-instance feature attribution heuristic

def get_top_features_for_instance(x_scaled_row, model, feature_names, top_k=5):
    """
    importance_for_patient = global_importance * |standardized_value|
    Returns list of (feature_name, score) pairs sorted by descending score.
    """
    global_importance = model.feature_importances_
    vals = np.abs(x_scaled_row.flatten())
    scores = global_importance * vals

    idx = np.argsort(scores)[::-1][:top_k]
    top_feats = []
    for i in idx:
        top_feats.append((feature_names[i], float(scores[i])))
    return top_feats



# %% LLM explanation tool (offline + optional Gemini)

def build_explanation_prompt(patient_id, pred_type, confidence,
                             top_features, cancer_type_true=None):
    conf_pct = confidence * 100
    feats_text = "\n".join(
        f"- {name} (influence score {score:.3f})"
        for name, score in top_features
    )

    correctness_note = ""
    if cancer_type_true is not None:
        correctness_note = f"The ground truth label for this case is {cancer_type_true}."

    prompt = f"""
You are a radiomics expert. A model has analyzed a tumor using handcrafted radiomic features.

Patient ID: {patient_id}
Predicted cancer type: {pred_type}
Model confidence: {conf_pct:.1f}%

Top contributing radiomic features:
{feats_text}

{correctness_note}

Write a short, clinician-friendly explanation (max 3 paragraphs) of why the model might have predicted this type based on these features. Mention concepts like texture heterogeneity, intensity distribution, and tumor shape if relevant, but keep language simple.
"""
    return prompt


def call_gemini_for_explanation(prompt: str) -> str:
    if not USE_GEMINI or gemini_client is None:
        return "LLM (Gemini) not available; using offline explanation."
    resp = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
    )
    return resp.text


def llm_explain_prediction(patient_id, pred_label, pred_type, confidence,
                           top_features, cancer_type_true=None):
    """
    Explanation tool:
    - If Gemini is available, call it.
    - Otherwise, use an offline explanation template (Kaggle-safe).
    """
    prompt = build_explanation_prompt(
        patient_id, pred_type, confidence, top_features, cancer_type_true
    )

    # Online LLM path (only for local runs)
    if USE_GEMINI and gemini_client is not None:
        return call_gemini_for_explanation(prompt)

    # Offline fallback: structured, readable explanation
    conf_pct = confidence * 100
    direction = "confident" if confidence >= 0.7 else "uncertain"
    feature_lines = "\n".join(
        f"- {name} (influence score {score:.3f})"
        for name, score in top_features
    )

    correctness_note = ""
    if cancer_type_true is not None:
        if cancer_type_true == pred_type:
            correctness_note = "The predicted type matches the known label for this case."
        else:
            correctness_note = (
                "The predicted type does NOT match the known label. "
                "This may be due to limited data and model uncertainty."
            )

    explanation = f"""
Patient **{patient_id}** is predicted as **{pred_type}** with {conf_pct:.1f}% confidence.
The model is relatively **{direction}** about this decision.

Top contributing radiomic patterns:
{feature_lines}

Interpretation:
- Shape and texture features are used as quantitative biomarkers of tumor phenotype.
- Differences in heterogeneity and morphology often distinguish breast tumors from head & neck tumors.
- This explanation is generated by an internal explanation tool (with an LLM-backed option) and is for research only.

{correctness_note}
"""
    return explanation



# %% RadiomicsAgent definition

class RadiomicsAgent:
    def __init__(self, df, model, scaler, feature_cols, label_map):
        self.df = df
        self.model = model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.label_map = label_map
        self.inv_label_map = {v: k for k, v in label_map.items()}

        self.history = []  # session memory
        self.logs = []     # observability

    # ----- logging helper -----
    def _log(self, tool, **kwargs):
        self.logs.append({"tool": tool, "args": kwargs})

    # ----- tools -----
    def load_patient_row(self, patient_id):
        self._log("load_patient_row", patient_id=patient_id)
        rows = self.df.loc[self.df["patient_id"] == patient_id]
        if rows.empty:
            return None
        return rows.iloc[0]

    def predict(self, x_scaled):
        self._log("predict", shape=x_scaled.shape)
        pred = self.model.predict(x_scaled)[0]
        proba = self.model.predict_proba(x_scaled)[0, int(pred)]
        return int(pred), float(proba)

    def analyze_patient(self, patient_id):
        """
        Full end-to-end analysis:
        1) Load patient row.
        2) Scale features.
        3) Predict cancer type.
        4) Compute per-instance top features.
        5) Generate explanation (LLM-style).
        6) Append to history and logs.
        """
        row = self.load_patient_row(patient_id)
        if row is None:
            return {"error": f"patient_id '{patient_id}' not found."}

        x_raw = row[self.feature_cols].values.reshape(1, -1)
        x_scaled = self.scaler.transform(x_raw)

        pred_label, confidence = self.predict(x_scaled)
        pred_type = self.inv_label_map[pred_label]

        top_feats = get_top_features_for_instance(
            x_scaled, self.model, self.feature_cols, top_k=5
        )

        true_type = row["cancer_type"]
        explanation = llm_explain_prediction(
            patient_id=patient_id,
            pred_label=pred_label,
            pred_type=pred_type,
            confidence=confidence,
            top_features=top_feats,
            cancer_type_true=true_type,
        )

        result = {
            "patient_id": patient_id,
            "predicted_type": pred_type,
            "predicted_label": pred_label,
            "confidence": confidence,
            "top_features": top_feats,
            "explanation": explanation,
            "true_type": true_type,
        }

        self.history.append(result)
        self._log("analyze_patient", patient_id=patient_id,
                  predicted_type=pred_type, confidence=confidence)

        return result

    def show_history(self, n=5):
        self._log("show_history", n=n)
        return self.history[-n:]

    def show_logs(self, n=10):
        return self.logs[-n:]

    def visualize_tumor(self, patient_id, slice_index=None):
        self._log("visualize_tumor", patient_id=patient_id, slice_index=slice_index)
        if patient_images_df is None:
            print("NIfTI visualization not configured.")
            return
        show_tumor_overlay(patient_id, slice_index=slice_index)



# %% instantiate agent

agent = RadiomicsAgent(df, clf, scaler, feature_cols, label_map)



# %% Simple natural language interface

def handle_query(agent: RadiomicsAgent, query: str):
    q = query.strip()
    q_lower = q.lower()
    agent._log("handle_query", query=q)

    if q_lower.startswith("analyze patient"):
        parts = q.split()
        patient_id = parts[-1]
        return agent.analyze_patient(patient_id)

    if q_lower.startswith("show history"):
        return agent.show_history()

    if q_lower.startswith("visualize patient"):
        parts = q.split()
        patient_id = parts[-1]
        agent.visualize_tumor(patient_id)
        return {"status": "visualized", "patient_id": patient_id}

    return {"error": "Unknown command. Try: 'analyze patient <ID>' or 'show history'."}



# %% Demonstration: agent in action

# Analyze all patients once
for pid in df["patient_id"]:
    res = agent.analyze_patient(pid)
    print("=" * 80)
    print(f"Patient: {pid}")
    print(f"True type: {res['true_type']}")
    print(f"Predicted: {res['predicted_type']} (conf {res['confidence']:.2f})")
    print(res["explanation"])

print("\n--- Example natural language queries ---")
example_id = df["patient_id"].iloc[0]
print("Query: analyze patient", example_id)
print(handle_query(agent, f"analyze patient {example_id}"))

print("\nQuery: show history")
print(handle_query(agent, "show history"))

# Optional: uncomment if you want to render NIfTI overlay in this run
# print("\nQuery: visualize patient", example_id)
# handle_query(agent, f"visualize patient {example_id}")



def auto_analyze_all_patients(agent: RadiomicsAgent,
                              confidence_threshold: float = 0.65,
                              visualize_uncertain: bool = False):
    """
    Autonomous workflow:
    - Iterate over all patients in the radiomics table.
    - For each patient:
      * Run analysis (prediction + explanation).
      * If confidence < threshold â†’ flag for review (+ optional visualization).
    - Return a summary DataFrame of all runs.
    """
    summary_rows = []

    for pid in agent.df["patient_id"]:
        res = agent.analyze_patient(pid)
        conf = res["confidence"]
        flagged = conf < confidence_threshold

        if flagged:
            agent._log("flag_review", patient_id=pid, confidence=conf)
            if visualize_uncertain:
                agent.visualize_tumor(pid)

        summary_rows.append({
            "patient_id": pid,
            "true_type": res.get("true_type"),
            "predicted_type": res.get("predicted_type"),
            "confidence": conf,
            "flagged_for_review": flagged,
        })

    summary_df = pd.DataFrame(summary_rows)
    return summary_df


def agent_goal_mode(agent: RadiomicsAgent, goal: str = "classify_all_patients",
                    **kwargs):
    """
    Simple goal-based interface:
    - 'classify_all_patients': run autonomous classification on all patients.
    - 'review_outliers': focus on previously flagged low-confidence cases.
    """
    agent._log("goal_start", goal=goal, kwargs=kwargs)

    if goal == "classify_all_patients":
        conf_threshold = kwargs.get("confidence_threshold", 0.65)
        visualize_uncertain = kwargs.get("visualize_uncertain", False)
        summary = auto_analyze_all_patients(
            agent,
            confidence_threshold=conf_threshold,
            visualize_uncertain=visualize_uncertain,
        )
        agent._log("goal_complete", goal=goal)
        return summary

    elif goal == "review_outliers":
        min_conf = kwargs.get("max_confidence", 0.65)
        rows = [h for h in agent.history if h["confidence"] < min_conf]
        for entry in rows:
            pid = entry["patient_id"]
            print(f"Reviewing low-confidence case: {pid} "
                  f"(pred={entry['predicted_type']}, conf={entry['confidence']:.2f})")
            # Show MRI + mask overlay if available
            agent.visualize_tumor(pid)
            print(entry["explanation"])
            print("-" * 60)
        agent._log("goal_complete", goal=goal)
        return rows

    else:
        agent._log("goal_unknown", goal=goal)
        return {"error": f"Unknown goal '{goal}'."}



summary_df = agent_goal_mode(agent, "classify_all_patients",
                             confidence_threshold=0.65,
                             visualize_uncertain=False)
summary_df



_ = agent_goal_mode(agent, "review_outliers", max_confidence=0.6)



def llm_decide_followup(patient_id: str,
                        confidence: float,
                        explanation: str) -> str:
    """
    Decide what to do with a low-confidence case.
    Returns one of: 'accept', 'visualize', 'flag'.
    - If Gemini is available, ask the LLM.
    - Otherwise, use a simple rule-based policy.
    """
    # Rule-based fallback first
    if not USE_GEMINI or gemini_client is None:
        if confidence >= 0.6:
            return "accept"
        elif 0.4 <= confidence < 0.6:
            return "visualize"
        else:
            return "flag"

    # Gemini-backed decision (for local runs)
    prompt = f"""
You are helping manage a radiomics classification pipeline.

For patient {patient_id}, the model produced a low-confidence prediction
with confidence {confidence:.2f}.

Explanation (for context):
{explanation}

Decide what the agent should do next:

1. accept  - accept the prediction as-is.
2. visualize - show MRI + mask overlay to a human reviewer.
3. flag    - flag this case as high-risk / needs manual review.

Respond with only one word: accept, visualize, or flag.
"""
    resp = call_gemini_for_explanation(prompt)
    # Normalize response
    resp_clean = (resp or "").strip().lower()
    if "visualize" in resp_clean:
        return "visualize"
    if "flag" in resp_clean:
        return "flag"
    return "accept"


def auto_analyze_with_llm_decision(agent: RadiomicsAgent,
                                   min_confidence: float = 0.65):
    """
    Autonomous loop that:
    - Analyzes all patients.
    - For each case below min_confidence:
      * Uses llm_decide_followup(...) to choose a policy:
          - 'accept'   â†’ do nothing special.
          - 'visualize'â†’ show MRI + mask overlay.
          - 'flag'     â†’ mark for manual review.
    Returns a summary DataFrame including the chosen actions.
    """
    rows = []

    for pid in agent.df["patient_id"]:
        res = agent.analyze_patient(pid)
        conf = res["confidence"]
        action = "none"

        if conf < min_confidence:
            action = llm_decide_followup(
                patient_id=pid,
                confidence=conf,
                explanation=res["explanation"],
            )

            if action == "visualize":
                agent.visualize_tumor(pid)
            if action == "flag":
                agent._log("flag_review_llm", patient_id=pid, confidence=conf)

        rows.append({
            "patient_id": pid,
            "true_type": res.get("true_type"),
            "predicted_type": res.get("predicted_type"),
            "confidence": conf,
            "action": action,
        })

    summary = pd.DataFrame(rows)
    return summary



summary_llm = auto_analyze_with_llm_decision(agent, min_confidence=0.9)
summary_llm


