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


"""
Smart Multi-Agent Capstone Project
File: Smart_MultiAgent_Capstone_Code.py
Description: A self-contained, runnable Python script that demonstrates a
multi-agent pipeline for health risk prediction and automatic document
generation (PDF, Word, Excel, PPT). Uses sklearn's diabetes dataset as an
example so it runs without external data. Designed to run in Kaggle or
locally. Output files are saved to ./agent_outputs by default.

Dependencies (pip install if missing):
    python-docx
    python-pptx
    reportlab
    openpyxl

Run:
    python Smart_MultiAgent_Capstone_Code.py

Note: On Kaggle, many of these libraries are preinstalled. If running
locally, install the packages first.
"""

import os
import sys
import traceback
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, classification_report

# Document libraries (wrapped imports to provide clearer error messages)
try:
    from reportlab.pdfgen import canvas
except Exception as e:
    canvas = None
    # We'll raise later if trying to use PDF functionality

try:
    from docx import Document
except Exception as e:
    Document = None

try:
    from openpyxl import Workbook
except Exception as e:
    Workbook = None

try:
    from pptx import Presentation
except Exception as e:
    Presentation = None


OUTPUT_DIR = os.path.join(os.getcwd(), "agent_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------ Agents ---------------------------------

class DataAgent:
    """Responsible for loading and summarizing datasets."""
    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()

    def summary(self) -> pd.DataFrame:
        return self.df.describe().T

    def get_features_and_target(self, target_col: str):
        X = self.df.drop(columns=[target_col])
        y = self.df[target_col]
        return X, y


class PredictionAgent:
    """Wraps model training and prediction functions."""
    def __init__(self, model=None, scaler: Optional[StandardScaler]=None, feature_names: Optional[List[str]]=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names

    def train(self, X: pd.DataFrame, y: pd.Series):
        # Fit scaler and model. Assumes classification task with binary target in y.
        self.feature_names = list(X.columns)
        self.scaler = StandardScaler()
        Xs = self.scaler.fit_transform(X)
        # Default model
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.model.fit(Xs, y)

    def predict(self, X: pd.DataFrame) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        if self.model is None or self.scaler is None:
            raise RuntimeError("Model or scaler not initialized. Call train() first.")
        Xs = self.scaler.transform(X[self.feature_names])
        preds = self.model.predict(Xs)
        probs = None
        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(Xs)
        return preds, probs


class AnalysisAgent:
    """Provides simple, interpretable explanations for individual instances.
    This implementation uses a very lightweight perturbation approach for
    demonstration purposes. For better explanations, integrate SHAP or LIME.
    """
    def explain_instance(self, model, scaler: StandardScaler, X_instance: pd.DataFrame, top_k: int = 8) -> List[Tuple[str, float]]:
        # X_instance is a single-row DataFrame
        if model is None or scaler is None:
            raise RuntimeError("Model and scaler required for explanation.")
        # Base prediction
        base_pred = model.predict(scaler.transform(X_instance))[0]
        importances: Dict[str, float] = {}
        # For each feature, replace with mean and measure change in predicted class
        for col in X_instance.columns:
            X_pert = X_instance.copy()
            X_pert[col] = X_pert[col].mean()
            pert_pred = model.predict(scaler.transform(X_pert))[0]
            # Importance as signed difference (coarse measure)
            importances[col] = float(base_pred) - float(pert_pred)
        sorted_imp = sorted(importances.items(), key=lambda x: abs(x[1]), reverse=True)
        return sorted_imp[:top_k]


class RecommendationAgent:
    """Maps predictions/probabilities to human-readable recommendations."""
    def recommend(self, pred_prob: Optional[np.ndarray] = None) -> List[str]:
        # If no probabilities available, return generic advice
        if pred_prob is None:
            return ["Follow-up recommended", "Monitor regularly"]
        # If prob array present for one instance
        try:
            if isinstance(pred_prob, np.ndarray):
                # handle shape (n_samples, n_classes)
                if pred_prob.ndim == 2 and pred_prob.shape[1] >= 2:
                    p = float(pred_prob[0, 1])
                else:
                    p = float(pred_prob.flatten()[0])
            else:
                p = float(pred_prob)
        except Exception:
            p = 0.0

        if p > 0.75:
            return ["High risk: Immediate clinical review", "Consider diagnostic imaging", "Lifestyle and medication review"]
        elif p > 0.45:
            return ["Moderate risk: Specialist consult suggested", "Lifestyle modification recommended"]
        else:
            return ["Low risk: Routine monitoring"]


class DocumentAgent:
    """Generates PDF, Word, Excel, and PPT reports. Each method checks that the
    corresponding library is available and raises an informative error if not.
    """

    def __init__(self, outdir: str = OUTPUT_DIR):
        self.outdir = outdir
        os.makedirs(self.outdir, exist_ok=True)

    def make_pdf(self, patient_id: int, summary_text: str, filename: Optional[str] = None) -> str:
        if canvas is None:
            raise RuntimeError("reportlab is not installed. Install it to generate PDFs.")
        if filename is None:
            filename = os.path.join(self.outdir, f'report_{patient_id}.pdf')
        c = canvas.Canvas(filename)
        try:
            c.setFont('Helvetica-Bold', 16)
        except Exception:
            pass
        c.drawString(50, 800, f'Patient Report - ID: {patient_id}')
        y = 760
        for line in summary_text.split('\n'):
            c.drawString(50, y, line)
            y -= 16
            if y < 60:
                c.showPage()
                y = 800
        c.save()
        return filename

    def make_word(self, patient_id: int, summary_text: str) -> str:
        if Document is None:
            raise RuntimeError("python-docx is not installed. Install it to generate Word documents.")
        fn = os.path.join(self.outdir, f'report_{patient_id}.docx')
        doc = Document()
        doc.add_heading(f'Patient Report - ID: {patient_id}', level=1)
        for line in summary_text.split('\n'):
            doc.add_paragraph(line)
        doc.save(fn)
        return fn

    def make_excel(self, patient_id: int, df_results: pd.DataFrame) -> str:
        if Workbook is None:
            raise RuntimeError("openpyxl is not installed. Install it to generate Excel files.")
        fn = os.path.join(self.outdir, f'report_{patient_id}.xlsx')
        wb = Workbook()
        ws = wb.active
        ws.title = "Patient Risk"
        # header
        ws.append(list(df_results.columns))
        for r in df_results.itertuples(index=False):
            ws.append(list(r))
        wb.save(fn)
        return fn

    def make_ppt(self, patient_id: int, summary_text: str) -> str:
        if Presentation is None:
            raise RuntimeError("python-pptx is not installed. Install it to generate PPT files.")
        fn = os.path.join(self.outdir, f'report_{patient_id}.pptx')
        prs = Presentation()
        # Title slide
        slide = prs.slides.add_slide(prs.slide_layouts[0])
        try:
            slide.shapes.title.text = f'Patient Report - ID: {patient_id}'
        except Exception:
            pass
        # Content slide
        slide = prs.slides.add_slide(prs.slide_layouts[1])
        title = slide.shapes.title
        body = slide.placeholders[1]
        title.text = 'Summary'
        tf = body.text_frame
        # First clear any default paragraphs
        tf.clear()
        for line in summary_text.split('\n'):
            p = tf.add_paragraph()
            p.text = line
        prs.save(fn)
        return fn


# ------------------------ Utility functions -----------------------

def create_binary_risk_target(df: pd.DataFrame, source_target_col: str = 'target_continuous', frac_high: float = 0.33) -> pd.DataFrame:
    """Creates a binary 'risk' column from a continuous target by marking the
    top frac_high portion as '1' (high risk) and the rest '0'."""
    df = df.copy()
    if source_target_col not in df.columns:
        raise ValueError(f"Source target column '{source_target_col}' not found in dataframe")
    threshold = df[source_target_col].quantile(1 - frac_high)
    df['risk'] = (df[source_target_col] >= threshold).astype(int)
    return df


def build_sample_dataset() -> pd.DataFrame:
    """Loads sklearn diabetes dataset and prepares it with a continuous target
    and a binary risk label for demonstration."""
    diab = load_diabetes(as_frame=True)
    df = diab.frame.copy()
    # Rename original target to keep it explicit
    df.rename(columns={"target": "target_continuous"}, inplace=True)
    df = create_binary_risk_target(df, source_target_col='target_continuous', frac_high=0.33)
    return df


# ------------------------ Main pipeline ---------------------------

def run_pipeline(num_examples_to_generate: int = 3):
    try:
        print("Preparing dataset...")
        df = build_sample_dataset()
        data_agent = DataAgent(df)
        print("Data summary (top rows):")
        print(df.head())

        # Prepare features
        feature_cols = [c for c in df.columns if c not in ['target_continuous', 'risk']]
        X, y = df[feature_cols], df['risk']

        # Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        # Train model
        print("Training model...")
        pred_agent = PredictionAgent()
        pred_agent.train(X_train, y_train)

        # Evaluate
        X_test_s = pred_agent.scaler.transform(X_test)
        y_pred = pred_agent.model.predict(X_test_s)
        print("Evaluation on test set:")
        print("Accuracy:", accuracy_score(y_test, y_pred))
        print("Precision:", precision_score(y_test, y_pred))
        print("Recall:", recall_score(y_test, y_pred))
        print(classification_report(y_test, y_pred))

        # Initialize other agents
        analysis_agent = AnalysisAgent()
        recommend_agent = RecommendationAgent()
        doc_agent = DocumentAgent(outdir=OUTPUT_DIR)

        # Run on a few examples
        X_test_df = X_test.reset_index(drop=True)
        y_test_df = y_test.reset_index(drop=True)

        results = []
        examples = min(num_examples_to_generate, len(X_test_df))
        print(f"Generating reports for {examples} test instances (saved to {OUTPUT_DIR})")

        for idx in range(examples):
            X_instance = X_test_df.loc[[idx]]
            preds, probs = pred_agent.predict(X_instance)
            pred = int(preds[0])
            prob_text = ''
            prob_val = None
            if probs is not None:
                # If binary prob available, take prob of class '1' (high risk)
                if probs.ndim == 2 and probs.shape[1] >= 2:
                    prob_val = float(probs[0, 1])
                else:
                    prob_val = float(probs.flatten()[0])
                prob_text = f'Prob(high risk)={prob_val:.3f}'

            explanation = analysis_agent.explain_instance(pred_agent.model, pred_agent.scaler, X_instance)
            recs = recommend_agent.recommend(probs)

            summary_lines = [
                f'Patient sample index: {int(idx)}',
                f'Predicted risk (binary): {pred}',
            ]
            if prob_text:
                summary_lines.append(prob_text)
            summary_lines.append('Top influences (feature: signed-change):')
            summary_lines += [f'{feat}: {score:.4f}' for feat, score in explanation]
            summary_lines.append('Recommendations:')
            summary_lines += recs
            summary_text = '\n'.join(summary_lines)

            # Prepare a small DataFrame for Excel
            df_results = X_instance.copy()
            df_results = df_results.reset_index(drop=True)
            df_results['predicted_risk'] = pred
            if prob_val is not None:
                df_results['prob_high_risk'] = prob_val

            # Generate documents safely (wrap each in try/except to continue on failures)
            pdf_fn = docx_fn = excel_fn = ppt_fn = None
            try:
                pdf_fn = doc_agent.make_pdf(idx, summary_text)
            except Exception as e:
                print(f"PDF generation failed for {idx}: {e}")
            try:
                docx_fn = doc_agent.make_word(idx, summary_text)
            except Exception as e:
                print(f"Word generation failed for {idx}: {e}")
            try:
                excel_fn = doc_agent.make_excel(idx, df_results)
            except Exception as e:
                print(f"Excel generation failed for {idx}: {e}")
            try:
                ppt_fn = doc_agent.make_ppt(idx, summary_text)
            except Exception as e:
                print(f"PPT generation failed for {idx}: {e}")

            results.append({
                'sample_idx': idx,
                'pred': pred,
                'prob': prob_val,
                'pdf': pdf_fn,
                'docx': docx_fn,
                'excel': excel_fn,
                'ppt': ppt_fn,
            })

        res_df = pd.DataFrame(results)
        print("Generated files summary:")
        print(res_df)
        print(f"All outputs saved to: {OUTPUT_DIR}")

        return res_df

    except Exception as exc:
        print("An error occurred in the pipeline:")
        traceback.print_exc()
        raise


if __name__ == '__main__':
    # Run the pipeline and create 3 example reports
    try:
        results = run_pipeline(num_examples_to_generate=3)
        print("Pipeline completed successfully.")
    except Exception as e:
        print("Pipeline failed:", e)
        sys.exit(1)


