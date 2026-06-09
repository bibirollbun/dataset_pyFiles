 !pip install --quiet openpyxl scikit-learn


import os, csv, shutil
import logging
from datetime import datetime
from typing import Any, Dict, Optional
from pathlib import Path
import numpy as np
import pandas as pd
from itertools import islice
from IPython.display import display

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("AutoCleanAI")

class SessionService:
    def __init__(self):
        self.sessions: Dict[str, Dict[str, Any]] = {}
    def create_session(self, session_id: Optional[str] = None) -> str:
        if session_id is None:
            session_id = datetime.utcnow().isoformat()
        self.sessions[session_id] = {"created_at": datetime.utcnow().isoformat(), "state": {}}
        logger.info(f"Session created: {session_id}")
        return session_id
    def get_state(self, session_id: str) -> Dict[str, Any]:
        return self.sessions[session_id]["state"]
    def update_state(self, session_id: str, key: str, value: Any):
        self.sessions[session_id]["state"][key] = value
        logger.debug(f"Session {session_id} update: {key} = {value}")

class BaseAgent:
    def __init__(self, name: str, session_service: SessionService):
        self.name = name
        self.session_service = session_service
    def run(self, *args, **kwargs):
        raise NotImplementedError

class DataIngestAgent(BaseAgent):
    def run(self, path: str) -> pd.DataFrame:
        logger.info(f"{self.name}: Loading dataset from {path}")
        if path.lower().endswith(".csv"):
            df = pd.read_csv(path)
        elif path.lower().endswith(('.xls', '.xlsx')):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path, engine='python')
        logger.info(f"{self.name}: Loaded DataFrame with shape {df.shape}")
        return df

class CleaningAgent(BaseAgent):
    def run(self, df: pd.DataFrame, impute_strategy: str = "median") -> pd.DataFrame:
        logger.info(f"{self.name}: Starting cleaning pipeline")
        df = df.copy()
        empty_cols = [c for c in df.columns if df[c].dropna().shape[0] == 0]
        if empty_cols:
            logger.info(f"{self.name}: Dropping empty columns: {empty_cols}")
            df.drop(columns=empty_cols, inplace=True)
        obj_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in obj_cols:
            df[c] = df[c].astype(str).str.strip()
        before = df.shape[0]
        df.drop_duplicates(inplace=True)
        after = df.shape[0]
        if before != after:
            logger.info(f"{self.name}: Dropped {before-after} duplicate rows")
        for c in df.columns:
            if df[c].dtype == object:
                try:
                    converted = pd.to_numeric(df[c], errors="coerce")
                    non_na_ratio = converted.notna().mean()
                    if non_na_ratio > 0.8:
                        df[c] = converted
                except Exception:
                    pass
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        for c in num_cols:
            na_count = df[c].isna().sum()
            if na_count > 0:
                if impute_strategy == "median":
                    fill = df[c].median()
                elif impute_strategy == "mean":
                    fill = df[c].mean()
                else:
                    fill = 0
                df[c].fillna(fill, inplace=True)
                logger.info(f"{self.name}: Imputed {na_count} NAs in {c} with {impute_strategy}={fill}")
        cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
        for c in cat_cols:
            na_count = (df[c] == "nan").sum() if df[c].dtype == object else df[c].isna().sum()
            if na_count > 0:
                df[c].replace("nan", np.nan, inplace=True)
                df[c].fillna("Unknown", inplace=True)
                logger.info(f"{self.name}: Filled missing in {c} with 'Unknown'")
        for c in num_cols:
            low = df[c].quantile(0.01)
            high = df[c].quantile(0.99)
            if low < high:
                df[c] = df[c].clip(lower=low, upper=high)
        logger.info(f"{self.name}: Cleaning complete. New shape {df.shape}")
        return df

class AnalysisAgent(BaseAgent):
    def run(self, df: pd.DataFrame, target: Optional[str] = None) -> Dict[str, Any]:
        logger.info(f"{self.name}: Running analysis")
        report = {}
        report['shape'] = df.shape
        report['dtypes'] = df.dtypes.apply(lambda x: str(x)).to_dict()
        report['describe'] = df.describe(include='all').to_dict()
        num = df.select_dtypes(include=[np.number])
        if num.shape[1] > 1:
            report['correlation'] = num.corr().to_dict()
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            from sklearn.model_selection import train_test_split
            from sklearn.ensemble import RandomForestRegressor
            from sklearn.metrics import mean_squared_error
            X = df.drop(columns=[target]).select_dtypes(include=[np.number]).fillna(0)
            y = df[target]
            if X.shape[1] > 0:
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                model = RandomForestRegressor(n_estimators=50, random_state=42)
                model.fit(X_train, y_train)
                preds = model.predict(X_test)
                mse = mean_squared_error(y_test, preds)
                report['model'] = {'type': 'RandomForestRegressor', 'mse': float(mse), 'features_used': X.columns.tolist()}
                logger.info(f"{self.name}: Trained simple model. MSE={mse:.4f}")
        logger.info(f"{self.name}: Analysis complete")
        return report

class ReportAgent(BaseAgent):
    def run(self, df: pd.DataFrame, analysis: Dict[str, Any], out_base: str = "report") -> Dict[str, str]:
        logger.info(f"{self.name}: Generating report files")
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        md_path = f"{out_base}_{timestamp}.md"
        xlsx_path = f"{out_base}_{timestamp}.xlsx"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(f"# AutoCleanAI Report\n\n")
            f.write(f"Generated at: {datetime.utcnow().isoformat()} UTC\n\n")
            f.write("## Dataset\n\n")
            f.write(f"- Shape: {analysis.get('shape')}\n")
            f.write("\n## Columns & Types\n\n")
            for col, t in analysis.get('dtypes', {}).items():
                f.write(f"- **{col}**: {t}\n")
            f.write("\n## Summary Statistics\n\n")
            try:
                numeric_describe = pd.DataFrame(analysis['describe']).select_dtypes(include=[np.number]).head()
                f.write(numeric_describe.to_markdown())
                f.write("\n\n")
            except Exception:
                f.write("(summary stats omitted)\n\n")
            if 'model' in analysis:
                f.write("## Simple Model\n\n")
                for k, v in analysis['model'].items():
                    f.write(f"- {k}: {v}\n")
            f.write("\n## Notes\n\n")
            f.write("- This is an automated prototype report. Check data and validate before production.\n")
        with pd.ExcelWriter(xlsx_path, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='cleaned_data', index=False)
            meta = pd.DataFrame.from_dict({k: [str(v)] for k, v in analysis.items() if k != 'describe'})
            try:
                meta.to_excel(writer, sheet_name='metadata', index=False)
            except Exception:
                pass
        logger.info(f"{self.name}: Report saved to {md_path} and {xlsx_path}")
        return {'markdown': md_path, 'excel': xlsx_path}

class AgentOrchestrator:
    def __init__(self):
        self.session_service = SessionService()
        self.agents = {
            'ingest': DataIngestAgent('DataIngestAgent', self.session_service),
            'clean': CleaningAgent('CleaningAgent', self.session_service),
            'analysis': AnalysisAgent('AnalysisAgent', self.session_service),
            'report': ReportAgent('ReportAgent', self.session_service)
        }
    def run(self, input_path: str, out_base: str = 'report', impute_strategy: str = 'median', target: Optional[str] = None):
        session = self.session_service.create_session()
        df = self.agents['ingest'].run(input_path)
        self.session_service.update_state(session, 'original_shape', df.shape)
        cleaned_df = self.agents['clean'].run(df, impute_strategy=impute_strategy)
        self.session_service.update_state(session, 'cleaned_shape', cleaned_df.shape)
        analysis_result = self.agents['analysis'].run(cleaned_df, target=target)
        self.session_service.update_state(session, 'analysis', analysis_result)
        report_result = self.agents['report'].run(cleaned_df, analysis=analysis_result, out_base=out_base)
        return {'session': session, 'analysis': analysis_result, 'report': report_result}

OUT_BASE = 'autoclean_report'
IMPUTE_STRATEGY = 'median'
TARGET_COLUMN = None

candidates = []
for p in Path('/kaggle/input').rglob('*'):
    if p.is_file() and p.suffix.lower() in ['.csv', '.txt', '.tsv', '.xlsx', '.xls']:
        try:
            if p.stat().st_size>0:
                candidates.append(str(p))
        except Exception:
            pass

if candidates:
    selected = candidates[0]
else:
    sample = pd.DataFrame({
        "name":["Nimisha","Riya","Arun","Riya"],
        "age":[22,25,None,25],
        "city":["Thrissur","Kochi  ","TVM","Kochi  "],
        "score":[88,92,75,92]
    })
    sample_path = '/kaggle/working/sample_for_autoclean.csv'
    sample.to_csv(sample_path, index=False)
    selected = sample_path

candidate = selected
text = Path(candidate).read_text(encoding='utf-8', errors='replace')
lines = text.splitlines()
sample_lines = "\n".join(lines[:100])
detected_sep = None
seps = [',','\t',';','|']
try:
    sniffer = csv.Sniffer()
    dialect = sniffer.sniff(sample_lines, delimiters=seps)
    detected_sep = dialect.delimiter
except Exception:
    counts = {s: sum(line.count(s) for line in lines[:50]) for s in seps}
    detected_sep = max(counts, key=counts.get) if any(v>0 for v in counts.values()) else None

if detected_sep:
    df = pd.read_csv(candidate, sep=detected_sep, engine='python')
else:
    try:
        df = pd.read_csv(candidate, sep=r'\s+', engine='python')
    except Exception:
        df = pd.read_csv(candidate, engine='python', header=None)

temp_csv = '/kaggle/working/hackathon_dataset_auto.csv'
df.to_csv(temp_csv, index=False)
INPUT_PATH = temp_csv
orchestrator = AgentOrchestrator()
result = orchestrator.run(INPUT_PATH, out_base=OUT_BASE, impute_strategy=IMPUTE_STRATEGY, target=TARGET_COLUMN)
print("Session ID:", result['session'])
print("Report files:", result['report'])
md_path = result['report']['markdown']
print("\n--- Markdown preview ---\n")
with open(md_path, 'r', encoding='utf-8') as f:
    print(f.read()[:2000])
xlsx_path = result['report']['excel']
try:
    cleaned = pd.read_excel(xlsx_path, sheet_name='cleaned_data')
    print("\nCleaned DataFrame preview:")
    display(cleaned.head(20))
except Exception:
    pass


#END

