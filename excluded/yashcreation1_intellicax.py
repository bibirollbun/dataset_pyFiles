# Basic imports and helper functions
import os, json, logging, datetime
from pathlib import Path
from pprint import pprint

# Configure a simple logger (outputs to notebook and a log file)
ROOT = Path('/kaggle/working/intellicaX_notebook') if 'kaggle' in os.getcwd() else Path('/mnt/data/intellicaX_notebook')
ROOT.mkdir(parents=True, exist_ok=True)
LOG_FILE = ROOT / 'intellicaX_notebook.log'

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)s | %(message)s',
                    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()])
logger = logging.getLogger('IntellicaX')
logger.info('Notebook environment ready. Logs -> %s', LOG_FILE)



# Create sample documents used by the mock Search tool
docs_dir = ROOT / 'sample_docs'
docs_dir.mkdir(exist_ok=True)
sample_texts = {
    'urban_farming_india.txt': 'Urban farming in India reduces supply-chain emissions and improves local nutrition. Pilot programs show positive outcomes.',
    'climate_resilient.txt': 'Climate-resilient practices include rainwater harvesting, drought tolerant crops, and micro-irrigation.',
    'hyderabad_case.txt': 'Hyderabad pilot showed community engagement, composting benefits, and small income generation for participants.'
}
for fname, txt in sample_texts.items():
    (docs_dir / fname).write_text(txt)
logger.info('Sample docs created at %s', docs_dir)



# LongTermMemory - simple keyword or TF-IDF retriever (uses sklearn if available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except Exception as e:
    logger.warning('sklearn not available; falling back to keyword matching. %s', e)
    SKLEARN_AVAILABLE = False

class LongTermMemory:
    def __init__(self, docs_path):
        self.docs_path = Path(docs_path)
        self._load_docs()
        if SKLEARN_AVAILABLE and self.docs:
            texts = list(self.docs.values())
            self.vectorizer = TfidfVectorizer(stop_words='english')
            self.tfidf = self.vectorizer.fit_transform(texts)
        logger.info('LongTermMemory initialized with %d docs', len(self.docs))

    def _load_docs(self):
        self.docs = {}
        for p in Path(self.docs_path).glob('*.txt'):
            self.docs[p.name] = p.read_text()

    def retrieve(self, query, k=3):
        logger.info('LTM: retrieve query=%s', query)
        if SKLEARN_AVAILABLE and hasattr(self, 'tfidf'):
            q_vec = self.vectorizer.transform([query])
            sims = cosine_similarity(q_vec, self.tfidf)[0]
            ranked = sorted(enumerate(sims), key=lambda x: x[1], reverse=True)[:k]
            results = []
            keys = list(self.docs.keys())
            for idx, score in ranked:
                doc_name = keys[idx]
                results.append({'doc_id': doc_name, 'score': float(score), 'text': self.docs[doc_name]})
            return results
        else:
            # simple keyword match fallback
            results = []
            for name, text in self.docs.items():
                score = sum(1 for w in query.lower().split() if w in text.lower())
                if score>0:
                    results.append((name, score, text))
            results = sorted(results, key=lambda x: x[1], reverse=True)[:k]
            return [{'doc_id': r[0], 'score': float(r[1]), 'text': r[2]} for r in results]

# Initialize memory
ltm = LongTermMemory(docs_dir)



# Tools: MockSearchTool and PDFReaderTool
class MockSearchTool:
    def __init__(self, memory):
        self.memory = memory
    def search(self, query, top_k=3):
        logger.info('[MockSearchTool] search: %s', query)
        return self.memory.retrieve(query, k=top_k)

class PDFReaderTool:
    def __init__(self, docs_path):
        self.docs_path = Path(docs_path)
    def read(self, doc_id):
        p = self.docs_path / doc_id
        if p.exists():
            logger.info('[PDFReaderTool] read: %s', doc_id)
            return p.read_text()
        logger.warning('[PDFReaderTool] missing: %s', doc_id)
        return ''
# Initialize tools
search_tool = MockSearchTool(ltm)
pdf_tool = PDFReaderTool(docs_dir)



# Session memory and basic observability collector
class SessionMemory:
    def __init__(self):
        self.history = []
    def add(self, entry):
        self.history.append({'time': str(datetime.datetime.utcnow()), 'entry': entry})
    def get(self):
        return list(self.history)

class Observability:
    def __init__(self):
        self.logs = []
        self.traces = []
    def log(self, message):
        self.logs.append({'ts': str(datetime.datetime.utcnow()), 'message': message})
        logger.info('[OBS] %s', message)
    def trace(self, record):
        self.traces.append({'ts': str(datetime.datetime.utcnow()), 'record': record})

session = SessionMemory()
obs = Observability()



# Agents: Planner, SearchAgent, AnalysisAgent, WriterAgent, EvaluatorAgent
class PlannerAgent:
    def __init__(self, obs):
        self.obs = obs
    def plan(self, query):
        plan = [
            {'step': 'retrieve', 'instruction': f'Find sources for: {query}'},
            {'step': 'analyze', 'instruction': 'Extract key claims and citations'},
            {'step': 'write', 'instruction': 'Produce a 1-2 page brief'},
            {'step': 'evaluate', 'instruction': 'Score and request revisions if needed'}
        ]
        self.obs.log(f'Planner created plan ({len(plan)} steps) for: {query}')
        return plan

class SearchAgent:
    def __init__(self, search_tool, pdf_tool, obs, session):
        self.search_tool = search_tool
        self.pdf_tool = pdf_tool
        self.obs = obs
        self.session = session
    def execute(self, instruction):
        query = instruction.replace('Find sources for: ', '')
        results = self.search_tool.search(query)
        sources = []
        for r in results:
            txt = self.pdf_tool.read(r['doc_id'])
            sources.append({'doc_id': r['doc_id'], 'score': r['score'], 'text': txt})
            self.session.add({'source': r['doc_id'], 'score': r['score']})
        self.obs.log(f'SearchAgent found {len(sources)} sources')
        return sources

class AnalysisAgent:
    def __init__(self, obs, memory):
        self.obs = obs
        self.memory = memory
    def execute(self, sources):
        claims = []
        citations = []
        for s in sources:
            sentences = [sent.strip() for sent in s['text'].split('.') if sent.strip()]
            key = ' | '.join(sentences[:2])
            claims.append({'doc_id': s['doc_id'], 'claim': key})
            citations.append(s['doc_id'])
            # store raw text to long-term memory (placeholder)
        self.obs.log(f'AnalysisAgent extracted {len(claims)} claims')
        return {'claims': claims, 'citations': citations}

class WriterAgent:
    def __init__(self, obs):
        self.obs = obs
    def execute(self, query, analysis):
        title = f'Brief: {query}'
        intro = f'This brief summarizes key findings on: {query}.\n'
        body = '\n'.join([f"- {c['claim']} (Source: {c['doc_id']})" for c in analysis['claims']])
        conclusion = 'Conclusion: These findings indicate practical interventions and benefits.'
        self.obs.log('WriterAgent produced draft report')
        return {'title': title, 'intro': intro, 'body': body, 'conclusion': conclusion}

class EvaluatorAgent:
    def __init__(self, obs):
        self.obs = obs
    def score(self, report, analysis):
        score = 0
        reasons = []
        if len(set(analysis.get('citations', []))) >= 2:
            score += 40
        else:
            reasons.append('Not enough citations')
        if len(report['body'].split()) > 20:
            score += 30
        else:
            reasons.append('Report too short')
        if report.get('intro') and report.get('conclusion'):
            score += 30
        else:
            reasons.append('Missing intro or conclusion')
        self.obs.log(f'Evaluator scored report {score}/100 - reasons: {reasons}')
        return {'score': score, 'reasons': reasons}



# Demo run: orchestrate agents end-to-end
def run_demo(user_query='Urban farming impact in India'):
    planner = PlannerAgent(obs)
    search_agent = SearchAgent(search_tool, pdf_tool, obs, session)
    analysis_agent = AnalysisAgent(obs, ltm)
    writer_agent = WriterAgent(obs)
    evaluator = EvaluatorAgent(obs)

    plan = planner.plan(user_query)
    sources = search_agent.execute(plan[0]['instruction'])
    analysis = analysis_agent.execute(sources)
    report = writer_agent.execute(user_query, analysis)
    eval_result = evaluator.score(report, analysis)

    # Simple revision loop
    rev = 0
    while eval_result['score'] < 80 and rev < 2:
        obs.log('Evaluator requested revision - applying simple augmentation')
        report['body'] += '\nAdditional synthesized insight derived from multiple sources.'
        eval_result = evaluator.score(report, analysis)
        rev += 1

    # Save outputs
    out = {'query': user_query, 'plan': plan, 'sources': sources, 'analysis': analysis, 'report': report, 'evaluation': eval_result, 'session': session.get(), 'observability': obs.logs}
    out_path = ROOT / 'demo_output.json'
    out_path.write_text(json.dumps(out, indent=2))
    logger.info('Demo finished. Output saved to %s', out_path)
    return out

# Run demo and show summary
demo_out = run_demo()
pprint({'title': demo_out['report']['title'], 'score': demo_out['evaluation']['score'], 'sources': [s['doc_id'] for s in demo_out['sources']]})    



run_demo("Climate resilient agriculture practices in Indian cities")


run_demo("Role of AI agents in modern education systems")



run_demo("Sustainable water management techniques")


