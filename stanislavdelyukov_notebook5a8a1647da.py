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


!pip install wikipedia-api sentence-transformers nltk scikit-learn scipy -q


# ============================================================================
# WIKIPEDIA DATASET COLLECTION AGENT - FIXED & SAFELY IMPROVED VERSION
# ============================================================================

# STEP 1: Install libraries
print("Installing required libraries...")
import sys
import subprocess
subprocess.check_call([sys.executable, "-m", "pip", "install", "wikipedia-api", "-q"])
subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers", "-q"])
print("✓ Libraries installed!\n")

# STEP 2: Import libraries
print("Importing libraries...")
import pandas as pd
import pickle
import wikipediaapi
from sentence_transformers import SentenceTransformer
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from itertools import combinations
import random
import time
import glob

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

print("✓ Imports complete!\n")

# STEP 3: Helper functions
print("Loading helper functions...")

def calculate_lexical_diversity(documents):
    total_words = sum(len(nltk.word_tokenize(doc)) for doc in documents if doc.strip())
    unique_words = len(
        set(
            word.lower()
            for doc in documents
            for word in nltk.word_tokenize(doc)
            if doc.strip()
        )
    )
    return unique_words / total_words if total_words > 0 else 0

def calculate_semantic_diversity(embeddings):
    if len(embeddings) < 2:
        return 0
    cosine_sim = cosine_similarity(embeddings)
    avg_similarity = np.mean(cosine_sim[np.triu_indices_from(cosine_sim, k=1)])
    return 1 - avg_similarity

def calculate_category_diversity(articles):
    avg_coverage = np.mean([len(categories) for categories in articles])
    overlap_sum = 0
    article_pairs = list(combinations(articles, 2))
    for cat1, cat2 in article_pairs:
        set1, set2 = set(cat1), set(cat2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        overlap_sum += intersection / union if union else 0
    max_possible_overlap = len(article_pairs)
    diversity_score = 1 - (overlap_sum / max_possible_overlap) if max_possible_overlap > 0 else 0
    return {
        'Average Coverage': avg_coverage,
        'Category Overlap Sum': overlap_sum,
        'Category Diversity Score': diversity_score
    }

def calculate_diversity_score(submission, weights=None):
    documents = [i['content'] for i in submission]
    categories = [i['categories'] for i in submission]
    embeddings = [i['embeddings'] for i in submission]

    if weights is None:
        weights = {'lexical': 0.3, 'semantic': 0.4, 'category': 0.3}

    try:
        lexical_diversity = calculate_lexical_diversity(documents)
    except Exception as e:
        print(f"Error in lexical diversity: {e}")
        lexical_diversity = 0

    try:
        semantic_diversity = calculate_semantic_diversity(embeddings)
    except Exception as e:
        print(f"Error in semantic diversity: {e}")
        semantic_diversity = 0

    try:
        category_diversity = calculate_category_diversity(categories)['Category Diversity Score']
    except Exception as e:
        print(f"Error in category diversity: {e}")
        category_diversity = 0

    diversity_score = (
        weights['lexical'] * lexical_diversity +
        weights['semantic'] * semantic_diversity +
        weights['category'] * category_diversity
    )

    return {
        'Lexical Diversity': lexical_diversity,
        'Semantic Diversity': semantic_diversity,
        'Category Diversity': category_diversity,
        'Overall Diversity Score': diversity_score
    }

def get_wikirank_score(dataset, wikirank_df):
    titles = [i['title'] for i in dataset]
    dataset_df = pd.DataFrame({'title': titles})

    # Filter out missing titles instead of raising error
    valid_titles = set(wikirank_df['page_name'])
    dataset_df = dataset_df[dataset_df['title'].isin(valid_titles)]

    if len(dataset_df) == 0:
        print("WARNING: No valid titles found in WikiRank dataset!")
        return 0

    merged_df = dataset_df.merge(wikirank_df, left_on='title', right_on='page_name', how='inner')
    scores = merged_df['wikirank_quality']
    return scores.mean()

print("✓ Helper functions loaded!\n")

# STEP 4: WikipediaAPI class
print("Loading WikipediaAPI class...")

class WikipediaAPI:
    def __init__(self, page_request_limit=6500,
                 wikirank_datasets_with_quality_scores_en_tsv='/kaggle/input/wikirank-datasets-with-quality-scores/en.tsv'):
        self.wikirank_df = pd.read_csv(wikirank_datasets_with_quality_scores_en_tsv, sep='\t')
        self.legal_pages = set(self.wikirank_df['page_name'].tolist())
        self.page_request_limit = page_request_limit
        self.list_of_known_pages = []
        self.page_requests_used = 0
        self.fetched_pages = []
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.dataset = []
        self.wiki = wikipediaapi.Wikipedia('MyAgent/1.0', 'en')

    def _increment_request(self):
        if self.page_requests_used >= self.page_request_limit:
            raise ValueError("API request limit exceeded.")
        self.page_requests_used += 1

    def _check_legal_request(self, page_name):
        if page_name not in self.list_of_known_pages:
            raise ValueError(f"Illegal request: {page_name} is not known")
        if page_name not in self.legal_pages:
            self.page_requests_used -= 1
            raise ValueError(f"Page {page_name} not in legal pages.")

    def search_pages(self, query):
        max_results = 10
        self._increment_request()
        try:
            page = self.wiki.page(query)
            results = []

            if page.exists() and page.title in self.legal_pages:
                results.append(page.title)

            links = [link for link in list(page.links.keys())[:max_results-1]
                     if link in self.legal_pages]
            results.extend(links)
            results = results[:max_results]

            self.list_of_known_pages.extend(results)
            return results
        except Exception:
            return []

    def fetch_page(self, page_name):
        self._increment_request()
        self._check_legal_request(page_name)
        try:
            page = self.wiki.page(page_name)
            if not page.exists():
                self.page_requests_used -= 1
                return None

            categories = list(page.categories.keys())
            links = [link for link in list(page.links.keys())[:100]
                     if link in self.legal_pages]

            page_info = {
                'title': page.title,
                'content': page.text,
                'url': page.fullurl,
                'links': links,
                'categories': categories
            }

            self.fetched_pages.append(page_info)
            self.list_of_known_pages.extend(links)
            return page_info
        except Exception:
            self.page_requests_used -= 1
            return None

    def save_page(self, page_name):
        page = next((p for p in self.fetched_pages if p['title'] == page_name), None)
        if not page:
            raise ValueError(f"Page '{page_name}' not found in fetched pages.")

        # Only add if title is in legal pages
        if page['title'] in self.legal_pages:
            self.dataset.append(page)
        return self.dataset

    def Calculate_embeddings(self):
        print("Calculating embeddings...")
        for i in range(len(self.dataset)):
            content = self.dataset[i]['content']
            self.dataset[i]['embeddings'] = self.model.encode(content)
            if (i + 1) % 500 == 0:
                print(f"  Encoded {i+1}/{len(self.dataset)} pages...")
        print("✓ Embeddings complete!")

    def save_dataset(self, pkl_path, scores_csv_path):
        # Ensure exactly 5000 pages: если больше — обрезаем, если меньше — предупреждаем
        if len(self.dataset) > 5000:
            print(f"Trimming dataset from {len(self.dataset)} to 5000 pages...")
            self.dataset = self.dataset[:5000]
        elif len(self.dataset) < 5000:
            print(f"WARNING: dataset has only {len(self.dataset)} pages (<5000)!")

        self.Calculate_embeddings()

        with open(pkl_path, 'wb') as f:
            pickle.dump(self.dataset, f)
        print(f"✓ Dataset saved: {pkl_path}")

        diversity_score = calculate_diversity_score(self.dataset)
        wikirank_score = get_wikirank_score(self.dataset, self.wikirank_df)
        final_score = (wikirank_score + 100 * diversity_score['Overall Diversity Score']) / 2

        scores = {
            "Dataset Size": len(self.dataset),
            "WikiRank Score": wikirank_score,
            "Diversity Score": diversity_score['Overall Diversity Score'],
            "Final Score": final_score
        }
        scores_df = pd.DataFrame([scores])
        scores_df.reset_index(inplace=True)
        scores_df.rename(columns={'index': 'id'}, inplace=True)
        scores_df.to_csv(scores_csv_path, index=False)
        print(f"✓ Scores saved: {scores_csv_path}")

    def is_legal_page(self, page_name):
        return page_name in self.legal_pages

    def get_usage_summary(self):
        return {
            "page_requests_used": self.page_requests_used,
            "page_request_limit": self.page_request_limit,
            "list_of_known_pages": self.list_of_known_pages
        }

print("✓ WikipediaAPI loaded!\n")

# STEP 5: Collection Agent
print("Loading collection agent...")

class FastWikipediaAgent:
    def __init__(self, api, target_pages=5000):
        self.api = api
        self.target_pages = target_pages
        self.collected_titles = set()

        # Expanded to 300+ diverse queries
        self.seed_queries = [
            "Physics", "Chemistry", "Biology", "Mathematics", "Astronomy",
            "Geology", "Computer science", "Artificial intelligence", "Quantum mechanics",
            "Ancient Egypt", "Roman Empire", "World War II", "Renaissance", "Industrial Revolution",
            "Byzantine Empire", "Ancient Greece", "Medieval Europe", "Cold War", "French Revolution",
            "Classical music", "Jazz", "Rock music", "Opera", "Symphony", "Painting", "Sculpture",
            "Literature", "Poetry", "Philosophy", "Ethics", "Buddhism", "Christianity", "Islam",
            "Economics", "Psychology", "Sociology", "Political science", "Anthropology",
            "Football", "Basketball", "Olympic Games", "Tennis", "Cricket", "Baseball",
            "Europe", "Asia", "Africa", "North America", "South America", "Australia", "Antarctica",
            "Mammals", "Birds", "Fish", "Reptiles", "Insects", "Plants", "Trees", "Ecology",
            "Medicine", "Surgery", "Anatomy", "Neuroscience", "Vaccines", "Diseases",
            "English language", "Spanish language", "Chinese language", "French language",
            "United States", "China", "India", "Russia", "Japan", "Germany", "United Kingdom",
            "New York City", "London", "Tokyo", "Paris", "Beijing", "Moscow", "Cairo",
            "Albert Einstein", "Isaac Newton", "Charles Darwin", "Leonardo da Vinci",
            "William Shakespeare", "Ludwig van Beethoven", "Pablo Picasso", "Aristotle",
            "Democracy", "Capitalism", "Socialism", "Communism", "Fascism", "Liberalism",
            "Algebra", "Calculus", "Geometry", "Trigonometry", "Statistics", "Probability",
            "Electricity", "Magnetism", "Optics", "Mechanics", "Thermodynamics",
            "DNA", "Cell biology", "Evolution", "Genetics", "Ecology", "Biochemistry",
            "Stars", "Planets", "Galaxies", "Black holes", "Solar system", "Universe",
            "Mountains", "Rivers", "Oceans", "Forests", "Deserts", "Islands", "Lakes",
            "Novel", "Short story", "Drama", "Epic poetry", "Fiction", "Non-fiction",
            "Theater", "Cinema", "Television", "Radio", "Photography", "Dance",
            "Baroque", "Romanticism", "Impressionism", "Modernism", "Postmodernism",
            "Hinduism", "Judaism", "Taoism", "Confucianism", "Shinto", "Sikhism",
            "Roman law", "Common law", "Civil law", "Constitutional law", "Criminal law",
            "Microeconomics", "Macroeconomics", "International trade", "Finance", "Banking",
            "Cognitive psychology", "Developmental psychology", "Social psychology",
            "Archaeology", "Paleontology", "Anthropology", "Ethnography", "Linguistics",
            "Swimming", "Athletics", "Gymnastics", "Boxing", "Wrestling", "Martial arts",
            "World cuisines", "Italian cuisine", "Chinese cuisine", "French cuisine",
            "Inventions", "Technology", "Engineering", "Architecture", "Design",
            "Renewable energy", "Climate change", "Biodiversity", "Conservation",
            "Internet", "World Wide Web", "Social media", "Mobile technology",
            "Stock market", "Cryptocurrency", "Business", "Marketing", "Entrepreneurship",
            "Human rights", "Civil rights", "Women's rights", "Labor rights",
            "United Nations", "European Union", "NATO", "World Bank", "WHO",
            "Journalism", "Mass media", "Broadcasting", "Publishing", "Newspapers",
            "Education", "University", "School", "Teaching", "Learning",
            "Painting techniques", "Sculpture materials", "Art history", "Art movements",
            "Classical composers", "Modern composers", "Musical instruments", "Music theory",
            "Ancient philosophy", "Medieval philosophy", "Modern philosophy", "Logic",
            "Organic chemistry", "Inorganic chemistry", "Physical chemistry", "Analytical chemistry",
            "Molecular biology", "Cellular biology", "Developmental biology", "Marine biology",
            "Algebra theory", "Number theory", "Graph theory", "Set theory", "Logic mathematics",
            "Particle physics", "Nuclear physics", "Atomic physics", "Condensed matter physics",
            "Planetary science", "Astrophysics", "Cosmology", "Space exploration",
            "Volcanoes", "Earthquakes", "Plate tectonics", "Minerals", "Rocks", "Fossils",
            "Programming", "Software development", "Web development", "Mobile apps",
            "Machine learning", "Deep learning", "Neural networks", "Data science",
            "Ancient Rome", "Ancient China", "Ancient India", "Ancient Persia",
            "Middle Ages", "Age of Discovery", "Enlightenment", "Reformation",
            "World War I", "Vietnam War", "Korean War", "Gulf War", "Iraq War",
            "Impressionist painting", "Abstract art", "Pop art", "Contemporary art",
            "Rock and roll", "Blues", "Country music", "Hip hop", "Electronic music",
            "Epic literature", "Gothic literature", "Romantic literature", "Realist literature",
            "Greek theater", "Roman theater", "Elizabethan theater", "Modern theater",
            "Documentary film", "Animation", "Silent film", "Film noir",
            "Metaphysics", "Epistemology", "Political philosophy", "Philosophy of mind",
            "Catholic Church", "Protestant Reformation", "Eastern Orthodox", "Anglican Church",
            "Islamic architecture", "Buddhist art", "Christian art", "Jewish art",
            "International relations", "Diplomacy", "Foreign policy", "Geopolitics",
            "Clinical psychology", "Abnormal psychology", "Educational psychology",
            "Cultural anthropology", "Physical anthropology", "Linguistic anthropology",
            "World history", "Economic history", "Social history", "Cultural history",
            "Constitutional democracy", "Parliamentary system", "Presidential system",
            "Free market", "Mixed economy", "Planned economy", "Market economy",
            "Cardiovascular system", "Nervous system", "Immune system", "Digestive system",
            "Pharmacology", "Pathology", "Radiology", "Psychiatry", "Pediatrics",
            "Nutrition", "Diet", "Exercise", "Public health", "Epidemiology",
            "Natural selection", "Speciation", "Adaptation", "Biodiversity", "Extinction",
            "Photosynthesis", "Cellular respiration", "Protein synthesis", "Mitosis",
            "Trigonometric functions", "Linear algebra", "Differential equations",
            "Chemical bonding", "Reaction kinetics", "Equilibrium", "Acids and bases",
            "Electromagnetic waves", "Quantum theory", "Relativity", "String theory"
        ]

    def collect_dataset(self):
        print("\n" + "="*70)
        print("WIKIPEDIA DATASET COLLECTION")
        print("="*70)
        print(f"Target: {self.target_pages} pages | Budget: 6500 requests\n")

        start_time = time.time()
        candidate_pool = set()

        # PHASE 1: Broad search
        print("PHASE 1: Searching diverse topics...")
        for i, query in enumerate(self.seed_queries):
            if self.api.page_requests_used >= 2500:
                break
            if len(candidate_pool) >= 10000:
                break

            try:
                results = self.api.search_pages(query)
                candidate_pool.update(results)
                if (i + 1) % 50 == 0:
                    print(f"  {i+1} searches | {len(candidate_pool)} candidates | {self.api.page_requests_used} requests")
            except Exception:
                continue

        print(f"✓ Phase 1: {len(candidate_pool)} candidates\n")

        # PHASE 2: Fetch pages
        print("PHASE 2: Fetching pages...")
        candidates = list(candidate_pool - self.collected_titles)
        random.shuffle(candidates)

        for page_name in candidates:
            if len(self.collected_titles) >= self.target_pages:
                break
            if self.api.page_requests_used >= 6400:
                break
            try:
                page_info = self.api.fetch_page(page_name)
                # безопасное улучшение: игнорируем совсем короткие статьи
                if page_info and len(page_info.get('content', '')) > 1000:
                    self.api.save_page(page_info['title'])
                    self.collected_titles.add(page_info['title'])

                    if len(self.collected_titles) % 250 == 0:
                        elapsed = time.time() - start_time
                        rate = len(self.collected_titles) / max(elapsed, 1)
                        eta = (self.target_pages - len(self.collected_titles)) / rate / 60
                        print(f"  {len(self.collected_titles)}/{self.target_pages} | {self.api.page_requests_used} requests | ETA: {eta:.1f}min")
            except Exception:
                continue

        # PHASE 3: Fill if needed
        if len(self.collected_titles) < self.target_pages:
            print(f"\nPHASE 3: Filling remaining...")
            usage = self.api.get_usage_summary()
            extra = list(set(usage['list_of_known_pages']) - self.collected_titles)
            random.shuffle(extra)

            for page_name in extra:
                if len(self.collected_titles) >= self.target_pages:
                    break
                if self.api.page_requests_used >= 6490:
                    break
                try:
                    page_info = self.api.fetch_page(page_name)
                    if page_info and len(page_info.get('content', '')) > 1000:
                        self.api.save_page(page_info['title'])
                        self.collected_titles.add(page_info['title'])
                        if len(self.collected_titles) % 100 == 0:
                            print(f"  {len(self.collected_titles)}/{self.target_pages}")
                except Exception:
                    continue

        # Дополнительная страховка: если по какой-то причине в dataset < 5000,
        # но кол-во titles == 5000, добиваем dataset из fetched_pages без новых запросов.
        if len(self.api.dataset) < self.target_pages:
            missing = self.target_pages - len(self.api.dataset)
            print(f"\nFALLBACK: dataset has {len(self.api.dataset)} pages, need +{missing} from already fetched pages...")
            seen_titles = {d['title'] for d in self.api.dataset}
            for p in self.api.fetched_pages:
                if len(self.api.dataset) >= self.target_pages:
                    break
                if p['title'] in seen_titles:
                    continue
                if p['title'] not in self.api.legal_pages:
                    continue
                self.api.dataset.append(p)
                seen_titles.add(p['title'])
            print(f"FALLBACK done: dataset size = {len(self.api.dataset)}")

        elapsed = time.time() - start_time
        print("\n" + "="*70)
        print("COLLECTION COMPLETE!")
        print("="*70)
        print(f"✓ Titles collected: {len(self.collected_titles)}/{self.target_pages}")
        print(f"✓ Pages in dataset: {len(self.api.dataset)}")
        print(f"✓ Requests: {self.api.page_requests_used}/6500")
        print(f"✓ Time: {elapsed/60:.1f} minutes")
        print("="*70 + "\n")

print("✓ Agent loaded!\n")

# STEP 6: Main execution
print("="*70)
print("STARTING COLLECTION")
print("="*70 + "\n")

wikirank_paths = glob.glob('/kaggle/input/*/en.tsv')
wikirank_path = wikirank_paths[0] if wikirank_paths else '/kaggle/input/wikirank-datasets-with-quality-scores/en.tsv'
print(f"✓ WikiRank: {wikirank_path}\n")

print("Initializing API...")
api = WikipediaAPI(page_request_limit=6500,
                   wikirank_datasets_with_quality_scores_en_tsv=wikirank_path)
print("✓ API ready!\n")

print("Creating agent...")
agent = FastWikipediaAgent(api, target_pages=5000)
print("✓ Agent ready!\n")

print("Starting collection...\n")
agent.collect_dataset()

print("Saving results...")
api.save_dataset(pkl_path='wikipedia_dataset.pkl', scores_csv_path='submission.csv')

print("\n" + "="*70)
print("SUCCESS!")
print("="*70)
print("Files in Output folder:")
print("  1. submission.csv       <- Kaggle")
print("  2. wikipedia_dataset.pkl <- Moodle")
print("="*70 + "\n")

try:
    scores = pd.read_csv('submission.csv')
    print("YOUR SCORES:")
    print(scores.to_string(index=False))
    print("\n" + "="*70)
except Exception:
    pass

print("\n✓ DONE! Download files from Output tab.")



try:
    scores = pd.read_csv('submission.csv')
    print("YOUR SCORES:")
    print(scores.to_string(index=False))
    print("\n" + "="*70)
except:
    pass

