import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
from wordcloud import WordCloud
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

try:
    import fitz  
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF not available")

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("textstat not available")

warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8')
sns.set_palette("husl")



class DataCitationEDA:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.train_labels = None
        self.documents = {}
        self.document_stats = {}
        self.citation_patterns = {
            # DOI patterns
            "doi": r"(?:doi:?\s*)?(?:https?://)?(?:dx\.)?doi\.org/(10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
            "doi_direct": r"(?:^|\s)(10\.\d{4,9}/[-._;()/:A-Z0-9]+)(?=\s|$|[.,;)\]])",
            
            # Repository URLs
            "zenodo": r"(?:https?://)?zenodo\.org/record/(\d+)",
            "figshare": r"(?:https?://)?figshare\.com/(?:articles/|s/)?(?:dataset/)?(?:\w+/)?(\d+)",
            "dryad": r"(?:https?://)?(?:datadryad\.org|dryad\.org)/(?:stash/dataset/)?(?:doi:)?(10\.\d+/[\w.-]+)",
            "github": r"(?:https?://)?github\.com/([\w\-_.]+/[\w\-_.]+)",
            "osf": r"(?:https?://)?osf\.io/([a-z0-9]{5})",
            
            # Biological databases
            "geo_gse": r"\bGSE\d+\b",
            "geo_gsm": r"\bGSM\d+\b",
            "sra": r"\b(?:SRA|SRR|SRX|SRP|SAMN|PRJNA|PRJEB)\d+\b",
            "pdb": r"\b(?:pdb\s+)?(\d[A-Z0-9]{3}|[1-9][A-Z0-9]{3})\b",
            "uniprot": r"\b(?:UniProt:?\s*)?([A-Z]\d[A-Z0-9]{3}\d|[OPQ]\d[A-Z0-9]{3}\d)\b",
            "ensembl": r"\bENS[A-Z]*[GT]\d{11}\b",
            "arrayexpress": r"\bE-[A-Z]+-\d+\b",
            
            # Chemical/drug databases
            "chembl": r"\bCHEMBL\d+\b",
            "pubchem": r"\b(?:PubChem|CID)[-:]?\s*(\d+)\b",
            
            # Other databases
            "ncbi_gene": r"\b(?:Gene ID|GeneID)[-:]?\s*(\d{4,})\b",
            "omim": r"\bOMIM[-:]?\s*(\d{6})\b",
            "dbsnp": r"\brs\d+\b",
            "clinicaltrials": r"\bNCT\d{8}\b",
        }
        
    def load_data(self):
        print("Loading training labels")
        self.train_labels = pd.read_csv(self.base_path / "train_labels.csv")
        
        print("Loading documents")
        self.load_documents()
        
        print(f"Loaded {len(self.train_labels)} training labels")
        print(f"Loaded {len(self.documents)} documents")
        
    def load_documents(self):
        xml_path = self.base_path / "train" / "XML"
        if xml_path.exists():
            for xml_file in xml_path.glob("*.xml"):
                text = self.extract_text_from_xml(xml_file)
                if text:
                    self.documents[xml_file.stem] = text
                    self.document_stats[xml_file.stem] = self.calculate_document_stats(text)
        
        pdf_path = self.base_path / "train" / "PDF"
        if pdf_path.exists() and HAS_PYMUPDF:
            for pdf_file in pdf_path.glob("*.pdf"):
                if pdf_file.stem not in self.documents:
                    text = self.extract_text_from_pdf(pdf_file)
                    if text:
                        self.documents[pdf_file.stem] = text
                        self.document_stats[pdf_file.stem] = self.calculate_document_stats(text)
    
    def extract_text_from_xml(self, xml_path):
        """Extract text from XML files"""
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()
            text_parts = []
            
            for elem in root.iter():
                if elem.text and elem.text.strip():
                    text_parts.append(elem.text.strip())
                if elem.tail and elem.tail.strip():
                    text_parts.append(elem.tail.strip())
            
            return " ".join(text_parts)
        except Exception as e:
            print(f"Error parsing XML {xml_path}: {e}")
            return ""
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF files"""
        if not HAS_PYMUPDF:
            return ""
        
        try:
            doc = fitz.open(pdf_path)
            full_text = ""
            for page in doc:
                page_text = page.get_text()
                if page_text.strip():
                    full_text += page_text + "\n"
            return full_text
        except Exception as e:
            print(f"Error parsing PDF {pdf_path}: {e}")
            return ""
    
    def calculate_document_stats(self, text):
        stats = {
            'length': len(text),
            'words': len(text.split()),
            'sentences': len(re.split(r'[.!?]+', text)),
            'paragraphs': len([p for p in text.split('\n\n') if p.strip()]),
            'urls': len(re.findall(r'https?://[^\s]+', text)),
            'citations_found': 0
        }
        for pattern_name, pattern in self.citation_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            stats['citations_found'] += len(matches)
        
        if HAS_TEXTSTAT and len(text) > 100:
            try:
                stats['flesch_ease'] = flesch_reading_ease(text)
                stats['flesch_grade'] = flesch_kincaid_grade(text)
            except:
                stats['flesch_ease'] = None
                stats['flesch_grade'] = None
        
        return stats
    
    def analyze_labels_distribution(self):
        """Analyze the distribution of citation types in training labels"""
        print("\n" + "="*60)
        print("TRAINING LABELS ANALYSIS")
        print("="*60)
        print(f"Total training samples: {len(self.train_labels):,}")
        print(f"Unique articles: {self.train_labels['article_id'].nunique():,}")
        print(f"Unique datasets: {self.train_labels['dataset_id'].nunique():,}")
        type_counts = self.train_labels['type'].value_counts()
        print(f"\nCitation Type Distribution:")
        for cite_type, count in type_counts.items():
            percentage = (count / len(self.train_labels)) * 100
            print(f"  {cite_type}: {count:,} ({percentage:.1f}%)")
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        type_counts.plot(kind='bar', ax=axes[0,0], color='skyblue')
        axes[0,0].set_title('Citation Type Distribution')
        axes[0,0].set_xlabel('Citation Type')
        axes[0,0].set_ylabel('Count')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        citations_per_article = self.train_labels.groupby('article_id').size()
        citations_per_article.hist(bins=30, ax=axes[0,1], alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Citations per Article Distribution')
        axes[0,1].set_xlabel('Number of Citations')
        axes[0,1].set_ylabel('Number of Articles')
        
        article_types = self.train_labels.groupby('article_id')['type'].apply(list)
        type_combinations = article_types.apply(lambda x: tuple(sorted(set(x))))
        top_combinations = Counter(type_combinations).most_common(10)
        
        combo_names = [str(combo) for combo, _ in top_combinations]
        combo_counts = [count for _, count in top_combinations]
        
        axes[1,0].barh(combo_names, combo_counts, color='lightgreen')
        axes[1,0].set_title('Top 10 Citation Type Combinations per Article')
        axes[1,0].set_xlabel('Number of Articles')
        
        dataset_lengths = self.train_labels['dataset_id'].str.len()
        dataset_lengths.hist(bins=50, ax=axes[1,1], alpha=0.7, color='gold')
        axes[1,1].set_title('Dataset ID Length Distribution')
        axes[1,1].set_xlabel('Character Length')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.show()
        
        return type_counts, citations_per_article
    
    def analyze_citation_patterns(self):
        print("\n" + "="*60)
        print("CITATION PATTERNS ANALYSIS")
        print("="*60)
        citation_categories = defaultdict(list)
        
        for _, row in self.train_labels.iterrows():
            dataset_id = str(row['dataset_id'])
            citation_type = row['type']
            matched_pattern = None
            for pattern_name, pattern in self.citation_patterns.items():
                if re.search(pattern, dataset_id, re.IGNORECASE):
                    matched_pattern = pattern_name
                    break
            
            if not matched_pattern:
                matched_pattern = 'other'
            
            citation_categories[matched_pattern].append({
                'dataset_id': dataset_id,
                'type': citation_type,
                'article_id': row['article_id']
            })
        print("Citation Pattern Distribution:")
        pattern_stats = {}
        for pattern, citations in citation_categories.items():
            pattern_stats[pattern] = len(citations)
            print(f"  {pattern}: {len(citations)} citations")
        
        pattern_type_matrix = defaultdict(lambda: defaultdict(int))
        for pattern, citations in citation_categories.items():
            for citation in citations:
                pattern_type_matrix[pattern][citation['type']] += 1
        
        patterns = list(pattern_type_matrix.keys())
        types = ['Primary', 'Secondary']
        
        heatmap_data = []
        for pattern in patterns:
            row = []
            for cite_type in types:
                row.append(pattern_type_matrix[pattern][cite_type])
            heatmap_data.append(row)
        
        plt.figure(figsize=(12, 8))
        sns.heatmap(heatmap_data, 
                   xticklabels=types, 
                   yticklabels=patterns,
                   annot=True, 
                   fmt='d',
                   cmap='YlOrRd')
        plt.title('Citation Pattern vs Type Heatmap')
        plt.xlabel('Citation Type')
        plt.ylabel('Citation Pattern')
        plt.tight_layout()
        plt.show()
        
        return citation_categories, pattern_type_matrix
    
    def analyze_document_characteristics(self):
        """Analyze characteristics of the documents"""
        print("\n" + "="*60)
        print("DOCUMENT CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        if not self.document_stats:
            print("No document statistics available")
            return
        
        stats_df = pd.DataFrame.from_dict(self.document_stats, orient='index')
        print("Document Statistics:")
        print(f"  Total documents: {len(stats_df):,}")
        print(f"  Average length: {stats_df['length'].mean():,.0f} characters")
        print(f"  Average words: {stats_df['words'].mean():,.0f}")
        print(f"  Average sentences: {stats_df['sentences'].mean():,.0f}")
        print(f"  Average paragraphs: {stats_df['paragraphs'].mean():,.0f}")
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        stats_df['length'].hist(bins=50, ax=axes[0,0], alpha=0.7, color='skyblue')
        axes[0,0].set_title('Document Length Distribution')
        axes[0,0].set_xlabel('Characters')
        axes[0,0].set_ylabel('Count')
    
        stats_df['words'].hist(bins=50, ax=axes[0,1], alpha=0.7, color='lightcoral')
        axes[0,1].set_title('Word Count Distribution')
        axes[0,1].set_xlabel('Words')
        axes[0,1].set_ylabel('Count')
 
        stats_df['citations_found'].hist(bins=50, ax=axes[0,2], alpha=0.7, color='lightgreen')
        axes[0,2].set_title('Citations Found per Document')
        axes[0,2].set_xlabel('Citation Count')
        axes[0,2].set_ylabel('Document Count')
      
        stats_df['sentences'].hist(bins=50, ax=axes[1,0], alpha=0.7, color='gold')
        axes[1,0].set_title('Sentence Count Distribution')
        axes[1,0].set_xlabel('Sentences')
        axes[1,0].set_ylabel('Count')
    
        stats_df['paragraphs'].hist(bins=50, ax=axes[1,1], alpha=0.7, color='purple')
        axes[1,1].set_title('Paragraph Count Distribution')
        axes[1,1].set_xlabel('Paragraphs')
        axes[1,1].set_ylabel('Count')
     
        if 'flesch_ease' in stats_df.columns and stats_df['flesch_ease'].notna().any():
            stats_df['flesch_ease'].dropna().hist(bins=30, ax=axes[1,2], alpha=0.7, color='orange')
            axes[1,2].set_title('Flesch Reading Ease Score')
            axes[1,2].set_xlabel('Score (higher = easier)')
            axes[1,2].set_ylabel('Count')
        else:
            axes[1,2].text(0.5, 0.5, 'Readability data\nnot available', 
                          ha='center', va='center', transform=axes[1,2].transAxes)
            axes[1,2].set_title('Readability Analysis')
        
        plt.tight_layout()
        plt.show()
        
        return stats_df
    
    def analyze_citation_context(self, sample_size=100):
        """Analyze the context around citations"""
        print("\n" + "="*60)
        print("CITATION CONTEXT ANALYSIS")
        print("="*60)
        
        context_analysis = defaultdict(list)
        sample_labels = self.train_labels.sample(min(sample_size, len(self.train_labels)))
        
        for _, row in sample_labels.iterrows():
            article_id = str(row['article_id'])
            dataset_id = str(row['dataset_id'])
            citation_type = row['type']
            
            if article_id not in self.documents:
                continue
            
            text = self.documents[article_id]
            context = self.get_citation_context(text, dataset_id)
            
            if context:
                context_analysis[citation_type].append(context)
        print("Context Analysis Results:")
        for cite_type, contexts in context_analysis.items():
            if not contexts:
                continue
                
            print(f"\n{cite_type} Citations ({len(contexts)} samples):")
            avg_length = np.mean([len(ctx) for ctx in contexts])
            print(f"  Average context length: {avg_length:.0f} characters")
            all_words = []
            for context in contexts:
                words = re.findall(r'\b\w+\b', context.lower())
                all_words.extend(words)
            
            if all_words:
                word_freq = Counter(all_words)
                common_words = [word for word, freq in word_freq.most_common(20) 
                              if len(word) > 3 and word not in ['the', 'and', 'for', 'are', 'with']]
                print(f"  Common context words: {', '.join(common_words[:10])}")

        if context_analysis:
            n_types = len(context_analysis)
            fig, axes = plt.subplots(1, n_types, figsize=(6*n_types, 6))
            
            if n_types == 1:
                axes = [axes]
            
            for idx, (cite_type, contexts) in enumerate(context_analysis.items()):
                if not contexts:
                    continue
        
                combined_text = ' '.join(contexts)
                wordcloud = WordCloud(width=400, height=300, 
                                    background_color='white',
                                    max_words=50).generate(combined_text)
                
                axes[idx].imshow(wordcloud, interpolation='bilinear')
                axes[idx].set_title(f'{cite_type} Citations\nContext Words')
                axes[idx].axis('off')
            
            plt.tight_layout()
            plt.show()
        
        return context_analysis
    
    def get_citation_context(self, text, citation, window=300):
        """Get context window around citation"""
        text_lower = text.lower()
        citation_lower = citation.lower()
        pos = text_lower.find(citation_lower)
        if pos == -1:
            citation_parts = citation_lower.split('/')
            for part in citation_parts:
                if len(part) > 8:
                    pos = text_lower.find(part)
                    if pos != -1:
                        break
        
        if pos == -1:
            return None
        start = max(0, pos - window)
        end = min(len(text), pos + len(citation) + window)
        return text[start:end]
    
    def analyze_data_availability_sections(self):
        print("\n" + "="*60)
        print("DATA AVAILABILITY SECTIONS ANALYSIS")
        print("="*60)
        
        da_sections = []
        da_pattern = r'(?:data\s+availability|data\s+access|supplementary\s+(?:data|materials?))(.*?)(?=\n\s*[A-Z][A-Z\s]+|$)'
        
        for doc_id, text in self.documents.items():
            matches = re.finditer(da_pattern, text, re.IGNORECASE | re.DOTALL)
            for match in matches:
                section_text = match.group(0)
                if len(section_text) > 50: 
                    da_sections.append({
                        'document_id': doc_id,
                        'section_text': section_text,
                        'length': len(section_text),
                        'has_url': bool(re.search(r'https?://[^\s]+', section_text)),
                        'has_doi': bool(re.search(r'doi\.org', section_text, re.IGNORECASE)),
                        'has_github': bool(re.search(r'github\.com', section_text, re.IGNORECASE))
                    })
        
        if da_sections:
            da_df = pd.DataFrame(da_sections)
            
            print(f"Found {len(da_sections)} data availability sections")
            print(f"Average section length: {da_df['length'].mean():.0f} characters")
            print(f"Sections with URLs: {da_df['has_url'].sum()} ({da_df['has_url'].mean()*100:.1f}%)")
            print(f"Sections with DOIs: {da_df['has_doi'].sum()} ({da_df['has_doi'].mean()*100:.1f}%)")
            print(f"Sections with GitHub: {da_df['has_github'].sum()} ({da_df['has_github'].mean()*100:.1f}%)")
            
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            da_df['length'].hist(bins=30, ax=axes[0], alpha=0.7, color='skyblue')
            axes[0].set_title('Data Availability Section Length')
            axes[0].set_xlabel('Characters')
            axes[0].set_ylabel('Count')
            
            features = ['has_url', 'has_doi', 'has_github']
            feature_counts = [da_df[feat].sum() for feat in features]
            feature_names = ['URLs', 'DOIs', 'GitHub Links']
            
            axes[1].bar(feature_names, feature_counts, color=['lightcoral', 'lightgreen', 'gold'])
            axes[1].set_title('Features in Data Availability Sections')
            axes[1].set_ylabel('Count')
            
            plt.tight_layout()
            plt.show()
            
            return da_df
        else:
            print("No data availability sections found")
            return None
    
    def generate_summary_insights(self):
        print("\n" + "="*60)
        print("KEY INSIGHTS")
        print("="*60)
        
        insights = []
        
        type_counts = self.train_labels['type'].value_counts()
        primary_pct = (type_counts.get('Primary', 0) / len(self.train_labels)) * 100
        secondary_pct = (type_counts.get('Secondary', 0) / len(self.train_labels)) * 100
        
        insights.append(f"Class Distribution: {primary_pct:.1f}% Primary, {secondary_pct:.1f}% Secondary")
        
        if abs(primary_pct - secondary_pct) > 20:
            insights.append("Class imbalance detected - consider balancing techniques")
        
        unique_patterns = len(set([self.identify_citation_pattern(cit) 
                                 for cit in self.train_labels['dataset_id']]))
        insights.append(f" Citation Diversity: {unique_patterns} different citation patterns detected")
        
        if self.document_stats:
            avg_length = np.mean([stats['length'] for stats in self.document_stats.values()])
            insights.append(f"Average Document Length: {avg_length:,.0f} characters")
            
            if avg_length > 50000:
                insights.append("Long documents detected - consider chunking for processing")
        
        citations_per_article = self.train_labels.groupby('article_id').size()
        avg_citations = citations_per_article.mean()
        insights.append(f"Average Citations per Article: {avg_citations:.1f}")
        
        if avg_citations > 10:
            insights.append("High citation density - rich dataset for training")
        
        print("Key Insights:")
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        return insights
    
    def identify_citation_pattern(self, citation):
        """Identify which pattern a citation matches"""
        citation_str = str(citation)
        for pattern_name, pattern in self.citation_patterns.items():
            if re.search(pattern, citation_str, re.IGNORECASE):
                return pattern_name
        return 'other'
    
    def run_complete_eda(self):
        print("\n" + "="*60)
        print(" Starting Comprehensive Data Citation Detection EDA")
        print("="*60)
        self.load_data()
       
        type_counts, citations_per_article = self.analyze_labels_distribution()
        citation_categories, pattern_type_matrix = self.analyze_citation_patterns()
        stats_df = self.analyze_document_characteristics()
        context_analysis = self.analyze_citation_context()
        da_analysis = self.analyze_data_availability_sections()
        insights= self.generate_summary_insights()
        
        print("\n" + "="*60)
        print(" EDA COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            'type_counts': type_counts,
            'citations_per_article': citations_per_article,
            'citation_categories': citation_categories,
            'pattern_type_matrix': pattern_type_matrix,
            'document_stats': stats_df,
            'context_analysis': context_analysis,
            'da_analysis': da_analysis,
            'insights': insights,
        }


if __name__ == "__main__":
  
    BASE_PATH = "/kaggle/input/make-data-count-finding-data-references"  
    
  
    eda = DataCitationEDA(BASE_PATH)
    results = eda.run_complete_eda()
    
    print(f"\n EDA complete! Results stored in 'results' dictionary")
    print(f" Dataset path: {BASE_PATH}")

