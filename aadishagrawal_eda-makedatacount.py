import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re
from collections import defaultdict, Counter
from wordcloud import WordCloud
import xml.etree.ElementTree as ET

# Optional imports with fallback
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    print("PyMuPDF not available - PDF processing will be skipped")

try:
    from textstat import flesch_reading_ease, flesch_kincaid_grade
    HAS_TEXTSTAT = True
except ImportError:
    HAS_TEXTSTAT = False
    print("textstat not available - readability analysis will be skipped")

class MakeDataEDA:
    def __init__(self, base_path):
        self.base_path = Path(base_path)
        self.train_labels = None
        self.documents = {}
        self.document_stats = {}
        self.citation_patterns = {
            # DOI patterns
            "doi": r"(?:doi:?\s*)?(?:https?://)?(?:dx\.)?doi\.org/(?:10\.\d{4,9}/[-._;()/:A-Z0-9]+)",
            "doi_direct": r"(?:^|\s)(?:10\.\d{4,9}/[-._;()/:A-Z0-9]+)(?=\s|$|[.,;)\]])",
            
            # Repository URLs
            "zenodo": r"(?:https?://)?zenodo\.org/record/(?:\d+)",
            "figshare": r"(?:https?://)?figshare\.com/(?:articles/|s/)?(?:dataset/)?(?:\w+/)?(?:\d+)",
            "dryad": r"(?:https?://)?(?:datadryad\.org|dryad\.org)/(?:stash/dataset/)?(?:doi:)?(?:10\.\d+/[\w.-]+)",
            "github": r"(?:https?://)?github\.com/(?:[\w\-_.]+/[\w\-_.]+)",
            "osf": r"(?:https?://)?osf\.io/(?:[a-z0-9]{5})",
            
            # Biological databases
            "geo_gse": r"\bGSE\d+\b",
            "geo_gsm": r"\bGSM\d+\b",
            "sra": r"\b(?:SRA|SRR|SRX|SRP|SAMN|PRJNA|PRJEB)\d+\b",
            "pdb": r"\b(?:pdb\s+)?(?:\d[A-Z0-9]{3}|[1-9][A-Z0-9]{3})\b",
            "uniprot": r"\b(?:UniProt:?\s*)?(?:[A-Z]\d[A-Z0-9]{3}\d|[OPQ]\d[A-Z0-9]{3}\d)\b",
            "ensembl": r"\bENS[A-Z]*[GT]\d{11}\b",
            "arrayexpress": r"\bE-[A-Z]+-\d+\b",
            
            # Chemical/drug databases
            "chembl": r"\bCHEMBL\d+\b",
            "pubchem": r"\b(?:PubChem|CID)[-:]?\s*(?:\d+)\b",
            
            # Other databases
            "ncbi_gene": r"\b(?:Gene ID|GeneID)[-:]?\s*(?:\d{4,})\b",
            "omim": r"\bOMIM[-:]?\s*(?:\d{6})\b",
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
        
        # Additional textual insights from the visualizations
        print(f"\nCitation Distribution Insights:")
        print(f"  Most common citation count per article: {citations_per_article.mode().iloc[0]}")
        print(f"  Articles with >5 citations: {(citations_per_article > 5).sum()}")
        print(f"  Articles with single citation: {(citations_per_article == 1).sum()}")
        
        print(f"\nType Combination Analysis:")
        print(f"  Most common combination: {combo_names[0]} ({combo_counts[0]} articles)")
        print(f"  Articles with mixed types: {sum(1 for combo in combo_names if len(eval(combo)) > 1)}")
        
        print(f"\nDataset ID Length Insights:")
        print(f"  Most common length range: {dataset_lengths.mode().iloc[0]} characters")
        print(f"  Short IDs (<10 chars): {(dataset_lengths < 10).sum()} ({(dataset_lengths < 10).mean()*100:.1f}%)")
        print(f"  Long IDs (>30 chars): {(dataset_lengths > 30).sum()} ({(dataset_lengths > 30).mean()*100:.1f}%)")
        
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
        
        # Additional textual insights from the visualizations
        print(f"\nDocument Distribution Insights:")
        print(f"  Length distribution: median={stats_df['length'].median():,.0f}, "
              f"90th percentile={stats_df['length'].quantile(0.9):,.0f}")
        print(f"  Word count distribution: median={stats_df['words'].median():,.0f}, "
              f"90th percentile={stats_df['words'].quantile(0.9):,.0f}")
        print(f"  Very long documents (>100k chars): {(stats_df['length'] > 100000).sum()} "
              f"({(stats_df['length'] > 100000).mean()*100:.1f}%)")
        print(f"  Documents with >20 citations: {(stats_df['citations_found'] > 20).sum()} "
              f"({(stats_df['citations_found'] > 20).mean()*100:.1f}%)")
        print(f"  Documents with minimal paragraphs (<5): {(stats_df['paragraphs'] < 5).sum()} "
              f"({(stats_df['paragraphs'] < 5).mean()*100:.1f}%)")
        
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
            
            # Additional textual insights from the visualizations
            print(f"\nData Availability Visualization Insights:")
            print(f"  Section length distribution: median={da_df['length'].median():.0f}, "
                  f"very long (>50k chars): {(da_df['length'] > 50000).sum()}")
            print(f"  Feature prevalence: URLs most common ({da_df['has_url'].mean()*100:.1f}%), "
                  f"DOIs moderate ({da_df['has_doi'].mean()*100:.1f}%), "
                  f"GitHub links least common ({da_df['has_github'].mean()*100:.1f}%)")
            print(f"  Well-structured sections (with URLs + DOIs): "
                  f"{(da_df['has_url'] & da_df['has_doi']).sum()} "
                  f"({(da_df['has_url'] & da_df['has_doi']).mean()*100:.1f}%)")
            
            return da_df
        else:
            print("No data availability sections found")
            return None
    
    def generate_summary_insights(self):
        print("\n" + "="*60)
        print("COMPREHENSIVE INSIGHTS FOR MODEL BUILDING")
        print("="*60)
        
        insights = []
        
        # Class distribution insights
        type_counts = self.train_labels['type'].value_counts()
        primary_pct = (type_counts.get('Primary', 0) / len(self.train_labels)) * 100
        secondary_pct = (type_counts.get('Secondary', 0) / len(self.train_labels)) * 100
        
        insights.append(f"ğŸ“Š Class Distribution: {primary_pct:.1f}% Primary, {secondary_pct:.1f}% Secondary")
        
        if abs(primary_pct - secondary_pct) > 20:
            insights.append("âš ï¸� Significant class imbalance - implement SMOTE, class weighting, or ensemble methods")
        
        # Pattern diversity insights
        unique_patterns = len(set([self.identify_citation_pattern(cit) 
                                 for cit in self.train_labels['dataset_id']]))
        insights.append(f"ğŸ”� Citation Diversity: {unique_patterns} different citation patterns detected")
        
        # Dataset ID complexity insights
        dataset_ids = self.train_labels['dataset_id'].astype(str)
        id_lengths = dataset_ids.str.len()
        length_cv = id_lengths.std() / id_lengths.mean()
        
        if length_cv > 1.0:
            insights.append(f"ğŸ“� High length variability (CV={length_cv:.2f}) - use dynamic padding and attention")
        
        # Document coverage insights
        if self.document_stats:
            avg_length = np.mean([stats['length'] for stats in self.document_stats.values()])
            insights.append(f"ğŸ“„ Average Document Length: {avg_length:,.0f} characters")
            
            if avg_length > 50000:
                insights.append("ğŸ“š Long documents detected - consider chunking or hierarchical processing")
        
        # Citation density insights
        citations_per_article = self.train_labels.groupby('article_id').size()
        avg_citations = citations_per_article.mean()
        insights.append(f"ğŸ“ˆ Average Citations per Article: {avg_citations:.1f}")
        
        if avg_citations > 10:
            insights.append("ğŸ�¯ High citation density - rich dataset for training complex models")
        
        # Quality indicators
        quality_issues = {
            'very_short': (id_lengths < 5).sum(),
            'very_long': (id_lengths > 200).sum(),
            'contains_whitespace': dataset_ids.str.contains(r'\s', regex=True).sum(),
            'potential_malformed': dataset_ids.str.contains(r'[<>"|{}\\^`]', regex=True).sum(),
        }
        
        total_quality_issues = sum(quality_issues.values())
        if total_quality_issues > len(dataset_ids) * 0.05:
            insights.append(f"âš ï¸� Quality issues detected in {total_quality_issues} citations - implement robust preprocessing")
        
        # Feature engineering opportunities
        char_diversity = len(set(''.join(dataset_ids)))
        if char_diversity > 100:
            insights.append(f"ğŸ”¤ High character diversity ({char_diversity}) - consider subword tokenization")
        
        # Pattern-based insights
        pattern_coverage = 0
        for pattern_name, pattern in self.citation_patterns.items():
            matches = dataset_ids.str.contains(pattern, case=False, na=False, regex=True).sum()
            pattern_coverage += matches
        
        coverage_rate = pattern_coverage / len(dataset_ids)
        if coverage_rate < 0.7:
            insights.append(f"ğŸ”� Low pattern coverage ({coverage_rate:.1%}) - need robust fallback features")
        
        # Cross-validation insights
        articles = self.train_labels['article_id'].unique()
        shared_datasets = self.train_labels['dataset_id'].value_counts()
        shared_count = (shared_datasets > 1).sum()
        
        if shared_count > len(articles) * 0.1:
            insights.append("ğŸ”„ Data leakage risk - use GroupKFold with article grouping")
        
        # Model complexity insights
        unique_tokens = len(set(re.findall(r'\w+|[^\w\s]', ' '.join(dataset_ids))))
        if unique_tokens > 10000:
            insights.append(f"ğŸ§  High vocabulary complexity ({unique_tokens:,} tokens) - consider ensemble methods")
        
        # Recommendations summary
        print("Key Insights and Recommendations:")
        for i, insight in enumerate(insights, 1):
            print(f"{i:2d}. {insight}")
        
        # Strategic recommendations
        print(f"\nğŸ�¯ STRATEGIC RECOMMENDATIONS:")
        print("   â€¢ Use stratified sampling to maintain class balance")
        print("   â€¢ Implement robust text preprocessing pipeline")
        print("   â€¢ Consider multi-stage modeling (pattern detection + classification)")
        print("   â€¢ Use ensemble methods for handling pattern diversity")
        print("   â€¢ Implement careful cross-validation to prevent leakage")
        print("   â€¢ Consider active learning for edge cases")
        
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
       
        # Core analyses
        type_counts, citations_per_article = self.analyze_labels_distribution()
        citation_categories, pattern_type_matrix = self.analyze_citation_patterns()
        stats_df = self.analyze_document_characteristics()
        context_analysis = self.analyze_citation_context()
        da_analysis = self.analyze_data_availability_sections()
        
        # Advanced analyses for model building
        print("\n" + "="*60)
        print(" ADVANCED ANALYSES FOR MODEL BUILDING")
        print("="*60)
        
        class_imbalance = self.analyze_class_imbalance_detailed()
        dataset_characteristics = self.analyze_dataset_id_characteristics()
        quality_indicators = self.analyze_citation_quality_indicators()
        coverage_analysis = self.analyze_article_document_coverage()
        preprocessing_req = self.analyze_text_preprocessing_requirements()
        leakage_analysis = self.analyze_potential_data_leakage()
        feature_opportunities = self.analyze_feature_engineering_opportunities()
        
        # Specialized analyses for edge cases and model optimization
        print("\n" + "="*60)
        print(" SPECIALIZED ANALYSES FOR MODEL OPTIMIZATION")
        print("="*60)
        
        edge_cases = self.analyze_edge_cases_and_anomalies()
        cv_strategy = self.analyze_cross_validation_strategy()
        model_complexity = self.analyze_model_complexity_indicators()
        
        # Generate insights
        insights = self.generate_summary_insights()
        
        print("\n" + "="*60)
        print(" EDA COMPLETED SUCCESSFULLY!")
        print("="*60)
        
        return {
            # Core analyses
            'type_counts': type_counts,
            'citations_per_article': citations_per_article,
            'citation_categories': citation_categories,
            'pattern_type_matrix': pattern_type_matrix,
            'document_stats': stats_df,
            'context_analysis': context_analysis,
            'da_analysis': da_analysis,
            'insights': insights,
            
            # Advanced analyses
            'class_imbalance': class_imbalance,
            'dataset_characteristics': dataset_characteristics,
            'quality_indicators': quality_indicators,
            'coverage_analysis': coverage_analysis,
            'preprocessing_requirements': preprocessing_req,
            'leakage_analysis': leakage_analysis,
            'feature_opportunities': feature_opportunities,
            
            # Specialized analyses
            'edge_cases': edge_cases,
            'cv_strategy': cv_strategy,
            'model_complexity': model_complexity,
        }
    
    def analyze_class_imbalance_detailed(self):
        """Detailed analysis of class imbalance and its implications"""
        print("\n" + "="*60)
        print("DETAILED CLASS IMBALANCE ANALYSIS")
        print("="*60)
        
        # Overall class distribution
        type_counts = self.train_labels['type'].value_counts()
        total_samples = len(self.train_labels)
        
        print("Overall Class Distribution:")
        for class_type, count in type_counts.items():
            percentage = (count / total_samples) * 100
            print(f"  {class_type}: {count:,} ({percentage:.2f}%)")
        
        # Imbalance ratio
        majority_class = type_counts.index[0]
        minority_class = type_counts.index[1] if len(type_counts) > 1 else None
        
        if minority_class:
            imbalance_ratio = type_counts[majority_class] / type_counts[minority_class]
            print(f"\nImbalance Ratio: {imbalance_ratio:.2f}:1 ({majority_class}:{minority_class})")
            
            if imbalance_ratio > 3:
                print("âš ï¸�  High imbalance detected - consider resampling techniques")
        
        # Per-article class distribution
        article_class_dist = self.train_labels.groupby('article_id')['type'].value_counts()
        articles_with_both = article_class_dist.groupby('article_id').size()
        mixed_articles = articles_with_both[articles_with_both > 1].count()
        
        print(f"\nArticle-level Analysis:")
        print(f"  Total articles: {self.train_labels['article_id'].nunique():,}")
        print(f"  Articles with both types: {mixed_articles:,}")
        print(f"  Articles with only one type: {self.train_labels['article_id'].nunique() - mixed_articles:,}")
        
        # Pattern-specific imbalance
        pattern_imbalance = {}
        for pattern in self.citation_patterns.keys():
            pattern_citations = self.train_labels[
                self.train_labels['dataset_id'].str.contains(
                    self.citation_patterns[pattern], case=False, na=False, regex=True
                )
            ]
            if len(pattern_citations) > 0:
                pattern_class_dist = pattern_citations['type'].value_counts()
                if len(pattern_class_dist) > 1:
                    pattern_imbalance[pattern] = pattern_class_dist[pattern_class_dist.index[0]] / pattern_class_dist[pattern_class_dist.index[1]]
        
        print(f"\nPattern-specific Imbalance (top 10):")
        for pattern, ratio in sorted(pattern_imbalance.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {pattern}: {ratio:.2f}:1")
        
        return {
            'overall_distribution': type_counts,
            'imbalance_ratio': imbalance_ratio if minority_class else None,
            'mixed_articles': mixed_articles,
            'pattern_imbalance': pattern_imbalance
        }
    
    def analyze_dataset_id_characteristics(self):
        """Comprehensive analysis of dataset ID characteristics"""
        print("\n" + "="*60)
        print("DATASET ID CHARACTERISTICS ANALYSIS")
        print("="*60)
        
        dataset_ids = self.train_labels['dataset_id'].astype(str)
        
        # Length analysis
        id_lengths = dataset_ids.str.len()
        print(f"Dataset ID Length Statistics:")
        print(f"  Mean: {id_lengths.mean():.1f}")
        print(f"  Median: {id_lengths.median():.1f}")
        print(f"  Std: {id_lengths.std():.1f}")
        print(f"  Min: {id_lengths.min()}")
        print(f"  Max: {id_lengths.max()}")
        
        # Character composition analysis
        char_stats = {
            'has_numbers': dataset_ids.str.contains(r'\d', regex=True).sum(),
            'has_letters': dataset_ids.str.contains(r'[a-zA-Z]', regex=True).sum(),
            'has_special_chars': dataset_ids.str.contains(r'[^a-zA-Z0-9]', regex=True).sum(),
            'has_url_structure': dataset_ids.str.contains(r'https?://', regex=True).sum(),
            'has_doi_structure': dataset_ids.str.contains(r'10\.\d+/', regex=True).sum(),
            'has_forward_slash': dataset_ids.str.contains(r'/', regex=True).sum(),
            'has_underscore': dataset_ids.str.contains(r'_', regex=True).sum(),
            'has_hyphen': dataset_ids.str.contains(r'-', regex=True).sum(),
            'has_period': dataset_ids.str.contains(r'\.', regex=True).sum(),
        }
        
        print(f"\nCharacter Composition:")
        for stat, count in char_stats.items():
            percentage = (count / len(dataset_ids)) * 100
            print(f"  {stat.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
        
        # Unique prefixes and suffixes
        prefixes = dataset_ids.str[:10].value_counts().head(10)
        suffixes = dataset_ids.str[-10:].value_counts().head(10)
        
        print(f"\nTop 10 Prefixes (first 10 chars):")
        for prefix, count in prefixes.items():
            print(f"  '{prefix}': {count}")
        
        print(f"\nTop 10 Suffixes (last 10 chars):")
        for suffix, count in suffixes.items():
            print(f"  '{suffix}': {count}")
        
        # URL domain analysis
        url_mask = dataset_ids.str.contains(r'https?://', regex=True)
        if url_mask.sum() > 0:
            urls = dataset_ids[url_mask]
            domains = urls.str.extract(r'https?://(?:www\.)?([^/]+)')[0].value_counts()
            print(f"\nTop URL Domains:")
            for domain, count in domains.head(10).items():
                print(f"  {domain}: {count}")
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Length distribution
        id_lengths.hist(bins=50, ax=axes[0,0], alpha=0.7, color='skyblue')
        axes[0,0].set_title('Dataset ID Length Distribution')
        axes[0,0].set_xlabel('Character Length')
        axes[0,0].set_ylabel('Count')
        
        # Character composition
        comp_names = list(char_stats.keys())[:8]  # Top 8 for readability
        comp_counts = [char_stats[name] for name in comp_names]
        axes[0,1].bar(range(len(comp_names)), comp_counts, color='lightcoral')
        axes[0,1].set_title('Character Composition')
        axes[0,1].set_xticks(range(len(comp_names)))
        axes[0,1].set_xticklabels([name.replace('_', ' ') for name in comp_names], rotation=45)
        
        # Length vs Type
        length_data = []
        type_data = []
        for cite_type in self.train_labels['type'].unique():
            type_subset = self.train_labels[self.train_labels['type'] == cite_type]
            lengths = type_subset['dataset_id'].str.len()
            length_data.extend(lengths.tolist())
            type_data.extend([cite_type] * len(lengths))
        
        length_df = pd.DataFrame({'length': length_data, 'type': type_data})
        length_df.boxplot(column='length', by='type', ax=axes[1,0])
        axes[1,0].set_title('Dataset ID Length by Citation Type')
        axes[1,0].set_ylabel('Character Length')
        axes[1,0].set_xlabel('Citation Type')
        
        # Pattern distribution
        pattern_matches = {}
        for pattern_name, pattern in self.citation_patterns.items():
            matches = dataset_ids.str.contains(pattern, case=False, na=False, regex=True).sum()
            pattern_matches[pattern_name] = matches
        
        sorted_patterns = sorted(pattern_matches.items(), key=lambda x: x[1], reverse=True)[:15]
        pattern_names = [item[0] for item in sorted_patterns]
        pattern_counts = [item[1] for item in sorted_patterns]
        
        axes[1,1].barh(range(len(pattern_names)), pattern_counts, color='lightgreen')
        axes[1,1].set_title('Pattern Match Distribution (Top 15)')
        axes[1,1].set_yticks(range(len(pattern_names)))
        axes[1,1].set_yticklabels(pattern_names)
        
        plt.tight_layout()
        plt.show()
        
        # Additional textual insights from the visualizations
        print(f"\nDataset ID Visualization Insights:")
        print(f"  Length distribution: {(id_lengths < 10).sum()} short (<10), "
              f"{((id_lengths >= 10) & (id_lengths <= 30)).sum()} medium (10-30), "
              f"{(id_lengths > 30).sum()} long (>30)")
        print(f"  Character composition diversity: {len([k for k, v in char_stats.items() if v > len(dataset_ids)*0.1])} "
              f"features present in >10% of IDs")
        print(f"  Length vs Type analysis: Primary citations tend to be {'longer' if self.train_labels[self.train_labels['type'] == 'Primary']['dataset_id'].str.len().mean() > self.train_labels[self.train_labels['type'] == 'Secondary']['dataset_id'].str.len().mean() else 'shorter'} "
              f"than Secondary citations")
        print(f"  Pattern coverage: Top 3 patterns cover {sum(pattern_counts[:3])} citations "
              f"({sum(pattern_counts[:3])/len(dataset_ids)*100:.1f}% of total)")
        
        return {
            'length_stats': id_lengths.describe(),
            'char_composition': char_stats,
            'pattern_matches': pattern_matches,
            'prefixes': prefixes,
            'suffixes': suffixes
        }
    
    def analyze_citation_quality_indicators(self):
        """Analyze quality indicators that might affect model performance"""
        print("\n" + "="*60)
        print("CITATION QUALITY INDICATORS ANALYSIS")
        print("="*60)
        
        dataset_ids = self.train_labels['dataset_id'].astype(str)
        
        # Potential quality issues
        quality_issues = {
            'empty_or_null': dataset_ids.isin(['', 'nan', 'null', 'None']).sum(),
            'very_short': (dataset_ids.str.len() < 5).sum(),
            'very_long': (dataset_ids.str.len() > 200).sum(),
            'contains_whitespace': dataset_ids.str.contains(r'\s', regex=True).sum(),
            'contains_brackets': dataset_ids.str.contains(r'[\[\]{}()]', regex=True).sum(),
            'contains_quotes': dataset_ids.str.contains(r'["\']', regex=True).sum(),
            'starts_with_number': dataset_ids.str.match(r'^\d', na=False).sum(),
            'all_uppercase': dataset_ids.str.isupper().sum(),
            'all_lowercase': dataset_ids.str.islower().sum(),
            'mixed_case': (~dataset_ids.str.isupper() & ~dataset_ids.str.islower()).sum(),
            'contains_unicode': dataset_ids.str.contains(r'[^\x00-\x7F]', regex=True).sum(),
            'potential_truncated': dataset_ids.str.endswith('...').sum(),
            'contains_multiple_urls': dataset_ids.str.count(r'https?://').gt(1).sum(),
        }
        
        print("Quality Issues Detection:")
        for issue, count in quality_issues.items():
            percentage = (count / len(dataset_ids)) * 100
            status = "âš ï¸�" if percentage > 5 else "âœ“"
            print(f"  {status} {issue.replace('_', ' ').title()}: {count:,} ({percentage:.1f}%)")
        
        # Duplicate analysis
        duplicate_ids = dataset_ids.duplicated().sum()
        duplicate_within_article = self.train_labels.groupby('article_id')['dataset_id'].apply(
            lambda x: x.duplicated().sum()
        ).sum()
        
        print(f"\nDuplicate Analysis:")
        print(f"  Total duplicates: {duplicate_ids:,}")
        print(f"  Duplicates within same article: {duplicate_within_article:,}")
        
        # Cross-reference with actual documents
        if self.documents:
            citation_findability = {}
            sample_size = min(100, len(self.train_labels))
            sample_df = self.train_labels.sample(sample_size)
            
            findable_count = 0
            for _, row in sample_df.iterrows():
                article_id = str(row['article_id'])
                dataset_id = str(row['dataset_id'])
                
                if article_id in self.documents:
                    text = self.documents[article_id].lower()
                    if dataset_id.lower() in text:
                        findable_count += 1
            
            findability_rate = (findable_count / sample_size) * 100
            print(f"\nCitation Findability in Documents:")
            print(f"  Findable citations: {findable_count}/{sample_size} ({findability_rate:.1f}%)")
            
            if findability_rate < 50:
                print("  âš ï¸�  Low findability rate - may indicate quality issues")
        
        return {
            'quality_issues': quality_issues,
            'duplicate_count': duplicate_ids,
            'duplicate_within_article': duplicate_within_article,
            'findability_rate': findability_rate if self.documents else None
        }
    
    def analyze_article_document_coverage(self):
        """Analyze coverage between labeled articles and available documents"""
        print("\n" + "="*60)
        print("ARTICLE-DOCUMENT COVERAGE ANALYSIS")
        print("="*60)
        
        labeled_articles = set(self.train_labels['article_id'].astype(str))
        available_documents = set(self.documents.keys()) if self.documents else set()
        
        # Coverage statistics
        total_labeled = len(labeled_articles)
        total_available = len(available_documents)
        intersection = len(labeled_articles & available_documents)
        
        print(f"Coverage Statistics:")
        print(f"  Labeled articles: {total_labeled:,}")
        print(f"  Available documents: {total_available:,}")
        print(f"  Intersection: {intersection:,}")
        print(f"  Coverage rate: {(intersection/total_labeled)*100:.1f}%")
        
        # Missing documents
        missing_docs = labeled_articles - available_documents
        extra_docs = available_documents - labeled_articles
        
        print(f"\nMissing/Extra Analysis:")
        print(f"  Missing documents: {len(missing_docs):,}")
        print(f"  Extra documents: {len(extra_docs):,}")
        
        if missing_docs:
            print(f"  Sample missing: {list(missing_docs)[:5]}")
        
        # Impact on different citation types
        if intersection > 0:
            covered_labels = self.train_labels[
                self.train_labels['article_id'].astype(str).isin(available_documents)
            ]
            coverage_by_type = covered_labels['type'].value_counts()
            total_by_type = self.train_labels['type'].value_counts()
            
            print(f"\nCoverage by Citation Type:")
            for cite_type in total_by_type.index:
                covered = coverage_by_type.get(cite_type, 0)
                total = total_by_type[cite_type]
                coverage_pct = (covered / total) * 100
                print(f"  {cite_type}: {covered:,}/{total:,} ({coverage_pct:.1f}%)")
        
        return {
            'labeled_articles': total_labeled,
            'available_documents': total_available,
            'intersection': intersection,
            'coverage_rate': (intersection/total_labeled)*100 if total_labeled > 0 else 0,
            'missing_docs': len(missing_docs),
            'extra_docs': len(extra_docs)
        }
    
    def analyze_text_preprocessing_requirements(self):
        """Analyze text characteristics to determine preprocessing needs"""
        print("\n" + "="*60)
        print("TEXT PREPROCESSING REQUIREMENTS ANALYSIS")
        print("="*60)
        
        if not self.documents:
            print("No documents available for text analysis")
            return None
        
        # Sample documents for analysis
        sample_size = min(50, len(self.documents))
        sample_docs = dict(list(self.documents.items())[:sample_size])
        
        preprocessing_stats = {
            'encoding_issues': 0,
            'non_ascii_chars': 0,
            'excessive_whitespace': 0,
            'html_tags': 0,
            'xml_artifacts': 0,
            'reference_sections': 0,
            'figure_captions': 0,
            'table_content': 0,
            'mathematical_content': 0,
            'special_symbols': 0,
            'line_breaks': 0,
            'multiple_languages': 0
        }
        
        patterns = {
            'html_tags': r'<[^>]+>',
            'xml_artifacts': r'&[a-zA-Z]+;',
            'reference_sections': r'(?i)references?\s*\n',
            'figure_captions': r'(?i)fig(?:ure)?\s*\d+',
            'table_content': r'(?i)table\s*\d+',
            'mathematical_content': r'[\$\\\{\}]|\\[a-zA-Z]+',
            'special_symbols': r'[^\w\s\.\,\!\?\;\:\-\(\)\[\]]',
            'excessive_whitespace': r'\s{3,}',
            'line_breaks': r'\n{2,}',
            'non_ascii_chars': r'[^\x00-\x7F]'
        }
        
        for doc_id, text in sample_docs.items():
            # Check for various preprocessing needs
            for pattern_name, pattern in patterns.items():
                if re.search(pattern, text):
                    preprocessing_stats[pattern_name] += 1
        
        print("Preprocessing Requirements (based on sample):")
        for requirement, count in preprocessing_stats.items():
            percentage = (count / sample_size) * 100
            status = "âš ï¸�" if percentage > 20 else "âœ“"
            print(f"  {status} {requirement.replace('_', ' ').title()}: {count}/{sample_size} ({percentage:.1f}%)")
        
        # Text quality metrics
        quality_metrics = {
            'avg_sentence_length': [],
            'avg_word_length': [],
            'vocabulary_richness': [],
            'punctuation_density': []
        }
        
        for text in sample_docs.values():
            sentences = re.split(r'[.!?]+', text)
            words = re.findall(r'\b\w+\b', text.lower())
            
            if sentences:
                avg_sent_len = np.mean([len(s.split()) for s in sentences if s.strip()])
                quality_metrics['avg_sentence_length'].append(avg_sent_len)
            
            if words:
                avg_word_len = np.mean([len(w) for w in words])
                quality_metrics['avg_word_length'].append(avg_word_len)
                
                unique_words = len(set(words))
                total_words = len(words)
                vocab_richness = unique_words / total_words if total_words > 0 else 0
                quality_metrics['vocabulary_richness'].append(vocab_richness)
            
            punct_count = len(re.findall(r'[.!?;:,]', text))
            punct_density = punct_count / len(text) if len(text) > 0 else 0
            quality_metrics['punctuation_density'].append(punct_density)
        
        print(f"\nText Quality Metrics:")
        for metric, values in quality_metrics.items():
            if values:
                mean_val = np.mean(values)
                print(f"  {metric.replace('_', ' ').title()}: {mean_val:.2f}")
        
        return {
            'preprocessing_stats': preprocessing_stats,
            'quality_metrics': quality_metrics,
            'sample_size': sample_size
        }
    
    def analyze_potential_data_leakage(self):
        """Analyze potential data leakage issues"""
        print("\n" + "="*60)
        print("POTENTIAL DATA LEAKAGE ANALYSIS")
        print("="*60)
        
        # Check for dataset IDs that appear in multiple articles
        id_article_mapping = self.train_labels.groupby('dataset_id')['article_id'].apply(set)
        multi_article_ids = id_article_mapping[id_article_mapping.str.len() > 1]
        
        print(f"Cross-Article Citation Analysis:")
        print(f"  Total unique dataset IDs: {len(id_article_mapping):,}")
        print(f"  IDs appearing in multiple articles: {len(multi_article_ids):,}")
        
        if len(multi_article_ids) > 0:
            print(f"  Max articles per ID: {max(len(articles) for articles in multi_article_ids)}")
            print(f"  âš ï¸�  Potential leakage risk from shared citations")
        
        # Check for similar dataset IDs (potential variations)
        dataset_ids = self.train_labels['dataset_id'].astype(str).unique()
        similar_pairs = []
        
        # Sample-based similarity check to avoid O(nÂ²) complexity
        sample_size = min(1000, len(dataset_ids))
        sample_ids = np.random.choice(dataset_ids, sample_size, replace=False)
        
        for i, id1 in enumerate(sample_ids):
            for id2 in sample_ids[i+1:]:
                # Simple similarity check
                if len(id1) > 10 and len(id2) > 10:
                    # Check if one is substring of another
                    if id1 in id2 or id2 in id1:
                        similar_pairs.append((id1, id2))
                    # Check edit distance for similar length IDs
                    elif abs(len(id1) - len(id2)) < 3:
                        common_chars = sum(1 for c1, c2 in zip(id1, id2) if c1 == c2)
                        similarity = common_chars / max(len(id1), len(id2))
                        if similarity > 0.8:
                            similar_pairs.append((id1, id2))
        
        print(f"\nSimilarity Analysis (sample-based):")
        print(f"  Similar ID pairs found: {len(similar_pairs)}")
        if similar_pairs:
            print(f"  Sample similar pairs: {similar_pairs[:3]}")
            print(f"  âš ï¸�  Potential variations of same dataset")
        
        # Temporal leakage check (if timestamps available)
        # This would require additional timestamp data
        
        return {
            'multi_article_ids': len(multi_article_ids),
            'similar_pairs': len(similar_pairs),
            'leakage_risk': len(multi_article_ids) > 0 or len(similar_pairs) > 0
        }
    
    def analyze_feature_engineering_opportunities(self):
        """Identify opportunities for feature engineering"""
        print("\n" + "="*60)
        print("FEATURE ENGINEERING OPPORTUNITIES")
        print("="*60)
        
        dataset_ids = self.train_labels['dataset_id'].astype(str)
        
        # Pattern-based features
        pattern_features = {}
        for pattern_name, pattern in self.citation_patterns.items():
            matches = dataset_ids.str.contains(pattern, case=False, na=False, regex=True)
            if matches.sum() > 0:
                # Analyze correlation with target
                correlation_data = self.train_labels[matches].copy()
                if len(correlation_data) > 10:
                    primary_rate = (correlation_data['type'] == 'Primary').mean()
                    pattern_features[pattern_name] = {
                        'count': matches.sum(),
                        'primary_rate': primary_rate,
                        'discriminative_power': abs(primary_rate - 0.5)
                    }
        
        # Sort by discriminative power
        top_patterns = sorted(pattern_features.items(), 
                             key=lambda x: x[1]['discriminative_power'], 
                             reverse=True)
        
        print("Most Discriminative Patterns:")
        for pattern, stats in top_patterns[:10]:
            print(f"  {pattern}: {stats['count']} samples, "
                  f"{stats['primary_rate']:.3f} primary rate, "
                  f"{stats['discriminative_power']:.3f} discriminative power")
        
        # Length-based features
        length_stats = {}
        for cite_type in self.train_labels['type'].unique():
            type_data = self.train_labels[self.train_labels['type'] == cite_type]
            lengths = type_data['dataset_id'].str.len()
            length_stats[cite_type] = {
                'mean': lengths.mean(),
                'std': lengths.std(),
                'median': lengths.median()
            }
        
        print(f"\nLength-based Features:")
        for cite_type, stats in length_stats.items():
            print(f"  {cite_type}: mean={stats['mean']:.1f}, "
                  f"std={stats['std']:.1f}, median={stats['median']:.1f}")
        
        # Character-based features
        char_features = {
            'digit_ratio': dataset_ids.str.count(r'\d') / dataset_ids.str.len(),
            'special_char_ratio': dataset_ids.str.count(r'[^\w\s]') / dataset_ids.str.len(),
            'uppercase_ratio': dataset_ids.str.count(r'[A-Z]') / dataset_ids.str.len(),
            'url_count': dataset_ids.str.count(r'https?://'),
            'slash_count': dataset_ids.str.count(r'/'),
            'dot_count': dataset_ids.str.count(r'\.'),
        }
        
        print(f"\nCharacter-based Feature Statistics:")
        for feature, values in char_features.items():
            print(f"  {feature}: mean={values.mean():.3f}, std={values.std():.3f}")
        
        # Context-based features (if documents available)
        if self.documents:
            context_features = {
                'citation_position': [],  # Position in document
                'section_type': [],       # Which section contains citation
                'surrounding_keywords': []  # Keywords around citation
            }
            
            # Sample analysis
            sample_size = min(100, len(self.train_labels))
            sample_df = self.train_labels.sample(sample_size)
            
            for _, row in sample_df.iterrows():
                article_id = str(row['article_id'])
                dataset_id = str(row['dataset_id'])
                
                if article_id in self.documents:
                    text = self.documents[article_id]
                    
                    # Find citation position
                    pos = text.lower().find(dataset_id.lower())
                    if pos != -1:
                        relative_pos = pos / len(text)
                        context_features['citation_position'].append(relative_pos)
                        
                        # Analyze section
                        context_window = text[max(0, pos-500):pos+500]
                        if re.search(r'(?i)reference|bibliograph', context_window):
                            context_features['section_type'].append('references')
                        elif re.search(r'(?i)method|approach', context_window):
                            context_features['section_type'].append('methods')
                        elif re.search(r'(?i)result|finding', context_window):
                            context_features['section_type'].append('results')
                        else:
                            context_features['section_type'].append('other')
            
            if context_features['citation_position']:
                print(f"\nContext-based Features:")
                print(f"  Average citation position: {np.mean(context_features['citation_position']):.3f}")
                section_dist = Counter(context_features['section_type'])
                for section, count in section_dist.most_common():
                    print(f"  {section} section: {count} citations")
        
        return {
            'pattern_features': pattern_features,
            'length_features': length_stats,
            'char_features': char_features,
            'context_features': context_features if self.documents else None
        }
    
    def analyze_edge_cases_and_anomalies(self):
        """Detect and analyze edge cases and anomalies in the data"""
        print("\n" + "="*60)
        print("EDGE CASES AND ANOMALIES ANALYSIS")
        print("="*60)
        
        dataset_ids = self.train_labels['dataset_id'].astype(str)
        
        # Detect outliers in dataset ID characteristics
        id_lengths = dataset_ids.str.len()
        length_q1, length_q3 = id_lengths.quantile([0.25, 0.75])
        length_iqr = length_q3 - length_q1
        length_outliers = id_lengths[(id_lengths < length_q1 - 1.5 * length_iqr) | 
                                    (id_lengths > length_q3 + 1.5 * length_iqr)]
        
        print(f"Length Outliers:")
        print(f"  Outliers found: {len(length_outliers)}")
        print(f"  Extreme short (< 3 chars): {(id_lengths < 3).sum()}")
        print(f"  Extreme long (> 500 chars): {(id_lengths > 500).sum()}")
        
        # Malformed citations
        malformed_patterns = {
            'incomplete_doi': dataset_ids.str.contains(r'10\.\d+/$', regex=True),
            'broken_url': dataset_ids.str.contains(r'https?://[^\s]*\s', regex=True),
            'multiple_protocols': dataset_ids.str.count(r'https?://').gt(1),
            'invalid_characters': dataset_ids.str.contains(r'[<>"|{}\\^`\[\]]', regex=True),
            'encoding_artifacts': dataset_ids.str.contains(r'%[0-9A-F]{2}', regex=True),
            'whitespace_issues': dataset_ids.str.contains(r'^\s|\s$', regex=True),
            'repeated_patterns': dataset_ids.str.contains(r'(.)\1{4,}', regex=True),
        }
        
        print(f"\nMalformed Citations:")
        for pattern_name, mask in malformed_patterns.items():
            count = mask.sum()
            if count > 0:
                print(f"  {pattern_name.replace('_', ' ').title()}: {count}")
                # Show examples
                examples = dataset_ids[mask].head(3).tolist()
                print(f"    Examples: {examples}")
        
        # Cross-validation edge cases
        article_citation_counts = self.train_labels.groupby('article_id').size()
        single_citation_articles = article_citation_counts[article_citation_counts == 1]
        high_citation_articles = article_citation_counts[article_citation_counts > 20]
        
        print(f"\nArticle Citation Distribution Edge Cases:")
        print(f"  Articles with single citation: {len(single_citation_articles)}")
        print(f"  Articles with >20 citations: {len(high_citation_articles)}")
        
        # Pattern conflicts (IDs matching multiple patterns)
        pattern_conflicts = []
        for idx, dataset_id in enumerate(dataset_ids):
            matching_patterns = []
            for pattern_name, pattern in self.citation_patterns.items():
                if re.search(pattern, dataset_id, re.IGNORECASE):
                    matching_patterns.append(pattern_name)
            if len(matching_patterns) > 1:
                pattern_conflicts.append((dataset_id, matching_patterns))
        
        print(f"\nPattern Conflicts:")
        print(f"  IDs matching multiple patterns: {len(pattern_conflicts)}")
        if pattern_conflicts:
            for dataset_id, patterns in pattern_conflicts[:5]:
                print(f"    '{dataset_id}' matches: {patterns}")
        
        # Check for potential mislabeled data
        mislabeling_indicators = []
        if self.documents:
            sample_size = min(200, len(self.train_labels))
            sample_df = self.train_labels.sample(sample_size)
            
            for _, row in sample_df.iterrows():
                article_id = str(row['article_id'])
                dataset_id = str(row['dataset_id'])
                cite_type = row['type']
                
                if article_id in self.documents:
                    text = self.documents[article_id]
                    context = self.get_citation_context(text, dataset_id, window=200)
                    
                    if context:
                        # Check for context clues about citation importance
                        primary_indicators = ['primary', 'main', 'central', 'key', 'original']
                        secondary_indicators = ['additional', 'supplementary', 'supporting', 'related']
                        
                        context_lower = context.lower()
                        primary_score = sum(1 for indicator in primary_indicators if indicator in context_lower)
                        secondary_score = sum(1 for indicator in secondary_indicators if indicator in context_lower)
                        
                        # Flag potential mislabeling
                        if cite_type == 'Primary' and secondary_score > primary_score:
                            mislabeling_indicators.append(('Primary->Secondary', dataset_id, context[:100]))
                        elif cite_type == 'Secondary' and primary_score > secondary_score:
                            mislabeling_indicators.append(('Secondary->Primary', dataset_id, context[:100]))
            
            print(f"\nPotential Mislabeling Indicators:")
            print(f"  Suspicious cases found: {len(mislabeling_indicators)}")
            for suggestion, dataset_id, context in mislabeling_indicators[:3]:
                print(f"    {suggestion}: '{dataset_id}' - '{context}...'")
        
        return {
            'length_outliers': len(length_outliers),
            'malformed_patterns': {k: v.sum() for k, v in malformed_patterns.items()},
            'pattern_conflicts': len(pattern_conflicts),
            'single_citation_articles': len(single_citation_articles),
            'high_citation_articles': len(high_citation_articles),
            'mislabeling_indicators': len(mislabeling_indicators) if self.documents else 0
        }
    
    def analyze_cross_validation_strategy(self):
        """Analyze data characteristics to recommend cross-validation strategy"""
        print("\n" + "="*60)
        print("CROSS-VALIDATION STRATEGY ANALYSIS")
        print("="*60)
        
        # Article-level analysis
        articles = self.train_labels['article_id'].unique()
        article_sizes = self.train_labels.groupby('article_id').size()
        
        print(f"Article-level Statistics:")
        print(f"  Total articles: {len(articles):,}")
        print(f"  Citations per article - Mean: {article_sizes.mean():.1f}, Median: {article_sizes.median():.1f}")
        print(f"  Min citations per article: {article_sizes.min()}")
        print(f"  Max citations per article: {article_sizes.max()}")
        
        # Check for temporal patterns (if available)
        # This would require additional timestamp information
        
        # Check for author/journal clustering
        # This would require additional metadata
        
        # Data leakage risks
        duplicate_datasets = self.train_labels['dataset_id'].value_counts()
        shared_datasets = duplicate_datasets[duplicate_datasets > 1]
        
        print(f"\nData Leakage Risks:")
        print(f"  Shared dataset IDs: {len(shared_datasets)}")
        print(f"  Max sharing frequency: {shared_datasets.max() if len(shared_datasets) > 0 else 0}")
        
        # Stratification analysis
        type_distribution = self.train_labels['type'].value_counts()
        min_class_size = type_distribution.min()
        
        print(f"\nStratification Considerations:")
        print(f"  Minimum class size: {min_class_size:,}")
        print(f"  Recommended min folds: {min(10, min_class_size // 50)}")
        
        # Pattern-based stratification
        pattern_distributions = {}
        for pattern_name, pattern in self.citation_patterns.items():
            pattern_mask = self.train_labels['dataset_id'].str.contains(
                pattern, case=False, na=False, regex=True
            )
            if pattern_mask.sum() > 0:
                pattern_type_dist = self.train_labels[pattern_mask]['type'].value_counts()
                pattern_distributions[pattern_name] = pattern_type_dist
        
        print(f"\nPattern-based Distribution:")
        for pattern, dist in list(pattern_distributions.items())[:5]:
            print(f"  {pattern}: {dict(dist)}")
        
        # Recommendations
        recommendations = []
        
        if len(shared_datasets) > len(articles) * 0.1:
            recommendations.append("Use GroupKFold to prevent data leakage from shared dataset IDs")
        
        if article_sizes.std() > article_sizes.mean():
            recommendations.append("Consider stratified sampling to balance article sizes")
        
        if min_class_size < 100:
            recommendations.append("Use StratifiedKFold with fewer folds due to small class sizes")
        
        if len(articles) < 100:
            recommendations.append("Consider Leave-One-Group-Out CV with articles as groups")
        
        print(f"\nRecommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return {
            'total_articles': len(articles),
            'article_size_stats': article_sizes.describe(),
            'shared_datasets': len(shared_datasets),
            'min_class_size': min_class_size,
            'recommendations': recommendations
        }
    
    def analyze_model_complexity_indicators(self):
        """Analyze indicators of model complexity requirements"""
        print("\n" + "="*60)
        print("MODEL COMPLEXITY INDICATORS")
        print("="*60)
        
        # Pattern complexity
        total_patterns = len(self.citation_patterns)
        pattern_matches = {}
        for pattern_name, pattern in self.citation_patterns.items():
            matches = self.train_labels['dataset_id'].str.contains(
                pattern, case=False, na=False, regex=True
            ).sum()
            pattern_matches[pattern_name] = matches
        
        active_patterns = sum(1 for count in pattern_matches.values() if count > 0)
        coverage = sum(pattern_matches.values()) / len(self.train_labels)
        
        print(f"Pattern Complexity:")
        print(f"  Total patterns defined: {total_patterns}")
        print(f"  Active patterns: {active_patterns}")
        print(f"  Pattern coverage: {coverage:.1%}")
        
        # Vocabulary complexity
        dataset_ids = self.train_labels['dataset_id'].astype(str)
        all_chars = ''.join(dataset_ids)
        unique_chars = len(set(all_chars))
        
        # Tokenization complexity
        all_tokens = []
        for dataset_id in dataset_ids:
            tokens = re.findall(r'\w+|[^\w\s]', dataset_id)
            all_tokens.extend(tokens)
        
        unique_tokens = len(set(all_tokens))
        avg_tokens_per_id = len(all_tokens) / len(dataset_ids)
        
        print(f"\nVocabulary Complexity:")
        print(f"  Unique characters: {unique_chars}")
        print(f"  Unique tokens: {unique_tokens:,}")
        print(f"  Average tokens per ID: {avg_tokens_per_id:.1f}")
        
        # Sequence length complexity
        id_lengths = dataset_ids.str.len()
        length_variance = id_lengths.var()
        length_range = id_lengths.max() - id_lengths.min()
        
        print(f"\nSequence Complexity:")
        print(f"  Length variance: {length_variance:.1f}")
        print(f"  Length range: {length_range}")
        print(f"  Length coefficient of variation: {id_lengths.std() / id_lengths.mean():.2f}")
        
        # Decision boundary complexity
        type_entropy = -sum(p * np.log2(p) for p in self.train_labels['type'].value_counts(normalize=True))
        
        # Pattern-type interaction complexity
        interaction_complexity = 0
        for pattern_name, pattern in self.citation_patterns.items():
            pattern_mask = dataset_ids.str.contains(pattern, case=False, na=False, regex=True)
            if pattern_mask.sum() > 10:
                pattern_labels = self.train_labels[pattern_mask]['type']
                if len(pattern_labels.unique()) > 1:
                    pattern_entropy = -sum(p * np.log2(p) for p in pattern_labels.value_counts(normalize=True))
                    interaction_complexity += pattern_entropy
        
        print(f"\nDecision Complexity:")
        print(f"  Overall type entropy: {type_entropy:.3f}")
        print(f"  Pattern-type interaction complexity: {interaction_complexity:.3f}")
        
        # Complexity recommendations
        complexity_indicators = {
            'high_vocabulary': unique_tokens > 10000,
            'high_length_variance': length_variance > 1000,
            'high_pattern_diversity': active_patterns > 15,
            'low_pattern_coverage': coverage < 0.5,
            'high_interaction_complexity': interaction_complexity > 10
        }
        
        print(f"\nComplexity Indicators:")
        for indicator, is_high in complexity_indicators.items():
            status = "âš ï¸�" if is_high else "âœ“"
            print(f"  {status} {indicator.replace('_', ' ').title()}: {is_high}")
        
        # Model recommendations
        recommendations = []
        
        if complexity_indicators['high_vocabulary']:
            recommendations.append("Consider subword tokenization (BPE) for handling large vocabulary")
        
        if complexity_indicators['high_length_variance']:
            recommendations.append("Use sequence padding and attention mechanisms")
        
        if complexity_indicators['low_pattern_coverage']:
            recommendations.append("Implement robust character-level or hybrid models")
        
        if complexity_indicators['high_interaction_complexity']:
            recommendations.append("Consider ensemble methods or multi-task learning")
        
        print(f"\nModel Recommendations:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
        
        return {
            'pattern_complexity': {
                'total_patterns': total_patterns,
                'active_patterns': active_patterns,
                'coverage': coverage
            },
            'vocabulary_complexity': {
                'unique_chars': unique_chars,
                'unique_tokens': unique_tokens,
                'avg_tokens_per_id': avg_tokens_per_id
            },
            'sequence_complexity': {
                'length_variance': length_variance,
                'length_range': length_range,
                'length_cv': id_lengths.std() / id_lengths.mean()
            },
            'decision_complexity': {
                'type_entropy': type_entropy,
                'interaction_complexity': interaction_complexity
            },
            'complexity_indicators': complexity_indicators,
            'recommendations': recommendations
        }



if __name__ == "__main__":

    BASE_PATH = "/kaggle/input/make-data-count-finding-data-references"

    eda = MakeDataEDA(BASE_PATH)
    results = eda.run_complete_eda()
    
    print(f"\n EDA complete! Results stored in 'results' dictionary")
    print(f" Dataset path: {BASE_PATH}")




