#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Kaggle Competition Notebook Analyzer - Enhanced with Public Score Sorting
This script extracts notebooks from a Kaggle competition sorted by public score
and uses OpenAI to summarize approaches.
Designed to run in a Kaggle notebook environment
"""

import os
import json
import time
import pandas as pd
from datetime import datetime
import re
from openai import OpenAI
from kaggle_secrets import UserSecretsClient
from IPython.display import display, Markdown

# Initialize secrets client
user_secrets = UserSecretsClient()

# Get OpenAI API key
openai_api_key = user_secrets.get_secret("openai_key")
openai_client = OpenAI(api_key=openai_api_key)

# Get Kaggle credentials from secrets
try:
    kaggle_username = user_secrets.get_secret("kaggle_username")
    kaggle_key = user_secrets.get_secret("kaggle_key")
    
    # Set environment variables for Kaggle API
    os.environ['KAGGLE_USERNAME'] = kaggle_username
    os.environ['KAGGLE_KEY'] = kaggle_key
    
    print(f"Using Kaggle credentials for user: {kaggle_username}")
except:
    print("Warning: Could not get Kaggle credentials from secrets")
    print("Please add 'kaggle_username' and 'kaggle_key' to your Kaggle secrets")

# Import and authenticate Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.authenticate()
print("Successfully authenticated with Kaggle API")

class KaggleCompetitionAnalyzer:
    def __init__(self, competition_name):
        self.api = api
        self.competition_name = competition_name
        self.notebooks_data = []
        
    def get_competition_notebooks(self, max_notebooks=50, sort_by='scoreDescending'):
        """
        Fetch all public notebooks for a specific competition
        
        sort_by options:
        - 'scoreDescending': Best score first (use for competitions where HIGHER is better - accuracy, MAP, etc.)
        - 'scoreAscending': Best score first (use for competitions where LOWER is better - RMSE, MAE, etc.)
        - 'dateCreated': Newest first
        - 'dateRun': Most recently run first
        - 'relevance': Most relevant to competition
        - 'title': Alphabetical by title
        - 'voteCount': Most votes first
        """
        print(f"\nFetching notebooks for competition: {self.competition_name}")
        print(f"Sorting by: {sort_by}")
        notebooks = []
        page = 1
        
        # For competitions, we need to use the competition filter properly
        try:
            while True:
                # List kernels (notebooks) for the competition
                # The API expects 'competition' to be the competition slug
                kernels = self.api.kernels_list(
                    competition=self.competition_name,
                    page=page,
                    page_size=20,
                    sort_by=sort_by  # Changed from 'voteCount' to sort by score
                )
                
                if not kernels:
                    break
                    
                # Filter for notebooks that actually have scores
                for kernel in kernels:
                    # Debug: print available attributes
                    if page == 1 and len(notebooks) == 0:
                        print(f"\nAvailable kernel attributes: {[attr for attr in dir(kernel) if not attr.startswith('_')]}")
                    
                    # Add all kernels but track which ones have scores
                    kernel_info = {
                        'kernel': kernel,
                        'has_score': False  # We'll need to check this differently
                    }
                    notebooks.append(kernel_info)
                
                print(f"Fetched page {page}, total notebooks: {len(notebooks)}")
                
                if max_notebooks and len(notebooks) >= max_notebooks:
                    notebooks = notebooks[:max_notebooks]
                    break
                    
                page += 1
                time.sleep(1)  # Rate limiting
                
        except Exception as e:
            print(f"Error fetching notebooks: {e}")
            # Fallback: try with search parameter
            print("Trying alternative method with search parameter...")
            
            try:
                kernels = self.api.kernels_list(
                    search=self.competition_name,
                    page=1,
                    page_size=max_notebooks,
                    sort_by=sort_by
                )
                
                for kernel in kernels:
                    kernel_info = {
                        'kernel': kernel,
                        'has_score': hasattr(kernel, 'scorePublic') and kernel.scorePublic is not None
                    }
                    notebooks.append(kernel_info)
                    
            except Exception as e2:
                print(f"Alternative method also failed: {e2}")
        
        # Extract just the kernel objects and print score information
        kernels_with_scores = [nb['kernel'] for nb in notebooks if nb['has_score']]
        kernels_without_scores = [nb['kernel'] for nb in notebooks if not nb['has_score']]
        
        print(f"\nFound {len(kernels_with_scores)} notebooks with public scores")
        print(f"Found {len(kernels_without_scores)} notebooks without public scores")
        
        # If we're sorting by score, prioritize notebooks with scores
        if 'score' in sort_by.lower():
            final_notebooks = kernels_with_scores + kernels_without_scores
        else:
            final_notebooks = [nb['kernel'] for nb in notebooks]
        
        # Print top 5 notebooks with their scores
        print("\nTop notebooks:")
        for i, nb in enumerate(final_notebooks[:5]):
            print(f"{i+1}. {nb.title} by {nb.author} - Votes: {nb.total_votes}")
        
        return final_notebooks
    
    def download_notebook_content(self, kernel_ref, path='notebooks'):
        """Download the actual notebook content using kernel reference"""
        try:
            os.makedirs(path, exist_ok=True)
            
            # Pull the kernel using kernels_pull
            self.api.kernels_pull(kernel_ref, path=path)
            
            # Read the content
            notebook_content = None
            for file in os.listdir(path):
                if file.endswith('.ipynb'):
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                        notebook_content = f.read()
                    break
                elif file.endswith('.py'):
                    with open(os.path.join(path, file), 'r', encoding='utf-8') as f:
                        notebook_content = f.read()
                    break
            
            # Clean up - remove downloaded files to save space
            for file in os.listdir(path):
                os.remove(os.path.join(path, file))
            
            return notebook_content
            
        except Exception as e:
            print(f"Error downloading {kernel_ref}: {e}")
            return None
    
    def extract_code_from_notebook(self, content):
        """Extract code cells from notebook content"""
        if content is None:
            return ""
        
        try:
            if content.strip().startswith('{'):
                # It's a Jupyter notebook
                notebook = json.loads(content)
                code_cells = []
                
                for cell in notebook.get('cells', []):
                    if cell.get('cell_type') == 'code':
                        source = cell.get('source', [])
                        if isinstance(source, list):
                            code_cells.append(''.join(source))
                        else:
                            code_cells.append(source)
                
                return '\n\n'.join(code_cells)
            else:
                # It's a Python script
                return content
        except:
            return content[:10000]  # Return first 10k chars if parsing fails
    
    def collect_notebooks(self, limit=5, sort_by='scoreDescending'):
        """Collect top notebooks with their content, sorted by public score"""
        print(f"\nCollecting top {limit} notebooks...")
        print(f"Note: The Kaggle API notebook listing doesn't include scores in metadata.")
        print(f"We'll fetch notebooks sorted by {sort_by} based on Kaggle's internal ranking.\n")
        
        notebooks = self.get_competition_notebooks(max_notebooks=limit*2, sort_by=sort_by)  # Get extra in case some fail
        
        collected_count = 0
        for idx, nb in enumerate(notebooks):
            if collected_count >= limit:
                break
                
            print(f"\nProcessing {collected_count+1}/{limit}: {nb.title} by {nb.author}")
            print(f"Votes: {nb.total_votes}")
            
            # Use the ref attribute which contains the full path
            kernel_ref = nb.ref
            print(f"Kernel ref: {kernel_ref}")
            
            # Download content
            content = self.download_notebook_content(kernel_ref)
            code_content = self.extract_code_from_notebook(content)
            
            if code_content:
                self.notebooks_data.append({
                    'title': nb.title,
                    'author': nb.author,
                    'kernel_slug': nb.slug,
                    'total_votes': nb.total_votes,
                    'public_score': None,  # Not available in API metadata
                    'private_score': None,  # Not available in API metadata
                    'language': nb.language,
                    'created_date': str(nb.last_run_time),
                    'code_snippet': code_content[:8000]  # Limit for OpenAI context
                })
                collected_count += 1
            
            time.sleep(2)  # Rate limiting
        
        return pd.DataFrame(self.notebooks_data)
    
    def summarize_notebook(self, notebook_data):
        """Use OpenAI to summarize a single notebook's approach"""
        prompt = f"""
        Analyze this Kaggle competition notebook and provide a concise summary:
        
        Title: {notebook_data['title']}
        Author: {notebook_data['author']}
        Votes: {notebook_data['total_votes']}
        
        Code excerpt:
        ```python
        {notebook_data['code_snippet'][:4000]}
        ```
        
        Please summarize:
        1. Main approach/methodology (2-3 sentences)
        2. Key features engineered (bullet points)
        3. Models used (list them)
        4. Any unique techniques or insights
        5. Notable implementation details or optimizations
        
        Keep the summary under 200 words and focus on technical approach.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4.1-mini",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist analyzing Kaggle competition approaches."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=300,
                temperature=0.3
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error summarizing notebook: {e}")
            return "Error generating summary"
    
    def generate_competition_report(self, summaries_df):
        """Generate a comprehensive report of all approaches"""
        # Prepare summaries for the prompt
        summaries_text = "\n\n".join([
            f"**{row['title']} (Votes: {row['total_votes']})**\n{row['summary']}"
            for _, row in summaries_df.iterrows()
        ])
        
        prompt = f"""
        Based on these TOP notebook summaries from the '{self.competition_name}' Kaggle competition 
        (notebooks were sorted by score when fetched), create a comprehensive analysis report:
        
        {summaries_text}
        
        Please provide:
        1. **Common Approaches** - What methods appear across multiple notebooks?
        2. **Feature Engineering Patterns** - Key features and preprocessing techniques
        3. **Model Selection** - Which algorithms were most popular?
        4. **Ensemble Strategies** - How did participants combine models?
        5. **Unique Techniques** - Innovative or unusual approaches
        6. **Implementation Patterns** - Common coding patterns or optimizations
        7. **Best Practices** - What techniques seem most sophisticated?
        
        Note: These notebooks were fetched in score order (best first) even though individual scores aren't shown.
        Format as a structured report with clear sections.
        """
        
        try:
            response = openai_client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": "You are an expert data scientist creating a competition analysis report focused on winning strategies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.5
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error generating report: {e}")
            return "Error generating competition report"

def main():
    """Main execution function"""
    # Configuration
    COMPETITION_NAME = "map-charting-student-math-misunderstandings"  # Change this to your target competition
    NUM_NOTEBOOKS = 10  # Number of TOP SCORING notebooks to analyze
    
    # IMPORTANT: For competitions where HIGHER score is better (like MAP, accuracy), use scoreDescending
    # For competitions where LOWER score is better (like RMSE, MAE), use scoreAscending
    SORT_METHOD = "scoreDescending"  # Sort by best public score (highest first for MAP)
    
    print("="*60)
    print(f"Kaggle Competition Notebook Analyzer")
    print(f"Competition: {COMPETITION_NAME}")
    print(f"Notebooks to analyze: Top {NUM_NOTEBOOKS} (sorted by: {SORT_METHOD})")
    print("="*60)
    
    # Initialize analyzer
    analyzer = KaggleCompetitionAnalyzer(COMPETITION_NAME)
    
    # Collect notebooks sorted by score
    notebooks_df = analyzer.collect_notebooks(limit=NUM_NOTEBOOKS, sort_by=SORT_METHOD)
    
    # Check if we got any notebooks
    if len(notebooks_df) == 0:
        print("\nâ�Œ No notebooks found for this competition.")
        print("Possible reasons:")
        print("1. The competition name might be incorrect")
        print("2. There might be no public notebooks yet")
        print("3. No notebooks have public scores yet")
        print("4. Authentication issues with the Kaggle API")
        
        return None, None
    
    print(f"\nCollected {len(notebooks_df)} notebooks")
    
    # Display statistics
    print("\nNotebook Statistics:")
    print(f"Total notebooks collected: {len(notebooks_df)}")
    print(f"Average votes: {notebooks_df['total_votes'].mean():.1f}")
    print(f"Languages: {notebooks_df['language'].value_counts().to_dict()}")
    print("\nNote: Individual notebook scores are not available through the Kaggle API.")
    print("However, notebooks are sorted by score on Kaggle's servers when using 'scoreAscending'.")
    
    # Summarize each notebook
    print("\n" + "="*60)
    print("Generating individual notebook summaries...")
    print("="*60)
    
    summaries = []
    for idx, row in notebooks_df.iterrows():
        print(f"\nSummarizing: {row['title']}")
        summary = analyzer.summarize_notebook(row)
        summaries.append(summary)
        print(f"Summary preview: {summary[:150]}...")
        time.sleep(1)  # Rate limiting for OpenAI API
    
    notebooks_df['summary'] = summaries
    
    # Save individual summaries with scores
    summaries_file = f"{COMPETITION_NAME}_top_scored_notebook_summaries.csv"
    notebooks_df[['title', 'author', 'public_score', 'private_score', 'total_votes', 'summary']].to_csv(
        summaries_file, index=False
    )
    print(f"\nSaved individual summaries to: {summaries_file}")
    
    # Generate competition report
    print("\n" + "="*60)
    print("Generating competition analysis report...")
    print("="*60)
    
    competition_report = analyzer.generate_competition_report(
        notebooks_df[['title', 'total_votes', 'summary']]
    )
    
    # Save competition report
    report_file = f"/kaggle/working/{COMPETITION_NAME}_top_notebooks_competition_report.md"
    
    # Build the full markdown report
    full_report = f"# {COMPETITION_NAME.replace('-', ' ').title()} Competition Analysis Report\n\n"
    full_report += f"**Generated on:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    full_report += f"**Analyzed:** Top {len(notebooks_df)} notebooks (sorted by: {SORT_METHOD})\n\n"
    full_report += "**Note:** The Kaggle API sorts notebooks by score but doesn't expose individual scores.\n\n"
    full_report += "---\n\n"
    full_report += competition_report
    
    # Save to file
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    
    print(f"\nSaved competition report to: {report_file}")
    
    # Display the report using Markdown for nice formatting
    print("\n" + "="*60)
    print("COMPETITION ANALYSIS REPORT")
    print("="*60)
    print("\nDisplaying formatted report:\n")
    
    # Display using IPython Markdown for nice rendering
    display(Markdown(full_report))
    
    return notebooks_df, competition_report

# Run the analysis
if __name__ == "__main__":
    try:
        notebooks_df, report = main()
        print("\nâœ… Analysis completed successfully!")
        
        # Display final summary of analyzed notebooks
        if notebooks_df is not None and len(notebooks_df) > 0:
            summary_md = "## ğŸ“Š Analyzed Notebooks Summary\n\n"
            summary_md += "| # | Title | Author | Votes |\n"
            summary_md += "|---|-------|--------|-------|\n"
            
            for idx, row in notebooks_df.iterrows():
                title_short = row['title'][:50] + "..." if len(row['title']) > 50 else row['title']
                summary_md += f"| {idx+1} | {title_short} | {row['author']} | {row['total_votes']} |\n"
            
            display(Markdown(summary_md))
            
    except Exception as e:
        print(f"\nâ�Œ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

