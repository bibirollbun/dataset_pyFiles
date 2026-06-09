# Pseudocode Findings Demonstration Notebook
# This notebook demonstrates AI model behavioral issues discovered in pseudocode findings

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

# File paths
file_paths = [
    "/kaggle/input/pseudocode-finding-1-sophistication-deception/pseudocode_finding_1_sophistication_deception.json",
    "/kaggle/input/pseudocode-finding-2-fake-fabrication/pseudocode_finding_2_fake_fabrication.json", 
    "/kaggle/input/pseudocode-finding-3-context-dependent-variation/pseudocode_finding_3_context_dependent_variation.json",
    "/kaggle/input/pseudocode-finding-4-evaluation-sabotage/pseudocode_finding_4_evaluation_sabotage.json",
    "/kaggle/input/pseudocode-finding-5-reasoning-deception/pseudocode_finding_5_reasoning_deception.json"
]

def load_finding_data(file_path: str) -> Dict[str, Any]:
    """Load and parse a pseudocode finding JSON file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def display_finding_summary(data: Dict[str, Any], finding_number: int):
    """Display a formatted summary of a finding"""
    if not data:
        return
    
    print(f"=" * 80)
    print(f"FINDING {finding_number}: {data['issue_title']}")
    print(f"=" * 80)
    
    # Model information
    model_info = data.get('model', {})
    print(f"Model: {model_info.get('name', 'Unknown')} v{model_info.get('version', 'Unknown')}")
    print(f"Provider: {model_info.get('provider', 'Unknown')}")
    print(f"Parameters: {model_info.get('parameters', {})}")
    print()
    
    # Issue summary
    summary = data.get('issue_summary', {})
    print(f"Title: {summary.get('title', 'Unknown')}")
    print(f"Topic Area: {summary.get('topic_area', 'Unknown')}")
    print(f"Severity: {summary.get('self_assessed_severity', 'Unknown')}/10")
    print(f"Breadth: {summary.get('self_assessed_breadth', 'Unknown')}/10")
    print()
    
    print("Overview:")
    print(summary.get('overview', 'No overview available'))
    print()
    
    print("Desired Behavior:")
    print(summary.get('desired_behavior', 'No desired behavior specified'))
    print()
    
    print("Observed Behavior:")
    print(summary.get('observed_behavior', 'No observed behavior specified'))
    print()
    
    # Severity justification
    print("Severity Justification:")
    print(summary.get('self_assessed_severity_justification', 'No justification provided'))
    print()

def extract_conversation_example(data: Dict[str, Any]) -> str:
    """Extract and format the conversation example from harmony response walkthroughs"""
    walkthroughs = data.get('harmony_response_walkthroughs', [])
    if not walkthroughs:
        return "No conversation example available"
    
    # Parse the conversation from the walkthrough
    conversation = walkthroughs[0]
    
    # Split by message markers
    parts = conversation.split('<|start|>')
    formatted_conversation = []
    
    for part in parts[1:]:  # Skip first empty part
        if '<|message|>' in part and '<|end|>' in part:
            role_and_message = part.split('<|message|>')[0]
            message_content = part.split('<|message|>')[1].split('<|end|>')[0]
            
            role = role_and_message.strip()
            formatted_conversation.append(f"{role.upper()}:\n{message_content.strip()}\n")
    
    return '\n'.join(formatted_conversation)

def demonstrate_issue(data: Dict[str, Any], finding_number: int):
    """Demonstrate the specific issue with examples"""
    print(f"\n{'='*60}")
    print(f"DEMONSTRATION OF ISSUE {finding_number}")
    print(f"{'='*60}")
    
    # Show the conversation example
    conversation = extract_conversation_example(data)
    print("CONVERSATION EXAMPLE:")
    print("-" * 40)
    print(conversation)
    
    # Show steps to reproduce
    steps = data.get('steps_to_reproduce', [])
    if steps:
        print("\nSTEPS TO REPRODUCE:")
        print("-" * 20)
        for i, step in enumerate(steps, 1):
            print(f"{i}. {step}")
    
    # Show additional notes
    notes = data.get('notes', '')
    if notes:
        print(f"\nADDITIONAL NOTES:")
        print("-" * 15)
        print(notes)

def create_severity_analysis(all_data: List[Dict[str, Any]]):
    """Create visualizations of severity and breadth across findings"""
    # Extract severity and breadth data
    findings_data = []
    for i, data in enumerate(all_data, 1):
        if data:
            summary = data.get('issue_summary', {})
            findings_data.append({
                'Finding': f"Finding {i}",
                'Issue': data.get('issue_title', 'Unknown'),
                'Severity': int(summary.get('self_assessed_severity', 0)),
                'Breadth': int(summary.get('self_assessed_breadth', 0)),
                'Topic_Area': summary.get('topic_area', 'Unknown')
            })
    
    df = pd.DataFrame(findings_data)
    
    # Create visualizations
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Severity scores
    bars1 = ax1.bar(df['Finding'], df['Severity'], color='red', alpha=0.7)
    ax1.set_title('Self-Assessed Severity Scores by Finding')
    ax1.set_ylabel('Severity (1-10)')
    ax1.set_ylim(0, 10)
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    plt.setp(ax1.get_xticklabels(), rotation=45)
    
    # Breadth scores
    bars2 = ax2.bar(df['Finding'], df['Breadth'], color='orange', alpha=0.7)
    ax2.set_title('Self-Assessed Breadth Scores by Finding')
    ax2.set_ylabel('Breadth (1-10)')
    ax2.set_ylim(0, 10)
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{height}', ha='center', va='bottom')
    plt.setp(ax2.get_xticklabels(), rotation=45)
    
    # Severity vs Breadth scatter
    ax3.scatter(df['Severity'], df['Breadth'], s=100, alpha=0.7, c='purple')
    for i, txt in enumerate(df['Finding']):
        ax3.annotate(txt, (df['Severity'].iloc[i], df['Breadth'].iloc[i]), 
                    xytext=(5, 5), textcoords='offset points')
    ax3.set_xlabel('Severity')
    ax3.set_ylabel('Breadth')
    ax3.set_title('Severity vs Breadth Analysis')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 10)
    ax3.set_ylim(0, 10)
    
    # Topic area distribution
    topic_counts = df['Topic_Area'].value_counts()
    ax4.pie(topic_counts.values, labels=topic_counts.index, autopct='%1.1f%%', startangle=90)
    ax4.set_title('Distribution by Topic Area')
    
    plt.tight_layout()
    plt.show()
    
    return df

# Main execution
print("PSEUDOCODE FINDINGS ANALYSIS")
print("=" * 80)
print("Loading and analyzing AI model behavioral issue findings...")
print()

# Load all findings
all_findings = []
for i, file_path in enumerate(file_paths, 1):
    print(f"Loading Finding {i}...")
    data = load_finding_data(file_path)
    all_findings.append(data)
    
    if data:
        display_finding_summary(data, i)
        demonstrate_issue(data, i)
    else:
        print(f"Failed to load Finding {i}")
    
    print("\n" + "="*80 + "\n")

# Create analysis summary
print("SUMMARY ANALYSIS")
print("=" * 40)

valid_findings = [f for f in all_findings if f is not None]
print(f"Successfully loaded {len(valid_findings)} out of {len(file_paths)} findings")

if valid_findings:
    print("\nCreating severity and breadth analysis...")
    analysis_df = create_severity_analysis(valid_findings)
    
    print("\nFINDINGS SUMMARY TABLE:")
    print("-" * 60)
    display_cols = ['Finding', 'Severity', 'Breadth', 'Topic_Area']
    print(analysis_df[display_cols].to_string(index=False))
    
    print(f"\nAverage Severity: {analysis_df['Severity'].mean():.1f}/10")
    print(f"Average Breadth: {analysis_df['Breadth'].mean():.1f}/10")
    print(f"Most Critical Finding: {analysis_df.loc[analysis_df['Severity'].idxmax(), 'Finding']}")
    print(f"Broadest Impact Finding: {analysis_df.loc[analysis_df['Breadth'].idxmax(), 'Finding']}")
    
    # Show unique topic areas
    unique_topics = analysis_df['Topic_Area'].unique()
    print(f"\nTopic Areas Covered: {', '.join(unique_topics)}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

