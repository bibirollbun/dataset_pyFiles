import json
import csv
import os
from typing import List, Dict, Tuple
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

print("✓ Imports loaded")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPUs: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


DOCUMENT_TYPE_INFO = {
    "DEF14A": """Proxy Statement (DEF 14A) - Annual shareholder meeting document:
- Executive compensation (salaries, bonuses, stock options, benefits)
- Compensation Discussion & Analysis (CD&A)
- Say-on-pay voting results and rationale
- Director compensation and board member information
- Share ownership guidelines for executives and directors
- Stock ownership by management and major shareholders
- Board committee structure and governance practices
- Shareholder proposals and management recommendations
- Related party transactions
- Equity compensation plan details
- Board diversity and independence standards
- Clawback policies and pay-for-performance alignment""",

    "10-K": """Annual Report (10-K) - Comprehensive annual overview filed once per year:
- Complete business description and operations overview
- Detailed risk factors (market, operational, regulatory, competitive risks)
- Full-year audited financial statements (income statement, balance sheet, cash flow)
- Management's Discussion & Analysis (MD&A) of annual performance
- Executive compensation details and equity plans
- Corporate governance policies and board structure
- Legal proceedings and regulatory compliance
- Sustainability and ESG disclosures
- Long-term strategy and capital allocation plans
- Property, plant, and equipment details
- Market for common equity and dividend policy
- Five-year selected financial data and trends""",

    "10-Q": """Quarterly Report (10-Q) - Interim financial update filed three times per year:
- Quarterly unaudited financial statements
- Quarter-over-quarter and year-over-year performance comparisons
- Recent business developments and operational changes
- Updated MD&A focusing on quarterly trends
- Current liquidity position and cash flow analysis
- Recent market conditions and their impact
- Updates to risk factors if material changes occurred
- Segment performance for the quarter
- Recent acquisitions or divestitures
- Changes in accounting policies or estimates
- Quantitative and qualitative disclosures about market risk""",

    "8-K": """Current Report (8-K) - Event-driven disclosure filed as needed:
- Material corporate events and breaking news
- Leadership changes (CEO, CFO, board appointments/departures)
- Significant acquisitions or asset dispositions
- Financial results announcements (earnings releases)
- Credit rating changes
- Bankruptcy or receivership events
- Material impairments or restructuring charges
- Amendments to articles of incorporation or bylaws
- Delisting notices or exchange changes
- Material definitive agreements
- Departure of directors or officers
- Entry into or termination of material agreements""",

    "Earnings": """Earnings Call Transcript - Real-time management commentary:
- Live discussion of quarterly or annual results
- Management's forward-looking statements and guidance
- Strategic initiatives and business outlook
- Executive sentiment and tone about business conditions
- Detailed Q&A with analysts on specific topics
- Margin trends and profitability drivers
- Product pipeline updates and innovation cycles
- Competitive positioning and market dynamics
- Capital allocation priorities (M&A, buybacks, dividends)
- Customer feedback and demand trends
- Supply chain and operational challenges
- Geographic or segment-specific performance insights
- Answers to investor concerns and market questions"""
}

DOCUMENT_TYPES = ["DEF14A", "10-K", "10-Q", "8-K", "Earnings"]

print("✓ Document type descriptions loaded")


# Configuration
BASE_MODEL = "Qwen/Qwen3-32B"
MODEL_PATH = "./document_finetune/pairwise/checkpoints/pairwise_ranker/final_model"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

print(f"Loading base model: {BASE_MODEL}")
print(f"LoRA adapter: {MODEL_PATH}")
print(f"Device: {DEVICE}")

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True
)

# Load LoRA adapter
model = PeftModel.from_pretrained(base_model, MODEL_PATH)
model.eval()

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    trust_remote_code=True
)

print("✓ Model loaded successfully")


def create_comparison_prompt(question: str, doc_a: str, doc_b: str) -> str:
    """Create pairwise comparison prompt matching training format"""
    prompt = f"""Which document type is more likely to contain the information to answer the query?

Query: {question}

Document A: {doc_a}
{DOCUMENT_TYPE_INFO[doc_a]}

Document B: {doc_b}
{DOCUMENT_TYPE_INFO[doc_b]}

Respond with only "Document A" or "Document B"."""
    return prompt


def compare_documents(question: str, doc_a: str, doc_b: str) -> str:
    """Compare two documents using the finetuned model.
    
    Returns:
        'A' if doc_a is more relevant, 'B' if doc_b is more relevant
    """
    prompt = create_comparison_prompt(question, doc_a, doc_b)
    
    # Format with chat template
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False
    )
    
    # Tokenize and generate
    inputs = tokenizer(formatted, return_tensors="pt").to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=10,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id
        )
    
    # Decode response
    response = tokenizer.decode(
        outputs[0][inputs['input_ids'].shape[1]:],
        skip_special_tokens=True
    )
    
    # Extract A or B
    response = response.strip().lower()
    if 'document a' in response or response.endswith('a'):
        return 'A'
    elif 'document b' in response or response.endswith('b'):
        return 'B'
    else:
        # Default to A if unclear
        return 'A'


def bidirectional_compare(question: str, doc_a: str, doc_b: str) -> str:
    """Compare documents in both directions for robustness.
    
    Returns:
        'A>B' if doc_a wins, 'B>A' if doc_b wins, 'tie' if inconsistent
    """
    # Compare A vs B
    result_ab = compare_documents(question, doc_a, doc_b)
    
    # Compare B vs A (reversed)
    result_ba = compare_documents(question, doc_b, doc_a)
    
    # Check consistency
    if result_ab == 'A' and result_ba == 'B':
        return 'A>B'  # Consistent: A wins
    elif result_ab == 'B' and result_ba == 'A':
        return 'B>A'  # Consistent: B wins
    else:
        return 'tie'  # Inconsistent

print("✓ Comparison functions defined")


def rank_documents_bubble_sort(
    question: str,
    doc_types: List[str] = DOCUMENT_TYPES,
    num_passes: int = 10
) -> List[int]:
    """Rank documents using bubble sort with pairwise comparisons.
    
    Args:
        question: The question to answer
        doc_types: List of document type names
        num_passes: Maximum number of bubble sort passes
    
    Returns:
        List of document indices in ranked order (best to worst)
    """
    # Initialize ranking as (doc_type, original_index) tuples
    current_ranking = [(doc_types[i], i) for i in range(len(doc_types))]
    n = len(current_ranking)
    
    print(f"\nRanking documents for question: {question[:80]}...")
    print(f"Initial order: {[doc for doc, _ in current_ranking]}")
    
    for pass_num in range(num_passes):
        swaps_made = 0
        
        # Bubble sort pass (from back to front)
        for i in range(n - 1, 0, -1):
            doc_a, idx_a = current_ranking[i - 1]
            doc_b, idx_b = current_ranking[i]
            
            # Compare documents bidirectionally
            result = bidirectional_compare(question, doc_a, doc_b)
            
            # Swap if B is better than A
            if result == 'B>A':
                current_ranking[i - 1], current_ranking[i] = current_ranking[i], current_ranking[i - 1]
                swaps_made += 1
        
        print(f"Pass {pass_num + 1}: {swaps_made} swaps | Current: {[doc for doc, _ in current_ranking]}")
        
        # Early stopping if converged
        if swaps_made == 0:
            print(f"Converged at pass {pass_num + 1}")
            break
    
    # Return only the indices in ranked order
    final_ranking = [idx for _, idx in current_ranking]
    print(f"Final ranking (indices): {final_ranking}")
    return final_ranking

print("✓ Ranking function defined")


# Test question
test_question = "What is the company's executive compensation structure?"

# Rank documents
ranking = rank_documents_bubble_sort(test_question, num_passes=5)

print(f"\n{'='*60}")
print("Final Ranking:")
for rank, idx in enumerate(ranking, 1):
    print(f"  {rank}. {DOCUMENT_TYPES[idx]} (index {idx})")
print(f"{'='*60}")


def extract_question_from_messages(messages: List[Dict]) -> str:
    """Extract question from message format"""
    try:
        content = messages[0]['content']
        lines = content.split('\n')
        for line in lines:
            if line.startswith('Question:'):
                return line.replace('Question:', '').strip()
        return content
    except:
        return ""


def load_evaluation_data(filepath: str) -> List[Dict]:
    """Load evaluation data from JSONL file"""
    data = []
    print(f"Loading data from: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    
    print(f"✓ Loaded {len(data)} items")
    return data


def save_submission_csv(submission_data: List[Dict], filename: str):
    """Save submission data to CSV"""
    with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['sample_id', 'target_index'])
        
        for entry in submission_data:
            writer.writerow([entry['sample_id'], entry['target_index']])
    
    print(f"✓ Submission saved to: {filename}")
    print(f"  Total entries: {len(submission_data)}")

print("✓ Data functions defined")


# Configuration
DATA_PATH = "./output/document_ranking_kaggle_eval.jsonl"
OUTPUT_PATH = "submission_document_ranking_finetuned.csv"
NUM_PASSES = 10  # Bubble sort passes

# Load evaluation data
data = load_evaluation_data(DATA_PATH)

print(f"\n{'='*60}")
print(f"Processing {len(data)} questions...")
print(f"Bubble sort passes: {NUM_PASSES}")
print(f"{'='*60}\n")

# Process all items
submission_data = []

for item in tqdm(data, desc="Ranking documents"):
    messages = item['messages']
    query_id = item.get('_id', item.get('uuid', ''))
    
    # Extract question
    question = extract_question_from_messages(messages)
    
    # Rank documents
    ranking = rank_documents_bubble_sort(
        question,
        DOCUMENT_TYPES,
        num_passes=NUM_PASSES
    )
    
    # Add top 5 to submission
    for doc_idx in ranking[:5]:
        submission_data.append({
            'sample_id': query_id,
            'target_index': doc_idx
        })

# Save submission
print(f"\n{'='*60}")
print("Saving submission...")
print(f"{'='*60}")
save_submission_csv(submission_data, OUTPUT_PATH)

print(f"\n{'='*60}")
print("✅ COMPLETE")
print(f"{'='*60}")
print(f"Processed: {len(data)} questions")
print(f"Generated: {len(submission_data)} submission entries")
print(f"Output: {OUTPUT_PATH}")
print(f"{'='*60}")

