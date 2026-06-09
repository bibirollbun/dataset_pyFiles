# =========================================
# Enhanced Smart Document Intelligence Agent
# Enterprise Track â€“ Capstone-ready
# Simulated Gemini LLM
# =========================================

from datetime import datetime
import json

# -----------------------------------------
# Helper: Logging
def log(msg):
    print(f"[{datetime.now()}] {msg}")

# -----------------------------------------
# Step 1: Reader Agent
def reader_agent(document_text):
    log("Reader Agent: Reading document...")
    pages = document_text.count("\n\n") + 1
    word_count = len(document_text.split())
    return {"text": document_text, "pages": pages, "word_count": word_count}

# -----------------------------------------
# Step 2: Extractor Agent (Simulated)
def extractor_agent(text):
    log("Extractor Agent: Simulating extraction...")
    # Simulated extraction with more fields
    return {
        "vendor": "Alpha Technologies",
        "invoice_no": "INV-9988",
        "date": "2025-01-12",
        "amount": 75000,
        "currency": "USD",
        "tax": 5000,
        "items": [
            {"item": "Laptop", "qty": 2, "price": 25000},
            {"item": "Monitor", "qty": 5, "price": 5000}
        ],
        "payment_terms": "Net 30"
    }

# -----------------------------------------
# Step 3: Verifier Agent
def verifier_agent(fields):
    log("Verifier Agent: Checking policy rules...")
    flags = []

    # Amount check
    if fields.get("amount") and fields["amount"] > 50000:
        flags.append("Amount exceeds company limit")

    # Vendor check
    if not fields.get("vendor"):
        flags.append("Vendor missing")

    # Tax consistency
    if fields.get("tax") != sum([i["qty"]*i["price"] for i in fields["items"]]) * 0.1:
        flags.append("Tax calculation mismatch")

    # Payment term check
    if fields.get("payment_terms") not in ["Net 15", "Net 30", "Net 45"]:
        flags.append("Unusual payment terms")

    return {"flags": flags, "num_flags": len(flags)}

# -----------------------------------------
# Step 4: Reporter Agent (Simulated)
def reporter_agent(fields, verification):
    log("Reporter Agent: Simulating report generation...")

    report = f"""
# ðŸ§¾ Smart Document Intelligence Report

**Document Summary:**  
- Vendor: {fields.get('vendor')}  
- Invoice No: {fields.get('invoice_no')}  
- Date: {fields.get('date')}  
- Total Amount: {fields.get('amount')} {fields.get('currency')}  
- Tax: {fields.get('tax')} {fields.get('currency')}  
- Payment Terms: {fields.get('payment_terms')}  
- Number of Items: {len(fields.get('items'))}  

**Itemized Details:**  
"""
    for item in fields["items"]:
        report += f"- {item['item']}: Quantity {item['qty']}, Price {item['price']} {fields.get('currency')}\n"

    report += "\n**Verification Flags:**\n"
    if verification["flags"]:
        for f in verification["flags"]:
            report += f"- {f}\n"
    else:
        report += "None\n"

    report += f"\n**Overall Status:** {'APPROVED' if verification['num_flags'] == 0 else 'REQUIRES APPROVAL'}\n"

    # Recommendations
    report += "\n**Recommendations:**\n"
    if verification["num_flags"] > 0:
        report += "- Review flagged issues before approval.\n"
    else:
        report += "- No action needed. Proceed with approval.\n"

    return report

# -----------------------------------------
# Step 5: Orchestrator Agent
def orchestrator(document_text):
    log("Orchestrator: Starting workflow...")
    read = reader_agent(document_text)
    fields = extractor_agent(read["text"])
    verification = verifier_agent(fields)
    report = reporter_agent(fields, verification)
    return report

# -----------------------------------------
# Step 6: Demo Input Document
document_text = """
Vendor: Alpha Technologies
Invoice Number: INV-9988
Date: 2025-01-12
Total: 75000 USD
Tax: 5000 USD
Payment Terms: Net 30
Items:
- Laptop x2 at 25000
- Monitor x5 at 5000
"""

# -----------------------------------------
# Step 7: Run Orchestrator
final_report = orchestrator(document_text)

# -----------------------------------------
# Step 8: Show Final Report
print(final_report)

