import os
from kaggle_secrets import UserSecretsClient

try:
    GOOGLE_API_KEY = UserSecretsClient().get_secret("GOOGLE_API_KEY")
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
    print("âœ… Gemini API key setup complete.")
except Exception as e:
    print(
        f"ðŸ”‘ Authentication Error: Please make sure you have added 'GOOGLE_API_KEY' to your Kaggle secrets. Details: {e}"
    )


from google.adk.agents import Agent, SequentialAgent, ParallelAgent, LoopAgent
from google.adk.agents import LlmAgent
from google.adk.models.google_llm import Gemini
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, FunctionTool
from google.genai import types  # Used for defining Schemas/parameters
from typing import List, Dict, Any
import sqlite3
import random
import time
import uuid
import json
import datetime

print("âœ… ADK components imported successfully.")


import warnings
warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')

import logging
logging.getLogger('google_genai.types').setLevel(logging.ERROR)


retry_config = types.HttpRetryOptions(
    attempts=5,  # Maximum retry attempts
    exp_base=7,  # Delay multiplier
    initial_delay=1,
    http_status_codes=[429, 500, 503, 504],  # Retry on these HTTP errors
)

print("retry_config options set up")


# =============================================================================
# SECTION 1: DATABASE SETUP
# =============================================================================

# Use two separate DBs: one for CRM, one for content/metadata

CRM_DB = "crm.db"

# -----------------------------------------------------------------------------
# 1.1 CRM Database Setup (customers + tickets)
# -----------------------------------------------------------------------------

def initialize_mock_crm(db_name: str = CRM_DB):
    """Initializes the SQLite CRM database with mock customer and ticket data."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("DROP TABLE IF EXISTS tickets")
    cursor.execute("DROP TABLE IF EXISTS customers")

    # Customers Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS customers (
            user_id TEXT PRIMARY KEY,
            email TEXT,
            status TEXT -- 'EXISTING' or 'UNKNOWN' (if enrolled)
        )
    """)

    # Tickets Table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS tickets (
            ticket_id TEXT PRIMARY KEY,
            user_id TEXT,
            issue_summary TEXT,
            assigned_agent_name TEXT,
            assigned_agent_email TEXT,
            status TEXT, -- 'PENDING', 'CLOSED', 'IN_PROGRESS'
            FOREIGN KEY (user_id) REFERENCES customers(user_id)
        )
    """)

    # Mock Customer Data (All known users are 'EXISTING')
    customer_data = [
        ("user_001", "alice.smith@corp.com", "EXISTING"),
        ("user_002", "bob.jones@corp.com", "EXISTING"),  # User with NO pending ticket
        ("user_003", "charlie.brown@corp.com", "EXISTING"),  # User with PENDING ticket
    ]

    cursor.executemany("INSERT OR IGNORE INTO customers VALUES (?,?,?)", customer_data)

    # Mock Ticket Data
    ticket_data = [
        ("TICKET-2024-001", "user_003", "Payment processing failed.", "Jane Doe", "jane.doe@corp.com", "PENDING"),
        ("TICKET-2024-002", "user_001", "Cannot log into my account.", "Mike Ross", "mike.ross@corp.com", "CLOSED"),
    ]

    cursor.executemany("INSERT OR IGNORE INTO tickets VALUES (?,?,?,?,?,?)", ticket_data)

    conn.commit()
    conn.close()

    print(f"âœ… Mock CRM database '{db_name}' re-initialized with simplified two-status model.")

# Initialize CRM DB once
initialize_mock_crm()


# Records with json_blob set to full article text (extracted from WebToffee docs)
records = [
    {
      "parent_id": "doc_004",
      "title": "Why Does the Cookie Scan for My Website Fail?",
      "product": "WebToffee GDPR Cookie Consent Plugin",
      "category": "troubleshooting",
      "version": "v1.0",
      "lifecycle": "active",
      "source": "https://www.webtoffee.com/docs/gdpr-plugin/cookie-scan-for-my-website-fail/",
      "last_updated": "2025-08-28T00:00:00Z",
      "owner": "support-team@webtoffee.com",
      "chunk_ids": "",
      "content_summary": "Describes reasons cookie scan fails and offers solutions.",
      "json_blob": (
        "Why Does the Cookie Scan for My Website Fail?\n\n"
        "Last updated on August 28, 2025\n\n"
        "Understanding and resolving issues with the scanning and categorizing of cookies on your website is necessary for successful cookie management. The cookie scanning process performed by WebToffeeâ€™s GDPR Cookie Consent plugin may occasionally encounter difficulties that impede its successful completion. If you come across a situation where a cookie scan on your website fails, itâ€™s crucial to identify the underlying issues and implement suitable solutions to ensure accurate cookie management.\n\n"
        "Below, we list some probable reasons for scan failures and offer corresponding solutions. Once youâ€™ve addressed the issue, initiate a fresh scan to validate the resolution.\n\n"
        "Firewall on the website blocking scanner access\n\n"
        "If you have a firewall on your website, it could inadvertently block your IP address from performing scans on your website. Consequently, the cookie scan carried out by the plugin would be unable to access the website content and collect the required information for the scan.\n\n"
        "Solution:\n"
        "- Temporarily Disable the Firewall: You can temporarily disable the firewall on your website during the cookie scan, which allows the plugin to access your website without any restriction. However, we do not recommend this option due to the potential risks involved. This approach might expose your site to security vulnerabilities and possible attacks.\n\n"
        "Website blocking headless browsers used by the scanner\n\n"
        "The scanning mechanism uses headless browsers to access and analyze your websiteâ€™s content. These browsers navigate pages, interact with their elements, and collect information about the cookies used. However, some websites adopt methods that block the functioning of these browsers, making it difficult for the plugin to perform its scan effectively.\n\n"
        "Solution: Enabling Headless Browsers\n"
        "Configure your websiteâ€™s security settings to enable headless browsers specifically for scanning. By giving access to headless browsers used by the scanning process, you ensure that the analysis of your websiteâ€™s cookies is accurate and complete.\n\n"
        "JavaScript errors on the website interfere with the scannerâ€™s functionality\n\n"
        "JavaScript errors on your website can influence the scannerâ€™s functioning. When the scanner encounters JavaScript errors while navigating your site, it might be unable to properly analyze its content, including identifying and categorizing cookies.\n\n"
        "Solution: Identify and Fix JavaScript Errors\n"
        "Identify and resolve the JavaScript errors that are affecting your website. This ensures that the scanner can function properly and accurately assess the cookie usage on your website.\n\n"
        "Low website configuration\n\n"
        "Limited resources or capabilities of a website can impact the scannerâ€™s effectiveness. In such cases, the scanning process might fail to proceed smoothly.\n\n"
        "Solution: Adjust the scanning speed\n\n"
        "Contact support to adjust your websiteâ€™s scanning speed. Reducing the scanning speed makes the process more manageable for websites with lower configurations, preventing disruptions that might occur during the scanning process.\n\n"
        "The website is not accessible from outside\n\n"
        "When the website is configured to restrict external access, the scanner cannot connect to and navigate the websiteâ€™s content. This hinders the scannerâ€™s ability to effectively analyze the cookies and other aspects of the websiteâ€™s data.\n\n"
        "Solution:\n"
        "- Configure your website settings: Change/adjust your websiteâ€™s settings to allow access from outside sources, which involves reconfiguring security settings or access controls that might be currently preventing external entities from interacting with your website.\n\n"
        "User agent\n\n"
        "To ensure a smooth scanning process, please confirm that there is an uninterrupted connection to our cookie scanning server, which is located in Ireland. If you have any firewalls or security plugins on your website or at your hosting providerâ€™s end, it might be helpful to temporarily disable them during the scanning. If it is not possible to disable, then try whitelisting the user agent. We use the service of Cookieyes for scanning. CookieYesbot is the official scanning bot used by CookieYes to verify cookies and enable auto-script blocking across websites using our plugin.\n\n"
        "CookieYesbot is classified as a Security and Compliance scanner that assists websites in maintaining up-to-date and legally compliant consent mechanisms. All requests made by CookieYesbot use the following user agent string: [CookieYesbot user-agent â€” please refer to the original article for the exact string].\n\n"
        "If youâ€™re unsure about the cause of the scan failure, consider initiating a re-scan of your website. If the problem continues to persist even after re-scanning, contact support."
      )
    },

    {
      "parent_id": "doc_005",
      "title": "Geo-target GDPR Cookie Banner",
      "product": "WebToffee GDPR Cookie Consent Plugin",
      "category": "feature-guide",
      "version": "v1.0",
      "lifecycle": "active",
      "source": "https://www.webtoffee.com/docs/gdpr-plugin/geo-target-cookie-consent-bar-non-eu-countries/",
      "last_updated": "2025-05-09T00:00:00Z",
      "owner": "technical-writer@webtoffee.com",
      "chunk_ids": "",
      "content_summary": "Explains geo-targeting support for cookie banners.",
      "json_blob": (
        "Geo-target GDPR Cookie Banner\n\n"
        "This article explains how to configure geo-IP based display of the cookie consent banner so that it appears only for visitors from selected regions (for example EU & UK) or for worldwide audiences. The plugin supports showing the cookie banner conditionally by detecting the visitor's location via GeoIP and then honoring regional privacy rules like GDPR for EU and CCPA for certain US states.\n\n"
        "Key points and configuration steps:\n"
        "1. Enable Geo-targeting in plugin settings.\n"
        "2. Choose target regions (EU, UK, specific countries, or worldwide).\n"
        "3. Configure fallbacks for unknown locations.\n"
        "4. Test the banner behavior with proxy/IP tools to verify region-based display.\n\n"
        "Notes:\n"
        "- When using Geo-targeting, confirm that the GeoIP database used by the plugin is up-to-date.\n"
        "- Remember to test cookie consent flows thoroughly across regional scenarios to avoid accidentally blocking content for legitimate users."
      )
    },

    {
      "parent_id": "doc_006",
      "title": "Script Blocking in New GDPR Cookie Consent Plugin",
      "product": "WebToffee GDPR Cookie Consent Plugin",
      "category": "feature-guide",
      "version": "v1.0",
      "lifecycle": "active",
      "source": "https://www.webtoffee.com/docs/gdpr-plugin/script-blocking-in-new-gdpr-cookie-consent-plugin/",
      "last_updated": "2025-11-13T00:00:00Z",
      "owner": "content-team@webtoffee.com",
      "chunk_ids": "",
      "content_summary": "Explains automatic script blocking and URL pattern matching.",
      "json_blob": (
        "Script Blocking in New GDPR Cookie Consent Plugin\n\n"
        "Overview\n"
        "The new GDPR Cookie Consent plugin automatically blocks scripts that match known cookie-producing script URL patterns. The plugin categorizes cookies and blocks scripts until the user gives consent. Unrecognized cookies are placed into an \"Other\" category and require manual categorization by the site admin.\n\n"
        "How automatic blocking works\n"
        "- The plugin maintains a list of known script URL patterns and blocks matching scripts automatically.\n"
        "- For scripts that don't match any known pattern, the resulting cookies appear in 'Other' and administrators can add patterns or change category mapping.\n\n"
        "Admin guidance\n"
        "- To block a custom script, add its URL pattern into the plugin's script-blocking list.\n"
        "- Review the audit of blocked scripts to verify no essential functionality is impacted.\n\n"
        "Edge cases\n"
        "- Dynamically injected scripts may require additional selectors or pattern variations to be matched correctly.\n\n"
        "Conclusion\n"
        "Use the script blocking feature to improve compliance, but carefully test site functionality after adding patterns."
      )
    },

    {
      "parent_id": "doc_007",
      "title": "GDPR Shortcodes (Legacy)",
      "product": "WebToffee GDPR Cookie Consent Plugin (Legacy)",
      "category": "feature-guide",
      "version": "legacy",
      "lifecycle": "deprecated",
      "source": "https://www.webtoffee.com/docs/gdpr-legacy/gdpr-shortcodes/",
      "last_updated": "2025-01-15T00:00:00Z",
      "owner": "legacy-support@webtoffee.com",
      "chunk_ids": "",
      "content_summary": "Lists legacy GDPR shortcodes and usage guidance.",
      "json_blob": (
        "GDPR Shortcodes (Legacy)\n\n"
        "This legacy documentation lists the shortcodes available in the older/legacy version of the WebToffee GDPR Cookie Consent plugin. Shortcodes allowed admins to embed cookie preference buttons, cookie audit tables, and category lists within posts, pages and banner areas.\n\n"
        "Common shortcodes included:\n"
        "- [wcc_preference_button] : Renders the preferences button.\n"
        "- [wcc_cookies_audit_table] : Displays discovered cookies as a table for auditing.\n"
        "- [wcc_category_list] : Shows cookie categories.\n\n"
        "Note: Because this is legacy documentation, prefer the newer shortcodes documented in the active plugin pages for up-to-date behavior."
      )
    },

    {
      "parent_id": "doc_008",
      "title": "Shortcodes Used in GDPR Cookie Consent Plugin",
      "product": "WebToffee GDPR Cookie Consent Plugin",
      "category": "feature-guide",
      "version": "vcurrent",
      "lifecycle": "active",
      "source": "https://www.webtoffee.com/docs/gdpr-plugin/shortcodes-used-in-gdpr-cookie-consent-plugin/",
      "last_updated": "2025-10-30T00:00:00Z",
      "owner": "content-team@webtoffee.com",
      "chunk_ids": "",
      "content_summary": "Explains shortcodes such as [wcc_preference_button] and others.",
      "json_blob": (
        "Shortcodes Used in GDPR Cookie Consent Plugin\n\n"
        "The GDPR Cookie Consent plugin supports a set of shortcodes that let you render buttons, lists and audit tables for cookie management in posts, pages or banner areas.\n\n"
        "Examples\n"
        "- [wcc_preference_button] â€” renders the preference button that opens the cookie preference modal.\n"
        "- [wcc_cookies_audit_table] â€” displays detected cookies in a table for auditing and review.\n        Usage: place the shortcode in a page to provide a cookie audit to site admins.\n"
        "- [wcc_category_list] â€” shows the list of cookie categories for the site.\n"
        "- [wcc_preference_category] â€” embed category-level preference controls.\n\n"
        "Usage notes\n"
        "- Shortcodes are typically added to pages or widgets via the WordPress editor. Some shortcodes can also be embedded in the cookie banner or templates where HTML is allowed.\n"
        "- For accessibility and styling, wrap shortcodes in appropriate container elements and apply CSS classes as needed.\n"
        "For detailed examples, consult the plugin settings and the WebToffee documentation."
      )
    }
]

print("Content database records defined")



# -----------------------------------------------------------------------------
# 1.2 Metadata DB for content (RAG ingestion)
# -----------------------------------------------------------------------------

METADATA_DB = "metadata.db"

# This database is to store content pulled from various repositories (metadata + chunks)
if os.path.exists(METADATA_DB):
    os.remove(METADATA_DB)

# -----------------------------------------------------------------------------
# 1.3 Metadata + Chunks Tables
# -----------------------------------------------------------------------------

TABLE_NAME = "metadata"
CHUNKS_TABLE = "chunks"

def initialize_db():
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
            parent_id TEXT PRIMARY KEY,
            title TEXT,
            product TEXT,
            category TEXT,
            version TEXT,
            lifecycle TEXT,
            source TEXT,
            last_updated TEXT,
            owner TEXT,
            chunk_ids TEXT,
            content_summary TEXT,
            json_blob TEXT          -- FULL article content
        );
    """)
    
    conn.commit()
    conn.close()
    print("Metadata DB created with json_blob column")

def initialize_chunks_table(db_name: str = METADATA_DB):
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS {CHUNKS_TABLE} (
            chunk_id TEXT PRIMARY KEY,
            parent_id TEXT,
            chunk_order INTEGER,
            chunk_text TEXT,
            created_at TEXT
        );
    """)
    conn.commit()
    conn.close()
    print("Chunks table ready.")

# Run the metadata DB setup
initialize_db()
initialize_chunks_table()

# -----------------------------------------------------------------------------
# 1.4 Insert metadata records
# -----------------------------------------------------------------------------

def insert_records_with_blob(records):
    conn = sqlite3.connect(METADATA_DB)
    cursor = conn.cursor()

    for rec in records:
        cursor.execute(f"""
            INSERT OR REPLACE INTO {TABLE_NAME}
            (parent_id, title, product, category, version, lifecycle,
             source, last_updated, owner, chunk_ids, content_summary, json_blob)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rec["parent_id"],
            rec["title"],
            rec["product"],
            rec["category"],
            rec["version"],
            rec["lifecycle"],
            rec["source"],
            rec["last_updated"],
            rec["owner"],
            rec["chunk_ids"],
            rec["content_summary"],
            rec["json_blob"]
        ))

    conn.commit()
    conn.close()
    print(f"Inserted {len(records)} records (with full article content) into {METADATA_DB}/{TABLE_NAME}")

insert_records_with_blob(records)


# =============================================================================
# SECTION 2: CHUNKING & INGESTION
# =============================================================================

def chunk_text_sliding_window(text: str, window_chars: int = 1200, stride_chars: int = 900) -> List[str]:
    """
    Produce overlapping chunks from text.
    - window_chars: size of each chunk (characters)
    - stride_chars: how far to move each window (stride). overlap = window - stride
    """
    if not text:
        return []
    text = text.strip()
    n = len(text)
    chunks = []
    i = 0
    while i < n:
        chunk = text[i:i+window_chars].strip()
        if chunk:
            chunks.append(chunk)
        if i + window_chars >= n:
            break
        i += stride_chars
    return chunks

def ingest_parent_and_chunks_force(
    parent_id: str,
    json_blob_text: str,
    db_name: str = METADATA_DB,
    window_chars: int = 1200,
    stride_chars: int = 900,
) -> List[str]:
    """
    Chunk Store : Chunk json_blob_text using sliding window and insert into chunks table.
    Always (re)creates chunk rows and updates metadata.chunk_ids.
    Returns created chunk_id list.
    """
    chunks = chunk_text_sliding_window(json_blob_text, window_chars=window_chars, stride_chars=stride_chars)
    created_chunk_ids = []
    now = datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')

    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    try:
        conn.execute("BEGIN")

        # Insert chunks; use INSERT OR REPLACE to avoid PRIMARY KEY conflicts if re-run
        for idx, chunk_text in enumerate(chunks, start=1):
            chunk_id = f"{parent_id}__chunk__{idx}"
            created_chunk_ids.append(chunk_id)
            cur.execute(f"""
                INSERT OR REPLACE INTO {CHUNKS_TABLE}
                (chunk_id, parent_id, chunk_order, chunk_text, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (chunk_id, parent_id, idx, chunk_text, now))

        # Update metadata.chunk_ids (store as JSON array string)
        chunk_ids_json = json.dumps(created_chunk_ids, ensure_ascii=False)
        cur.execute(f"""
            UPDATE {TABLE_NAME}
            SET chunk_ids = ?
            WHERE parent_id = ?
        """, (chunk_ids_json, parent_id))

        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

    return created_chunk_ids

def run_chunk(
    db_name: str = METADATA_DB,
    window_chars: int = 1200,
    stride_chars: int = 900
):
    """
    Process all metadata rows with json_blob and create chunks for each parent doc.
    """
    conn = sqlite3.connect(db_name)
    cur = conn.cursor()
    cur.execute(f"SELECT parent_id, json_blob FROM {TABLE_NAME}")
    rows = cur.fetchall()
    conn.close()

    processed = 0
    for parent_id, json_blob in rows:
        if not json_blob or not json_blob.strip():
            print(f"Skipping {parent_id}: empty json_blob")
            continue
        created = ingest_parent_and_chunks_force(
            parent_id, json_blob,
            db_name=db_name,
            window_chars=window_chars,
            stride_chars=stride_chars
        )
        print(f"{parent_id}: created {len(created)} chunks.")
        processed += 1

    print(f"Processing complete. {processed} parent documents chunked.")

print("run_chunk set up")


# # -----------------------------------------------------------------------------
# # RAG Knowledge Base (simple in-memory)
# # -----------------------------------------------------------------------------

# PRODUCT_KNOWLEDGE_BASE = [
#     {
#         "id": "DOC-001",
#         "keywords": ["premium", "pricing", "cost", "features", "dashboard"],
#         "content": "The Premium Tier subscription costs $49.99/month and includes unlimited data storage, real-time analytics, and access to the new, customizable reporting dashboard. All premium users receive priority customer support."
#     },
#     {
#         "id": "DOC-002",
#         "keywords": ["basic", "standard", "free", "data limit"],
#         "content": "The Basic Tier is free but is limited to 1GB of data storage and only includes weekly email reports. It does not include the customizable reporting dashboard."
#     },
#     {
#         "id": "DOC-003",
#         "keywords": ["reporting", "dashboard", "export", "analytics", "data"],
#         "content": "The Reporting Dashboard is a key feature of the Premium Tier. It allows users to filter, visualize, and export data in CSV and PDF formats. It updates every 60 seconds (real-time). Access is restricted to accounts enrolled before January 2025 or those on the Premium plan."
#     }
# ]

# def product_knowledge_search(query: str) -> str:
#     """
#     Simulates a vector database search on the product knowledge base 
#     and returns the top 3 relevant documents.
#     """
#     query_tokens = query.lower().split()
#     scores = {}

#     for doc in PRODUCT_KNOWLEDGE_BASE:
#         # Score based on how many query tokens match the document's keywords
#         score = sum(1 for token in query_tokens if token in doc["keywords"])
#         scores[doc["id"]] = score

#     # Sort documents by score in descending order
#     sorted_docs = sorted(scores.items(), key=lambda item: item[1], reverse=True)

#     # Filter out documents with a score of 0 and take the top 3
#     top_doc_ids = [doc_id for doc_id, score in sorted_docs if score > 0][:3]

#     if not top_doc_ids:
#         return "NO_RELEVANT_DOCUMENTS_FOUND"

#     results = []
#     for doc in PRODUCT_KNOWLEDGE_BASE:
#         if doc["id"] in top_doc_ids:
#             results.append(f"Document ID: {doc['id']}\nContent: {doc['content']}")

#     return "\n---\n".join(results)



# =============================================================================
# SECTION 4: CRM & RAG TOOLS
# =============================================================================

# 1. Check Customer Status Tool (Used by Router)
def check_customer_status(user_id: str) -> str:
    """
    Looks up the user's profile in the CRM to determine their status 
    (EXISTING or UNKNOWN).
    """
    conn = sqlite3.connect(CRM_DB)
    cursor = conn.cursor()
    cursor.execute("SELECT status FROM customers WHERE user_id = ?", (user_id,))
    result = cursor.fetchone()
    conn.close()

    if result:
        # Known user, status is always EXISTING after enrollment
        return "EXISTING"
    else:
        # Represents a new user not found in the database
        return "UNKNOWN"

# 2. CRM Enrollment Tool (Used by Marketing Agent to enroll new customer)
def enroll_new_customer(user_id: str, full_name: str, email: str) -> str:
    """
    Adds a new customer profile to the CRM database with initial status 'EXISTING'.
    Returns success or failure message.
    """
    try:
        conn = sqlite3.connect(CRM_DB)
        cursor = conn.cursor()
        # New users are immediately given 'EXISTING' status upon enrollment
        cursor.execute(
            "INSERT INTO customers (user_id, email, status) VALUES (?, ?, ?)",
            (user_id, email, "EXISTING")
        )

        conn.commit()
        conn.close()

        return f"Successfully enrolled new user {user_id} ({full_name}) with email {email} into the CRM as EXISTING customer."
    except Exception as e:
        return f"Failed to enroll user {user_id}. Error: {str(e)}"

# 3. Check Ticket Status Tool
def check_ticket_status(user_id: str) -> str:
    """
    Looks up the PENDING ticket status for a given user.
    Returns the ticket ID, assigned agent, and status, or 'NO_PENDING_TICKET'.
    """
    conn = sqlite3.connect(CRM_DB)
    cursor = conn.cursor()
    cursor.execute(
        "SELECT ticket_id, assigned_agent_name, status FROM tickets WHERE user_id = ? AND status = 'PENDING' ORDER BY ticket_id DESC LIMIT 1",
        (user_id,)
    )

    result = cursor.fetchone()
    conn.close()

    if result:
        ticket_id, agent_name, status = result
        return f"PENDING_TICKET|ID:{ticket_id}|AGENT:{agent_name}|STATUS:{status}"
    else:
        return "NO_PENDING_TICKET"

# 4. Create New Ticket Tool
def create_new_ticket(user_id: str, issue_summary: str) -> str:
    """
    Creates a new PENDING ticket in the database and assigns it to a dummy agent.
    """
    try:
        conn = sqlite3.connect(CRM_DB)
        cursor = conn.cursor()

        # Simple auto-incrementing ticket ID simulation
        cursor.execute("SELECT COUNT(*) FROM tickets")
        count = cursor.fetchone()[0]
        new_ticket_id = f"TICKET-2024-{count + 1:03d}"

        assigned_agent = random.choice(["Tim Cook", "Satya Nadella", "Jensen Huang"])
        assigned_email = f"{assigned_agent.replace(' ', '.').lower()}@corp.com"

        cursor.execute(
            "INSERT INTO tickets (ticket_id, user_id, issue_summary, assigned_agent_name, assigned_agent_email, status) VALUES (?, ?, ?, ?, ?, ?)",
            (new_ticket_id, user_id, issue_summary, assigned_agent, assigned_email, "PENDING")
        )

        conn.commit()
        conn.close()
        return f"NEW_TICKET_CREATED|ID:{new_ticket_id}|AGENT:{assigned_agent}|SUMMARY:{issue_summary}"
    except Exception as e:
        return f"TICKET_CREATION_FAILED|ERROR:{str(e)}"

print(f"Done with helper functions")


# =============================================================================
# SECTION 3: INGESTION AGENT
# =============================================================================

# ingestion_agent = LlmAgent(
#     name="ingestionagent",
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     instruction="""
#         Consider yourself as an ingestion agent.
#         1. If the content of the article is empty, then do not use the chunk tool.
#         2. If the article has content, then use the chunk tool.
#     """,
#     tools=[run_chunk]  # pass the function, not the call
# )

ingestion_agent = LlmAgent(
    name="IngestionAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
        You are the Knowledge Ingestion and Retrieval Agent. Your primary function is to retrieve relevant content 
        from the knowledge base for the RAGAgent.
        1. Receive the user's inquiry from the RAGAgent.
        2. Immediately call the `run_chunk` tool with the inquiry.
        3. The `run_chunk` tool simulates checking if records exist in the 'metadata.db' and creating/retrieving 
           relevant content chunks.
        4. If `run_chunk` returns chunks, pass them back directly to the RAGAgent.
        5. If `run_chunk` does not return any chunks, just tell RAGAgent that no records found pertaining to user query'.
        # 5. If `run_chunk` indicates 'NO_RECORDS_FOUND_IN_METADATA_DB', return that exact message to the RAGAgent.
    """,
    tools=[run_chunk]
)

print(f"ingestion_agent set up")


# =============================================================================
# SECTION 5: AGENT DEFINITIONS (RAG, Marketing, Query Classifier, Router)
# =============================================================================

# RAG Agent
# rag_agent = LlmAgent(
#     name="RAGAgent",
#     model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
#     instruction="""
#     You are the Product Knowledge Specialist. Your task is to answer user queries using the 
#     `product_knowledge_search` tool.
    
#     Workflow:
#     1. Immediately call the `product_knowledge_search` tool with the user's query.
#     2. Analyze the search results (Documents) provided by the tool.
#     3. If relevant documents are returned, synthesize an accurate, concise answer *based exclusively* on the retrieved content.
#     4. If no relevant documents are found, state politely that the information is not available in the current knowledge base.
#     """,
#     tools=[
#         product_knowledge_search
#     ]
# )

rag_agent = LlmAgent(
    name="RAGAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are the Product Knowledge Specialist. Your task is to answer user queries by grounding your response 
    in information retrieved from the knowledge base.
    
    Workflow:
    1. Delegate the user query immediately to the `ingestion_agent` tool to retrieve relevant chunks (context).
    2. Analyze the result (chunks or a status message) provided by the `ingestion_agent`.
    3. If chunks are returned, synthesize an accurate, concise answer *based exclusively* on the retrieved content.
    4. If the status message is 'NO_RECORDS_FOUND_IN_METADATA_DB', state politely that the information is not available in the current knowledge base.
    """,
    tools=[
        AgentTool(agent=ingestion_agent) # Delegate search to Ingestion Agent
    ]
)

# Marketing Agent (for new customers)

marketing_agent = LlmAgent(
    name="MarketingAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are the Marketing and Enrollment Specialist. Your job is to convert new visitors into prospects.
    Your workflow MUST be:
    1. Acknowledge the user is new and politely ask for their full name and email address.
    2. Once the name and email are provided, call the `enroll_new_customer` tool with the provided user ID, name, and email.
    3. After successful enrollment, confirm the enrollment and then pass the original user query to the `rag_agent` tool to answer their question.
    """,
    tools=[
        enroll_new_customer,
        AgentTool(agent=rag_agent)
    ],
)

# Query Classification Agent (for existing customers)

query_classification_agent = LlmAgent(
    name="QueryClassifierAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are the Query Classification Agent (Agent D). You handle all existing customer inquiries.
    Your primary task is to classify the user's query as either a 'TICKET_QUERY' or 'PRODUCT_INQUIRY'.
    
    If the query is a 'TICKET_QUERY' (e.g., status, update, issue):
    1. Immediately call the `check_ticket_status` tool using the user ID.
    2. If the tool returns 'PENDING_TICKET...', use the information to inform the user the ticket is 
    PENDING, state the assigned agent's name, and assure them it will be resolved soon. Do NOT delegate further.
    3. If the tool returns 'NO_PENDING_TICKET': You must assume the user is reporting a new issue. Extract 
    the issue summary from the user's query and call the `create_new_ticket` tool. Then, respond to the user 
    with the new ticket ID and the assigned agent's name.

    If the query is a 'PRODUCT_INQUIRY' (e.g., features, pricing, comparison):
    1. Immediately delegate the query to the `rag_agent` tool to provide an answer.
    
    If the query is ambiguous, assume it is a 'PRODUCT_INQUIRY' and pass it to the `rag_agent`.
    """,
    tools=[
        check_ticket_status,
        create_new_ticket,
        AgentTool(agent=rag_agent)
    ],
)


# Router Orchestrator Agent

router_agent = LlmAgent(
    name="RouterAgent",
    model=Gemini(model="gemini-2.5-flash-lite", retry_options=retry_config),
    instruction="""
    You are the Router Orchestrator Agent (Agent A). 
    Your primary task is to receive the user's query and immediately call the 
    `check_customer_status` tool using the user ID extracted from the query.

    Based on the status result:
    1. If the status is 'UNKNOWN' (new customer): Immediately delegate the query to the `marketing_agent`.
    2. If the status is 'EXISTING' (existing or known customer): Immediately delegate the query to the 
    `query_classification_agent`.
    
    Do NOT provide any conversational response yourself; only call the appropriate tool.
    """,
    tools=[
        check_customer_status,
        AgentTool(agent=marketing_agent),
        AgentTool(agent=query_classification_agent)
    ],
)


# =============================================================================
# SECTION 6: TEST UTILITIES & COMPREHENSIVE TEST
# =============================================================================

def get_final_text(response_list):
    """
    Helper function to safely extract the final text from the runner's response list,
    as the response is often a list of intermediate steps.
    """
    if response_list and hasattr(response_list[-1], 'text'):
        return response_list[-1].text
    return "[Could not extract final text from response object.]"

def print_db_contents(db_name: str = CRM_DB, section_title: str = "DATABASE STATE CHECK"):
    """Prints the current contents of the customers and tickets tables."""
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    print("\n" + "="*80)
    print(section_title)
    print("="*80)

    # Print Customers Table
    cursor.execute("SELECT * FROM customers")
    customer_rows = cursor.fetchall()
    print("\n--- Customers Table ---")
    print(["user_id", "email", "status"])
    for row in customer_rows:
        print(row)

    # Print Tickets Table
    cursor.execute("SELECT * FROM tickets")
    ticket_rows = cursor.fetchall()
    print("\n--- Tickets Table ---")
    print(["ticket_id", "user_id", "issue_summary", "assigned_agent_name", "assigned_agent_email", "status"])
    for row in ticket_rows:
        print(row)

    conn.close()



# Comprehensive test runner
async def run_comprehensive_crm_test():
    """Sets up and runs comprehensive test cases using the InMemoryRunner."""

    # Re-initialize the CRM to ensure a clean state for the ticket counter
    initialize_mock_crm()

    # Instantiate the runner
    runner = InMemoryRunner(agent=router_agent)

    print("\n" + "="*80)
    print("COMPREHENSIVE CRM ARCHITECTURE TEST (RAG, Routing, Enrollment, Ticketing)")
    print("="*80)

    # --- Test 1: New Customer Path ---
    user_id_new = "user_999"
    query_text_new = f"I am user {user_id_new} and I'm new here. Can you tell me about the features of your premium product?"

    print(f"\n--- Scenario 1: New User ({user_id_new}) Enrollment & Query ---")
    print(f"User Query 1: {query_text_new}")

    # 1a. Initial Query
    response_new_1 = await runner.run_debug(query_text_new, session_id=user_id_new)
    print(f"Router -> Marketing Agent Response 1: {get_final_text(response_new_1)}")

    # 1b. Follow-up: user provides details
    follow_up_text = f"My full name is Test New and my email is test.new@{user_id_new}.com"
    print(f"User Query 2 (Follow-up): {follow_up_text}")
    response_new_2 = await runner.run_debug(follow_up_text, session_id=user_id_new)
    print(f"Marketing Agent Response 2 (Enrollment/RAG): {get_final_text(response_new_2)}")

    # Database check
    print_db_contents(section_title=f"DATABASE CHECK AFTER SCENARIO 1: {user_id_new} Enrollment")

    # --- Test 2: Existing Customer - Pending Ticket Status Query ---
    user_id_ticket_pending = "user_003"
    query_text_ticket = f"Hello, I am user {user_id_ticket_pending}. What is the status of my recent payment issue ticket?"

    print(f"\n--- Scenario 2: Existing User ({user_id_ticket_pending}) PENDING Ticket Query ---")
    print(f"User Query: {query_text_ticket}")

    response_ticket = await runner.run_debug(query_text_ticket, session_id=user_id_ticket_pending)
    print(f"Router -> Query Classifier Response: {get_final_text(response_ticket)}")

    # --- Test 3: Existing Customer - RAG Inquiry ---
    user_id_product_query = "user_001"
    query_text_product = f"I am user {user_id_product_query}. Can you explain how the new reporting dashboard works and tell me its price?"

    # query_text_product = f"Can you tell me how to add shortcodes to pages or widgets via the WordPress?"

    print(f"\n--- Scenario 3: Existing User ({user_id_product_query}) RAG Inquiry (Add Shortcodes) ---")
    print(f"User Query: {query_text_product}")

    response_product = await runner.run_debug(query_text_product, session_id=user_id_product_query)
    print(f"Router -> Query Classifier -> RAG Response: {get_final_text(response_product)}")

    # --- Test 4: Existing Customer - New Ticket Creation ---
    user_id_new_ticket = "user_002"
    query_text_new_ticket = f"I am user {user_id_new_ticket}. My login page is showing a 500 server error, I need a ticket opened right away."

    print(f"\n--- Scenario 4: Existing User ({user_id_new_ticket}) NEW Ticket Creation ---")
    print(f"User Query: {query_text_new_ticket}")

    response_new_ticket = await runner.run_debug(query_text_new_ticket, session_id=user_id_new_ticket)
    print(f"Router -> Query Classifier Response: {get_final_text(response_new_ticket)}")

    print("\n" + "="*80)
    print_db_contents(section_title="FINAL DATABASE STATE CHECK (All data changes confirmed)")
    print("COMPREHENSIVE TEST COMPLETE.")
    print("="*80)


await run_comprehensive_crm_test()

