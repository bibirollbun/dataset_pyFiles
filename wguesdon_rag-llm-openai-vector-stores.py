!pip install --upgrade openai  > /dev/null 2>&1
!pip install faiss-cpu==1.7.4  > /dev/null 2>&1
!pip install tiktoken==0.5.2  > /dev/null 2>&1
!pip install numpy==1.24.3 > /dev/null 2>&1


!cp /kaggle/input/creating-a-knowledge-base-paid-models/*.md /kaggle/working/
!cp /kaggle/input/openai-to-z-challenge-deep-research-reports/*.md /kaggle/working/


!rm __notebook__.ipynb
!rm __output__.json
!rm __results__.html
!rm custom.css


!ls /kaggle/working/


import os
import glob
import itertools
import time
import httpx
from openai import OpenAI, BadRequestError
from datetime import datetime
from IPython.display import display, Markdown
from typing import Optional, List

from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
os.environ["OPENAI_API_KEY"] = user_secrets.get_secret("openai_key")


# Configuration
DIR = "/kaggle/working/"
timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
VECTOR_STORE_NAME = f"OpenAI_challenge_{timestamp}"
print(VECTOR_STORE_NAME)
SEARCH_K = 5
CHAT_MODEL = "gpt-4.1" #gpt-4o o3-mini
TEMPERATURE = 0.2
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI()

def get_or_create_vector_store(name: str) -> str:
    """
    Retrieves an existing vector store by name or creates a new one.
    Returns the vector_store_id.
    """
    stores = client.vector_stores.list()
    for store in stores.data:
        if store.name == name:
            print(f"Found existing vector store '{name}' (id={store.id})")
            return store.id
    vs = client.vector_stores.create(name=name)
    print(f"Created new vector store '{name}' (id={vs.id})")
    return vs.id

def upload_to_vector_store(
    dir: str,
    vector_store_id: str,
    extensions: tuple[str, ...] = ("pdf", "md", "markdown", "txt"),
    min_bytes: int = 1,  # skip files smaller than this
):
    """
    Upload every file in *dir* whose extension appears in *extensions* to OpenAI
    and index it into the given vector store, skipping empty files.

    Parameters
    ----------
    dir : str
        Folder to scan.
    vector_store_id : str
        ID of the target vector store.
    extensions : tuple[str, ...], optional
        Allowed file extensions (case-insensitive, *without* the leading dot).
    min_bytes : int, optional
        Minimum file size to upload; defaults to 1 byte (skip blank files).
    """
    exts = [ext.lower().lstrip(".") for ext in extensions]
    patterns = (os.path.join(dir, f"*.{ext}") for ext in exts)
    file_paths = list(
        itertools.chain.from_iterable(glob.glob(p, recursive=False) for p in patterns)
    )

    if not file_paths:
        print(f"No files with extensions {exts} found in {dir}")
        return

    for path in file_paths:
        filename = os.path.basename(path)

        # â”€â”€ Skip zero-byte (or very small) files â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        size = os.path.getsize(path)
        if size < min_bytes:
            print(f"âš ï¸�  Skipping {filename} (size {size} bytes)")
            continue

        print(f"Uploading {filename} ({size} bytes)â€¦")
        try:
            with open(path, "rb") as fp:
                file_resp = client.files.create(file=fp, purpose="assistants")

            client.vector_stores.files.create(
                vector_store_id=vector_store_id,
                file_id=file_resp.id,
            )
            print(f"Indexed {filename} (file_id={file_resp.id})")
            time.sleep(0.2)  # gentle rate limit

        except BadRequestError as e:
            # Catch other per-file errors so the loop keeps going
            print(f"â�Œ  OpenAI rejected {filename}: {e}")

def search_vector_store(vector_store_id: str, query: str) -> list:
    """
    Queries the OpenAI Vector Store via HTTP and returns up to SEARCH_K items.
    """
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    url = f"https://api.openai.com/v1/vector_stores/{vector_store_id}/search"
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    payload = {"query": query}
    resp = httpx.post(url, headers=headers, json=payload)
    resp.raise_for_status()
    data = resp.json().get("data", [])
    # limit to top SEARCH_K
    return data[:SEARCH_K]


def answer_question(vector_store_id: str, question: str) -> str:
    """
    Retrieves relevant passages and asks the chat model to synthesize an answer.
    """
    # 1. Retrieve top passages
    items = search_vector_store(vector_store_id, question)
    # 2. Build context string
    context_parts = []
    for item in items:
        filename = item.get("filename", "unknown")
        for passage in item.get("content", []):
            context_parts.append(f"Source: {filename}\n{passage.get('text', '')}")
    context = "\n---\n".join(context_parts)

    # 3. Create chat messages
    system_msg = {
        "role": "system",
        "content": (
            "You are an expert in Maya archaeology and remote sensing. "
            "Use the provided context to answer precisely, citing sources when relevant."
        )
    }
    user_msg = {
        "role": "user",
        "content": (
            f"Context passages:\n{context}\n\n"
            f"Answer the following question: {question}"
        )
    }

    # 4. Call chat completion
    response = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[system_msg, user_msg],
        temperature=TEMPERATURE,
        max_tokens=800
    )

    return response.choices[0].message.content

def delete_vector_store_files(vector_store_id: str) -> int:
    """
    Delete all files associated with a vector store.
    
    Parameters
    ----------
    vector_store_id : str
        ID of the vector store whose files should be deleted.
    
    Returns
    -------
    int
        Number of files deleted.
    """
    deleted_count = 0
    
    try:
        # List all files in the vector store
        files = client.vector_stores.files.list(vector_store_id=vector_store_id)
        
        for file in files.data:
            try:
                # First, remove the file from the vector store
                client.vector_stores.files.delete(
                    vector_store_id=vector_store_id,
                    file_id=file.id
                )
                
                # Then delete the actual file
                client.files.delete(file.id)
                print(f"âœ“ Deleted file {file.id}")
                deleted_count += 1
                time.sleep(0.1)  # Rate limiting
                
            except Exception as e:
                print(f"â�Œ Error deleting file {file.id}: {e}")
                
    except Exception as e:
        print(f"â�Œ Error listing files for vector store {vector_store_id}: {e}")
    
    return deleted_count

def delete_vector_store(vector_store_id: str) -> bool:
    """
    Delete a vector store.
    
    Parameters
    ----------
    vector_store_id : str
        ID of the vector store to delete.
    
    Returns
    -------
    bool
        True if successfully deleted, False otherwise.
    """
    try:
        client.vector_stores.delete(vector_store_id)
        print(f"âœ“ Deleted vector store {vector_store_id}")
        return True
    except Exception as e:
        print(f"â�Œ Error deleting vector store {vector_store_id}: {e}")
        return False

def cleanup_vector_store(vector_store_id: str) -> None:
    """
    Complete cleanup: delete all files and the vector store.
    
    Parameters
    ----------
    vector_store_id : str
        ID of the vector store to completely remove.
    """
    print(f"\nğŸ§¹ Starting cleanup for vector store {vector_store_id}")
    
    # Step 1: Delete all files
    print("\nğŸ“„ Deleting files...")
    files_deleted = delete_vector_store_files(vector_store_id)
    print(f"Deleted {files_deleted} files")
    
    # Step 2: Delete the vector store
    print("\nğŸ—‘ï¸�  Deleting vector store...")
    if delete_vector_store(vector_store_id):
        print("\nâœ… Cleanup complete!")
    else:
        print("\nâš ï¸�  Cleanup completed with errors")

def cleanup_all_vector_stores_by_pattern(name_pattern: str = "OpenAI_challenge_") -> None:
    """
    Delete all vector stores whose names start with the given pattern.
    Useful for cleaning up multiple test runs.
    
    Parameters
    ----------
    name_pattern : str
        Pattern to match vector store names (default: "OpenAI_challenge_")
    """
    stores = client.vector_stores.list()
    matching_stores = [s for s in stores.data if s.name.startswith(name_pattern)]
    
    if not matching_stores:
        print(f"No vector stores found matching pattern '{name_pattern}'")
        return
    
    print(f"Found {len(matching_stores)} vector stores matching '{name_pattern}'")
    for store in matching_stores:
        print(f"\nProcessing: {store.name} (id={store.id})")
        cleanup_vector_store(store.id)

def list_all_files(purpose: Optional[str] = None) -> List:
    """
    List all files in OpenAI storage.
    
    Parameters
    ----------
    purpose : str, optional
        Filter by purpose ('assistants', 'fine-tune', etc.)
        If None, returns all files.
    
    Returns
    -------
    List
        List of file objects
    """
    all_files = []
    
    try:
        # OpenAI's list method might paginate, so we need to handle that
        has_more = True
        after = None
        
        while has_more:
            if after:
                files = client.files.list(purpose=purpose, after=after)
            else:
                files = client.files.list(purpose=purpose) if purpose else client.files.list()
            
            all_files.extend(files.data)
            
            # Check if there are more files
            has_more = files.has_more if hasattr(files, 'has_more') else False
            if has_more and files.data:
                after = files.data[-1].id
            
        return all_files
        
    except Exception as e:
        print(f"â�Œ Error listing files: {e}")
        return []

def delete_all_files(purpose: Optional[str] = None, dry_run: bool = True) -> dict:
    """
    Delete all files in OpenAI storage.
    
    Parameters
    ----------
    purpose : str, optional
        Only delete files with this purpose ('assistants', 'fine-tune', etc.)
        If None, deletes all files.
    dry_run : bool
        If True, only shows what would be deleted without actually deleting.
    
    Returns
    -------
    dict
        Statistics about the deletion process
    """
    files = list_all_files(purpose=purpose)
    
    if not files:
        print("No files found to delete.")
        return {"total": 0, "deleted": 0, "failed": 0}
    
    stats = {
        "total": len(files),
        "deleted": 0,
        "failed": 0,
        "total_bytes": sum(f.bytes for f in files if hasattr(f, 'bytes') and f.bytes)
    }
    
    print(f"\n{'ğŸ”� DRY RUN - ' if dry_run else ''}Found {stats['total']} files")
    print(f"Total size: {stats['total_bytes'] / (1024*1024):.2f} MB")
    
    if dry_run:
        print("\nFiles that would be deleted:")
        for f in files[:10]:  # Show first 10 files
            created_date = datetime.fromtimestamp(f.created_at).strftime('%Y-%m-%d %H:%M:%S')
            print(f"  - {f.filename} (ID: {f.id}, Purpose: {f.purpose}, Created: {created_date})")
        if len(files) > 10:
            print(f"  ... and {len(files) - 10} more files")
        print("\nâš ï¸�  Run with dry_run=False to actually delete these files")
        return stats
    
    # Actual deletion
    print("\nğŸ—‘ï¸�  Deleting files...")
    for i, file in enumerate(files):
        try:
            client.files.delete(file.id)
            stats["deleted"] += 1
            print(f"âœ“ Deleted {file.filename} ({i+1}/{stats['total']})")
            time.sleep(0.1)  # Rate limiting
            
        except Exception as e:
            stats["failed"] += 1
            print(f"â�Œ Failed to delete {file.filename}: {e}")
    
    print(f"\nâœ… Deletion complete!")
    print(f"   Deleted: {stats['deleted']}")
    print(f"   Failed: {stats['failed']}")
    
    return stats

def delete_files_by_name_pattern(pattern: str, dry_run: bool = True) -> dict:
    """
    Delete files whose names contain a specific pattern.
    
    Parameters
    ----------
    pattern : str
        Delete files whose names contain this pattern
    dry_run : bool
        If True, only shows what would be deleted
    
    Returns
    -------
    dict
        Statistics about the deletion process
    """
    all_files = list_all_files()
    matching_files = [f for f in all_files if pattern in f.filename]
    
    if not matching_files:
        print(f"No files found matching pattern '{pattern}'")
        return {"total": 0, "deleted": 0, "failed": 0}
    
    stats = {
        "total": len(matching_files),
        "deleted": 0,
        "failed": 0
    }
    
    print(f"\n{'ğŸ”� DRY RUN - ' if dry_run else ''}Found {stats['total']} files matching '{pattern}'")
    
    if dry_run:
        print("\nFiles that would be deleted:")
        for f in matching_files[:10]:
            print(f"  - {f.filename} (ID: {f.id})")
        if len(matching_files) > 10:
            print(f"  ... and {len(matching_files) - 10} more files")
        print("\nâš ï¸�  Run with dry_run=False to actually delete these files")
        return stats
    
    # Actual deletion
    print("\nğŸ—‘ï¸�  Deleting matching files...")
    for i, file in enumerate(matching_files):
        try:
            client.files.delete(file.id)
            stats["deleted"] += 1
            print(f"âœ“ Deleted {file.filename} ({i+1}/{stats['total']})")
            time.sleep(0.1)
            
        except Exception as e:
            stats["failed"] += 1
            print(f"â�Œ Failed to delete {file.filename}: {e}")
    
    return stats

def show_storage_summary():
    """
    Display a summary of all files in storage grouped by purpose.
    """
    all_files = list_all_files()
    
    if not all_files:
        print("No files in storage.")
        return
    
    # Group by purpose
    by_purpose = {}
    total_size = 0
    
    for f in all_files:
        purpose = f.purpose
        if purpose not in by_purpose:
            by_purpose[purpose] = {"count": 0, "size": 0, "files": []}
        
        by_purpose[purpose]["count"] += 1
        if hasattr(f, 'bytes') and f.bytes:
            by_purpose[purpose]["size"] += f.bytes
            total_size += f.bytes
        by_purpose[purpose]["files"].append(f.filename)
    
    print("\nğŸ“Š Storage Summary")
    print("=" * 50)
    print(f"Total files: {len(all_files)}")
    print(f"Total size: {total_size / (1024*1024):.2f} MB")
    print("\nBy purpose:")
    
    for purpose, info in by_purpose.items():
        print(f"\n  {purpose}:")
        print(f"    Count: {info['count']}")
        print(f"    Size: {info['size'] / (1024*1024):.2f} MB")
        print(f"    Files: {', '.join(info['files'][:3])}" + 
              (f" ... and {info['count'] - 3} more" if info['count'] > 3 else ""))


# 1. Get or create vector store
vs_id = get_or_create_vector_store(VECTOR_STORE_NAME)

# 2. Upload and index Documents (skip if already done)
upload_to_vector_store(DIR, vs_id)


# 3. Answer Questions

question = """
What are the locations of known archaeological sites in the Amazon Rainforest?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What is the location of Kuhikugu (GPS coordinate)?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What are the locations of suspected archaeological sites
in the Brazilian Amazon Rainforest from the
literature that have not yet been confirmed?"
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What locations in the Amazon Rainforest match descriptions from historical accounts but haven't been verified on the ground?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What techniques have been used to discover new archaeological sites in the Amazon rainforest in recent years?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
Were historical and indigenous texts analyzed using machine
learning or a large language model to identify potential archaeological site locations?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What is the impact of the presence of Amazonian dark earth on vegetation?
How could this be used to detect potential
unexplored archaeological sites using satellite imagery and machine learning? 
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What open-access LIDAR datasets cover portions of the Brazilian Amazon?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What machine learning models have successfully identified archaeological features beneath forest canopy?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
How are radar (e.g., Sentinel-1 SAR) and hyperspectral data being combined with LIDAR for sub-canopy detection?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What benchmark datasets and evaluation metrics are commonly used to compare canopy-penetrating algorithms?
"""

display(Markdown(answer_question(vs_id, question)))


question = """
What common geographical features are shared among verified archaeological sites in the Amazon?
"""

display(Markdown(answer_question(vs_id, question)))


# Clean up
# Option 1: Clean up specific vector store
cleanup_vector_store(vs_id)
    
# Option 2: Clean up all test vector stores (uncomment to use)
#cleanup_all_vector_stores_by_pattern("OpenAI_challenge_")


#print("Current storage status:")
#show_storage_summary()


# Example 1: Delete ALL files (dry run first for safety)
#print("\n" + "="*50)
#print("Preview of deleting ALL files:")
#delete_all_files(dry_run=True)


delete_all_files(dry_run=False)


# Example 2: Delete only assistant files
# delete_all_files(purpose="assistants", dry_run=False)
    
# Example 3: Delete files by name pattern
delete_files_by_name_pattern("question_", dry_run=False)
delete_files_by_name_pattern("deep_research_", dry_run=False)

