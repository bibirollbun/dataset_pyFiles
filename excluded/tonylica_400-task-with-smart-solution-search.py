# Auto-detect environment: local vs Kaggle
import os
import zipfile
import shutil
import json
import importlib.util
import sys
import copy
import traceback
import numpy as np
import re
from collections import defaultdict
import csv
import glob
from pathlib import Path

if os.path.exists("./tasks"):
    # Local environment
    print("Detected LOCAL environment - using local tasks folder")
    ENVIRONMENT = "local"
    sources = [
        "./tasks"
    ]
else:
    # Kaggle environment - treat input paths as direct dataset sources
    print("Detected KAGGLE environment - using dataset sources")
    ENVIRONMENT = "kaggle"
    sources = [
        "/kaggle/input/google-code-golf-2025-submit"
    ]

print(f"Environment: {ENVIRONMENT}")
print(f"Sources: {sources}")


def find_submission_zips():
    """Find all submission.zip files in source directories and their subdirectories"""
    submission_zips = {}
    
    for source in sources:
        if not os.path.exists(source):
            print(f"Warning: Source {source} does not exist")
            continue
            
        # Use glob to find all submission.zip files recursively
        pattern = os.path.join(source, "**/submission.zip")
        found_zips = glob.glob(pattern, recursive=True)
        
        if found_zips:
            # Use the first submission.zip found in this source
            submission_zips[source] = found_zips[0]
            print(f"Found submission.zip in {source}: {found_zips[0]}")
        else:
            print(f"No submission.zip found in {source}")
    
    return submission_zips

submission_zips = find_submission_zips()
print(f"\nFound {len(submission_zips)} submission.zip files")


def setup_working_directories():
    """Create working directory structure"""
    if ENVIRONMENT == "kaggle":
        working_dir = "/kaggle/working"
        submission_dir = "/kaggle/working/submission"
        source_dir = "/kaggle/working/source"
    else:
        working_dir = "./working"
        submission_dir = "./working/submission"
        source_dir = "./working/source"
    
    # Clean specific subdirectories and leftover files
    cleanup_items = [
        submission_dir,
        source_dir,
        os.path.join(working_dir, "submission_final"),
        os.path.join(working_dir, "submission_final.zip"),
        os.path.join(working_dir, "submission.zip"),
        os.path.join(working_dir, "others.zip"),
        os.path.join(working_dir, "others_metadata.csv")
    ]
    
    for item_path in cleanup_items:
        if os.path.exists(item_path):
            try:
                if os.path.isdir(item_path):
                    shutil.rmtree(item_path)
                    print(f"Cleaned existing directory: {item_path}")
                else:
                    os.remove(item_path)
                    print(f"Cleaned existing file: {item_path}")
            except OSError as e:
                print(f"Warning: Could not clean {item_path}: {e}")
                # Try to clean contents if it's a directory
                if os.path.isdir(item_path):
                    try:
                        for item in os.listdir(item_path):
                            sub_item_path = os.path.join(item_path, item)
                            if os.path.isdir(sub_item_path):
                                shutil.rmtree(sub_item_path)
                            else:
                                os.remove(sub_item_path)
                        print(f"Cleaned contents of: {item_path}")
                    except OSError as e2:
                        print(f"Warning: Could not clean contents of {item_path}: {e2}")
    
    # Create directories
    os.makedirs(submission_dir, exist_ok=True)
    os.makedirs(source_dir, exist_ok=True)
    
    return working_dir, submission_dir, source_dir

def read_file_as_text(file_path):
    """Read file content as text for verification purposes only"""
    try:
        # Try UTF-8 first (most common)
        with open(file_path, 'r') as f:
            return f.read()
    except UnicodeDecodeError:
        # Fallback to latin-1 which can handle any byte sequence
        with open(file_path, 'r', encoding='latin-1') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def copy_task_files_from_dataset(source_path, dest_dir):
    """Copy task files directly from dataset folder as binary to preserve exact bytes"""
    print(f"Copying task files from dataset: {source_path}")
    
    # Look for task files with pattern task001.py to task400.py
    task_files_copied = 0
    for task_num in range(1, 401):
        task_filename = f"task{task_num:03d}.py"
        source_file = os.path.join(source_path, task_filename)
        
        if os.path.exists(source_file):
            dest_file = os.path.join(dest_dir, task_filename)
            try:
                # Simple binary copy to preserve exact bytes
                shutil.copy2(source_file, dest_file)
                task_files_copied += 1
            except Exception as e:
                print(f"Error copying {task_filename}: {e}")
    
    print(f"Copied {task_files_copied} task files from {source_path}")
    return task_files_copied

def extract_default_submission(submission_zips, submission_dir):
    """Extract default source (first in sources list) to submission directory"""
    if not sources:
        raise ValueError("No sources defined")
    
    default_source = sources[0]
    
    # ALWAYS prioritize direct task files over submission.zip for default source
    if os.path.exists(default_source):
        print(f"Using direct task files from default source: {default_source}")
        
        # Check how many task files are actually available in the source
        available_tasks = 0
        for task_num in range(1, 401):
            task_file = os.path.join(default_source, f"task{task_num:03d}.py")
            if os.path.exists(task_file):
                available_tasks += 1
        
        print(f"Found {available_tasks} task files in default source")
        
        task_files_copied = copy_task_files_from_dataset(default_source, submission_dir)
        
        if task_files_copied > 0:
            print(f"Copied {task_files_copied} task files directly from default source")
            
            # Verify what was actually copied
            copied_tasks = 0
            for task_num in range(1, 401):
                dest_file = os.path.join(submission_dir, f"task{task_num:03d}.py")
                if os.path.exists(dest_file):
                    copied_tasks += 1
            
            print(f"Verified {copied_tasks} task files in submission directory")
            
            if copied_tasks < 400:
                print(f"WARNING: Only {copied_tasks}/400 tasks copied from default source!")
            
            return None
    
    # Fallback to submission.zip if direct files not available
    if default_source in submission_zips:
        default_zip = submission_zips[default_source]
        print(f"Fallback: Extracting default submission from: {default_zip}")
        
        with zipfile.ZipFile(default_zip, 'r') as zip_ref:
            zip_ref.extractall(submission_dir)
        
        print(f"Default submission extracted to: {submission_dir}")
        return default_zip
    
    # No files found at all
    raise ValueError(f"No task files or submission.zip found in default source {default_source}")

def extract_extra_submissions(submission_zips, source_dir):
    """Extract extra sources to their own directories"""
    extra_sources = {}
    
    for i, source in enumerate(sources[1:], 1):  # Skip first source (default)
        # Get top-level folder name from the source path
        source_name = Path(source).parts[-1] if Path(source).parts else f"source_{i}"
        source_path = os.path.join(source_dir, source_name)
        
        os.makedirs(source_path, exist_ok=True)
        
        if source in submission_zips:
            # Found submission.zip, extract it
            print(f"Processing extra source {source} with submission.zip")
            
            # Copy submission.zip to the source directory
            zip_dest = os.path.join(source_path, "submission.zip")
            shutil.copy2(submission_zips[source], zip_dest)
            
            # Extract submission.zip in the source directory
            with zipfile.ZipFile(submission_zips[source], 'r') as zip_ref:
                zip_ref.extractall(source_path)
            
            extra_sources[source] = {
                'path': source_path,
                'name': source_name,
                'zip_path': zip_dest
            }
            
            print(f"Extra source {source} extracted to: {source_path}")
            
        elif os.path.exists(source):
            # No submission.zip found, copy task files directly from dataset
            print(f"Processing extra source {source} as task dataset")
            
            task_files_copied = copy_task_files_from_dataset(source, source_path)
            
            if task_files_copied > 0:
                extra_sources[source] = {
                    'path': source_path,
                    'name': source_name,
                    'zip_path': None  # No zip file for direct copy
                }
                print(f"Extra source {source} copied {task_files_copied} files to: {source_path}")
            else:
                print(f"Warning: No task files found in extra source {source}")
                # Remove empty directory
                if os.path.exists(source_path):
                    shutil.rmtree(source_path)
        else:
            print(f"Warning: Extra source {source} does not exist")
    
    return extra_sources

working_dir, submission_dir, source_dir = setup_working_directories()
default_zip = extract_default_submission(submission_zips, submission_dir)
extra_sources = extract_extra_submissions(submission_zips, source_dir)

print(f"\nSetup complete:")
print(f"  Working directory: {working_dir}")
print(f"  Default submission: {submission_dir}")
print(f"  Extra sources: {len(extra_sources)}")


def load_task_examples(task_num):
    """Load examples for a task (simplified version of verify1.py logic)"""
    # Try to find task data file
    task_data_paths = [
        f"/kaggle/input/google-code-golf-2025/task{task_num:03d}.json",
        f"google-code-golf-2025/task{task_num:03d}.json",
        f"./task{task_num:03d}.json"
    ]
    
    for path in task_data_paths:
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
    
    print(f"Warning: No test data found for task {task_num}")
    return None

def verify_solution_code(solution_code, task_num, examples=None):
    """Verify if a solution works correctly"""
    if examples is None:
        examples = load_task_examples(task_num)
        if examples is None:
            return False, "No test data available"
    
    try:
        # Execute the solution code
        namespace = {}
        exec(solution_code, namespace)
        
        if 'p' not in namespace:
            return False, "No function 'p' found"
        
        solution_func = namespace['p']
        if not callable(solution_func):
            return False, "Function 'p' is not callable"
        
        # Test on all available test cases
        test_cases = []
        for category in ['test', 'train', 'arc-gen']:
            if category in examples:
                test_cases.extend(examples[category])
        
        if not test_cases:
            return False, "No test cases available"
        
        correct_count = 0
        total_count = len(test_cases)
        
        for i, test_case in enumerate(test_cases):
            try:
                result = solution_func(copy.deepcopy(test_case['input']))
                
                # Convert result to comparable format
                result_json = json.dumps(result)
                result_json = result_json.replace("true", "1").replace("false", "0")
                
                # Check for unsafe characters
                unsafe_chars = re.compile(r"[^0-9,\[\]\s\.]")
                if unsafe_chars.search(result_json):
                    return False, f"Invalid output format: {result_json[:100]}"
                
                result_array = np.array(json.loads(result_json))
                expected_array = np.array(test_case['output'])
                
                if np.array_equal(result_array, expected_array):
                    correct_count += 1
                else:
                    return False, f"Wrong output on test case {i+1}/{total_count}"
                    
            except Exception as e:
                return False, f"Exception on test case {i+1}: {str(e)[:100]}"
        
        return True, f"Passed all {correct_count}/{total_count} test cases"
        
    except Exception as e:
        return False, f"Execution error: {str(e)[:100]}"


def get_task_candidates(task_num, submission_dir, extra_sources):
    """Get all candidate solutions for a task, sorted by size"""
    candidates = []
    
    # Default solution
    default_path = os.path.join(submission_dir, f"task{task_num:03d}.py")
    if os.path.exists(default_path):
        # Use actual file size on disk
        size = os.path.getsize(default_path)
        
        # Read content for verification only
        content = read_file_as_text(default_path)
        if content is not None:
            candidates.append({
                'path': default_path,
                'content': content,
                'size': size,
                'source': 'default',
                'source_name': 'default'
            })
        else:
            print(f"Could not read default task {task_num} from {default_path}")
    
    # Extra source solutions
    for source, info in extra_sources.items():
        extra_path = os.path.join(info['path'], f"task{task_num:03d}.py")
        if os.path.exists(extra_path):
            # Use actual file size on disk
            size = os.path.getsize(extra_path)
            
            # Read content for verification only
            content = read_file_as_text(extra_path)
            if content is not None:
                candidates.append({
                    'path': extra_path,
                    'content': content,
                    'size': size,
                    'source': source,
                    'source_name': info['name']
                })
    
    # Sort by size (smallest first)
    candidates.sort(key=lambda x: x['size'])
    return candidates

def select_best_solution(task_num, submission_dir, extra_sources):
    """Select the best (smallest working) solution for a task"""
    candidates = get_task_candidates(task_num, submission_dir, extra_sources)
    
    if not candidates:
        print(f"No candidates found for task {task_num}")
        return None
    
    # Load examples once for this task
    examples = load_task_examples(task_num)
    
    default_candidate = next((c for c in candidates if c['source'] == 'default'), None)
    if not default_candidate:
        print(f"No default solution found for task {task_num}")
        return None
    
    # Find smaller candidates than default
    smaller_candidates = [c for c in candidates if c['size'] < default_candidate['size']]
    
    if not smaller_candidates:
        # No smaller solutions, use default
        return {
            'candidate': default_candidate,
            'used_default': True,
            'verification_result': None,
            'alternatives_tested': 0
        }
    
    print(f"Task {task_num}: Default size {default_candidate['size']}, testing {len(smaller_candidates)} smaller alternatives")
    
    # Test smaller candidates in order (smallest first)
    for i, candidate in enumerate(smaller_candidates):
        print(f"  Testing candidate {i+1}/{len(smaller_candidates)}: {candidate['source_name']} ({candidate['size']} bytes)")
        
        is_working, message = verify_solution_code(candidate['content'], task_num, examples)
        
        if is_working:
            print(f"  âœ“ Working! Using {candidate['source_name']} solution ({candidate['size']} bytes, saved {default_candidate['size'] - candidate['size']} bytes)")
            return {
                'candidate': candidate,
                'used_default': False,
                'verification_result': message,
                'alternatives_tested': i + 1,
                'bytes_saved': default_candidate['size'] - candidate['size']
            }
        else:
            print(f"  âœ— Failed: {message}")
    
    # No smaller solution worked, use default
    print(f"  Using default solution ({default_candidate['size']} bytes)")
    return {
        'candidate': default_candidate,
        'used_default': True,
        'verification_result': "No smaller working solution found",
        'alternatives_tested': len(smaller_candidates),
        'bytes_saved': 0
    }


print(f"\nProcessing tasks 1-400...")

results = {}
total_bytes_saved = 0
tasks_improved = 0
tasks_with_alternatives = 0
verification_stats = defaultdict(int)

for task_num in range(1, 401):
    if task_num % 50 == 0:
        print(f"Progress: {task_num}/400 tasks processed")
    
    result = select_best_solution(task_num, submission_dir, extra_sources)
    
    if result is None:
        verification_stats['no_candidates'] += 1
        continue
    
    results[task_num] = result
    
    if not result['used_default']:
        tasks_improved += 1
        total_bytes_saved += result.get('bytes_saved', 0)
    
    if result['alternatives_tested'] > 0:
        tasks_with_alternatives += 1
    
    verification_stats['processed'] += 1

print(f"\nProcessing complete!")
print(f"Tasks processed: {verification_stats['processed']}")
print(f"Tasks with smaller alternatives: {tasks_with_alternatives}")
print(f"Tasks improved: {tasks_improved}")
print(f"Total bytes saved: {total_bytes_saved}")


def create_final_submission(results, working_dir):
    """Create the final submission with best solutions"""
    # Create submission.zip directly in working directory
    submission_zip_path = os.path.join(working_dir, "submission.zip")
    
    tasks_written = 0
    total_size = 0
    
    # Create submission.zip directly without intermediate directory
    missing_tasks = []
    with zipfile.ZipFile(submission_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for task_num in range(1, 401):
            if task_num not in results:
                missing_tasks.append(task_num)
                continue
            
            result = results[task_num]
            candidate = result['candidate']
            
            # Read the original binary file and add it to zip
            task_filename = f"task{task_num:03d}.py"
            try:
                with open(candidate['path'], 'rb') as f:
                    file_data = f.read()
                zipf.writestr(task_filename, file_data)
                tasks_written += 1
                total_size += len(file_data)
            except Exception as e:
                print(f"Error reading {candidate['path']}: {e}")
                missing_tasks.append(task_num)
    
    # Report missing tasks
    if missing_tasks:
        print(f"WARNING: {len(missing_tasks)} tasks missing from results!")
        print(f"Missing tasks: {missing_tasks[:10]}..." if len(missing_tasks) > 10 else f"Missing tasks: {missing_tasks}")
    else:
        print("All 400 tasks included in submission")
    
    zip_size = os.path.getsize(submission_zip_path)
    
    print(f"\nFinal submission created:")
    print(f"  Zip file: {submission_zip_path}")
    print(f"  Tasks: {tasks_written}")
    print(f"  Total uncompressed size: {total_size:,} bytes")
    print(f"  Zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")
    print(f"  Compression ratio: {zip_size/total_size:.3f}")
    
    return submission_zip_path

def create_others_zip_and_metadata(results, working_dir):
    """Create others.zip with improved solutions and metadata CSV"""
    improved_tasks = {task_num: result for task_num, result in results.items() 
                     if not result['used_default']}
    
    if not improved_tasks:
        print("No improved tasks found - skipping others.zip creation")
        return None, None
    
    others_zip_path = os.path.join(working_dir, "others.zip")
    others_csv_path = os.path.join(working_dir, "others_metadata.csv")
    
    # Create CSV metadata
    with open(others_csv_path, 'w', newline='') as csvfile:
        fieldnames = [
            'task_number', 'source', 'source_name',
            'default_size', 'chosen_size', 'bytes_saved',
            'alternatives_tested', 'verification_result'
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        for task_num, result in sorted(improved_tasks.items()):
            # Find default size for comparison
            default_size = None
            candidates = get_task_candidates(task_num, submission_dir, extra_sources)
            default_candidate = next((c for c in candidates if c['source'] == 'default'), None)
            if default_candidate:
                default_size = default_candidate['size']
            
            candidate = result['candidate']
            writer.writerow({
                'task_number': f"task{task_num:03d}",
                'source': candidate['source'],
                'source_name': candidate['source_name'],
                'default_size': default_size or '',
                'chosen_size': candidate['size'],
                'bytes_saved': result.get('bytes_saved', 0),
                'alternatives_tested': result['alternatives_tested'],
                'verification_result': result['verification_result'] or ''
            })
    
    # Create others.zip with improved solutions
    with zipfile.ZipFile(others_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for task_num, result in sorted(improved_tasks.items()):
            task_filename = f"task{task_num:03d}.py"
            try:
                with open(result['candidate']['path'], 'rb') as f:
                    file_data = f.read()
                zipf.writestr(task_filename, file_data)
            except Exception as e:
                print(f"Error reading {result['candidate']['path']} for others.zip: {e}")
    
    others_zip_size = os.path.getsize(others_zip_path)
    others_csv_size = os.path.getsize(others_csv_path)
    
    print(f"\nOthers files created:")
    print(f"  Others.zip: {others_zip_path} ({others_zip_size:,} bytes)")
    print(f"  Metadata CSV: {others_csv_path} ({others_csv_size:,} bytes)")
    print(f"  Improved tasks: {len(improved_tasks)}")
    print(f"  Total bytes saved: {sum(r.get('bytes_saved', 0) for r in improved_tasks.values())}")
    
    return others_zip_path, others_csv_path


# Create final submission
submission_zip_path = create_final_submission(results, working_dir)

# Create others.zip and metadata  
others_zip_path, others_csv_path = create_others_zip_and_metadata(results, working_dir)

# Calculate final score estimate
total_score = 0
for task_num, result in results.items():
    task_size = result['candidate']['size']
    task_score = max(1, 2500 - task_size)
    total_score += task_score

print(f"\nFinal Summary:")
print("=" * 60)
print(f"Environment: {ENVIRONMENT}")
print(f"Sources processed: {len(sources)}")
print(f"Submission zips found: {len(submission_zips)}")
print("")
print(f"Tasks processed: {len(results)}")
print(f"Tasks with smaller alternatives: {tasks_with_alternatives}")
print(f"Tasks improved: {tasks_improved}")
print(f"Total bytes saved: {total_bytes_saved:,}")
print("")
print(f"Final submission: {submission_zip_path}")
if others_zip_path:
    print(f"Others archive: {others_zip_path}")
    print(f"Metadata file: {others_csv_path}")
print("")
print(f"Estimated score: {total_score:,}")
print("=" * 60)

