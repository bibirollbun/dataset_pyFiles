!pip install python-minifier
!mkdir -p /kaggle/working/arc-code-golf-starter-solutions
!cp -r /kaggle/input/arc-code-golf-starter-solutions/*.py /kaggle/working/arc-code-golf-starter-solutions
!pyminify /kaggle/working/arc-code-golf-starter-solutions/*.py --in-place


source = "/kaggle/working/arc-code-golf-starter-solutions"


from zlib import compress

def zip_src(src):
    compression_level = 9  # Max Compression
    
    # We prefer that compressed source not end in a quotation mark
    while (compressed := compress(src, compression_level))[-1] == ord('"'): 
        src += b"#"
    
    def sanitize(b_in):
        """Clean up problematic bytes in compressed b-string"""
        b_out = bytearray()
        for b in b_in:
            if b == 0:         
                b_out += b"\\x00"
            elif b == ord("\r"): 
                b_out += b"\\r"
            elif b == ord("\\"): 
                b_out += b"\\\\"
            else: 
                b_out.append(b)
        return b"" + b_out
    
    compressed = sanitize(compressed)
    
    delim = b'"""' if ord("\n") in compressed or ord('"') in compressed else b'"'
    
    return b"#coding:L1\nimport zlib\nexec(zlib.decompress(bytes(" + \
        delim + compressed + delim + \
        b',"L1")))'



import os

total_save = 0
processed_tasks = 0

submission = "/kaggle/working/submission"
os.makedirs(submission, exist_ok=True)

# Process tasks 1-400
for task_num in range(1, 401):
    path_in = f"{source}/task{task_num:03d}.py"
    path_out = f"{submission}/task{task_num:03d}.py"
    
    if not os.path.exists(path_in):
        continue
    
    try:
        with open(path_in, "rb") as task_in:
            task_src = task_in.read()
        
        # Only compress if file has content
        if len(task_src) > 0:
            zipped_src = zip_src(task_src)
            improvement = len(task_src) - len(zipped_src)
            
            # Use compressed version if it saves space
            if improvement > 0:
                task_src = zipped_src
                total_save += improvement
            
            with open(path_out, "wb") as task_out:
                task_out.write(task_src)
            
            processed_tasks += 1
    
    except Exception as e:
        print(f"Error processing task{task_num:03d}: {e}")
        continue

print(f"Processed {processed_tasks} tasks")
print(f"Saved {total_save}b with zlib compression")


import zipfile

submission_zip = f"{submission}.zip"

with zipfile.ZipFile(submission_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    task_count = 0
    for task_num in range(1, 401):
        task_id = f"{task_num:03d}"
        src_path = f"{submission}/task{task_id}.py"
        
        if os.path.exists(src_path):
            zipf.write(src_path, arcname=f"task{task_id}.py")
            task_count += 1

print(f"Created submission zip with {task_count} tasks: {submission_zip}")

# Display zip file size
zip_size = os.path.getsize(submission_zip)
print(f"Submission zip size: {zip_size:,} bytes ({zip_size/1024:.1f} KB)")


def compute_lb_score(submission_path):
    lb_score = 0
    for submission_py in os.listdir(submission_path):
        try: 
            with open(os.path.join(submission_path, submission_py), 'r') as f:
                l = len(f.read().strip())
                lb_score += max(1, 2500 - l)
        except:
            with open(os.path.join(submission_path, submission_py), 'r', encoding='L1') as f:
                l = len(f.read().strip())
                lb_score += max(1, 2500 - l)
    return lb_score

compute_lb_score("/kaggle/working/submission")

