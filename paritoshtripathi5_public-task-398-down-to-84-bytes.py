import os
import shutil

# Define source and destination base directories
src_dir = "/kaggle/input/r30-neurips-golf-lessons-learned/submission"
dst_dir = "/kaggle/working/submission"

# Create destination folder if it doesn't exist
os.makedirs(dst_dir, exist_ok=True)

# Loop over all task files from 000 to 400
for i in range(1, 401):  # 0 through 400 inclusive
    filename = f"task{i:03d}.py"
    src_path = os.path.join(src_dir, filename)
    dst_path = os.path.join(dst_dir, filename)
    
    # Copy file if it exists
    if os.path.exists(src_path):
        shutil.copy(src_path, dst_path)
    else:
        print(f"Missing: {filename}")


%%writefile /kaggle/working/submission/task398.py
def p(r):R=r[0];S=len({*R}-{0})*5;return[([0]*d+R+[0]*S)[:S]for d in range(S)][::-1]


import os
import zipfile

# Define output zip and folder containing files
output_zip = "submission.zip"
source_dir = "/kaggle/working/submission"  # Change this path as needed

# Create ZIP file
with zipfile.ZipFile(output_zip, "w", zipfile.ZIP_DEFLATED) as zipf:
    for f in range(1, 401):
        filename = f"task{f:03d}.py"
        filepath = os.path.join(source_dir, filename)
        if os.path.exists(filepath):
            zipf.write(filepath, arcname=filename)
        else:
            print(f"Warning: {filename} not found, skipping.")

print("Submission file generated:", output_zip)

