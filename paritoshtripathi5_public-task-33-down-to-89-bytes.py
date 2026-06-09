import os
import shutil

# Define source and destination base directories
src_dir = "/kaggle/input/google-code-golf-full-400-solutions-verification/submission"
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


%%writefile /kaggle/working/submission/task033.py
z=range(17)
p=lambda g:[[g[y][x]if g[y][x]==g[y%6][x%6]else g[5][0]for x in z]for y in z]


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

