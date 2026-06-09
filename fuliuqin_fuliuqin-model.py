import sys
import os
import shutil

# å®‰è£…RDKit
!pip install rdkit-pypi

# è�·å�–å½“å‰�Pythonç‰ˆæœ¬
python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
print(f"å½“å‰�Pythonç‰ˆæœ¬: {python_version}")

# æ�„å»ºsite-packagesè·¯å¾„
site_packages_path = f"/usr/local/lib/python{python_version}/dist-packages"
print(f"site-packagesè·¯å¾„: {site_packages_path}")

# åˆ›å»ºå­˜æ”¾ç›®å½•
output_dir = "/kaggle/working/offline_deps/rdkit"
os.makedirs(output_dir, exist_ok=True)

# éœ€è¦�å¤�åˆ¶çš„æ ¸å¿ƒç›®å½•å’Œæ–‡ä»¶
required_dirs = [
    "rdkit",  # ä¸»åŒ…ç›®å½•
    "rdkit/Chem",  # åŒ–å­¦æ¨¡å�—
    "rdkit/Data",  # æ•°æ�®æ–‡ä»¶
]

required_files = [
    # æ ¸å¿ƒC++åº“
    "rdkit/rdBase.so",
    "rdkit/rdChemReactions.so",
    "rdkit/rdchem.so",
    "rdkit/rdMolDescriptors.so",
    "rdkit/rdDepictor.so",
    "rdkit/rdForceFieldHelpers.so",
    "rdkit/rdGeometry.so",
    "rdkit/rdmolops.so",
    "rdkit/rdmolfiles.so",
    "rdkit/rdFingerprintGenerator.so",
    # Pythonæ¨¡å�—
    "rdkit/__init__.py",
    "rdkit/Chem/__init__.py",
    "rdkit/Chem/Descriptors.py",
    "rdkit/Chem/rdMolDescriptors.py",
]

# å¤�åˆ¶å¿…è¦�çš„ç›®å½•
for dir_path in required_dirs:
    src = os.path.join(site_packages_path, dir_path)
    dst = os.path.join(output_dir, dir_path)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copytree(src, dst, dirs_exist_ok=True)
        print(f"âœ… å¤�åˆ¶ç›®å½•: {dir_path}")
    else:
        print(f"â�Œ ç›®å½•ä¸�å­˜åœ¨: {dir_path}")

# å¤�åˆ¶å¿…è¦�çš„æ–‡ä»¶
for file_path in required_files:
    src = os.path.join(site_packages_path, file_path)
    dst = os.path.join(output_dir, file_path)
    if os.path.exists(src):
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)
        print(f"âœ… å¤�åˆ¶æ–‡ä»¶: {file_path}")
    else:
        print(f"â�Œ æ–‡ä»¶ä¸�å­˜åœ¨: {file_path}")

# å¤�åˆ¶ä¾�èµ–çš„Booståº“ï¼ˆå…³é”®æ­¥éª¤ï¼‰
print("\nğŸ”� æŸ¥æ‰¾å¹¶å¤�åˆ¶Boostä¾�èµ–åº“...")
boost_dir = os.path.join(output_dir, "boost")
os.makedirs(boost_dir, exist_ok=True)

# ä½¿ç”¨lddå‘½ä»¤æ£€æŸ¥ä¾�èµ–
!ldd {os.path.join(output_dir, 'rdkit', 'rdBase.so')} > /kaggle/working/rdkit_deps.txt
with open("/kaggle/working/rdkit_deps.txt", "r") as f:
    deps = f.readlines()

# å¤�åˆ¶æ‰€æœ‰ä»¥libboost_pythonå¼€å¤´çš„åº“
for line in deps:
    if "libboost_python" in line:
        parts = line.split("=>")
        if len(parts) > 1:
            lib_path = parts[1].split("(")[0].strip()
            if os.path.exists(lib_path):
                shutil.copy2(lib_path, boost_dir)
                print(f"âœ… å¤�åˆ¶Booståº“: {os.path.basename(lib_path)}")

print(f"\nğŸ�‰ RDKitæ ¸å¿ƒæ¨¡å�—å·²æˆ�åŠŸä¸‹è½½åˆ°: {output_dir}")
print("ç›®å½•å†…å®¹:", os.listdir(output_dir))
print("å�¯ä»¥å°†æ­¤ç›®å½•ä¸Šä¼ åˆ°Kaggle Datasetå¹¶é€šè¿‡ä»¥ä¸‹æ–¹å¼�å¯¼å…¥:")
print("""
import sys
import os
sys.path.append("/kaggle/input/your-dataset/offline_deps/rdkit")
os.environ["LD_LIBRARY_PATH"] += ":/kaggle/input/your-dataset/offline_deps/rdkit/boost"
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors
""")


import zipfile
import os

# å�‹ç¼©Boostç›®å½•
boost_zip = "/kaggle/working/boost_libs.zip"
with zipfile.ZipFile(boost_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
    for root, dirs, files in os.walk("/kaggle/working/rdkit_with_boost/boost"):
        for file in files:
            file_path = os.path.join(root, file)
            zipf.write(file_path, os.path.relpath(file_path, "/kaggle/working/rdkit_with_boost/boost"))

print(f"Booståº“å·²å�‹ç¼©ä¸º: {boost_zip}")


# å®‰è£…transformersï¼ˆç¡®ä¿�ç‰ˆæœ¬å…¼å®¹ï¼‰
!pip install transformers

# ä¸‹è½½æ¨¡å�‹
from transformers import AutoTokenizer, AutoModel
import os

model_dir = "/kaggle/working/offline_deps/chemberta"
os.makedirs(model_dir, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")
model = AutoModel.from_pretrained("seyonec/ChemBERTa-zinc-base-v1")

# ä¿�å­˜æ¨¡å�‹åˆ°æœ¬åœ°
tokenizer.save_pretrained(model_dir)
model.save_pretrained(model_dir)


# import os
# import sys

# # è§£å�‹é¢„å®‰è£…çš„RDKitåŒ…
# !unzip -o /kaggle/input/rdkit-for-kaggle/rdkit-package.zip -d /tmp/

# # æ·»åŠ åˆ°Pythonè·¯å¾„
# sys.path.append('/tmp/rdkit_install')

# # éªŒè¯�å®‰è£…
# try:
#     import rdkit
#     print(f"RDKitç‰ˆæœ¬: {rdkit.__version__}")
# except ImportError:
#     print("RDKitå¯¼å…¥å¤±è´¥ï¼Œè¯·æ£€æŸ¥è·¯å¾„æˆ–é‡�æ–°åˆ›å»ºæ•°æ�®é›†ã€‚")





# import os
# import sys
# from transformers import AutoTokenizer, AutoModel

# # è®¾ç½®è®¾å¤‡
# device = "cuda" if torch.cuda.is_available() else "cpu"

# # è§£å�‹æ¨¡å�‹æ–‡ä»¶
# print("è§£å�‹ChemBERTaæ¨¡å�‹...")
# !unzip -o /kaggle/input/chemberta-model-offline/chemberta-model.zip -d /tmp/chemberta/

# # åŠ è½½æ¨¡å�‹å’Œtokenizer
# print("åŠ è½½ChemBERTaæ¨¡å�‹...")
# tokenizer = AutoTokenizer.from_pretrained("/tmp/chemberta")
# model = AutoModel.from_pretrained("/tmp/chemberta").to(device)
# model.eval()

# print("æ¨¡å�‹åŠ è½½å®Œæˆ�ï¼Œå�¯ä»¥ä½¿ç”¨äº†ï¼�")




