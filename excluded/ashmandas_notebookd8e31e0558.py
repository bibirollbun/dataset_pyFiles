
!git clone https://github.com/IronManRox228228/local-agentic-blog-writer.git

import os
print("ğŸ“‚ Project Structure:")
for root, dirs, files in os.walk("local-agentic-blog-writer"):
    level = root.replace("local-agentic-blog-writer", "").count(os.sep)
    indent = " " * 4 * (level)
    print(f"{indent}{os.path.basename(root)}/")
    subindent = " " * 4 * (level + 1)
    for f in files:
        print(f"{subindent}{f}")

