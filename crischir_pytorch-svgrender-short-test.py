


import os
import shutil


import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


import re


%%bash

#!/bin/bash

set -eu

# Detect the shell from which the script was called
parent=$(ps -o comm $PPID |tail -1)
parent=${parent#-}  # remove the leading dash that login shells have
case "$parent" in
  # shells supported by `micromamba shell init`
  bash|fish|xonsh|zsh)
    shell=$parent
    ;;
  *)
    # use the login shell (basename of $SHELL) as a fallback
    shell=${SHELL##*/}
    ;;
esac

# Define default values to avoid user input
BIN_FOLDER="${BIN_FOLDER:-${HOME}/.local/bin}"
INIT_YES="${INIT_YES:-yes}"  # Automatically initialize shell
CONDA_FORGE_YES="${CONDA_FORGE_YES:-yes}"  # Automatically configure conda-forge
PREFIX_LOCATION="${PREFIX_LOCATION:-${HOME}/micromamba}"  # Default prefix location

# Computing artifact location
case "$(uname)" in
  Linux)
    PLATFORM="linux" ;;
  Darwin)
    PLATFORM="osx" ;;
  *NT*)
    PLATFORM="win" ;;
esac

ARCH="$(uname -m)"
case "$ARCH" in
  aarch64|ppc64le|arm64)
      ;;  # pass
  *)
    ARCH="64" ;;
esac

case "$PLATFORM-$ARCH" in
  linux-aarch64|linux-ppc64le|linux-64|osx-arm64|osx-64|win-64)
      ;;  # pass
  *)
    echo "Failed to detect your OS" >&2
    exit 1
    ;;
esac

if [ "${VERSION:-}" = "" ]; then
  RELEASE_URL="https://github.com/mamba-org/micromamba-releases/releases/latest/download/micromamba-${PLATFORM}-${ARCH}"
else
  RELEASE_URL="https://github.com/mamba-org/micromamba-releases/releases/download/${VERSION}/micromamba-${PLATFORM}-${ARCH}"
fi

# Downloading artifact
mkdir -p "${BIN_FOLDER}"
if hash curl >/dev/null 2>&1; then
  curl "${RELEASE_URL}" -o "${BIN_FOLDER}/micromamba" -fsSL --compressed ${CURL_OPTS:-}
elif hash wget >/dev/null 2>&1; then
  wget ${WGET_OPTS:-} -qO "${BIN_FOLDER}/micromamba" "${RELEASE_URL}"
else
  echo "Neither curl nor wget was found" >&2
  exit 1
fi
chmod +x "${BIN_FOLDER}/micromamba"

# Initializing shell
case "$INIT_YES" in
  y|Y|yes)
    case $("${BIN_FOLDER}/micromamba" --version) in
      1.*|0.*)
        shell_arg=-s
        prefix_arg=-p
        ;;
      *)
        shell_arg=--shell
        prefix_arg=--root-prefix
        ;;
    esac
    "${BIN_FOLDER}/micromamba" shell init $shell_arg "$shell" $prefix_arg "$PREFIX_LOCATION"

    echo "Please restart your shell to activate micromamba or run the following:\n"
    echo "  source ~/.bashrc (or ~/.zshrc, ~/.xonshrc, ~/.config/fish/config.fish, ...)"
    ;;
  *)
    echo "You can initialize your shell later by running:"
    echo "  micromamba shell init"
    ;;
esac

# Initializing conda-forge
case "$CONDA_FORGE_YES" in
  y|Y|yes)
    "${BIN_FOLDER}/micromamba" config append channels conda-forge
    "${BIN_FOLDER}/micromamba" config append channels nodefaults
    "${BIN_FOLDER}/micromamba" config set channel_priority strict
    ;;
esac

export MAMBA_ROOT_PREFIX=~/micromamba

eval "$(/root/.local/bin/micromamba shell hook -s posix)"


%%bash
source ~/.bashrc
git clone https://github.com/jiejiezi0v0/PyTorch-SVGRender.git
cd PyTorch-SVGRender
chmod +x script/install.sh
bash script/install.sh


def printfun():
    # Define a variable to store the working space path
    working_space_path = None
    
    # Use regex to find the line containing "-> Working Space:" and extract the path
    # The pattern looks for "-> Working Space: " followed by a single quote,
    # then captures any characters until the next single quote.
    match = re.search(r"-> Working Space: '(.*)'", result.stdout)
    
    # Check if a match was found
    if match:
        # Extract the captured group (the path)
        working_space_path = match.group(1)
        print(f"Extracted Working Space Path: {working_space_path}")
    else:
        print("Working Space path not found in the provided output.")
    # Define the directory name to search for
    directory_name = 'png_logs'
    # Define the keyword that a parent directory name in the path must contain
    parent_keyword = 'diffvg'
    parent_keyword = working_space_path
    
    def find_directory_with_keyword_in_path(start_dir, target_name, keyword_in_path):
        """
        Recursively searches for a directory with the target_name where any directory
        in the path from start_dir to the target_name contains the keyword_in_path.
        Returns the full path if found, otherwise returns None.
        """
        print(f"Searching for directory '{target_name}' with '{keyword_in_path}' in its path, starting from '{start_dir}'...")
        for root, dirs, files in os.walk(start_dir):
            # Check if the target directory name is in the current list of directories
            if target_name in dirs:
                # Construct the full path to the potential target directory
                potential_target_path = os.path.join(root, target_name)
                # Check if the keyword is present anywhere in the path (case-insensitive)
                if keyword_in_path.lower() in potential_target_path.lower():
                    print(f"Found matching directory at: {potential_target_path}")
                    return potential_target_path
                else:
                    print(f"Found '{target_name}' in '{root}', but the path '{potential_target_path}' does not contain '{keyword_in_path}'. Continuing search.")
    
        print(f"Directory '{target_name}' with '{keyword_in_path}' in its path not found in '{start_dir}' or its subdirectories.")
        return None
    
    # Get the current working directory in Kaggle
    # In a Kaggle notebook, the working directory is usually /kaggle/working/
    current_directory = os.getcwd()
    
    # Search for the target directory starting from the current working directory
    # Use the new function that checks for the keyword anywhere in the path
    target_directory_path = find_directory_with_keyword_in_path(current_directory, directory_name, parent_keyword)
    
    # Check if the directory was found
    if target_directory_path:
        print(f"Processing directory: {target_directory_path}")
    
        # List all files in the directory
        all_files = os.listdir(target_directory_path)
    
        # Filter for potential image files (you might want to add more extensions)
        image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
        if not image_files:
            print(f"No image files found in '{directory_name}' at '{target_directory_path}'.")
        else:
            print(f"Found {len(image_files)} image files.")
    
            # Sort the image files to get a consistent order (e.g., by name)
            image_files.sort()
    
            # Select the last up to 10 files
            num_files_to_plot = min(10, len(image_files))
            # Select the last 'num_files_to_plot' files
            latest_image_files = image_files[-num_files_to_plot:]
    
    
            print(f"Plotting the last {num_files_to_plot} image files.")
    
            # Plot the selected images
            # Adjust figure size based on the number of images to plot
            # Calculate rows needed for plotting, assuming up to 5 columns
            cols = min(num_files_to_plot, 5)
            rows = (num_files_to_plot + cols - 1) // cols if cols > 0 else 0
            fig_height = 5 * rows if rows > 0 else 5 # Default height if no images
    
            plt.figure(figsize=(15, fig_height))
    
            for i, image_file_name in enumerate(latest_image_files):
                image_path = os.path.join(target_directory_path, image_file_name)
    
                try:
                    # Read the image
                    img = mpimg.imread(image_path)
    
                    # Create a subplot - dynamically adjust grid based on number of images
                    if rows > 0 and cols > 0:
                        plt.subplot(rows, cols, i + 1)
                        plt.imshow(img)
                        plt.title(image_file_name, fontsize=8, wrap=True) # Wrap long titles
                        plt.axis('off') # Hide axes
                    else:
                         print(f"Skipping plot for {image_file_name} due to zero rows or columns.")
    
    
                except Exception as e:
                    print(f"Could not read or plot image {image_file_name}: {e}")
    
            plt.tight_layout() # Adjust layout to prevent titles overlapping
            plt.show()
    
    else:
        # Message is already printed by the find_directory_with_keyword_in_path function
        pass


%%time
import subprocess

result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=diffvg', "target='/kaggle/input/lighthouseimage/__results___28_1.png'", 'x.num_paths=512', 'x.num_iter=20'], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
print('stdout: ', result.stdout)
print('stderr: ', result.stderr)


printfun()


%%time
import subprocess

result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=diffvg', "target='/kaggle/input/lighthouseimage/__results___28_1.png'", 'x.num_paths=512', 'x.num_iter=100'], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
print('stdout: ', result.stdout)
print('stderr: ', result.stderr)


printfun()


import kagglehub


pytorch_svgrender_models_pytorch_1_1_path = kagglehub.model_download(
    'crischir/pytorch-svgrender-models/PyTorch/1/1'
)


try:
    downloaded_model_root_path = kagglehub.model_download(
        'crischir/pytorch-svgrender-models/PyTorch/1/1'
    )
    print(f"Model artifact downloaded by kagglehub to: {downloaded_model_root_path}")
except Exception as e:
    print(f"Error downloading model with kagglehub: {e}")
    # Exit or handle the error if download fails
    exit()

# Define the relative path to the u2net.pth file *within* the downloaded artifact
# Based on common structure and your previous error pattern
relative_path_within_artifact = os.path.join('u2net.pth')

# Construct the full source path in the input directory
source_file_path = os.path.join(downloaded_model_root_path, relative_path_within_artifact)
print(f"Source file path in input: {source_file_path}")

# --- 2. Define the target relative path for the copy operation ---
target_relative_path = './checkpoint/u2net/u2net.pth'
print(f"Target relative path for copy: {target_relative_path}")


# --- 3. Ensure the target directory structure exists in the current working directory ---
# We need to create the directory './checkpoint/u2net/'
# os.path.dirname() gets the directory part of the path
target_directory = os.path.dirname(target_relative_path)

# os.makedirs() creates directories recursively. exist_ok=True prevents errors if they already exist.
os.makedirs(target_directory, exist_ok=True)

print(f"Ensured target directory exists relative to CWD: {target_directory}")
print(f"Current Working Directory (CWD) is: {os.getcwd()}") # Important to know for relative paths!


# --- 4. Copy the file from the source to the target relative path ---

print(f"Copying '{source_file_path}' to '{target_relative_path}'...")
try:
    # shutil.copy2 copies the file, using the target_relative_path as the destination filename
    shutil.copy2(source_file_path, target_relative_path)
    print("File copied successfully to the target relative path!")

    # --- Verification steps (optional, but good for debugging) ---
    # Construct the absolute path where the file *should* be now
    absolute_copied_path = os.path.abspath(target_relative_path)
    print(f"File should now be at absolute path: {absolute_copied_path}")
    print(f"Does file exist at this absolute path? {os.path.exists(absolute_copied_path)}")
    # --- End Verification ---

except FileNotFoundError:
    print(f"Error: Source file not found at {source_file_path}. Cannot perform copy.")
except Exception as e:
    print(f"An error occurred during the copy operation: {e}")

# --- After this code runs, the rest of your script that tries to load
# --- './checkpoint/u2net/u2net.pth' should ideally find the file,
# --- assuming the CWD doesn't change before the load attempt.


%%time
import subprocess

result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=clipasso', "target='/kaggle/input/lighthouseimage/__results___28_1.png'"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
print('stdout: ', result.stdout)
print('stderr: ', result.stderr)


printfun()


# Define the directory name to search for
directory_name = 'png_logs'

def find_directory(start_dir, target_name):
    """
    Recursively searches for a directory with the target_name starting from start_dir.
    Returns the full path if found, otherwise returns None.
    """
    print(f"Searching for directory '{target_name}' starting from '{start_dir}'...")
    for root, dirs, files in os.walk(start_dir):
        if target_name in dirs:
            found_path = os.path.join(root, target_name)
            print(f"Found directory at: {found_path}")
            return found_path
    print(f"Directory '{target_name}' not found in '{start_dir}' or its subdirectories.")
    return None

# Get the current working directory in Kaggle
# In a Kaggle notebook, the working directory is usually /kaggle/working/
current_directory = os.getcwd()

# Search for the target directory starting from the current working directory
target_directory_path = find_directory(current_directory, directory_name)

# Check if the directory was found
if target_directory_path:
    print(f"Processing directory: {target_directory_path}")

    # List all files in the directory
    all_files = os.listdir(target_directory_path)

    # Filter for potential image files (you might want to add more extensions)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in '{directory_name}'.")
    else:
        print(f"Found {len(image_files)} image files.")

        # Select up to 10 random files
        num_files_to_plot = min(10, len(image_files))
        random_image_files = random.sample(image_files, num_files_to_plot)

        print(f"Plotting {num_files_to_plot} random image files.")

        # Plot the selected images
        # Adjust figure size based on the number of images to plot
        fig_height = 5 * ((num_files_to_plot + 4) // 5) # Roughly 5 images per row
        plt.figure(figsize=(15, fig_height))

        for i, image_file_name in enumerate(random_image_files):
            image_path = os.path.join(target_directory_path, image_file_name)

            try:
                # Read the image
                img = mpimg.imread(image_path)

                # Create a subplot - dynamically adjust grid based on number of images
                rows = (num_files_to_plot + 4) // 5 # Calculate number of rows
                cols = min(num_files_to_plot, 5) # Max 5 columns per row
                plt.subplot(rows, cols, i + 1)
                plt.imshow(img)
                plt.title(image_file_name, fontsize=8, wrap=True) # Wrap long titles
                plt.axis('off') # Hide axes

            except Exception as e:
                print(f"Could not read or plot image {image_file_name}: {e}")

        plt.tight_layout() # Adjust layout to prevent titles overlapping
        plt.show()

else:
    # Message is already printed by the find_directory function
    pass


import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Define the directory name to search for
directory_name = 'png_logs'
# Define the keyword that a parent directory name in the path must contain
parent_keyword = 'clipasso'

def find_directory_with_keyword_in_path(start_dir, target_name, keyword_in_path):
    """
    Recursively searches for a directory with the target_name where any directory
    in the path from start_dir to the target_name contains the keyword_in_path.
    Returns the full path if found, otherwise returns None.
    """
    print(f"Searching for directory '{target_name}' with '{keyword_in_path}' in its path, starting from '{start_dir}'...")
    for root, dirs, files in os.walk(start_dir):
        # Check if the target directory name is in the current list of directories
        if target_name in dirs:
            # Construct the full path to the potential target directory
            potential_target_path = os.path.join(root, target_name)
            # Check if the keyword is present anywhere in the path (case-insensitive)
            if keyword_in_path.lower() in potential_target_path.lower():
                print(f"Found matching directory at: {potential_target_path}")
                return potential_target_path
            else:
                print(f"Found '{target_name}' in '{root}', but the path '{potential_target_path}' does not contain '{keyword_in_path}'. Continuing search.")

    print(f"Directory '{target_name}' with '{keyword_in_path}' in its path not found in '{start_dir}' or its subdirectories.")
    return None

# Get the current working directory in Kaggle
# In a Kaggle notebook, the working directory is usually /kaggle/working/
current_directory = os.getcwd()

# Search for the target directory starting from the current working directory
# Use the new function that checks for the keyword anywhere in the path
target_directory_path = find_directory_with_keyword_in_path(current_directory, directory_name, parent_keyword)

# Check if the directory was found
if target_directory_path:
    print(f"Processing directory: {target_directory_path}")

    # List all files in the directory
    all_files = os.listdir(target_directory_path)

    # Filter for potential image files (you might want to add more extensions)
    image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

    if not image_files:
        print(f"No image files found in '{directory_name}' at '{target_directory_path}'.")
    else:
        print(f"Found {len(image_files)} image files.")

        # Select up to 10 random files
        num_files_to_plot = min(10, len(image_files))
        random_image_files = random.sample(image_files, num_files_to_plot)

        print(f"Plotting {num_files_to_plot} random image files.")

        # Plot the selected images
        # Adjust figure size based on the number of images to plot
        # Calculate rows needed for plotting, assuming up to 5 columns
        cols = min(num_files_to_plot, 5)
        rows = (num_files_to_plot + cols - 1) // cols if cols > 0 else 0
        fig_height = 5 * rows if rows > 0 else 5 # Default height if no images

        plt.figure(figsize=(15, fig_height))

        for i, image_file_name in enumerate(random_image_files):
            image_path = os.path.join(target_directory_path, image_file_name)

            try:
                # Read the image
                img = mpimg.imread(image_path)

                # Create a subplot - dynamically adjust grid based on number of images
                if rows > 0 and cols > 0:
                    plt.subplot(rows, cols, i + 1)
                    plt.imshow(img)
                    plt.title(image_file_name, fontsize=8, wrap=True) # Wrap long titles
                    plt.axis('off') # Hide axes
                else:
                     print(f"Skipping plot for {image_file_name} due to zero rows or columns.")


            except Exception as e:
                print(f"Could not read or plot image {image_file_name}: {e}")

        plt.tight_layout() # Adjust layout to prevent titles overlapping
        plt.show()

else:
    # Message is already printed by the find_directory_with_keyword_in_path function
    pass



!pip install kornia


import kornia


# import subprocess
# import shlex # Useful for splitting command strings safely

# # The micromamba executable path from previous output
# micromamba_executable = '/root/.local/bin/micromamba'

# # The command to install kornia and albumentations into the svgrender environment
# # Added 'albumentations' to the list of packages
# # Added version specifiers to try and resolve potential compatibility issues
# # You might need to adjust these versions based on the specific requirements of pytorch_svgrender
# command_str = f"{micromamba_executable} install -n svgrender kornia<=0.6 albumentations<=1.0.0 -c conda-forge"

# # Convert the command string into a list of arguments
# # shlex.split handles quotes and spaces correctly
# command_args = shlex.split(command_str)

# print(f"Running installation command: {' '.join(command_args)}")

# try:
#     # Execute the command using subprocess.run
#     # capture_output=True captures stdout and stderr
#     # text=True decodes stdout and stderr as text
#     # check=True will raise CalledProcessError if the command fails
#     result = subprocess.run(
#         command_args,
#         capture_output=True,
#         text=True,
#         check=True
#     )

#     # Print the output from the installation process
#     print("\nstdout:")
#     print(result.stdout)
#     print("\nstderr:")
#     print(result.stderr)
#     print("\nKornia and Albumentations installation command completed successfully.")
#     print("Please try running your svg_render.py script again.")


# except subprocess.CalledProcessError as e:
#     print(f"\nError during package installation:")
#     print(f"Command failed with exit code {e.returncode}")
#     print("stdout:", e.stdout)
#     print("stderr:", e.stderr)
# except FileNotFoundError:
#     print(f"\nError: The micromamba executable was not found at {micromamba_executable}.")
#     print("Please ensure micromamba is installed and the path is correct.")
# except Exception as e:
#     print(f"\nAn unexpected error occurred: {e}")





# %%time
# result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=clipascene', "target='/kaggle/input/lighthouseimage/__results___28_1.png'"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
# print('stdout: ', result.stdout)
# print('stderr: ', result.stderr)


#printfun()


%%time
result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=clipdraw',"prompt='a photo of a cat'"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
print('stdout: ', result.stdout)
print('stderr: ', result.stderr)


printfun()


%%time
import subprocess

result = subprocess.run(
    [
        '/root/.local/bin/micromamba', 'run', '-n', 'svgrender', 'python',
        '/kaggle/working/PyTorch-SVGRender/svg_render.py',
        'x=diffsketcher',
        "prompt='a photo of a cat'",
        'x.token_ind=5',
        'seed=801',
        'x.num_paths=96',
        'diffuser.download=True',
        'x.enable_xformers=False'  # Add this argument to disable xformers
        # Add back any other specific arguments you were using
    ],
    capture_output=True,
    text=True,
    executable='/root/.local/bin/micromamba'
)

print("stdout:", result.stdout)
print("stderr:", result.stderr)


printfun()


%%time
result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=vectorfusion',"x.style='iconography'","prompt='a lighthouse near ocean,minimal flat 2d vector icon. lineal color. trending on artstation.'",   'x.enable_xformers=False'], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
print('stdout: ', result.stdout)
print('stderr: ', result.stderr)


printfun()


%%time
result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=styleclipdraw',"prompt='a lighthouse near ocean'", "target='/kaggle/input/lighthouseimage/__results___28_1.png'"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
print('stdout: ', result.stdout)
print('stderr: ', result.stderr)


printfun()


# import subprocess
# import shlex

# # The micromamba executable path (adjust if yours is different)
# micromamba_executable = '/root/.local/bin/micromamba'

# # The Python code to execute for clearing CUDA memory
# # We include imports and the necessary calls
# python_clear_script = """
# import torch
# import gc

# print('Attempting to clear CUDA memory...')
# try:
#     # Run Python's garbage collector
#     gc.collect()

#     # Empty the PyTorch CUDA cache
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         print('CUDA memory cleared.')
#     else:
#         print('CUDA not available, skipping memory clear.')
# except Exception as e:
#     print(f'Error during CUDA memory clear: {e}')
# """

# # Construct the command to run the Python script using micromamba
# # We use python -c "..." to execute the script string directly
# command_str = f"{micromamba_executable} run -n svgrender python -c \"{python_clear_script}\""

# # Split the command string into arguments
# command_args = shlex.split(command_str)

# print(f"Running memory clearing command: {' '.join(command_args)}")

# try:
#     # Execute the command
#     result = subprocess.run(
#         command_args,
#         capture_output=True,
#         text=True,
#         check=False # Set to False so it doesn't raise an error if the clear fails for some reason
#     )

#     # Print the output
#     print("\nstdout:")
#     print(result.stdout)
#     print("\nstderr:")
#     print(result.stderr)

# except FileNotFoundError:
#     print(f"\nError: The micromamba executable was not found at {micromamba_executable}.")
#     print("Please ensure micromamba is installed and the path is correct.")
# except Exception as e:
#     print(f"\nAn unexpected error occurred: {e}")



# %%time
# result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=svgdreamer', 'state.mprec=fp16', "prompt='A colorful German shepherd in vector art. tending on artstation.'", "save_step=50", "x.guidance.n_particle=6", "x.guidance.vsd_n_particle=4", "x.guidance.phi_n_particle=2", "result_path='svgdreamer/GermanShepherd'","x.enable_xformers=False"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
# print('stdout: ', result.stdout)
# print('stderr: ', result.stderr)


# import random
# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg

# # Define the directory name to search for
# directory_name = 'png_logs'
# # Define the keyword that a parent directory name in the path must contain
# parent_keyword = 'GermanShepherd'

# def find_directory_with_keyword_in_path(start_dir, target_name, keyword_in_path):
#     """
#     Recursively searches for a directory with the target_name where any directory
#     in the path from start_dir to the target_name contains the keyword_in_path.
#     Returns the full path if found, otherwise returns None.
#     """
#     print(f"Searching for directory '{target_name}' with '{keyword_in_path}' in its path, starting from '{start_dir}'...")
#     for root, dirs, files in os.walk(start_dir):
#         # Check if the target directory name is in the current list of directories
#         if target_name in dirs:
#             # Construct the full path to the potential target directory
#             potential_target_path = os.path.join(root, target_name)
#             # Check if the keyword is present anywhere in the path (case-insensitive)
#             if keyword_in_path.lower() in potential_target_path.lower():
#                 print(f"Found matching directory at: {potential_target_path}")
#                 return potential_target_path
#             else:
#                 print(f"Found '{target_name}' in '{root}', but the path '{potential_target_path}' does not contain '{keyword_in_path}'. Continuing search.")

#     print(f"Directory '{target_name}' with '{keyword_in_path}' in its path not found in '{start_dir}' or its subdirectories.")
#     return None

# # Get the current working directory in Kaggle
# # In a Kaggle notebook, the working directory is usually /kaggle/working/
# current_directory = os.getcwd()

# # Search for the target directory starting from the current working directory
# # Use the new function that checks for the keyword anywhere in the path
# target_directory_path = find_directory_with_keyword_in_path(current_directory, directory_name, parent_keyword)

# # Check if the directory was found
# if target_directory_path:
#     print(f"Processing directory: {target_directory_path}")

#     # List all files in the directory
#     all_files = os.listdir(target_directory_path)

#     # Filter for potential image files (you might want to add more extensions)
#     image_files = [f for f in all_files if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

#     if not image_files:
#         print(f"No image files found in '{directory_name}' at '{target_directory_path}'.")
#     else:
#         print(f"Found {len(image_files)} image files.")

#         # Select up to 10 random files
#         num_files_to_plot = min(10, len(image_files))
#         random_image_files = random.sample(image_files, num_files_to_plot)

#         print(f"Plotting {num_files_to_plot} random image files.")

#         # Plot the selected images
#         # Adjust figure size based on the number of images to plot
#         # Calculate rows needed for plotting, assuming up to 5 columns
#         cols = min(num_files_to_plot, 5)
#         rows = (num_files_to_plot + cols - 1) // cols if cols > 0 else 0
#         fig_height = 5 * rows if rows > 0 else 5 # Default height if no images

#         plt.figure(figsize=(15, fig_height))

#         for i, image_file_name in enumerate(random_image_files):
#             image_path = os.path.join(target_directory_path, image_file_name)

#             try:
#                 # Read the image
#                 img = mpimg.imread(image_path)

#                 # Create a subplot - dynamically adjust grid based on number of images
#                 if rows > 0 and cols > 0:
#                     plt.subplot(rows, cols, i + 1)
#                     plt.imshow(img)
#                     plt.title(image_file_name, fontsize=8, wrap=True) # Wrap long titles
#                     plt.axis('off') # Hide axes
#                 else:
#                      print(f"Skipping plot for {image_file_name} due to zero rows or columns.")


#             except Exception as e:
#                 print(f"Could not read or plot image {image_file_name}: {e}")

#         plt.tight_layout() # Adjust layout to prevent titles overlapping
#         plt.show()

# else:
#     # Message is already printed by the find_directory_with_keyword_in_path function
#     pass


#printfun()


# %%time
# # result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=svgdreamer','state.mprec='fp16'', "prompt='A colorful German shepherd in vector art. tending on artstation.. low-ploy. polygon'", "x.style='low-poly'","x.guidance.n_particle=6","save_step=50", "x.guidance.n_particle=6", "x.guidance.vsd_n_particle=4","x.guidance.phi_n_particle=2","x.grid=30","x.guidance.num_iter=1000","result_path='svgdreamer/BaldEagle'","x.enable_xformers=False"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
# result = subprocess.run(['micromamba', 'run', '-n', 'svgrender', 'python', '/kaggle/working/PyTorch-SVGRender/svg_render.py', 'x=svgdreamer', 'state.mprec="fp16"', "prompt='A colorful German shepherd in vector art. tending on artstation.. low-ploy. polygon'", "x.style='low-poly'","x.guidance.n_particle=6","save_step=50", "x.guidance.n_particle=6", "x.guidance.vsd_n_particle=4","x.guidance.phi_n_particle=2","x.grid=30","x.guidance.num_iter=1000","result_path='svgdreamer/BaldEagle'","x.enable_xformers=False"], capture_output=True, text=True, executable='/root/.local/bin/micromamba')
# print('stdout: ', result.stdout)
# print('stderr: ', result.stderr)


# printfun()




