!ls '/kaggle/input/install-all-notebook/packages'
wheel_dir = '/kaggle/input/install-all-notebook/packages'

import importlib
import subprocess
import sys
import os

def install_if_not_available(package, import_as=None, quiet=False):
    module_name = import_as or package

    try:
        pkg = importlib.import_module(module_name)
    except ImportError:
        print(f"{package}: Downloading and installing locally...")

        #os.makedirs(wheel_dir, exist_ok=True) 
        # download_cmd = [
        #     sys.executable, "-m", "pip", "download", package, "-d", wheel_dir
        # ]
        install_cmd = [
            sys.executable, "-m", "pip", "install", "--no-index",
            f"--find-links={wheel_dir}", package
        ]

        if quiet:
            #subprocess.run(download_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(install_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            #subprocess.check_call(download_cmd)
            subprocess.check_call(install_cmd)

        pkg = importlib.import_module(module_name)

    version = getattr(pkg, "__version__", None)
    if version is not None:
        print(f"{module_name} version: {version}")
    else:
        print(f"{module_name} imported (version unknown)")

    return pkg

# try:
#     import mordred
#     print('mordred', mordred.__version)__)
# else:
#     !pip insall mordredcommunity

print('')
print('INSTALL HELPER OK!')


mordred = install_if_not_available("mordredcommunity", "mordred")

if 1:   
    print('')
    print('let\'s check installion ...')
    from mordred import Calculator, descriptors
    num_descriptor_2d3d = len(Calculator(descriptors, ignore_3D=False).descriptors)
    print('num_descriptor_2d3d:', num_descriptor_2d3d)

