import zipfile
konwinski= zipfile.ZipFile('../input/konwinski-prize/data.a_zip')
konwinski.extractall()


pip_packages_path='/kaggle/working/data/pip_packages/astropy__astropy-16898'


%cd /kaggle/working/data/repos/repo__astropy__astropy-16898


!dpkg -i /kaggle/input/konwinski-prize/kprize_setup/python3.11/ubuntu_22.04/*.deb


!uv venv --python python3.11
!source .venv/bin/activate
!SETUPTOOLS_SCM_PRETEND_VERSION=1.2.3 uv pip install --no-index --find-links=$pip_packages_path --link-mode=symlink -e .[test] --verbose && SETUPTOOLS_SCM_PRETEND_VERSION=1.2.3 uv pip install --no-index --find-links=$pip_packages_path --link-mode=symlink attrs==23.1.0 exceptiongroup==1.1.3 execnet==2.0.2 hypothesis==6.82.6 iniconfig==2.0.0 numpy==1.25.2 packaging==23.1 pluggy==1.3.0 psutil==5.9.5 pyerfa==2.0.0.3 pytest-arraydiff==0.5.0 pytest-astropy-header==0.2.2 pytest-astropy==0.10.0 pytest-cov==4.1.0 pytest-doctestplus==1.0.0 pytest-filter-subpackage==0.1.2 pytest-mock==3.11.1 pytest-openfiles==0.5.0 pytest-remotedata==0.4.0 pytest-xdist==3.3.1 pytest==7.4.0 PyYAML==6.0.1 setuptools==68.0.0 sortedcontainers==2.4.0 tomli==2.0.1


%pwd


!pip uninstall -y astropy
!pip show astropy


%%file test.py
import numpy as np
import astropy

from astropy.io import fits
from astropy.table import QTable, Table

print('numpy', np.__version__)
print('astropy',astropy.__version__)
data = np.array([("", 12)], dtype=[("a", "S"), ("b", "i4")])
fits.BinTableHDU(data).writeto("zerodtable.fits", overwrite=True)
t = Table.read("zerodtable.fits")
print(t)


!uv run pip show astropy


!uv run python test.py

