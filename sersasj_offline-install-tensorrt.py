!mkdir packages
!pip download -d ./packages tensorrt-cu12 tensorrt-cu12-bindings tensorrt-cu12-libs --extra-index-url https://pypi.nvidia.com
!pip download -d ./packages onnxruntime-gpu onnxslim
!tar -czvf packages.tar.gz ./packages


