!cp /kaggle/input/llama-server-l4/packages/var/cache/apt/archives/* .


!dpkg-scanpackages . /dev/null | gzip -9c > Packages.gz


!rm /etc/apt/sources.list


!echo "deb [trusted=yes] file:$(pwd) ./" | sudo tee /etc/apt/sources.list


!sudo apt-get update -o Dir::Etc::sourcelist="sources.list" -o Dir::Etc::sourceparts="-"


!sudo apt install cuda-compat-12-2


!sudo apt install nvidia-cuda-dev


!rm *.deb


!cp -r /kaggle/input/llama-ccp/llama.cpp .


!rm -r llama.cpp/build


!cd llama.cpp/examples/server/public && gzip -k index.html loading.html


!CUDA_HOME=/usr/local/cuda-12.2 PATH=$CUDA_HOME/bin:$PATH LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH cmake llama.cpp -B llama.cpp/build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="89-real" -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.2/bin/nvcc


!cmake --build llama.cpp/build --config Release -j --clean-first --target llama-server


!cp llama.cpp/build/bin/llama-* /kaggle/working/


# !CUDA_VISIBLE_DEVICES=0 ./llama-server --model /kaggle/input/qwq-unsloth/QwQ-32B-Q4_K_M.gguf --threads 12 --ctx-size 14000 --n-gpu-layers 99 --seed 340 --prio 2 --temp 0.6 --repeat-penalty 1.0 --dry-multiplier 0.5 --min-p 0.1 --top-k 40 --top-p 0.95 --samplers "top_k;top_p;min_p;temperature;dry;typ_p;xtc" -fa




