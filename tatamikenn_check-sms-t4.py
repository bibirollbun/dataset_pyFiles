! nvidia-smi


! nvcc -V


import pycuda.driver as cuda
import pycuda.autoinit

device = cuda.Device(0)  # 0番目のGPU
props = device.get_attributes()
print(f"Streaming Multiprocessors: {props[cuda.device_attribute.MULTIPROCESSOR_COUNT]}")

