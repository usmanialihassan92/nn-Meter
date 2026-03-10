import sys
import platform
import subprocess

print("===== SYSTEM INFO =====")
print("Python version:", sys.version)
print("Platform:", platform.platform())
print("Processor:", platform.processor())

print("\n===== PYTORCH INFO =====")
try:
    import torch
    print("PyTorch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("CUDA version (PyTorch build):", torch.version.cuda)
    print("cuDNN version:", torch.backends.cudnn.version())

    if torch.cuda.is_available():
        print("GPU count:", torch.cuda.device_count())
        print("GPU name:", torch.cuda.get_device_name(0))
except ImportError:
    print("PyTorch not installed")

print("\n===== CUDA (SYSTEM) =====")
try:
    output = subprocess.check_output("nvcc --version", shell=True).decode()
    print(output)
except:
    print("nvcc not found (CUDA toolkit may not be installed)")

print("\n===== PIP VERSION =====")
try:
    output = subprocess.check_output("pip --version", shell=True).decode()
    print(output)
except:
    print("pip not found")

print("\n===== INSTALLED PACKAGES =====")
try:
    output = subprocess.check_output("pip list", shell=True).decode()
    print(output)
except:
    print("Could not list packages")