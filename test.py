import psutil
import GPUtil

def get_memory_info():
    memory = psutil.virtual_memory()
    print(f"Total Memory: {memory.total / (1024 ** 3):.2f} GB")
    print(f"Available Memory: {memory.available / (1024 ** 3):.2f} GB")
    print(f"Used Memory: {memory.used / (1024 ** 3):.2f} GB")
    print(f"Memory Usage: {memory.percent}%")

def get_gpu_info():
    gpus = GPUtil.getGPUs()
    for gpu in gpus:
        print(f"GPU: {gpu.name}")
        print(f"  Load: {gpu.load * 100:.2f}%")
        print(f"  Free Memory: {gpu.memoryFree:.2f} MB")
        print(f"  Used Memory: {gpu.memoryUsed:.2f} MB")
        print(f"  Total Memory: {gpu.memoryTotal:.2f} MB")
        print(f"  Temperature: {gpu.temperature} °C")

if __name__ == "__main__":
    print("Memory Information:")
    get_memory_info()
    print("\nGPU Information:")
    get_gpu_info()