!pip install psutil
# It provides a Process class that allows us to check the memory usage of the current process as follows:
import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")
