!pip install psutil
# It provides a Process class that allows us to check the memory usage of the current process as follows:
import psutil

# Process.memory_info is expressed in bytes, so convert to megabytes
print(f"RAM used: {psutil.Process().memory_info().rss / (1024 * 1024):.2f} MB")

# Here the rss attribute refers to the resident set size, 
# which is the fraction of memory that a process occupies in RAM. 
# This measurement also includes the memory used by the Python interpreter and the libraries we’ve loaded, 
# so the actual amount of memory used to load the dataset is a bit smaller. 
