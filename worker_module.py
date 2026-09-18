# worker_module.py
from multiprocessing import shared_memory
import numpy as np

def process_array(shared_name, shape, dtype, index):
    existing_shm = shared_memory.SharedMemory(name=shared_name)
    array = np.ndarray(shape, dtype=dtype, buffer=existing_shm.buf)
    array[index] *= 2
    print(f"Processed array at index {index}:\n{array[index]}")
    existing_shm.close()
