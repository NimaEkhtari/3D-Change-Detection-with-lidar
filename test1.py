# test1.py
from multiprocessing import Process, shared_memory
import multiprocessing as mp
import numpy as np
from worker_module import process_array

if __name__ == "__main__":
    if not hasattr(mp.get_context(), '_start_method') or mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)

    data = [np.random.randint(0, 10, (3, 3), dtype=np.int32) for _ in range(5)]

    total_size = sum(arr.nbytes for arr in data)
    shm = shared_memory.SharedMemory(create=True, size=total_size)

    try:
        shared_buffer = np.ndarray((total_size,), dtype=np.uint8, buffer=shm.buf)
        offsets, shapes, dtypes = [], [], []
        current_offset = 0

        for arr in data:
            offsets.append(current_offset)
            shapes.append(arr.shape)
            dtypes.append(arr.dtype)
            shared_buffer[current_offset:current_offset + arr.nbytes] = np.frombuffer(arr.tobytes(), dtype=np.uint8)
            current_offset += arr.nbytes

        processes = []
        for idx, (offset, shape, dtype) in enumerate(zip(offsets, shapes, dtypes)):
            p = Process(target=process_array, args=(shm.name, shape, dtype, idx))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

        for idx, (offset, shape, dtype) in enumerate(zip(offsets, shapes, dtypes)):
            result_array = np.ndarray(shape, dtype=dtype, buffer=shm.buf, offset=offset)
            print(f"Reconstructed array {idx}:\n{result_array}")

    finally:
        shm.close()
        shm.unlink()
