

import numpy as np


def save_large_csv(data, path, chunk_size=10000):
    """Save large DataFrame to CSV in chunks."""
    num_chunks = int(np.ceil(data.shape[0] / chunk_size))
    for i in range(num_chunks):
        start = i * chunk_size
        end = (i + 1) * chunk_size
        chunk = data[start:end]
        if i == 0:
            chunk.to_csv(path, index=False, mode='w', header=True)
        else:
            chunk.to_csv(path, index=False, mode='a', header=False)