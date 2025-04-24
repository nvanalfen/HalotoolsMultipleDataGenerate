import multiprocessing as mp
import numpy as np
import os

# Testing doing multiprocessing in mpi with dummy examples
# This should simulate multiprocessing correlations on an mpi rank

def func1(x1, x2):
    """
    Dummy function to simulate some computation.
    """
    return x1 + x2

def func2(x1, x2):
    """
    Dummy function to simulate some computation.
    """
    return x1 * x2

def func3(x1, x2):
    """
    Dummy function to simulate some computation.
    """
    return x1 - x2

def calculate(row):
    func, args = row
    return func(*args)

def calculate_parallel(x1, x2):

    pairs = [
        (func1, (x1, x2)),
        (func2, (x1, x2)),
        (func3, (x1, x2))
    ]

    with mp.Pool(processes=2) as pool:
        results = pool.map(calculate, pairs)
    return results

def calculate_all(x1s, x2s):
    results = []
    for x1, x2 in zip(x1s, x2s):
        result = calculate_parallel(x1, x2)
        results.append(result)
    return np.array(results)

def determine_size(shape, rank, min_rank, max_rank):
    """
    Determine the size of the array to be sent to each rank.
    This function is used to determine the start and end indices for each rank.
    these indices will determine buffer size.
    """
    rows, cols = shape
    # split up the indices by which rank will be in charge
    chunk_size = int(rows // (max_rank - min_rank + 1))

    assert rank >= min_rank, "Rank must be greater than lowest rank"

    start = (rank - min_rank) * chunk_size
    end = min(start + chunk_size, rows)

    # If this is the last rank, it will take care of the rest of the rows
    if rank == max_rank:
        end = rows
    return start, end, cols

def root(comm):
    # Make dummy inputs
    x1 = np.ones((10,5))*np.arange(10)[:,np.newaxis]
    x2 = x1-1

    # Broadcast shape of the data
    shape = np.array(x1.shape)
    comm.Bcast(shape, root=0)

    ownership = {}

    # Send the data to the other ranks
    requests = []
    for i in range(1, comm.Get_size()):
        start, end, _ = determine_size(x1.shape, i, 0, comm.Get_size()-1)
        req1 = comm.Isend(x1[start:end], dest=i, tag=0)
        req_2 = comm.Isend(x2[start:end], dest=i, tag=1)
        ownership[i] = (start, end)
        requests.append(req1)
        requests.append(req_2)
    
    # Wait for all sends to complete
    MPI.Request.Waitall(requests)

    # Root's inputs
    start, end, cols = determine_size(x1.shape, comm.Get_rank(), 0, comm.Get_size()-1)
    root_x1 = x1[start:end]
    root_x2 = x2[start:end]

    # Allocate master result array
    results = np.zeros((x1.shape[0], 3, x1.shape[1]), dtype=float)
    results[start:end] = calculate_all(root_x1, root_x2)

    # Receive results from other ranks
    requests = []
    for i in range(1, comm.Get_size()):
        start, end = ownership[i]
        buffer = results[start:end]
        req = comm.Irecv(buffer, source=i, tag=0)
        requests.append(req)

    # Wait for all sends to complete
    MPI.Request.Waitall(requests)

    print("X1:", flush=True)
    print(x1, flush=True)
    print("\n>>>>>>>>>>", flush=True)
    print("X2:", flush=True)
    print(x2, flush=True)
    print("\n>>>>>>>>>>", flush=True)
    print("Results:", flush=True)
    print(results, flush=True)

    return 0


def nonroot(comm):
    # Get broadcast shape

    shape = np.empty(2, dtype=int)
    comm.Bcast(shape, root=0)

    # Get the data from the root
    start, end, cols = determine_size(shape, comm.Get_rank(), 0, comm.Get_size()-1)
    x1 = np.empty((end-start, cols), dtype=float)
    x2 = np.empty((end-start, cols), dtype=float)
    req1 = comm.Irecv(x1, source=0, tag=0)
    req2 = comm.Irecv(x2, source=0, tag=1)
    requests = [req1, req2]
    MPI.Request.Waitall(requests)

    results = calculate_all(x1, x2)
    req = comm.Isend(results, dest=0, tag=0)
    req.Wait()

    return 0

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        return root(comm)
    else:
        return nonroot(comm)

if __name__ == "__main__":
    import sys


    spawn = False
    if len(sys.argv) > 1:
        spawn = bool(int(sys.argv[1]))

    if spawn:
        mp.set_start_method("spawn", force=True)

    from mpi4py import MPI

    main()
    
