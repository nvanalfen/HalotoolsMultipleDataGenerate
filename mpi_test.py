from mpi4py import MPI
import os
import numpy as np

# A dummy test based on the mpi example in https://github.com/akkornel/mpi4py/blob/master/mpi4.py

def determine_size(shape, rank, lowest_rank=1, highest_rank=1):
    rows, cols = shape
    # split up the indices by which rank will be in charge
    chunk_size = int(rows // (highest_rank - lowest_rank + 1))

    assert rank >= lowest_rank, "Rank must be greater than lowest rank"

    start = (rank - lowest_rank) * chunk_size
    end = min(start + chunk_size, rows)

    # If this is the last rank, it will take care of the rest of the rows
    if rank == highest_rank:
        end = rows
    return start, end, cols

def main() -> int:
    # This function is the exact same as the main in the example code. Pretty simple and easy. I like it.

    # Get our MPI communicator, our rank, and the world size.
    mpi_comm = MPI.COMM_WORLD
    mpi_rank = mpi_comm.Get_rank()
    mpi_size = mpi_comm.Get_size()

    # Do we only have one process?  If yes, then exit.
    if mpi_size == 1:
        print('You are running an MPI program with only one slot/task!')
        print('Are you using `mpirun` (or `srun` when in SLURM)?')
        print('If you are, then please use an `-n` of at least 2!')
        print('(Or, when in SLURM, use an `--ntasks` of at least 2.)')
        print('If you did all that, then your MPI setup may be bad.')
        return 1

    # Is our world size over 999?  The output will be a bit weird.
    # NOTE: Only rank zero does anything, so we don't get duplicate output.
    if mpi_size >= 1000 and mpi_rank == 0:
        print('WARNING:  Your world size {} is over 999!'.format(mpi_size))
        print("The output formatting will be a little weird, but that's it.")

    # Sanity checks complete!

    # Call the appropriate function, based on our rank
    if mpi_rank == 0:
        return mpi_root(mpi_comm)
    else:
        return mpi_nonroot(mpi_comm)

def mpi_root(mpi_comm):

    # The input array will be global so there's no need to broadcast it
    ins = np.ones((15, 3)).astype(float)
    values = np.arange(len(ins)).astype(float)
    ins *= values[:, np.newaxis]

    # Print the input array
    print('Rank {}:  I am the root rank!'.format(mpi_comm.Get_rank()), flush=True)
    print('Rank {}:  The inputs are shape {}'.format(mpi_comm.Get_rank(), ins.shape), flush=True)
    print("Rank {}: Inputs:".format(mpi_comm.Get_rank()), flush=True)
    print(ins, flush=True)

    # Broadcast the input shape to all ranks
    shape = np.array(ins.shape)
    mpi_comm.Bcast(shape, root=0)

    rank_ownership = {}

    # Send the chunked inputs to each rank
    requests = []
    for i in range(1, mpi_comm.Get_size()):
        # Get the start and end indices for this rank
        start_ind, end_ind, cols = determine_size(ins.shape, i, lowest_rank=1, highest_rank=mpi_comm.Get_size()-1)
        rank_ownership[i] = (start_ind, end_ind)

        # Send the inputs to the rank
        sendbuf = ins[start_ind:end_ind]
        print("Rank {}:  I am rank {} and I will send rank {} inputs:\n{}".format(mpi_comm.Get_rank(), 0, i, sendbuf), flush=True)
        req = mpi_comm.ISend(sendbuf, dest=i, tag=0)
        requests.append(req)

    # Wait for all sends to complete
    MPI.Request.Waitall(requests)
    print("Rank {}:  I am rank {} and I sent all inputs!".format(mpi_comm.Get_rank(), 0), flush=True)

    # TODO

    results = np.zeros(ins.shape, dtype=float)
    requests = []
    for i in range(1, mpi_comm.Get_size()):
        start_ind, end_ind = rank_ownership[i]
        buffer = results[start_ind:end_ind]
        req = mpi_comm.Irecv(buffer, source=i, tag=i)
        requests.append(req)

    # Wait for all receives to complete
    MPI.Request.Waitall(requests)
    print("Rank {}:  I am rank {} and I received all responses!".format(mpi_comm.Get_rank(), 0), flush=True)
    print("Responses:\n", results, flush=True)

    return 0
    

def mpi_nonroot(mpi_comm):
    # Get our MPI rank.
    # This is a unique number, in the range [0, MPI_size), which identifies us
    # in this MPI world.
    mpi_rank = mpi_comm.Get_rank()

    print('Rank {}:  I am a non-root rank!'.format(mpi_comm.Get_rank()), flush=True)

    # Gather the size of the input array from the root Bcast
    shape = np.empty(2)
    mpi_comm.Bcast(shape, root=0)
    start_ind, end_ind, cols = determine_size(shape, mpi_rank, lowest_rank=1, highest_rank=mpi_comm.Get_size()-1)
    print("Rank {}: taking care of inputs {} to {}".format(mpi_comm.Get_rank(), start_ind, end_ind), flush=True)

    rows = end_ind - start_ind

    # Wait for the inputs
    ins = np.empty((rows, cols), dtype=float)
    req = mpi_comm.Irecv(ins, source=0, tag=0)
    req.Wait()
    print("Rank {}:  I am rank {} and I received inputs:\n{}".format(mpi_comm.Get_rank(), mpi_rank, ins), flush=True)

    # Perform trivial calculation
    ins *= 2.5

    # Return the rows we are responsible for
    print("Rank {}:  I am rank {} and I will send a response:\n{}".format(mpi_comm.Get_rank(), mpi_rank, ins), flush=True)
    mpi_comm.Isend(ins, dest=0, tag=mpi_rank)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())        # Use the return code from main() as the exit code