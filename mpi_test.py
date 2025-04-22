from mpi4py import MPI
import os
import numpy as np

# A dummy test based on the mpi example in https://github.com/akkornel/mpi4py/blob/master/mpi4.py

# The input array will be global so there's no need to broadcast it
ins = np.ones((15, 3)).astype(float)
values = np.arange(len(ins)).astype(float)
ins *= values[:, np.newaxis]

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
        return mpi_root(mpi_comm, len(ins))
    else:
        return mpi_nonroot(mpi_comm, len(ins))

def mpi_root(mpi_comm, size):

    # Print the input array
    print('Rank {}:  I am the root rank!'.format(mpi_comm.Get_rank()), flush=True)
    print('Rank {}:  The inputs are shape {}'.format(mpi_comm.Get_rank(), ins.shape), flush=True)
    print("Rank {}: Inputs:".format(mpi_comm.Get_rank()), flush=True)
    print(ins, flush=True)

    # Broadcast the inputs to all ranks
    mpi_comm.Bcast(ins, root=0)

    # Gather responses
    sendbuf = np.empty((0, 3), dtype=float)
    recvbuf_arr = np.empty([size, 3], dtype=float)
    sendcounts = mpi_comm.gather(0, root=0)
    displs = [0] + list(np.cumsum(sendcounts[:-1]))
    recvbuf = (recvbuf_arr, sendcounts, displs, MPI.DOUBLE)
    mpi_comm.Gatherv(sendbuf, recvbuf, root=0)

    # # Did we get all responses?
    mpi_size = mpi_comm.Get_size()
    # if len(recvbuf) != mpi_size:
    #     print('Rank {}:  I did not get all responses!'.format(mpi_comm.Get_rank()), flush=True)
    #     print('Rank {}:  I got {} responses, but expected {}!'.format(mpi_comm.Get_rank(), len(recvbuf), mpi_size), flush=True)
    #     return 1
    
    # Output each rank's response
    for i in range(mpi_size):
        if i == 0:
            print("Rank {}:  I am rank 0", flush=True)
        else:
            print("Rank {}:  I got a response from rank {}".format(mpi_comm.Get_rank(), i), flush=True)
    print("Rank {}:  The responses are:".format(mpi_comm.Get_rank()), flush=True)
    print(recvbuf[0], flush=True)

    return 0
    

def mpi_nonroot(mpi_comm, size):
    # Get our MPI rank.
    # This is a unique number, in the range [0, MPI_size), which identifies us
    # in this MPI world.
    mpi_rank = mpi_comm.Get_rank()

    print('Rank {}:  I am a non-root rank!'.format(mpi_comm.Get_rank()), flush=True)

    # Wait for the inputs
    ins = np.empty((size, 3), dtype=float)
    mpi_comm.Bcast(ins, root=0)

    # Only grab the porton of the inputs corresponding to what this rank is responsible for
    # This will simulate the MPI ranks taking care of different parts of the input
    chunk_size = int(np.ceil(len(ins) / (mpi_comm.Get_size()-1) ))          # Master rank is not involved
    start = (mpi_rank-1) * chunk_size
    end = min(start + chunk_size, size)
    print("Rank {}: taking care of inputs {} to {}".format(mpi_comm.Get_rank(), start, end), flush=True)

    # Return the rows we are responsible for
    sendbuf = np.array(ins[start:end])*2.5
    print("Rank {}:  I am rank {} and I will send a response:\n{}".format(mpi_comm.Get_rank(), mpi_rank, sendbuf), flush=True)
    sendcounts = mpi_comm.gather((end-start)*3, root=0)
    recvbuf = (None, sendcounts, None, MPI.DOUBLE)
    mpi_comm.Gatherv(sendbuf, None, root=0)

    return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())        # Use the return code from main() as the exit code