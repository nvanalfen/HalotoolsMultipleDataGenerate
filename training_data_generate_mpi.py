import os
import numpy as np
import multiprocessing as mp
import h5py
from halotools.sim_manager import CachedHaloCatalog
from generate_median_training_data import build_model_instance
from generate_median_training_data import generate_data
from data_utils import load_yaml_config

def setup_generation(config):
    """
    Setup the generation of training data.
    This function will unpack configuration file and set up the parameters for the training data generation.
    """

    # Build halocat
    catalog = config['catalog']
    halo_finder = config['halo_finder']
    redshift = config['redshift']
    version_name = config['version_name']

    halocat = CachedHaloCatalog(simname=catalog, redshift=redshift,
                                halo_finder=halo_finder, version_name=version_name)
    
    # Build the model instance
    sat_bins = config['sat_bins']
    constant_alignment_strength = config['constant_alignment_strength']
    seed = config['seed']

    model_dict = {
        'sat_bins': sat_bins,
        'constant_alignment_strength': constant_alignment_strength,
        'seed': seed,
    }

    # Build the model
    # pass in 1.0 for both alignment strengths since this will be overwritten anyway
    # model = build_model_instance(1.0, 1.0, sat_bins, halocat, constant=constant_alignment_strength, seed=seed)

    # return model, halocat
    return model_dict, halocat

def generate(config, keys, inputs):
    rank = MPI.COMM_WORLD.Get_rank()
    num_ranks = MPI.COMM_WORLD.Get_size()

    model_dict, halocat = setup_generation(config)
    job = config.get('job', 0)  # Get the job number if it exists, default to 0
    num_jobs = config.get('num_jobs', 1)  # Get the number of jobs if it exists, default to 1
    rbins = config['rbins']
    runs = config['runs']
    max_attempts = config['max_attempts']
    subset_dir = config['subset_dir']
    processes = config['processes']
    parallel_method = config["parallelization"]
    store_columns = config['store_columns']
    store_correlations = config['store_correlations']
    column_labels = config['column_labels']

    subset_file = f"job_{job}_subset_{rank}.h5"
    start_index = 0
    if os.path.exists( os.path.join(subset_dir, subset_file) ):
        with h5py.File(os.path.join(subset_dir, subset_file), "r") as f:
            group_names = [name for name in f if isinstance(f[name], h5py.Group)]
            start_index = len(group_names)
            print(f"Rank {rank} found subset file. Loading...", flush=True)
    else:
        print(f"Rank {rank} creating subset file...", flush=True)
        with h5py.File(os.path.join(subset_dir, subset_file), "w") as f:
            if store_columns:
                if not column_labels or column_labels == "all" or len(column_labels) == 0:
                    # Use all the columns in the model
                    # Make a quick model to get the column names
                    model = build_model_instance(1.0, 1.0, model_dict['sat_bins'], halocat,
                                                constant=model_dict['constant_alignment_strength'],
                                                seed=model_dict['seed'])
                    column_labels = np.array( [ col for col in model.mock.galaxy_table.columns ] )

                f.attrs["columns"] = column_labels
            f.attrs["simname"] = halocat.simname
            f.attrs["redshift"] = halocat.redshift
            f.attrs["halo_finder"] = halocat.halo_finder
            f.attrs["version_name"] = halocat.version_name
            f.attrs["particle_mass"] = halocat.particle_mass
            keys_array = np.array(keys, dtype=h5py.string_dtype(encoding='utf-8'))
            f.attrs["input_params"] = keys_array

    # Loop through the inputs and create catalogs/correlations
    for i in range(len(inputs))[start_index:]:
        input_num = (i*num_ranks) + rank
        input_row = inputs[i]
        input_dict = {keys[j]: input_row[j] for j in range(len(keys))}

        # Call the generation for this set of inputs
        generate_data(model_dict, halocat, input_dict, rbins, subset_file, input_num, runs=runs, max_attempts=max_attempts,
             output_dir=subset_dir, processes=processes, parallel_method=parallel_method,
             store_columns=store_columns, store_correlations=store_correlations, column_labels=column_labels)

def merge_hdf5_files(output_dir, job):
    """
    Merge all the hdf5 files in the output directory into a single file.
    This is useful for cleaning up the output directory and making it easier to work with.
    """
    rank = MPI.COMM_WORLD.Get_rank()
    num_ranks = MPI.COMM_WORLD.Get_size()
    if rank == 0:
        # Create a new hdf5 file to hold the merged data
        with h5py.File(os.path.join(output_dir, f"job_{job}_merged_data.h5"), "w") as f:
            for i in range(num_ranks):
                subset_file = f"job_{job}_subset_{i}.h5"
                with h5py.File(os.path.join(output_dir, subset_file), "r") as g:
                    # Copy all attrs fields
                    for key in g.attrs.keys():
                        f.attrs[key] = g.attrs[key]
                    # Copy the groups from the subset file to the merged file
                    for name in g:
                        group = g[name]
                        f.copy(group, name)
        print(f"Rank {rank} merged hdf5 files into {os.path.join(output_dir, f'job_{job}_merged_data.h5')}", flush=True)

def remove_subset_files(output_dir, job, subset_pattern="job_{}_subset_{}.h5"):
    """
    Clean up the subset files in the output directory.
    This is useful for cleaning up the output directory after merging the files.
    """
    num_ranks = MPI.COMM_WORLD.Get_size()
    for i in range(num_ranks):
        subset_file = subset_pattern.format(job, i)
        if os.path.exists(os.path.join(output_dir, subset_file)):
            os.remove(os.path.join(output_dir, subset_file))
    print("Removed subset files.", flush=True)


def determine_size(shape, rank, min_rank, max_rank):
    """
    Determine the size of the array to be sent to each rank.
    This function is used to determine the range to split data on.
    To avoid loading all small values to one rank or all large values to another,
    start at the index matching the rank (accounting for if rank 0 is involved)
    and use step sizes equal to the maximum number of ranks involved.
    """
    rows, cols = shape
    
    start = rank - min_rank                     # Start at the index matching the rank (one lower if rank 0 isn't helping)
    stop = rows                                 # End at the end of the array
    step = max_rank - min_rank + 1              # Step size equal to the maximum number of ranks involved

    return range(start, stop, step)

def broadcast_keys(comm, keys):
    # Encode into byte array
    maxlen = max(len(key) for key in keys)
    arr = np.zeros((len(keys), maxlen), dtype="S1")
    for i, s in enumerate(keys):
        arr[i, :len(s)] = np.frombuffer(s.encode('utf-8'), dtype='S1')

    # Broadcast the shape of the array
    shape = np.array(arr.shape, dtype=int)
    comm.Bcast(shape, root=0)

    # Broadcast the array
    comm.Bcast(arr, root=0)

def receive_keys(comm):
    # Get the shape of the chararray from the root
    shape = np.empty(2, dtype=int)
    comm.Bcast(shape, root=0)

    # Create an empty array to hold the keys
    arr = np.empty(shape, dtype="S1")
    comm.Bcast(arr, root=0)

    # decode byte array into keys
    return [b''.join(row).decode('utf-8').rstrip('\x00') for row in arr]

def root(comm, param_loc):

    config = load_yaml_config(param_loc)
    job = config.get('job', 0)  # Get the job number if it exists, default to 0
    num_jobs = config.get('num_jobs', 1)  # Get the number of
    data = np.load(  config['param_loc'], allow_pickle=True)
    keys = data['keys']
    inputs = data['values']

    # Reduce inputs to just what this job will take care of. Evenly space the slices so no job gets all the hard inputs
    inputs = inputs[job::num_jobs]

    # return_product = config['return_product']
    # assert return_product in ["correlation", "columns"], f"Invalid return product {return_product}. Must be 'correlation' or 'columns'."

    # Broadcast config to all ranks
    comm.bcast(config, root=0)

    # Send the keys to all ranks
    broadcast_keys(comm, keys)

    # Broadcast shape of inputs to all ranks
    input_shape = np.array(inputs.shape, dtype=int)
    comm.Bcast(input_shape, root=0)

    rank_ownership = {}                 # Which ranks take care of which rows of data
    requests = []                       # List of requests for non-blocking sends
    for i in range(1, comm.Get_size()):
        # Set lowest rank to 0 as root rank will be involved as well
        span = determine_size(input_shape, i, min_rank=0, max_rank=comm.Get_size()-1)
        rank_ownership[i] = span

        # Send the inputs to the rank
        sendbuf = inputs[span]
        req = comm.Isend(sendbuf, dest=i, tag=0)
        requests.append(req)

    # Wait for all sends to complete
    MPI.Request.Waitall(requests)

    # Now that the relevant inputs have been sent off to other ranks, gather our own to be done by root
    span = determine_size(input_shape, 0, 0, comm.Get_size()-1)
    root_inputs = inputs[span]

    generate(config, keys, root_inputs)
    requests = []
    # Get "results" from non-root (no actual returns, but will signal completion)
    for i in range(1, comm.Get_size()):
        dummy = np.empty(1, dtype='i')  # dummy buffer
        req = comm.Irecv(dummy, source=i, tag=0)
        requests.append(req)
    # Wait for all receives to complete
    MPI.Request.Waitall(requests)

    # Merge files
    merge_hdf5_files(config['subset_dir'], job)
    # Clean up subset files
    remove_subset_files(config['subset_dir'], job)

    return 0

def nonroot(comm):
    rank = comm.Get_rank()

    # Receive the config from root
    config = comm.bcast(None, root=0)
    job = config.get('job', 0)  # Get the job number if it exists, default to 0
    num_jobs = config.get('num_jobs', 1)  # Get the number of

    # Receive the keys from root
    keys = receive_keys(comm)
    
    # Receive shape of inputs from root
    input_shape = np.empty(2, dtype=int)
    comm.Bcast(input_shape, root=0)
    _, cols = input_shape

    # Get the size to allocate a buffer for the relevant inputs
    span = determine_size(input_shape, rank, min_rank=0, max_rank=comm.Get_size()-1)
    rows = len(span)

    # Create a buffer to hold the inputs
    inputs = np.empty((rows, cols), dtype=float)
    # Receive the inputs from root
    req = comm.Irecv(inputs, source=0, tag=0)
    req.Wait()

    generate(config, keys, inputs)
    dummy = np.array(0, dtype='i')
    req = comm.Isend(dummy, dest=0, tag=0)
    req.Wait()

    # close and end
    return 0

def main(param_loc):
    # Get our MPI communicator, our rank, and the world size.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    if rank == 0:
        return root(comm, param_loc)
    else:
        return nonroot(comm)
    
if __name__ == "__main__":

    # Get the config location from the command line
    import sys
    config_loc = sys.argv[1]

    spawn = False
    if len(sys.argv) > 2:
        spawn = bool(int(sys.argv[2]))

    if spawn:
        # Spawn for safe multiprocessing on each rank
        mp.set_start_method("spawn", force=True)
    else:
        # Fork for faster multiprocessing on each rank
        mp.set_start_method("fork", force=True)

    from mpi4py import MPI

    # Call the main function
    sys.exit( main(config_loc) )