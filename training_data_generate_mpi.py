import os
import numpy as np
import multiprocessing as mp
from halotools.sim_manager import CachedHaloCatalog
from generate_median_training_data import build_model_instance
from generate_median_training_data import calculate_all_iterations
from data_utils import load_yaml_config

def atomic_save(dirname, filename, data_dict):
    """
    Save data to a file using an atomic pattern.
    This prevents issues if there is a failure during file save
    """
    temp_filename = "tmp_" + filename
    np.savez( os.path.join(dirname, temp_filename), **data_dict)
    os.rename(os.path.join(dirname, temp_filename), os.path.join(dirname, filename))

def clear_checkpoints(config):
    # TODO: Delete all checkpoint files
    # Only call this after the output file has been saved
    checkpoint_dir = config['checkpoint_dir']
    if not os.path.exists(checkpoint_dir):
        print(f"Checkpoint directory {checkpoint_dir} does not exist. Skipping cleanup.", flush=True)
        return
    for filename in os.listdir(checkpoint_dir):
        if filename.startswith("checkpoint_"):
            os.remove(os.path.join(checkpoint_dir, filename))

def run_generation(config, keys, inputs):
    """
    Run the generation of training data.
    This function will unpack configuration file and set up the parameters for the training data generation.
    """

    model, halocat = setup_generation(config)
    rbins = config['rbins']
    runs = config['runs']
    max_attempts = config['max_attempts']
    save_every = config['save_every']
    checkpoint_dir = config['checkpoint_dir']
    processes = config['processes']
    parallel_method = config["parallelization"]

    _, _, outputs = generate_training_data(model, rbins, halocat, keys, inputs,
                                                   runs=runs, save_every=save_every,
                                                   output_dir=checkpoint_dir, suffix="",
                                                   max_attempts=max_attempts,
                                                   processes=processes,
                                                   parallel_method=parallel_method)
    
    print(f"Rank {MPI.COMM_WORLD.Get_rank()} finished generation", flush=True)
    
    return outputs

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

    # Build the model
    # pass in 1.0 for both alignment strengths since this will be overwritten anyway
    model = build_model_instance(1.0, 1.0, sat_bins, halocat, constant=constant_alignment_strength, seed=seed)

    return model, halocat

def generate_training_data(model, rbins, halocat, keys, all_inputs, runs=10, save_every=5, 
                           output_dir="checkpoints", suffix="", max_attempts=5, processes=3,
                           parallel_method="correlation"):

    # Create an empty input list and output array
    # The shape of the output array is (number of inputs, number of inner runs, 3, number of rbin_centers)
    # The 3 corresponds to the three different correlation functions
    inputs = []
    outputs = np.zeros((len(all_inputs), runs, 3, len(rbins)-1))
    start_index = 0

    # check if a checkpoint file exists for this rank (i.e. if this is picking up from a previous run)
    rank = MPI.COMM_WORLD.Get_rank()
    checkpoint_file = f"checkpoint_{rank}.npz"
    if os.path.exists( os.path.join(output_dir, checkpoint_file) ):
        print(f"Rank {rank} found checkpoint file. Loading...", flush=True)
        checkpoint = np.load( os.path.join(output_dir, checkpoint_file), allow_pickle=True)
        inputs = checkpoint['inputs'].tolist()
        outputs = checkpoint['outputs']
        start_index = len(inputs)
    
    # Loop over the inputs
    for i in range(len(all_inputs))[start_index:]:
        print(f"Rank {rank} processing input {i+1}/{len(all_inputs)}", flush=True)
        # Get the input for this iteration
        input_dict = {keys[j]: all_inputs[i][j] for j in range(len(keys))}

        # Calculate the outputs for this input
        # Fortunately, the hard work is already taken care of in the imported function
        # and it even uses multprocessing to speed things up
        try:
            result = calculate_all_iterations(model, rbins, halocat, runs=runs, input_dict=input_dict,
                                       max_attempts=max_attempts, processes=processes, parallel_method=parallel_method)
            outputs[i] = result
            inputs.append(all_inputs[i])
        except Exception as e:
            print(f"Rank {rank} failed on input {input_dict}: {e}")

        # Save the outputs every save_every iterations
        # Only save after full chunks of inputs. All or nothing on the iterations within an input
        if (i + 1) % save_every == 0:
            atomic_save(output_dir, checkpoint_file, {'keys':keys, 'inputs':inputs, 'outputs':outputs})

    return keys, inputs, outputs

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
    data = np.load(  config['param_loc'], allow_pickle=True)
    keys = data['keys']
    inputs = data['values']

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

    # Allocate the full output array
    # Shape of Nxmx3xlen(rbins)-1
    outputs = np.zeros((input_shape[0], config['runs'], 3, len(config['rbins'])-1), dtype=float)

    # TODO: Enter calculation loop. Non-root ranks will also do this
    outputs[span] = run_generation(config, keys, root_inputs)

    # TODO: use Irecv to receive the results from non-root ranks
    requests = []

    # MPI needs contiguous buffers, so we need to create a temporary buffer for each rank
    temp_buffers = []
    for i in range(1, comm.Get_size()):
        temp = np.zeros((len(rank_ownership[i]), config['runs'], 3, len(config['rbins'])-1), dtype=float)
        temp_buffers.append(temp)

    for i in range(1, comm.Get_size()):
        # Receive the inputs from the rank
        buffer = temp_buffers[i-1]
        req = comm.Irecv(buffer, source=i, tag=0)
        requests.append(req)

    # Wait for all receives to complete
    MPI.Request.Waitall(requests)

    # Populate the outputs array with the results in buffers
    for i in range(1, comm.Get_size()):
        span = rank_ownership[i]
        outputs[span] = temp_buffers[i-1]

    # Save the outputs to a file
    output_f_name = config["output"]
    np.savez(output_f_name, keys=keys, inputs=inputs, outputs=outputs)

    # TODO: Clean up checkpoint files
    clear_checkpoints(config)

    return 0

def nonroot(comm):
    rank = comm.Get_rank()

    # Receive the config from root
    config = comm.bcast(None, root=0)

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

    # Now we have the inputs, we can do whatever we want with them
    outputs = run_generation(config, keys, inputs)
    req = comm.Isend(outputs, dest=0, tag=0)
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