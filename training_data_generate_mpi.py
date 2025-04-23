from mpi4py import MPI
import os
import numpy as np
from generate_median_training_data import calculate_all_iterations

def atomic_save(filename, data_dict):
    """
    Save data to a file using an atomic pattern.
    This prevents issues if there is a failure during file save
    """
    temp_filename = filename + ".tmp"
    np.savez(temp_filename, **data_dict)
    os.rename(temp_filename+".npz", filename+".npz")

def generate_training_data(model, rbins, halocat, keys, all_inputs, inner_runs=10, save_every=5, 
                           output_dir="checkpoints", suffix="", max_attempts=5):

    # Create an empty input list and output array
    # The shape of the output array is (number of inputs, number of inner runs, 3, number of rbins)
    # The 3 corresponds to the three different correlation functions
    inputs = []
    outputs = np.zeros((len(all_inputs), inner_runs, 3, len(rbins)))
    start_index = 0

    # check if a checkpoint file exists for this rank (i.e. if this is picking up from a previous run)
    rank = MPI.COMM_WORLD.Get_rank()
    checkpoint_file = os.path.join(output_dir, f"checkpoint_{suffix}_{rank}")
    extension = ".npz"
    if os.path.exists(checkpoint_file+extension):
        print(f"Rank {rank} found checkpoint file. Loading...")
        checkpoint = np.load(checkpoint_file+extension)
        inputs = checkpoint['inputs'].tolist()
        outputs = checkpoint['outputs']
        start_index = len(inputs)
    
    # Loop over the inputs
    for i in range(len(all_inputs)[start_index:]):
        # Get the input for this iteration
        input_dict = {key: all_inputs[key][i] for key in keys}
        inputs.append(input_dict)

        # Calculate the outputs for this input
        # Fortunately, the hard work is already taken care of in the imported function
        # and it even uses multprocessing to speed things up
        try:
            result = calculate_all_iterations(model, rbins, halocat, input_dict, inner_runs=inner_runs, 
                                       max_attempts=max_attempts)
            outputs[i] = result
            inputs.append(all_inputs[i])
        except Exception as e:
            print(f"Rank {rank} failed on input {input_dict}: {e}")

        # Save the outputs every save_every iterations
        if (i + 1) % save_every == 0:
            atomic_save(checkpoint_file, {'keys':keys, 'inputs': inputs, 'outputs': outputs})

    return keys, inputs, outputs

def root(comm, keys, inputs):
    input_length = len(inputs)

def nonroot(comm, keys, input_length):
    pass

def main(param_loc):
    # Get our MPI communicator, our rank, and the world size.
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    data = np.load(param_loc)
    keys = data['keys']
    inputs = data['inputs']

    if rank == 0:
        return root(comm, keys, inputs)
    else:
        return nonroot(comm, keys, len(inputs))
    
if __name__ == "__main__":
    # Get the parameter location from the command line
    import sys
    param_loc = sys.argv[1]

    # Call the main function
    sys.exit( main(param_loc) )