import yaml
import numpy as np
import os

def load_yaml_config(config_file):
    """
    Load a YAML configuration file.
    """
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    config["rbins"] = parse_array_field(config, "rbins")
    config["sat_bins"] = parse_array_field(config, "sat_bins")
    parse_processes(config)

    return config

def parse_array_field(config, key):
    """
    Parse a field in the config file that either is or will be used to define an array.
    """
    method = config[key].get('method', -1)
    if method == -1:
        raise ValueError(f"Method not specified for {key} in config file.")
    elif method == "logspace":
        # Use the args as start, stop, num and build logspace
        start, stop, num = config[key]["args"]
        return np.logspace(start, stop, num)
    elif method == "file":
        # Load the array from a file
        path = config[key]["file"]
        if path is None:
            raise ValueError(f"File path not specified for {key} in config file.")
        if not os.path.exists(path):
            raise FileNotFoundError(f"File {path} does not exist.")
        return np.load(path)
    elif method == "list":
        # Load the array from a list
        return np.array(config[key]["args"])
    else:
        raise ValueError(f"Unknown method {method} for {key} in config file.")
    

def parse_processes(config):
    """
    Adjust the values of processes and ntasks in the config file.
    Because this can be relatively complex, we do it in a separate function.

    Five relevant values:
    config["processes"] = int. Number of processes to use
    config["cap_processes"] = bool. Whether to limit processes based on the value of --ntasks (below)
    config["parallelization"] = str. "iteration" or "correlation". Which aspect to parallelize.
    config["runs"] = int. Number of runs to use in the slurm job for each set of inputs.
    config["slurm"]["--cpus-per-task"] = int. Number of cpus to allocate per task in slurm

    At base, the following happen:
    config["processes"] if a value is given, is used as the number of processes. If null value is given,
    the number depends onf parallelization method:
        - If config["parallelization"] is "iteration", the number of processes is equal to config["runs"]
        - If config["parallelization"] is "correlation", the number of processes is equal to the number of correlations (3)
    config["cap_processes"] is used to limit the number of processes to the value of --cpus-per-task.
        - e.g. if processes is 5, but cpus-per-task is only 3, the number of processes is set to 3.
    
    config["slurm"]["--cpus-per-task"] is used to set the number of cpus per task in slurm.
        - If value is null, inherit the value of processes. If not null, use this as an upper bound for processes.
    """

    # Ensure processes has a value
    if config["processes"] is None:
        # Set the number of processes based on the parallelization method
        if config["parallelization"] == "iteration":
            config["processes"] = config["runs"]            # Use as many processes as iterations
        elif config["parallelization"] == "correlation":
            config["processes"] = 3                         # Use as many processes as correlations
        else:
            raise ValueError(f"Unknown parallelization method {config['parallelization']}")
        
    # Now, runs has a definite value. Check that cpus-per-task has one as well
    if config["slurm"]["--cpus-per-task"] is None:
        # Set the number of cpus per task to the number of processes
        config["slurm"]["--cpus-per-task"] = config["processes"]

    # Now, check if we wish to cap processes at cpus-per-task
    # If cpus-per-task was set using processes, this won't change anything.
    if config["cap_processes"]:
        # Set the number of processes to the value of --cpus-per-task
        config["processes"] = min(config["processes"], config["slurm"]["--cpus-per-task"])