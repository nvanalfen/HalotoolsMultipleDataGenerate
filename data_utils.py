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