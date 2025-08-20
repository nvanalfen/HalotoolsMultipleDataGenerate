import h5py
import numpy as np

# Old structure:
# File
#   - attrs
#       - columns
#       - halo_finder
#       - input_params
#       - particle_mass
#       - redshift
#       - simname
#       - version_name
#   - keys
#       - input_0 (group)
#           - attrs
#               - alpha
#               - central_alignment_strength
#               - logM0
#               - logM1
#               - logMmin
#               - satellite_alignment_strength
#               - sigma_logM
#           - keys
#               - iteration_0 (group)
#                   - keys
#                       - table_subset (dataset)
#                       - correlations (dataset)
#               ...
#               - iteration_N (group)
#       ...
#       - input_N (group)

# New structure:
# File
#   - attrs
#       - columns
#       - halo_finder
#       - input_params
#       - particle_mass
#       - redshift
#       - simname
#       - version_name
#   - keys
#       - input_0 (group)
#           - attrs
#               - alpha
#               - central_alignment_strength
#               - logM0
#               - logM1
#               - logMmin
#               - satellite_alignment_strength
#               - sigma_logM
#           - keys
#               - iteration_0 (group)
#                   - keys
#                       - table_subset (group)                  # BEGIN CHANGE
#                           - col_1 (dataset)
#                           ...
#                           - col_m (dataset)                   # END CHANGE
#                       - correlations (dataset)
#               ...
#               - iteration_N (group)
#       ...
#       - input_N (group)

def unpack_old(f_name):
    with h5py.File(f_name, "r") as f:
        input_params = f.attrs["input_params"]
        columns = f.attrs["columns"]
        N = len(f.keys())
        m = len(input_params)
        input_arr = np.zeros((N,m))
        galaxy_tables = []
        for i in range(len(f.keys())):
            print(i)
            key = list(f.keys())[i]
            subset = f[key]
            input_arr[i] = np.array([ subset.attrs[param] for param in input_params ])
            table = []
            if len(subset["iteration_0/table_subset"][:]) > 0:
                table = np.vstack([ subset["iteration_0/table_subset"][:][col] for col in columns ]).T
            galaxy_tables.append(table)
        return_block = {"input_params":input_params,"inputs": input_arr, "galaxy_tables": galaxy_tables, "columns": columns}
        return return_block
    
def unpack_reformatted(f_name):
    with h5py.File(f_name, "r") as f:
        input_params = f.attrs["input_params"]
        columns = f.attrs["columns"]
        N = len(f.keys())
        m = len(input_params)
        input_arr = np.zeros((N,m))
        galaxy_tables = []
        for i in range(len(f.keys())):
            print(i)
            key = list(f.keys())[i]
            subset = f[key]
            input_arr[i] = np.array([ subset.attrs[param] for param in input_params ])
            table = []
            if len(subset["iteration_0/table_subset"].keys()) > 0:
                table = np.vstack([ subset[f"iteration_0/table_subset/{col}"] for col in columns ]).T
            galaxy_tables.append(table)
        return_block = {"input_params":input_params,"inputs": input_arr, "galaxy_tables": galaxy_tables, "columns": columns}
        return return_block