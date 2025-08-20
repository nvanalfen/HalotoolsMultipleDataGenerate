import h5py
import numpy as np
import time

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

def write_new(f_name, f_old):
    columns = f_old.attrs["columns"]
    with h5py.File(f_name, "w") as f_new:
        # Copy attributes
        for key in f_old.attrs.keys():
            f_new.attrs[key] = f_old.attrs[key]
        
        # Iterate through inputs
        for input_key in f_old.keys():
            print(f"Processing input: {input_key}", flush=True)
            input_group = f_old[input_key]
            new_input_group = f_new.create_group(input_key)
            
            # Copy attributes for input
            for attr_key in input_group.attrs.keys():
                new_input_group.attrs[attr_key] = input_group.attrs[attr_key]
            
            # Iterate through iterations
            for iteration_key in input_group.keys():
                iteration_group = input_group[iteration_key]
                new_iteration_group = new_input_group.create_group(iteration_key)
                
                # Create table_subset group
                if "table_subset" in iteration_group:
                    table_subset_group = new_iteration_group.create_group("table_subset")

                    if len(iteration_group["table_subset"][:]) == 0:
                        # Skip failed table saves
                        continue
                    
                    # Copy datasets to the new structure
                    for col in columns:
                        if not iteration_group["table_subset"]:
                            continue
                        table_subset_group.create_dataset(col, data=iteration_group["table_subset"][col][:])
                
                # Copy correlations dataset
                if "correlations" in iteration_group:
                    new_iteration_group.create_dataset("correlations", data=iteration_group["correlations"][:])

if __name__ == "__main__":
    f_name = "Fully_merged_data.h5"
    with h5py.File(f_name, "r") as f_old:
        write_new("Converted_data.h5", f_old)
    print("Conversion complete.", flush=True)