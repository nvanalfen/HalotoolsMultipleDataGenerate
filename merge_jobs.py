import h5py
import os

def remove_job_files(output_dir, num_jobs, job_pattern="job_{}_merged_data.h5"):
    for i in range(num_jobs):
        subset_file = job_pattern.format(i)
        if os.path.exists(os.path.join(output_dir, subset_file)):
            os.remove(os.path.join(output_dir, subset_file))
    print("Removed job files.", flush=True)

def merge_job_files(output_dir, num_jobs, remove=True):
    """
    Merge all the hdf5 files in the output directory into a single file.
    This is useful for cleaning up the output directory and making it easier to work with.

    The File format of the merged h5 files (for each job, which this function will then merge into a final file) is:

    File:
        attrs:
            columns
            halo_finder
            input_params
            particle_mass
            redshift
            simname
            version_name
        keys:
            input_0:
                attrs:
                    central_alignment_strength
                    satellite_alignment_strength
                    logMmin
                    sigma_logM
                    logM0
                    logM1
                    alpha
                keys:
                    iteration_0:
                        table_subset:
                            (columns from the mock catalog created)
                        correlations:
                            (correlations from the mock catalog created)
                    ...
                    iteration_m
            ...
            input_N
    """
    job_pattern = "job_{}_merged_data.h5"           # File name format for job merged files

    master_file = os.path.join(output_dir, "Fully_merged_data.h5")

    try:
        with h5py.File(master_file, "w") as f:
            attrs_written = False
            for i in range(num_jobs):
                job_file = job_pattern.format(i)
                try:
                    with h5py.File(os.path.join(output_dir, job_file), "r") as g:
                        if not attrs_written:
                            # Copy all attrs fields
                            for key in g.attrs.keys():
                                f.attrs[key] = g.attrs[key]
                            attrs_written = True
                        # Copy the groups from the subset file to the merged file
                        for name in g:
                            group = g[name]
                            f.copy(group, name)
                except:
                    print(f"Could not merge job {i} file.")

        print(f"Merged job files into master file: {master_file}", flush=True)
        
        # Erase old files
        if remove:
            try:
                remove_job_files(output_dir, num_jobs, job_pattern)
            except:
                print("Error removing old job files", flush=True)
    except:
        print("Error creating fully merged file.", flush=True)