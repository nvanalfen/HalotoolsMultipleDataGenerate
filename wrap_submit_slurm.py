import sys
from data_utils import load_yaml_config, save_yaml_config
from submit_slurm import submit_slurm_job
import os

def generate_multijob_configs(base_config):
    """
    Taking a base config, use the num_jobs category to generate multiple
    config objects for submitting multple jobs.
    """
    num_jobs = base_config.get("num_jobs", 1)
    configs = []
    for i in range(num_jobs):
        config_f_name = os.path.join("temp_configs", f"config_{i}.yaml")
        new_config = dict(base_config)  # Create a copy of the base config
        new_config["job"] = i

        # Use new config script name in the slurm script_args
        script_args = new_config["slurm"]["script_args"]
        script_args[0] = config_f_name
        new_config["slurm"]["script_args"] = script_args

        # Update error and output file locations
        new_config["slurm"]["--error"] = f"error_logs/job_{i}.err"
        new_config["slurm"]["--output"] = f"output_logs/job_{i}.out"

        # Update job name
        new_config["slurm"]["--job-name"] = f"job_{i}"

        # Save the new config to a file
        save_yaml_config(new_config, config_f_name)
        # Load config back to process it
        new_config = load_yaml_config(config_f_name, restructure=True)
        configs.append(new_config)
    return configs

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 submit_slurm.py <config_file>"
    # Load the configuration file
    config_file = sys.argv[1]
    config = load_yaml_config(config_file, restructure=False)

    # Generate multiple job configs
    configs = generate_multijob_configs(config)

    # Submit the jobs
    for job_config in configs:
        submit_slurm_job(job_config, use_temp_file=True)