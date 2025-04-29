import subprocess
import os
import sys
import tempfile
from data_utils import load_yaml_config

def build_slurm_job(config):
    pass
    # Set up the --<field> options for the sbatch command
    lines = ["#!/bin/bash"]
    for key, value in config["slurm"].items():
        if len(key) > 2 and key[:2] == "--":
            # If the key starts with --, add it to the sbatch command
            lines.append(f"#SBATCH\t{key}={value}")
        elif len(key) > 1 and key[0] == "-":
            # If the key starts with -, add it to the sbatch command
            lines.append(f"#SBATCH\t{key} {value}")

    lines.append("")        # Empty line for readability if we want to keep the file

    # Load modules
    modules = config["slurm"].get("modules", [])
    for module in modules:
        lines.append(f"module load {module}")
    # Activate the base conda (if applicable)
    source = config["slurm"].get("source", None)
    if source is not None:
        lines.append(f"source {source}")
    # Activate the conda environment
    conda_env = config["slurm"].get("conda_env", None)
    if conda_env is not None:
        lines.append(f"conda activate {conda_env}")

    lines.append("")        # Empty line for readability if we want to keep the file

    # Build the command to run
    cli_command = config["slurm"].get("cli_command", None)
    cli_args = config["slurm"].get("cli_args", [])
    if not cli_command is None and cli_command == "mpirun" and "--ntasks" in config["slurm"]:
        # Make sure to automatically add the number of tasks to the command
        cli_args.append(f"-n {config['slurm']['--ntasks']}")
    python_command = config["slurm"].get("python_command", None)
    script = config["slurm"].get("script", None)
    script_args = config["slurm"].get("script_args", [])

    cmd = []
    if cli_command is not None:
        cmd.append(cli_command)
    cmd += cli_args
    if python_command is not None:
        cmd.append(python_command)
    if script is not None:
        cmd.append(script)
    cmd += script_args

    cmd = [str(x) for x in cmd]                     # Convert all arguments to strings
    cmd = " ".join(cmd)                             # Join the command into a single string
    lines.append(cmd)                               # Add the command to the script

    if not script:
        raise ValueError("No script specified in the config file.")
    
    return "\n".join(lines)                         # Join the lines into a single string

def submit_slurm_job(config, use_temp_file=True, persistent_f_name=None):
    """
    Submit a SLURM job using the configuration file.
    """
    if not use_temp_file:
        assert persistent_f_name is not None, "Persistent file name must be provided if use_temp_file is False."
    # Build the SLURM job script
    slurm_script = build_slurm_job(config)

    # Write the script to a temporary file
    if use_temp_file:
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".sh") as temp_script:
            temp_script.write(slurm_script)
            temp_script_path = temp_script.name

        try:
            print("Submitting SLURM job...")
            result = subprocess.run(["sbatch", temp_script_path], capture_output=True, text=True)
            if result.returncode == 0:
                print("SLURM job submitted successfully.")
                print(result.stdout)
            else:
                print("Error submitting SLURM job:")
                print(result.stderr)
        except subprocess.CalledProcessError as e:
            print("Error submitting SLURM job:")
            print(e.stderr)
        except Exception as e:
            print("An unexpected error occurred while submitting the SLURM job.")
            print(e)
        finally:
            os.remove(temp_script_path)  # Clean up the temporary file
    else:
        f_name = persistent_f_name
        with open(f_name, "w") as f:
            f.write(slurm_script)

if __name__ == "__main__":
    assert len(sys.argv) == 2, "Usage: python3 submit_slurm.py <config_file>"
    # Load the configuration file
    config_file = sys.argv[1]
    config = load_yaml_config(config_file)

    # Submit the job
    submit_slurm_job(config, use_temp_file=True)