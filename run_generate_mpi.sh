# Description: This script is used to run the training data generation script using MPI with 5 processes.
# --mca btl vader,self: This option specifies the use of the "vader" (shared memory)
# and "self" (loopback) BTL (Byte Transfer Layer) components for communication between processes.
# the arguments passed to the script are:
# 1. config.yaml: The configuration file for the training data generation script.
# 2. 1: This tells the process to set the multiprocess start mode to spawn instead of fork. Safer
mpirun --mca btl vader,self -n 5 python3 training_data_generate.py config.yaml 1