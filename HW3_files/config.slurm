#!/bin/bash
#----------------------------------------------------
# Sample Slurm job script
# for TACC Lonestar6  nodes
#----------------------------------------------------

#SBATCH -J HW3_gr0                        # Job name
#SBATCH -o HW3_gr0.o%j                    # Name of stdout output file (%j corresponds to the job id)
#SBATCH -e HW3_gr0.e%j                    # Name of stderr error file (%j corresponds to the job id)
#SBATCH -p gpu-a100-small                 # Queue (partition) name
#SBATCH -N 1                              # Total # of nodes (must be 1 for serial)
#SBATCH -n 1                              # Total # of mpi tasks (should be 1 for serial)
#SBATCH -t 24:00:00                       # Run time (hh:mm:ss)
#SBATCH --mail-user=ryaneli@utexas.edu
#SBATCH --mail-type=all                   # Send email at begin and end of job (can assign begin or end as well)
#SBATCH -A CCR24007                       # Allocation name

# Other commands must follow all #SBATCH directives...
module load python3

# Environment code
source $WORK/HW3_virtualenv/bin/activate

# Change to HW3_files so data path and outputs work correctly
cd $WORK/HW3_files

# Launch code (run VGG11 and VGG16 in parallel on different GPUs)
CUDA_VISIBLE_DEVICES=0 python main.py --model=vgg11 --batch_size=128 --epochs=100 > vgg11_out 2>&1 &
CUDA_VISIBLE_DEVICES=1 python main.py --model=vgg16 --batch_size=128 --epochs=100 > vgg16_out 2>&1 &
wait

# After VGG models finish, train MobileNet-v1 (reuse one GPU)
CUDA_VISIBLE_DEVICES=0 python main.py --model=mobilenet --batch_size=128 --epochs=100 > mobilenet_out 2>&1

# ---------------------------------------------------

