#!/bin/bash -l

#SBATCH -N 1

#SBATCH -t 00:20:00

#SBATCH -p debug

#SBATCH -L SCRATCH

#SBATCH --qos=premium

#SBATCH -C haswell
module load python

module load deeplearning

python tf_script.py --trace_flag=1 --epoch=10


