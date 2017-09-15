#!/bin/bash -l

#SBATCH -N 1

#SBATCH -t 00:20:00

#SBATCH -p debug

#SBATCH -L SCRATCH

#SBATCH --qos=premium

#SBATCH -C haswell
module load python
echo "running script: tf_script_yisha.py"
module load deeplearning

tart_date=`date +"%Y-%m-%d %T.%6N"`
echo "$start_date #1:Script starts"
echo "SUBMIT DIR: $SLURM_SUBMIT_DIR"
echo "Tensorflow location"
python -u -c "import tensorflow; print(tensorflow.__file__)"

python tf_script_yisha.py --trace_flag=0 --epoch=20

#python confusion_matrix.py


