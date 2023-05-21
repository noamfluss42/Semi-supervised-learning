#!/bin/bash
#SBATCH --mem=5g
#SBATCH -c 8
#SBATCH --time=6-0
#SBATCH --gres=gpu:0,vmem:5g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-0
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./find_output/d-%j.out # STDOUT

dir=/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/temp

cd $dir

source /cs/labs/daphna/noam.fluss/usb_new_venv/bin/activate

echo "Script Name: $0"
echo "First Parameter of the script is $1"
echo "The second Parameter is $2"
echo "Total Number of Parameters: $#"
echo "The process ID is $$"
echo "Exit code for the script: $?"
python cancel_after_job.py


