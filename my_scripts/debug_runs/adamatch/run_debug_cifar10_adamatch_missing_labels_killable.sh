#!/bin/bash
#SBATCH --mem=15g
#SBATCH -c 8
#SBATCH --time=6-0
#SBATCH --gres=gpu:2,vmem:15g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-0
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./debug_runs_out/d-%j.out # STDOUT
#SBATCH --killable
#SBATCH --requeue
dir=/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning
cd $dir

source /cs/labs/daphna/noam.fluss/usb_venv/bin/activate

echo "Script Name: $0"
echo "First Parameter of the script is $1"
echo "The second Parameter is $2"
echo "Total Number of Parameters: $#"
echo "The process ID is $$"
echo "Exit code for the script: $?"
python3.7 train_v2.py --c config/classic_cv/adamatch/my_scripts/debug_full_run_adamatch_cifar10_40_0.yaml --random_missing_labels_num 1 --slurm_job_id ${SLURM_JOB_ID}


