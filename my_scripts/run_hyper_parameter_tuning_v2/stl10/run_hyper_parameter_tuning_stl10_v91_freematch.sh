#!/bin/bash
#SBATCH --mem=30g
#SBATCH -c 4
#SBATCH --time=3-5
#SBATCH --gres=gpu:1,vmem:18g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-3
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./hyper_parameter_out_debug_freematch/d_stl10_v91_freematch-%j.out # STDOUT
#SBATCH --killable
#SBATCH --requeue
dir=/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning

cd $dir

source /cs/labs/daphna/noam.fluss/usb_new_venv/bin/activate
echo "Script Name: $0"
echo "First Parameter of the script is $1"
echo "The second Parameter is $2"
echo "Total Number of Parameters: $#"
echo "The process ID is $$"
echo "Exit code for the script: $?"
##echo "missing_labels: $missing_labels?"
echo "slurm_job_id: ${SLURM_JOB_ID}?"
echo "run d_stl10_v91_freematch.sh"
epoch=1024  # Replace 10 with your desired value
algorithm="freematch"
random_missing_labels_num=4
choose_random_labeled_training_set=-1
choose_random_labeled_training_set_duplicate=-1
lambda_entropy=0
python_code_version=6
project_wandb="Flexmatch_10_project_missing_labels"
num_labels=400
delete=0
net_new="wrn_var_37_2"
split_to_superclasses=0
# Calculate the value of num_train_iter using epoch
num_train_iter=$((epoch * 1024))
seed_1=$((SLURM_ARRAY_TASK_ID + 1))

python train_v7.py --c config/classic_cv/freematch/freematch_stl10_hyperparameter_tuning.yaml\
                        --seed $seed_1\
                        --epoch $epoch --num_train_iter $num_train_iter --num_eval_iter 1024\
                        --num_classes 10 --algorithm $algorithm\
                        --dataset stl10 --slurm_job_id ${SLURM_JOB_ID}\
                        --random_missing_labels_num $random_missing_labels_num\
                         --choose_random_labeled_training_set $choose_random_labeled_training_set\
                        --lambda_entropy $lambda_entropy\
                        --project_wandb $project_wandb\
                        --num_labels $num_labels\
                        --python_code_version $python_code_version --delete $delete\
                        --net_new $net_new\
                        --split_to_superclasses $split_to_superclasses\
                        --choose_random_labeled_training_set_duplicate $choose_random_labeled_training_set_duplicate\
                        --comment 7009