#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 4
#SBATCH --time=1-5
#SBATCH --gres=gpu:1,vmem:18g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-2
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./hyper_parameter_out_debug/d_cifar100_not_killable_v2-%j.out # STDOUT
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
echo "run run_hyper_parameter_tuning_cifar100_not_killable_v6.sh"
epoch=50  # Replace 10 with your desired value
algorithm="flexmatch"
random_missing_labels_num=40
choose_random_labeled_training_set=-1
choose_random_labeled_training_set_duplicate=-1
lambda_entropy=1.5
python_code_version=7
project_wandb="test"
num_labels=2500
delete=0
net_new="wrn_28_2"
split_to_superclasses=0
# Calculate the value of num_train_iter using epoch
num_train_iter=$((epoch * 1024))

python train_v7.py --c /cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/config/classic_cv/flexmatch/debug_flexmatch_cifar100_0.yaml\
                        --seed ${SLURM_ARRAY_TASK_ID}\
                        --epoch $epoch --num_train_iter $num_train_iter --num_eval_iter 1024\
                        --num_classes 100 --algorithm $algorithm\
                        --dataset cifar100 --slurm_job_id ${SLURM_JOB_ID}\
                        --random_missing_labels_num $random_missing_labels_num\
                         --choose_random_labeled_training_set $choose_random_labeled_training_set\
                        --lambda_entropy $lambda_entropy\
                        --project_wandb $project_wandb\
                        --num_labels $num_labels\
                        --python_code_version $python_code_version --delete $delete\
                        --net_new $net_new\
                        --split_to_superclasses $split_to_superclasses\
                        --choose_random_labeled_training_set_duplicate $choose_random_labeled_training_set_duplicate\
                        --choose_last_classes_as_unseen 1\
                        --comment 8001