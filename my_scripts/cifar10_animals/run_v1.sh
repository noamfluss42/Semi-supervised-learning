#!/bin/bash
#SBATCH --mem=15g
#SBATCH -c 4
#SBATCH --time=3-0
#SBATCH --gres=gpu:1,vmem:15g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-1
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./out_v1/d_v1-%j.out # STDOUT
dir=/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning

cd $dir

source /cs/labs/daphna/noam.fluss/usb_new_venv/bin/activate
while [ $# -gt 0 ]; do

   if [[ $1 == *"--"* ]]; then
        v="${1/--/}"
        declare $v="$2"
   fi

  shift
done

echo "Script Name: $0"
echo "First Parameter of the script is $1"
echo "The second Parameter is $2"
echo "Total Number of Parameters: $#"
echo "The process ID is $$"
echo "Exit code for the script: $?"

echo "slurm_job_id: ${SLURM_JOB_ID}?"
echo "run run_v1.sh"
python train_v6.py --c config/classic_cv/flexmatch/my_flexmatch/debug_hyper_parameter_tuning/run_hyper_parameter_tuning/flexmatch_cifar10_40_0_hyperparameter_tuning_new.yaml\
                        --load_path ./saved_models/classic_cv/flexmatch_cifar10_animals/latest_model.pth\
                         --seed ${SLURM_ARRAY_TASK_ID} --save_name flexmatch_cifar10_animals\
                        --epoch 500 --num_train_iter 512000 --num_eval_iter 1024\
                        --num_classes 10 --algorithm flexmatch --save_dir ./saved_models/classic_cv/tuning\
                        --dataset cifar10 --slurm_job_id ${SLURM_JOB_ID}\
                        --lambda_entropy 1.5\
                        --project_wandb test\
                        --num_labels 400\
                        --python_code_version 6 --delete 0\
                        --net_new wrn_28_2\
                        --comment 123401\
                        --missing_labels 0 1 8 9

