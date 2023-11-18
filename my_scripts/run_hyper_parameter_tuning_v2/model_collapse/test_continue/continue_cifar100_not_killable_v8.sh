#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 8
#SBATCH --time=5-0
#SBATCH --gres=gpu:1,vmem:20g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-0
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./continue_cifar100_v8_out_dir/d_17126773.out # STDOUT
#SBATCH --killable
#SBATCH --requeue
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


##echo "missing_labels: $missing_labels?"
echo "slurm_job_id: ${SLURM_JOB_ID}?"
echo "continue_cifar100_v8_out_dir.sh"
# 861 epoch
# time 214.747
# write the same line as above, but with line breaks so the user can read it
python train_v8.py\
  --c config/classic_cv/flexmatch/my_flexmatch/debug_hyper_parameter_tuning/run_hyper_parameter_tuning/flexmatch_cifar100_40_0_hyperparameter_tuning.yaml \
  --load_path ./saved_models/classic_cv/flexmatch_cifar100_0/latest_model.pth \
  --seed 0 \
  --save_name flexmatch_cifar100_0 \
  --epoch 500 \
  --num_train_iter 512000  \
  --num_eval_iter 1024 \
  --num_classes 100 \
  --algorithm flexmatch \
  --save_dir ./saved_models/classic_cv/tuning \
  --dataset cifar100 \
  --slurm_job_id 17126773 \
  --random_missing_labels_num 35 \
  --choose_random_labeled_training_set -1 \
  --lambda_entropy 1.5 \
  --lb_imb_ratio 1 \
  --MNAR_round_type ceil \
  --threshold 0.95 \
  --lambda_datapoint_entropy 0 \
  --project_wandb Flexmatch_10_project_missing_labels \
  --num_labels 400 \
  --ulb_loss_ratio 1.0 \
  --python_code_version 5 \
  --delete 0 \
  --weight_decay 0.001 \
  --new_p_cutoff -1 \
  --net_new wrn_28_8 \
  --lambda_entropy_with_labeled_data 0 \
  --lambda_entropy_with_labeled_data_v2 0 \
  --new_ent_loss_ratio -1 \
  --split_to_superclasses 0
  --batch_size \
  --choose_random_labeled_training_set_duplicateb \
  --save_all_models \
  --comment
