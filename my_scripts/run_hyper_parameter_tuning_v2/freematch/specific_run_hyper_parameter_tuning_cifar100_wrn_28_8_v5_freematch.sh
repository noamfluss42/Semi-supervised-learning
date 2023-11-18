#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 8
#SBATCH --time=5-0
#SBATCH --gres=gpu:1,vmem:20g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-0
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./freematch/hyper_parameter_out_cifar100_wrn_28_8_v5_freematch/d-%j.out # STDOUT
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

echo "load_path: $load_path?"
echo "seed: $seed?"
echo "save_name: $save_name?"
echo "epoch: $epoch?"
echo "num_train_iter: $num_train_iter?"
echo "num_eval_iter: $num_eval_iter?"
echo "num_classes: $num_classes?"
echo "algorithm: $algorithm?"
echo "save_dir: $save_dir?"
echo "dataset: $dataset?"

##echo "missing_labels: $missing_labels?"
echo "slurm_job_id: ${SLURM_JOB_ID}?"
echo "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8_v5_freematch.sh"
python train_v5.py --c config/classic_cv/freematch/my_scripts/freematch_cifar100_40_0_hyperparameter_tuning.yaml\
                        --load_path $load_path --seed $seed --save_name $save_name\
                        --epoch $epoch --num_train_iter $num_train_iter --num_eval_iter $num_eval_iter\
                        --num_classes $num_classes --algorithm $algorithm --save_dir $save_dir\
                        --dataset $dataset --slurm_job_id ${SLURM_JOB_ID}\
                        --random_missing_labels_num $random_missing_labels_num\
                         --choose_random_labeled_training_set $choose_random_labeled_training_set\
                        --lambda_entropy $lambda_entropy --lb_imb_ratio $lb_imb_ratio\
                        --MNAR_round_type $MNAR_round_type --threshold $threshold\
                        --lambda_datapoint_entropy $lambda_datapoint_entropy --project_wandb $project_wandb\
                        --num_labels $num_labels --ulb_loss_ratio $ulb_loss_ratio\
                        --python_code_version $python_code_version --delete $delete\
                        --weight_decay $weight_decay --new_p_cutoff $new_p_cutoff --net_new $net_new\
                        --lambda_entropy_with_labeled_data $lambda_entropy_with_labeled_data\
                        --lambda_entropy_with_labeled_data_v2 $lambda_entropy_with_labeled_data_v2\
                        --new_ent_loss_ratio $new_ent_loss_ratio\
                        --split_to_superclasses $split_to_superclasses
                        ##--missing_labels $missing_labels

