#!/bin/bash
#SBATCH --mem=20g
#SBATCH -c 8
#SBATCH --time=4-0
#SBATCH --gres=gpu:1,vmem:20g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-0
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./debug_runs_out/d-%j.out # STDOUT
##SBATCH --killable
##SBATCH --requeue
##torch 1.9+cuda111, torchaudio 090 torchvision 0100+cuda11
## what we had before
## torch==1.13.0
##torchaudio==0.13.0
##torchvision==0.13.0
dir=/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning

cd $dir

source /cs/labs/daphna/noam.fluss/usb_venv/bin/activate
echo "Script Name:my_flexmatch_cifar10_40_0.yaml"
python3.7 train_v2.py --c config/classic_cv/flexmatch/my_flexmatch/my_flexmatch_cifar10_40_0.yaml --slurm_job_id ${SLURM_JOB_ID}


