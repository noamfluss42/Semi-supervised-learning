#!/bin/bash
#SBATCH --mem=3g
#SBATCH -c 4
#SBATCH --time=6-0
#SBATCH --gres=gpu:0,vmem:1g
## 10g selects a better GPU, if we're paying for it. 9153189
## 0-9%4 limits to 4 jobs symultanously, 0-9:4 will run jobs 0,4,8 ${SLURM_JOB_ID}
#SBATCH --array=0-0
#SBATCH --exclude=gsm-04,gsm-01,gsm-03
#SBATCH --output ./main2/d-%j.out # STDOUT
##SBATCH --killable
##SBATCH --requeue
##torch 1.9+cuda111, torchaudio 090 torchvision 0100+cuda11
## what we had before
## torch==1.13.0
##torchaudio==0.13.0
##torchvision==0.13.0
dir=/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/my_scripts/run_hyper_parameter_tuning_v2

cd $dir

source /cs/labs/daphna/noam.fluss/usb_venv/bin/activate

echo "Script Name: $0"
echo "try v2"
python3.7 hyper_parameter_tuning_v2.py


