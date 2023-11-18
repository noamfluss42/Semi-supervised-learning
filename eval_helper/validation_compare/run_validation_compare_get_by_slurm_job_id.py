import sys
import time

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
sys.path.append("/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper/")
from part_of_eval.util import get_out_path_by_slurm_job_id,EVAL_PATH,path_out_tmp_file



def get_eval_diff(slurm_job_id):
    model_path = get_out_path_by_slurm_job_id(slurm_job_id,
                                              "/cs/labs/daphna/noam.fluss/project/"
                                              "SSL_Benchmark/new_forked_semi_supervised_learning/"
                                              "Semi-supervised-learning/saved_models")
    command_run_eval = fr"{EVAL_PATH} --dataset cifar100 --num_classes 100 --save_test_val 1 --load_path {model_path}"
    os.system(command_run_eval)
    time.sleep(0.5)
    while not os.path.isfile(f"{path_out_tmp_file}{slurm_job_id}.out"):
        time.sleep(0.5)
    with open(f"{path_out_tmp_file}{slurm_job_id}.out", "r") as f:
        first_half_accuracy = float(f.readline()[20:])
        second_half_accuracy = float(f.readline()[21:-1])
        diff = abs(float(f.readline()[5:]))
    os.remove(f"{path_out_tmp_file}{slurm_job_id}.out")
    return [first_half_accuracy, second_half_accuracy, diff]


def get_eval_diff_list(slurm_job_id_list):
    result = []
    for slurm_job_id in slurm_job_id_list:
        result.append(get_eval_diff(slurm_job_id))
    return result


def get_eval_diff_list_string(slurm_job_id_list):
    result = []
    for slurm_job_id in slurm_job_id_list.split("_"):
        tmp = get_eval_diff(slurm_job_id)
        result.append("-".join([str(tmp[0]), str(tmp[1]), str(tmp[2])]))
    return "_".join(result)
