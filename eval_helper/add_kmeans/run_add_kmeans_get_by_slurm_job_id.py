import sys
import time

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt

sys.path.append(
    "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper")
from part_of_eval.utils import get_out_path_by_slurm_job_id, EVAL_PATH, path_out_tmp_file


def run_eval_command(slurm_job_id, version, result_df_my_algorithm_unseen_count, is_transductive=0):
    model_path = get_out_path_by_slurm_job_id(slurm_job_id,
                                              "/cs/labs/daphna/noam.fluss/project/"
                                              "SSL_Benchmark/new_forked_semi_supervised_learning/"
                                              "Semi-supervised-learning/saved_models")
    if result_df_my_algorithm_unseen_count is None:
        command_run_eval = fr"{EVAL_PATH} --dataset cifar100 --num_classes 100 --run_threshold_range 1 --is_transductive {is_transductive} --load_path {model_path} --version {version}"
    else:
        print("command_run_eval", result_df_my_algorithm_unseen_count)
        command_run_eval = fr"{EVAL_PATH} --dataset cifar100 --num_classes 100 --run_threshold_range 1 --run_threshold_range_get_same_unseen_count 1 --is_transductive {is_transductive} --my_algorithm_unseen_count {result_df_my_algorithm_unseen_count} --load_path {model_path} --version {version}"

    os.system(command_run_eval)
    time.sleep(0.5)


def get_eval_results(slurm_job_id, kmean_option):
    while not os.path.isfile(f"{path_out_tmp_file}{kmean_option}_{slurm_job_id}.out"):
        time.sleep(0.5)
    with open(f"{path_out_tmp_file}{kmean_option}_{slurm_job_id}.out", "r") as f:
        clustering_accuracy_with_permutation = float(f.readline()[56:])
        seen_classes_accuracy = float(f.readline()[22:])
        unseen_classes_recall = float(f.readline()[22:])
        unseen_classes_precision = float(f.readline()[25:])
        chosen_threshold = float(f.readline()[17:])
        balanced_accuracy = float(f.readline()[18:])
    os.remove(f"{path_out_tmp_file}{kmean_option}_{slurm_job_id}.out")
    return [clustering_accuracy_with_permutation, seen_classes_accuracy, unseen_classes_recall,
            unseen_classes_precision,balanced_accuracy], chosen_threshold


def get_kmeans(slurm_job_id, version, kmean_option="optimal_threshold", result_df_my_algorithm_unseen_count=None,
               is_transductive=0):
    run_eval_command(slurm_job_id, version, result_df_my_algorithm_unseen_count, is_transductive=is_transductive)
    return get_eval_results(slurm_job_id, kmean_option)


# TODO continue from here
def get_eval_diff_list_add_kmean(slurm_job_id_list, version, kmean_option="optimal_threshold",
                                 result_df_my_algorithm_unseen_count_list=None):
    result = []
    threshold_list = []
    for slurm_job_id_index in range(len(slurm_job_id_list)):
        if result_df_my_algorithm_unseen_count_list is not None:
            print("slurm_job_id_list",slurm_job_id_list)
            print("get_eval_diff_list_add_kmean - slurm_job_id_index",slurm_job_id_index)
            print("get_eval_diff_list_add_kmean - result_df_my_algorithm_unseen_count_list",result_df_my_algorithm_unseen_count_list)
            current_result, current_threshold = get_kmeans(slurm_job_id_list[slurm_job_id_index], version,
                                                           kmean_option=kmean_option,
                                                           result_df_my_algorithm_unseen_count=
                                                           result_df_my_algorithm_unseen_count_list[slurm_job_id_index])
        else:
            current_result, current_threshold = get_kmeans(slurm_job_id_list[slurm_job_id_index], version,
                                                           kmean_option=kmean_option)
        result.append(current_result)
        threshold_list.append(current_threshold)
    return result, np.array(threshold_list)
