import sys
import time

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from run_add_kmeans_get_by_slurm_job_id import get_eval_diff_list_add_kmean
from functools import *
import sys

columns_to_update = ["accuracy/iteration validation clustering accuracy with missing labels - mean",
                     "accuracy/iteration validation clustering accuracy with missing labels - standard error",
                     "accuracy/iteration validation accuracy appearing lables - mean",
                     "accuracy/iteration validation accuracy appearing lables - standard error",
                     "accuracy/iteration validation accuracy missing labels recall - mean",
                     "accuracy/iteration validation accuracy missing labels recall - standard error",
                     "accuracy/iteration validation accuracy missing labels precision - mean",
                     "accuracy/iteration validation accuracy missing labels precision - standard error",
                     "accuracy/balanced accuracy - mean",
                     "accuracy/balanced accuracy - standard error"]
use_k_means_dict = {"optimal_threshold": "optimal",
                    "same_unseen_count_threshold": "same_unseen"}
name_dict = {"optimal_threshold": f"\nadd kmeans\noptimal\nt-",
             "same_unseen_count_threshold": f"\nadd kmeans\nby unseen\nt-"}


def get_slurm_job_id_list_from_str(slurm_job_id_list_str):
    return slurm_job_id_list_str[2:-2].split("' '")


def analyze_result_validation_test(result_test):
    result_test = np.array(result_test)
    result = []
    for i in range(result_test.shape[1]):
        result.append(result_test[:, i].mean())
        result.append(np.std(result_test[:, i], ddof=1) / np.sqrt(np.size(result_test[:, i])))
    return result


def add_kmeans_optimal_or_unseen_count(result_df, kmean_option="optimal_threshold", version=-1,
                                       result_df_my_algorithm_unseen_count_list=None):
    if result_df_my_algorithm_unseen_count_list is None or len(result_df_my_algorithm_unseen_count_list) == 0:
        result_df_my_algorithm_unseen_count_list = None
    print("add_kmeans_optimal_or_unseen_count",result_df["slurm_job_id"])
    slurm_job_id_list = result_df["slurm_job_id"].apply(get_slurm_job_id_list_from_str).iloc[0]

    if result_df_my_algorithm_unseen_count_list is None:
        eval_diff_list_add_kmean, threshold_list = get_eval_diff_list_add_kmean(slurm_job_id_list, version=version,
                                                                                kmean_option=kmean_option)
    else:
        eval_diff_list_add_kmean, threshold_list = get_eval_diff_list_add_kmean(slurm_job_id_list, version=version,
                                                                                kmean_option=kmean_option,
                                                                                result_df_my_algorithm_unseen_count_list=result_df_my_algorithm_unseen_count_list)

    result_validation_test = analyze_result_validation_test(eval_diff_list_add_kmean)
    for column_index in range(len(columns_to_update)):
        result_df[columns_to_update[column_index]] = result_validation_test[column_index]

    result_df["use_k_means"] = use_k_means_dict[kmean_option]
    threshold_chosen_mean = round(threshold_list.mean(), 3)
    result_df["name"] = result_df["name"] + name_dict[kmean_option] + str(threshold_chosen_mean)
    return result_df


def create_new_csv(path, version):
    result_df = pd.read_csv(path, index_col=0)
    new_path = f"/cs/labs/daphna/noam.fluss/project/create_graphs/output/debug_v{version}_with_kmeans.csv"
    result_df.to_csv(new_path)


def filter_result_df_only_clean_algorithms(result_df):
    result_df["lambda_entropy_with_labeled_data_non_nan"] = result_df["lambda_entropy_with_labeled_data"].replace(
        np.nan, -1)
    result_df["lambda_entropy_with_labeled_data_v2_non_nan"] = result_df["lambda_entropy_with_labeled_data_v2"].replace(
        np.nan, -1)
    result_df = result_df[result_df["lambda_entropy"] == 0]
    result_df = result_df[
        (result_df["lambda_entropy_with_labeled_data"] == 0) | (
                result_df["lambda_entropy_with_labeled_data_non_nan"] == -1)]
    result_df = result_df[
        (result_df["lambda_entropy_with_labeled_data_v2"] == 0) | (
                result_df["lambda_entropy_with_labeled_data_v2_non_nan"] == -1)]
    result_df.drop(['lambda_entropy_with_labeled_data_non_nan', 'lambda_entropy_with_labeled_data_v2_non_nan'], axis=1,
                   inplace=True)
    result_df = result_df[result_df["algorithm"] != "openmatch"]
    return result_df


def add_all_kmeans(result_df, version, result_df_my_algorithm_unseen_count_list):
    result_df_optimal_threshold = add_kmeans_optimal_or_unseen_count(result_df.copy(), kmean_option="optimal_threshold",
                                                                     version=version)
    print("result_df after 1 add_kmeans_optimal", result_df_optimal_threshold["name"])
    # result_df_same_unseen_count_threshold = add_kmeans_optimal_or_unseen_count(result_df.copy(),
    #                                                                            kmean_option="same_unseen_count_threshold",
    #                                                                            version=version,
    #                                                                            result_df_my_algorithm_unseen_count_list=result_df_my_algorithm_unseen_count_list)
    # print("result_df_same_unseen_count_threshold columns", result_df_same_unseen_count_threshold.columns)
    # print("result_df after 2 add_kmeans_optimal", result_df_same_unseen_count_threshold["name"])

    # result_df = pd.concat([result_df_optimal_threshold, result_df_same_unseen_count_threshold],
    #                       ignore_index=True).reset_index()
    result_df = pd.concat([result_df_optimal_threshold], ignore_index=True).reset_index()
    return result_df


def save_result_df_with_kmeans(result_df, path, version):
    result_df.drop(columns=["index"], inplace=True)
    result_df.fillna("null", inplace=True)
    result_df.to_csv(path, mode='a', index=True, header=False)
    time.sleep(1)


def get_specific_unseen_count_by_slurm_job_id_from_file(slurm_job_id):

    path = f"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper/result_output/tmp_res_unseen_count_{slurm_job_id}.out"
    while not os.path.isfile(path):
        time.sleep(0.5)
    with open(path, "r") as f:
        unseen_count = int(f.readline()[13:])

    os.remove(path)
    return unseen_count


def main():
    if len(sys.argv) > 1:
        version = sys.argv[1]
        print(f"start run_add_kmeans_to_csv.py, version {version}")
        path = f"/cs/labs/daphna/noam.fluss/project/create_graphs/output/debug_v{version}.csv"
        path_read = path
    else:
        version = 100
        path_read = f"/cs/labs/daphna/noam.fluss/project/create_graphs/output/debug_v{version}_try.csv"
        path = f"/cs/labs/daphna/noam.fluss/project/create_graphs/output/debug_v{version}.csv"
    print("path_read",path_read)
    result_df = pd.read_csv(path_read, index_col=0)
    my_algorithm_slurm_job_id_list = result_df[(result_df["name"] == "Ours") | (result_df["name"] == "Ours wrn_28_8")]
    if len(my_algorithm_slurm_job_id_list) > 0:
        my_algorithm_slurm_job_id_list = my_algorithm_slurm_job_id_list.iloc[0]["slurm_job_id"][2:-2].split("' '")

    result_df_my_algorithm_unseen_count_list = [get_specific_unseen_count_by_slurm_job_id_from_file(slurm_job_id) for
                                                slurm_job_id in
                                                my_algorithm_slurm_job_id_list]
    result_df = filter_result_df_only_clean_algorithms(result_df)
    for i in range(len(result_df)):
        result_df_current = add_all_kmeans(
            result_df[result_df["slurm_job_id"] == result_df.iloc[i]["slurm_job_id"]].copy(), version,
            result_df_my_algorithm_unseen_count_list)
        save_result_df_with_kmeans(result_df_current, path, version)
    create_new_csv(path, version)
    print("finish,run_add_kmeans_to_csv.py")


if __name__ == '__main__':
    main()
