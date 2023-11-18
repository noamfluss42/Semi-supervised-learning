import time

import numpy as np
import os
import pandas as pd
from matplotlib import pyplot as plt
from run_validation_compare_get_by_slurm_job_id import *

version = 3


def get_first_mean(row):
    result = []
    for slurm_job_id_info in row.split("_"):
        result.append(float(slurm_job_id_info.split("-")[0]))
    return np.array(result).mean()


def get_second_mean(row):
    result = []
    for slurm_job_id_info in row.split("_"):
        result.append(float(slurm_job_id_info.split("-")[1]))
    return np.array(result).mean()


def get_diff_mean(row):
    result = []
    for slurm_job_id_info in row.split("_"):
        result.append(float(slurm_job_id_info.split("-")[2]))
    return np.array(result).mean()


def get_diff_max(row):
    result = []
    for slurm_job_id_info in row.split("_"):
        result.append(abs(float(slurm_job_id_info.split("-")[2])))
    return max(result)


def main1(result_df):
    a = result_df["slurm_job_id"].apply(get_eval_diff_list_string)
    result_df["first_mean"] = a.apply(get_first_mean)
    result_df["second_mean"] = a.apply(get_second_mean)
    result_df["diff_mean"] = a.apply(get_diff_mean)
    result_df["diff_max"] = a.apply(get_diff_max)
    plt.bar(result_df["name"], result_df["first_mean"], label="first_mean")
    plt.savefig("1.png")
    plt.clf()
    plt.bar(result_df["name"], result_df["second_mean"], label="second_mean")
    plt.savefig("2.png")
    plt.clf()
    plt.bar(result_df["name"], result_df["diff_mean"], label="diff_mean")
    plt.savefig("3.png")
    plt.clf()
    plt.bar(result_df["name"], result_df["diff_max"], label="diff_max")
    plt.savefig("4.png")
    plt.clf()


def get_slurm_job_id_list_from_str(slurm_job_id_list_str):
    return slurm_job_id_list_str[2:-2].split("' '")


def analyze_result_validation_test(result_test):
    result_test = np.array(result_test)
    first_half_accuracy_mean = result_test[:, 0].mean()
    first_half_accuracy_standard_error = np.std(result_test[:, 0], ddof=1) / np.sqrt(np.size(result_test[:, 0]))
    second_half_accuracy_mean = result_test[:, 1].mean()
    second_half_accuracy_standard_error = np.std(result_test[:, 1], ddof=1) / np.sqrt(np.size(result_test[:, 1]))
    diff_mean = result_test[:, 2].mean()
    diff_mean_standard_error = np.std(result_test[:, 2], ddof=1) / np.sqrt(np.size(result_test[:, 2]))
    diff_max = max(result_test[:, 2])
    return [first_half_accuracy_mean, first_half_accuracy_standard_error,
            second_half_accuracy_mean, second_half_accuracy_standard_error,
            diff_mean, diff_mean_standard_error, diff_max]


def main2(result_df):
    a = result_df["slurm_job_id"].apply(get_slurm_job_id_list_from_str).apply(get_eval_diff_list)
    # result_df[["first_half_accuracy", "second_half_accuracy", "abs_diff"]] = result_df["slurm_job_id"].apply(
    #    get_eval_diff_list)
    b = a.apply(analyze_result_validation_test)
    result_df[["first_half_accuracy_mean", "first_half_accuracy_standard_error",
               "second_half_accuracy_mean", "second_half_accuracy_standard_error",
               "diff_mean", "diff_mean_standard_error", "diff_max"]] = pd.DataFrame(b.tolist(), index=b.index)
    return result_df


if __name__ == '__main__':
    # result = {"name": ['Ours', 'flexmatch', 'rssl', 'softmatch'],
    #           "slurm_job_id": ["_".join(['15537394', '15537430', '15537466', '15537502', '15537538']),
    #                            "_".join(['15535259', '15615211', '15535331', '15535367', '15537357']),
    #                            "_".join(['15594917', '15594919', '15594921', '15594923', '15594925']),
    #                            "_".join(['15576286', '15576292', '15576298', '15576304', '15576310'])]}
    # main1(pd.DataFrame(result))
    path = f"/cs/labs/daphna/noam.fluss/project/create_graphs/output/debug_v{version}.csv"
    result_df = main2(pd.read_csv(path, index_col=0))
    result_df.to_csv(path)
    print("finish")
