import os.path
from os import listdir
from os.path import isfile, join
import subprocess

import numpy as np


def find_parameter(content, parameter):
    parameter_index = content.find(parameter)
    if parameter_index == -1:
        return None

    if "seed" not in parameter:
        if "invalid float value: " in content[parameter_index + len(parameter): content.find("\n", parameter_index)]:
            return None
        else:
            return float(content[parameter_index + len(parameter): content.find("\n", parameter_index)])
    else:
        return float(content[parameter_index + len(parameter): content.find("\n", parameter_index) - 1])

def cancel_job(file_name):
    print("file_name",file_name)
    jobid = file_name[file_name.find("-") + 1:file_name.find(".")]
    subprocess.call(f"scancel {jobid}", shell=True)

def main():
    mypath = r"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/" \
             r"Semi-supervised-learning/my_scripts/run_hyper_parameter_tuning_v2/hyper_parameter_out_cifar100_v2"
    i = 0
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    compare_dict_name = {}
    compare_dict_content = {}
    cuda_out_of_memory = "RuntimeError: CUDA out of memory"
    for file_name in onlyfiles:
        current_path = os.path.join(mypath, file_name)
        with open(current_path, 'r') as file:
            # read all content of a file
            content = file.read()
            # check if string present in a file
            print("file_name",file_name)
            current_num_labels = find_parameter(content, "num_labels: ")
            current_random_missing_labels_num = find_parameter(content, "random_missing_labels_num: ")
            current_lambda_entropy = find_parameter(content, "lambda_entropy: ")
            current_threshold = find_parameter(content, "threshold: ")
            current_seed = find_parameter(content, "seed: ")
            current_new_p_cutoff = find_parameter(content, "new_p_cutoff: ")
            current_net_new = find_parameter(content, "net_new: ")
            key = str([current_num_labels, current_random_missing_labels_num, current_lambda_entropy, current_threshold,
                       current_seed])
            if "None" in key:
                continue
            if cuda_out_of_memory in content:
                if current_net_new is not None:
                    print(current_net_new,"?",file_name)
            if float(current_lambda_entropy) == 0.1 or float(current_lambda_entropy) == 5:
                print("delete",file_name)
                cancel_job(file_name)
            elif current_threshold is not None and float(current_threshold) != 0.95:
                print("delete", file_name)
                cancel_job(file_name)
            elif current_new_p_cutoff is not None and float(current_new_p_cutoff) == 0.95:
                cancel_job(file_name)
            if float(current_lambda_entropy) == 0 and (current_new_p_cutoff is not None and 0.4 <= float(current_new_p_cutoff) < 0.95):
                cancel_job(file_name)

            if key in compare_dict_name:
                compare_dict_name[key].append(file_name)
                compare_dict_content[key].append(content)
            else:
                compare_dict_name[key] = [file_name]
                compare_dict_content[key] = [content]

                # jobid = file_name[file_name.find("-") + 1:file_name.find(".")]
                # subprocess.call(f"scancel {jobid}",shell=True)
                # os.system(f"scancel {jobid}")
    searched_str_state = "iteration, USE_EMA: True, train/sup_loss:"
    # for key in compare_dict_content:
    #     min_index = np.argmin([len(content) for content in compare_dict_content[key]])
    #     if searched_str_state not in compare_dict_content[key][min_index]:
    #         continue
    #     else:
    #         searched_state_line = [line for line in compare_dict_content[key][min_index].split("\n") if searched_str_state in line][-1]
    #         searched_state_line = searched_state_line[searched_state_line.find("iteration"):searched_state_line.find("train/run_time")]
    #     for content in compare_dict_content[key]:
    #         if searched_state_line not in content:
    #             print("?",key)

    # for key in compare_dict_name:
    #     current_iteration = []
    #     if len(compare_dict_content[key]) == 1:
    #         continue
    #     if len(compare_dict_content[key][0]) > len(compare_dict_content[key][1]):
    #
    #         cancel_job(compare_dict_name[key][1])
    #     else:
    #         cancel_job(compare_dict_name[key][0])

if __name__ == "__main__":
    main()
