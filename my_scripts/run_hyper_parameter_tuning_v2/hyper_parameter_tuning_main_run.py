import pprint
import subprocess
import torch

NEEDED = ["load_path", "seed", "save_name"]

algorithm_dataset_net_new_killable_to_shell = {
    ("flexmatch", "cifar10", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                            "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                            "my_scripts/run_hyper_parameter_tuning_v2/"
                                            "specific_run_hyper_parameter_tuning_v2.sh",
    ("flexmatch", "cifar10", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                             "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                             "my_scripts/run_hyper_parameter_tuning_v2/"
                                             "specific_run_hyper_parameter_tuning_not_killable_v2.sh",
    ("flexmatch", "cifar100", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                             "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                             "my_scripts/run_hyper_parameter_tuning_v2/"
                                             "specific_run_hyper_parameter_tuning_cifar100_v2.sh",
    ("flexmatch", "cifar100", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                              "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                              "my_scripts/run_hyper_parameter_tuning_v2/"
                                              "specific_run_hyper_parameter_tuning_cifar100_not_killable_v2.sh",
    ("flexmatch", "cifar100", "wrn_28_8", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                                 "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                                 "my_scripts/run_hyper_parameter_tuning_v2/"
                                                 "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8_v2.sh",

    ("softmatch", "cifar10", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                            "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                            "my_scripts/run_hyper_parameter_tuning_v2/"
                                            "specific_run_hyper_parameter_tuning_v2.sh",
    ("softmatch", "cifar10", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                             "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                             "my_scripts/run_hyper_parameter_tuning_v2/"
                                             "specific_run_hyper_parameter_tuning_not_killable_v2.sh",
    ("softmatch", "cifar100", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                             "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                             "my_scripts/run_hyper_parameter_tuning_v2/"
                                             "specific_run_hyper_parameter_tuning_cifar100_v2.sh",
    ("softmatch", "cifar100", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                              "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                              "my_scripts/run_hyper_parameter_tuning_v2/"
                                              "specific_run_hyper_parameter_tuning_cifar100_not_killable_v2.sh",
    ("softmatch", "cifar100", "wrn_28_8", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                                 "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                                 "my_scripts/run_hyper_parameter_tuning_v2/"
                                                 "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8_v2.sh",

    ("adamatch", "cifar100", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                            "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                            "my_scripts/run_hyper_parameter_tuning_v2/adamatch/"
                                            "specific_run_hyper_parameter_tuning_cifar100_v2_adamatch.sh",
    ("adamatch", "cifar100", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                             "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                             "my_scripts/run_hyper_parameter_tuning_v2/adamatch/"
                                             "specific_run_hyper_parameter_tuning_cifar100_v2_adamatch_not_killable.sh",

    ("adamatch", "cifar100", "wrn_28_8", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                            "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                            "my_scripts/run_hyper_parameter_tuning_v2/adamatch/"
                                            "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8_v4_adamatch.sh",



    ("comatch", "cifar100", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                           "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                           "my_scripts/run_hyper_parameter_tuning_v2/comatch/"
                                           "specific_run_hyper_parameter_tuning_cifar100_v2_comatch.sh",
    ("comatch", "cifar100", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                            "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                            "my_scripts/run_hyper_parameter_tuning_v2/comatch/"
                                            "specific_run_hyper_parameter_tuning_cifar100_v2_comatch_not_killable.sh",
    ("comatch", "cifar100", "wrn_28_8", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                           "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                           "my_scripts/run_hyper_parameter_tuning_v2/comatch/"
                                           "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8_v4_comatch.sh",



    ("freematch", "cifar100", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                             "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                             "my_scripts/run_hyper_parameter_tuning_v2/freematch/"
                                             "specific_run_hyper_parameter_tuning_cifar100_v4_freematch.sh",
    ("freematch", "cifar100", "None", False): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                              "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                              "my_scripts/run_hyper_parameter_tuning_v2/freematch/"
                                              "specific_run_hyper_parameter_tuning_cifar100_v5_freematch_not_killable.sh",
    ("freematch", "cifar100", "None", True): "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
                                              "new_forked_semi_supervised_learning/Semi-supervised-learning/"
                                              "my_scripts/run_hyper_parameter_tuning_v2/freematch/"
                                              "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8_v4_freematch.sh",

}


def run_batch_list_option(parameters_dict, keys, current_keys_list_index, current_subprocess_run_list):
    """

    Args:
        parameters_dict:
        keys:
        current_keys_list_index:
        current_subprocess_run_list:

    Returns:
        -1 = return also
        1 = continue

    """
    print("run_batch_list_option")
    # print("run current_subprocess_run_list",current_subprocess_run_list)
    if current_keys_list_index == len(keys):
        print("\n\nv2 - run current_subprocess_run_list current_keys_list_index == len(keys)",
              " - ".join(current_subprocess_run_list))
        p = subprocess.run(current_subprocess_run_list)
        print("p", p)
        # print("\nstart", current_subprocess_run_list)
        # print("len(current_subprocess_run_list)",len(current_subprocess_run_list))
        return 1

    for i in range(current_keys_list_index, len(keys)):
        if type(parameters_dict[keys[i]]) == list and len(parameters_dict[keys[i]]) == 0:
            continue
        current_subprocess_run_list.append("--" + keys[i])
        if type(parameters_dict[keys[i]]) != list:
            current_subprocess_run_list.append(str(parameters_dict[keys[i]]))
        else:
            current_subprocess_run_list.append("")
            for option_key in parameters_dict[keys[i]]:
                current_subprocess_run_list[-1] = str(option_key)
                option_key_current_subprocess_run_list = current_subprocess_run_list.copy()
                run_batch_list_option(parameters_dict, keys, i + 1,
                                      option_key_current_subprocess_run_list)
            return -1
    print("\n\nv2 - run current_subprocess_run_list", " - ".join(current_subprocess_run_list))
    print(" ".join(current_subprocess_run_list))
    p = subprocess.run(current_subprocess_run_list)
    # print("\nend", current_subprocess_run_list)
    # print("len(current_subprocess_run_list)", len(current_subprocess_run_list))
    return 1
    # print("p", p)


def complete_values(parameters_dict):
    if "net_new" not in parameters_dict or parameters_dict["net_new"] == "":
        parameters_dict["net_new"] = "None"
    if "new_p_cutoff" not in parameters_dict:
        parameters_dict["new_p_cutoff"] = -1
    if "threshold" not in parameters_dict:
        parameters_dict["threshold"] = 0.95
    if "MNAR_round_type" not in parameters_dict:
        parameters_dict["MNAR_round_type"] = "ceil"
    if "python_code_version" not in parameters_dict:
        parameters_dict["python_code_version"] = 4
    if parameters_dict["python_code_version"] >= 3 and "lambda_entropy_with_labeled_data" not in parameters_dict:
        parameters_dict["lambda_entropy_with_labeled_data"] = 0
    if parameters_dict["python_code_version"] >= 4 and "lambda_entropy_with_labeled_data_v2" not in parameters_dict:
        parameters_dict["lambda_entropy_with_labeled_data_v2"] = 0
    if parameters_dict["python_code_version"] >= 5 and "new_ent_loss_ratio" not in parameters_dict:
        parameters_dict["new_ent_loss_ratio"] = -1
    if parameters_dict["python_code_version"] >= 5 and "split_to_superclasses" not in parameters_dict:
        parameters_dict["split_to_superclasses"] = 0
    # if "comment" not in parameters_dict:
    #     parameters_dict["comment"] = 0


def update_file_by_python_code_version(shell_file, parameters_dict):
    print("start update_file_by_python_code_version")
    print("original shell_file",shell_file)
    print(f"python_code_version - {parameters_dict['python_code_version']}")
    if f"v{parameters_dict['python_code_version']}" in shell_file:
        return shell_file
    else:
        for v in range(parameters_dict['python_code_version']):
            if f"v{v}" in shell_file:
                start_replace_index = len("/cs/labs/daphna/noam.fluss/"
                                          "project/SSL_Benchmark/new_forked_semi_supervised_learning/"
                                          "Semi-supervised-learning/my_scripts/run_hyper_parameter_tuning_v2") + 1
                return shell_file[:start_replace_index] + \
                       shell_file[start_replace_index:].replace(f"v{v}", f"v{parameters_dict['python_code_version']}")
    exit(f"version number ({parameters_dict['python_code_version']}) not in name - {shell_file}")
    return ""


def main(parameters_dict, killable=True, seed_start=0, seed_end=5):
    complete_values(parameters_dict)
    keys = [key for key in list(parameters_dict.keys()) if parameters_dict[key] is not None]
    print("len keys_update", len(keys))
    for i in range(seed_start, seed_end):
        print("seed - ", i)
        parameters_dict["seed"] = i
        parameters_dict[
            "save_name"] = f"{parameters_dict['algorithm']}_{parameters_dict['dataset']}_{i}"
        parameters_dict["load_path"] = f"./saved_models/classic_cv/{parameters_dict['save_name']}/latest_model.pth"
        shell_file = algorithm_dataset_net_new_killable_to_shell[
            (parameters_dict["algorithm"], parameters_dict["dataset"], parameters_dict["net_new"], killable)]
        shell_file = update_file_by_python_code_version(shell_file, parameters_dict)
        print("shell_file", shell_file)
        run_batch_list_option(parameters_dict, keys, 0,
                              ["sbatch", shell_file])
        #
        # if parameters_dict["algorithm"] == "flexmatch" and parameters_dict["net_new"] == "wrn_28_8":
        #     run_batch_list_option(parameters_dict, keys, 0,
        #                           ["sbatch", "specific_run_hyper_parameter_tuning_cifar100_wrn_28_8.sh"])
        # elif parameters_dict["algorithm"] == "comatch":
        #     run_batch_list_option(parameters_dict, keys, 0,
        #                           ["sbatch", "specific_run_hyper_parameter_tuning_v2_comatch.sh"])
        # elif parameters_dict["algorithm"] == "adamatch":
        #     if parameters_dict["dataset"] == "cifar100":
        #         if killable:
        #             run_batch_list_option(parameters_dict, keys, 0,
        #                                   ["sbatch",
        #                                    "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
        #                                    "new_forked_semi_supervised_learning/Semi-supervised-learning/"
        #                                    "my_scripts/run_hyper_parameter_tuning_v2/adamatch/"
        #                                    "specific_run_hyper_parameter_tuning_cifar100_v2_adamatch.sh"])
        #         else:
        #
        #             run_batch_list_option(parameters_dict, keys, 0,
        #                                   ["sbatch",
        #                                    "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
        #                                    "new_forked_semi_supervised_learning/Semi-supervised-learning/"
        #                                    "my_scripts/run_hyper_parameter_tuning_v2/adamatch/"
        #                                    "specific_run_hyper_parameter_tuning_cifar100_v2_adamatch_not_killable.sh"])
        #     else:
        #         run_batch_list_option(parameters_dict, keys, 0,
        #                               ["sbatch", "specific_run_hyper_parameter_tuning_v2_adamatch.sh"])
        # elif parameters_dict["dataset"] == "cifar10" and killable:
        #     run_batch_list_option(parameters_dict, keys, 0, ["sbatch", "specific_run_hyper_parameter_tuning_v2.sh"])
        # elif parameters_dict["dataset"] == "cifar100" and killable:
        #     run_batch_list_option(parameters_dict, keys, 0,
        #                           ["sbatch", "specific_run_hyper_parameter_tuning_cifar100_v2.sh"])
        # elif parameters_dict["dataset"] == "cifar100" and not killable:
        #     print("test 1")
        #     run_batch_list_option(parameters_dict, keys, 0,
        #                           ["sbatch", "specific_run_hyper_parameter_tuning_cifar100_not_killable_v2.sh"])
        # elif parameters_dict["dataset"] == "cifar10" and not killable:
        #     run_batch_list_option(parameters_dict, keys, 0,
        #                           ["sbatch", "specific_run_hyper_parameter_tuning_not_killable_v2.sh"])
        print(f"end seed - {i}")
