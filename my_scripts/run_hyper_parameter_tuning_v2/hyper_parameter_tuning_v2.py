import pprint
import subprocess

NEEDED = ["load_path", "seed", "save_name"]


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
    p = subprocess.run(current_subprocess_run_list)
    # print("\nend", current_subprocess_run_list)
    # print("len(current_subprocess_run_list)", len(current_subprocess_run_list))
    return 1
    # print("p", p)


def main(parameters_dict, killable=True, seed_start=0, seed_end=5):
    parameters_dict["python_code_version"] = 2
    keys = [key for key in list(parameters_dict.keys()) if parameters_dict[key] is not None]
    print("len keys_update", len(keys))
    for i in range(seed_start, seed_end):
        print("seed - ", i)
        parameters_dict["seed"] = i
        parameters_dict[
            "save_name"] = f"{parameters_dict['algorithm']}_{parameters_dict['dataset']}_{parameters_dict['num_labels']}_{i}"
        parameters_dict["load_path"] = f"./saved_models/classic_cv/{parameters_dict['save_name']}/latest_model.pth"
        if parameters_dict["algorithm"] == "comatch":
            run_batch_list_option(parameters_dict, keys, 0,
                                  ["sbatch", "specific_run_hyper_parameter_tuning_v2_comatch.sh"])
        if parameters_dict["algorithm"] == "adamatch":
            run_batch_list_option(parameters_dict, keys, 0,
                                  ["sbatch", "specific_run_hyper_parameter_tuning_v2_adamatch.sh"])
        if parameters_dict["dataset"] == "cifar10" and killable:
            run_batch_list_option(parameters_dict, keys, 0, ["sbatch", "specific_run_hyper_parameter_tuning_v2.sh"])
        elif parameters_dict["dataset"] == "cifar100" and killable:
            run_batch_list_option(parameters_dict, keys, 0,
                                  ["sbatch", "specific_run_hyper_parameter_tuning_cifar100_v2.sh"])
        elif parameters_dict["dataset"] == "cifar100" and not killable:
            print("test 1")
            run_batch_list_option(parameters_dict, keys, 0,
                                  ["sbatch", "specific_run_hyper_parameter_tuning_cifar100_not_killable_v2.sh"])
        elif parameters_dict["dataset"] == "cifar10" and not killable:
            run_batch_list_option(parameters_dict, keys, 0,
                                  ["sbatch", "specific_run_hyper_parameter_tuning_not_killable_v2.sh"])


def call_flexmatch_missing_labels(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 614400,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [4],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [2],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,

        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels)


def call_flexmatch_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 30,  # 500,
        "num_train_iter": 30720,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 2, 5],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 2],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "test",
        "python_code_version": 2,

        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels, killable=False)


def call_free_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 30,  # 500,
        "num_train_iter": 30720,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 2, 4],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "test",
        "python_code_version": 2,

        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels, killable=False)


def call_softmatch_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 30,  # 500,
        "num_train_iter": 30720,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "softmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 2, 4],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "test",
        "python_code_version": 2,

        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels, killable=False)


def call_flexmatch_missing_labels_lambda_u(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 614400,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [4, 6],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [2, 5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": [0.1, 0.5, 0.8, 1.2]
    }
    main(parameters_dict_missing_labels)


def call_flexmatch_missing_labels_250(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 614400,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 250,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [1, 4, 6],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 2, 5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels)


def call_flexmatch_few_labels(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 614400,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": [10, 20, 30, 40],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": 0,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels)


def call_flexmatch_missing_labels_debug():
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 614400,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [4, 6],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.1, 8],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "Flexmatch_9_project_missing_labels",
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def call_flexmatch_MNAR():
    parameters_dict_MNAR = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 0.1, 3, 5],
        "lb_imb_ratio": [20, 50, 100],
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "Flexmatch_second_project_MNAR",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_MNAR)


def call_adamatch_missing_labels(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 1048576,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "adamatch",
        "save_dir": "./saved_models/classic_cv/tuning/adamatch/",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 1],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def call_comatch_missing_labels(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,  # 500,
        "num_train_iter": 1048576,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 10,
        "num_labels": 40,
        "algorithm": "comatch",
        "save_dir": "./saved_models/classic_cv/tuning/comatch/",
        "dataset": "cifar10",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 1],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def debug_call_flexmatch_missing_labels_cifar100(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 3,  # 500,
        "num_train_iter": 3072,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.1, 1, 2],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "thresh_warmup": True,
        "delete": 1,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)



def debug_call_flexmatch_missing_labels_cifar100_single_run(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 3,  # 500,
        "num_train_iter": 3072,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.1],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "thresh_warmup": True,
        "delete": 1,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)




if __name__ == "__main__":
    # version = 10
    # print("version - ",version)
    # call_adamatch_missing_labels(version)
    version = "debug"
    print("version - ", version)
    debug_call_flexmatch_missing_labels_cifar100_single_run(version)
    # call_flexmatch_few_labels(version)
