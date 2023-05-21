import pprint
import subprocess
import torch
from hyper_parameter_tuning_main_run import main



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



def flexmatch_missing_labels_cifar100_v2(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
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
        "lambda_entropy": [1, 2],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": [0.7, 0.6, 0.95],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def debug_call_flexmatch_missing_labels_cifar100(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
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
        "lambda_entropy": [0, 0.1, 5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": [0.5, 0.95],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v3(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "",
        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 1, 2, 5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [0.5, 0.8],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v3_single(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 20,
        "num_train_iter": 20480,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "",
        "missing_labels": None,
        "random_missing_labels_num": [0],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [0.5, 0.95],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def debug_call_flexmatch_missing_labels_cifar100_single_run(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 50,  # 500,
        "num_train_iter": 51200,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [0],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 1,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=2)


def debug_call_flexmatch_missing_labels_cifar100_single_run_missing_classes(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 50,  # 500,
        "num_train_iter": 51200,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [4],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 1,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v4(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 200,
        "num_train_iter": 204800,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 2, 4],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [0.6, 0.8, 0.95],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)


def flexmatch_missing_labels_cifar100_v5(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 200,
        "num_train_iter": 204800,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [3, 3.5, 4.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)


def flexmatch_missing_labels_cifar100_v6(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 200,
        "num_train_iter": 204800,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [3, 3.5, 4.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=3)


def cifar100_flexmatch_missing_labels_test_killable_v3(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 10,  # 500,
        "num_train_iter": 10240,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 100,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": ["None"],
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1, 2, 3],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "test",
        "python_code_version": 2,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "weight_decay": 0.001,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


