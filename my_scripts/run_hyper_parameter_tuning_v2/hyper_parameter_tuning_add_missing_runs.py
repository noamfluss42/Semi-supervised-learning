import pprint
import subprocess
import torch
from hyper_parameter_tuning_v2 import main


def add_missing_runs_v1(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)


def add_missing_runs_v2(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0.5,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=4, seed_end=5)


def add_missing_runs_v3(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)
    main(parameters_dict_missing_labels, killable=False, seed_start=4, seed_end=5)


def add_missing_runs_v4(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=2, seed_end=3)
    main(parameters_dict_missing_labels, killable=False, seed_start=4, seed_end=5)


def add_missing_runs_v5(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 20,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)

def add_missing_runs_v6(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 20,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0.5,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)
    main(parameters_dict_missing_labels, killable=False, seed_start=3, seed_end=4)

def add_missing_runs_v7(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 10000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 20,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=4, seed_end=5)


def add_missing_runs_v8(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 5000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, seed_start=1, seed_end=2)


def add_missing_runs_v9(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 5000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.5,1,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, seed_start=1, seed_end=2)
    main(parameters_dict_missing_labels, seed_start=3, seed_end=4)


def add_missing_runs_v10(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 5000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 20,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.5,1],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, seed_start=1, seed_end=2)


def add_missing_runs_v11(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 5000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 20,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, seed_start=3, seed_end=4)

def add_missing_runs_v12(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0.2,
        "lambda_entropy_with_labeled_data_v2": 0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def add_missing_runs_v13(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)



def add_missing_runs_v14(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 3,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)
    main(parameters_dict_missing_labels, seed_start=2, seed_end=4)



def add_missing_runs_v15(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)

def add_missing_runs_v16(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)