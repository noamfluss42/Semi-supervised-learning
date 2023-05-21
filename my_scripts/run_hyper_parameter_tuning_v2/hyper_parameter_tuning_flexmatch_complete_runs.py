import pprint
import subprocess
import torch
from hyper_parameter_tuning_v2 import main


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
        "project_wandb": "test",  # f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,

        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels)


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


def flexmatch_missing_labels_cifar100_v7(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 200,
        "num_train_iter": 204800,
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
        "lambda_entropy": [1.5, 2.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [-1, 0.55, 0.65],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)


def flexmatch_missing_labels_cifar100_v8(version):
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
        "lambda_entropy": [1.5, 2.5, 3],  # [0, 0.1, 1, 3, 5],
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


def flexmatch_missing_labels_cifar100_v9(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 200,
        "num_train_iter": 204800,
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
        "lambda_entropy": [1.95, 2.05],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [0.9, -1],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)


def flexmatch_missing_labels_cifar100_v10(version):
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
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [2.5, 3],  # [0, 0.1, 1, 3, 5],
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


def flexmatch_missing_labels_cifar100_v11(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 200,
        "num_train_iter": 204800,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1.5, 2.5, 3, 3.5],  # [0, 0.1, 1, 3, 5],
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


def flexmatch_missing_labels_cifar100_v12(version):
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
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1.5, 2.3, 2.8, 3.5, 4],  # [0, 0.1, 1, 3, 5],
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


def flexmatch_missing_labels_cifar100_v13(version):
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
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 10,
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
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=5, seed_end=10)


def flexmatch_missing_labels_cifar100_v14(version):
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
        "lambda_entropy": 4,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [0.7, 0.8, 0.9],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=3, seed_end=5)


def flexmatch_missing_labels_cifar100_v15(version):
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
        "lambda_entropy": [0.9, 1.3],  # [0, 0.1, 1, 3, 5],
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


def flexmatch_missing_labels_cifar100_v16(version):
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
        "lambda_entropy": [0.9, 1, 1.3, 1.5],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=3, seed_end=5)


def flexmatch_missing_labels_cifar100_v17(version):
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
        "lambda_entropy": [0.9, 1, 1.3, 1.5],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=3, seed_end=5)


def flexmatch_missing_labels_cifar100_v18(version):
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
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1.2, 1.4],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [-1, 0.7, 0.8],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)


def flexmatch_missing_labels_cifar100_v19(version):
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
        "lambda_entropy": [2.05, 1.95],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": [-1, 0.7, 0.8],
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=3, seed_end=5)


def flexmatch_missing_labels_cifar100_v20(version):
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
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.5, 0.9, 1.5, 2.5],  # [0, 0.1, 1, 3, 5],
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


def flexmatch_missing_labels_cifar100_complete_runs(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 400,
        "num_train_iter": 409600,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "",
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
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)


def flexmatch_wrn_28_8_cifar100(version):
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
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
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
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=3)


def flexmatch_missing_labels_cifar100_wrn_28_8_v1(version):
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
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1, 1.5],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_wrn_28_8_v2(version):
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
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_wrn_28_8_v3(version):
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
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [2],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v21(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [5000, 10000],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [20, 45],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 1, 1.5, 2],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v22(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0, 1],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v23(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1.5, 2],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v24(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
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
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 3,
        "lambda_entropy_with_labeled_data": [0.1, 1, 3]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v25(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
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
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 3,
        "lambda_entropy_with_labeled_data": [0.5, 1.5, 2]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v26(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
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
        "python_code_version": 3,
        "lambda_entropy_with_labeled_data": 1
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v27(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
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
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 3,
        "lambda_entropy_with_labeled_data": [0.3]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v28(version):
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
        "python_code_version": 3,
        "lambda_entropy_with_labeled_data": 0.5
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v29(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
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
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": [0.1, 0.5, 1, 1.5, 2]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v30(version):
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
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.6, 0.7, 0.8, 1.1],  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v31(version):
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
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=3, seed_end=5)


def flexmatch_missing_labels_cifar100_v32(version):
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
        "lambda_entropy_with_labeled_data": [0.2, 1]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v33(version):
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
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)


def flexmatch_missing_labels_cifar100_v34(version):
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
        "lambda_entropy_with_labeled_data": [1]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v35(version):
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
        "lambda_entropy": [1.9, 1.95, 2, 2.05, 2.1],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0
    }
    main(parameters_dict_missing_labels, killable=True, seed_start=0, seed_end=2)


def flexmatch_missing_labels_cifar100_v37(version):
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
        "lambda_entropy": [1.9, 1.95, 2, 2.05, 2.1],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "python_code_version": 4,
        "lambda_entropy_with_labeled_data": 0
    }
    main(parameters_dict_missing_labels, killable=True, seed_start=0, seed_end=2)


def flexmatch_missing_labels_cifar100_v38(version):
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
        "lambda_entropy": [1.9, 1.95, 2.05, 2.1],  # [0, 0.1, 1, 3, 5],
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
        "lambda_entropy_with_labeled_data": 0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=4)


def flexmatch_missing_labels_cifar100_v39(version):
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
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=3)


def flexmatch_missing_labels_cifar100_v40(version):
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
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0.2,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v41(version):
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
        "lambda_entropy": 1,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v42(version):
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
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1, 1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v43(version):
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
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
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
    main(parameters_dict_missing_labels, seed_start=3, seed_end=5)


def flexmatch_missing_labels_cifar100_redo_15614365(version):
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
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
    }
    main(parameters_dict_missing_labels, seed_start=1, seed_end=2)


def flexmatch_missing_labels_cifar100_redo_15614379(version):
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
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,

        "lambda_entropy_with_labeled_data": 0.2,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v45(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": [0.1, 0.5, 1, 3],
    }
    main(parameters_dict_missing_labels,killable=False, seed_start=0, seed_end=2)



def flexmatch_missing_labels_cifar100_v46(version):
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
        "python_code_version": 2,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
    }
    main(parameters_dict_missing_labels, seed_start=1, seed_end=3)



def flexmatch_missing_labels_cifar100_v47(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0,0.5,1,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v48(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [5000,10000],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [20,45],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0,0.5,1,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v49(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0,0.5,1,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)



def flexmatch_missing_labels_cifar100_v50(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def flexmatch_missing_labels_cifar100_v51(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 30,
        "num_train_iter": 30720,
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
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 4,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": [0,0.5,1,3],
    }
    main(parameters_dict_missing_labels,killable=False, seed_start=0, seed_end=1)


def flexmatch_missing_labels_cifar100_v52(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 30,
        "num_train_iter": 30720,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [10,40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "split_to_superclasses":1,
        "new_ent_loss_ratio":-1
    }
    main(parameters_dict_missing_labels,killable=False, seed_start=0, seed_end=1)



def flexmatch_missing_labels_cifar100_v53(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [5000, 10000],
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [20,45],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0,1,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "split_to_superclasses":1,
        "new_ent_loss_ratio":-1
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)

def flexmatch_missing_labels_cifar100_v54(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "flexmatch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0,1,1.5],  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "delete": 0,
        "ulb_loss_ratio": 1.0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "split_to_superclasses":1,
        "new_ent_loss_ratio":-1
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)