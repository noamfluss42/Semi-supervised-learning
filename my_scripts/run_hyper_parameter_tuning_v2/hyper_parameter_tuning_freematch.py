import pprint
import subprocess
import torch
from hyper_parameter_tuning_main_run import main


def call_freematch_cifar100_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 3,  # 500,
        "num_train_iter": 3072,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [0, 4, 6],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "test",
        "python_code_version": 2,
        "ulb_loss_ratio": 1.0,
        "delete": 1,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def call_freematch_cifar100_missing_labels_v1(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "ulb_loss_ratio": 1.0,
        "delete": 1,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_softmatch_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 3,  # 500,
        "num_train_iter": 3072,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "softmatch",
        "save_dir": "./saved_models/classic_cv/tuning/softmatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [0, 10],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,  # [0, 0.1, 1, 3, 5],
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": "test",
        "python_code_version": 2,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def call_softmatch_missing_labels_v1(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "softmatch",
        "save_dir": "./saved_models/classic_cv/tuning/softmatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_softmatch_missing_labels_v2(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "softmatch",
        "save_dir": "./saved_models/classic_cv/tuning/softmatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 2,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_adamatch_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 3,  # 500,
        "num_train_iter": 3072,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "adamatch",
        "save_dir": "./saved_models/classic_cv/tuning/adamatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 10],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)


def call_adamatch_missing_labels_v1(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "adamatch",
        "save_dir": "./saved_models/classic_cv/tuning/adamatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "python_code_version": 5,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_comatch_missing_labels_test(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 3,  # 500,
        "num_train_iter": 3072,  # 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "comatch",
        "save_dir": "./saved_models/classic_cv/tuning/comatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 10],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)


def call_comatch_missing_labels_v1(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "comatch",
        "save_dir": "./saved_models/classic_cv/tuning/comatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,

        "missing_labels": None,
        "random_missing_labels_num": [0, 10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "delete": 0,
        "ulb_loss_ratio": 1.0
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_comatch_missing_labels_v2(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 10,
        "num_train_iter": 10240,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "comatch",
        "save_dir": "./saved_models/classic_cv/tuning/comatch/",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [0, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "MNAR_round_type": "ceil",
        "threshold": 0.95,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 4,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def call_freematch_cifar100_missing_labels_v2(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
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
        "lambda_entropy": [0.5, 1, 3, 5, 10],
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 4,
        "ulb_loss_ratio": 1.0,
        "delete": 1,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def call_freematch_cifar100_missing_labels_v3(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [1, 3],
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
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v4(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 50,
        "num_train_iter": 51200,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [0.001, 0.002, 0.005, 0.0001, 0.01, 0.05]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=2)


def call_freematch_cifar100_missing_labels_v5(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [0.1, 0.5, 1, 1.5, 2, 2.5]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=2)


def call_freematch_cifar100_missing_labels_v6(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [0.001, 0.01, 0.05, 0.1]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v7(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.5, 1.5, 2],
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [0.001, 0.01, 0.05, 0.1]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v8(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": [0.5, 1.5, 2],
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [0.001, 0.01, 0.05, 0.1]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v9(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [10]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def call_freematch_cifar100_missing_labels_v10(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 40,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [1, 3, 5, 10]
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=1, seed_end=2)


def call_freematch_cifar100_missing_labels_v11(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,

        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,

        "new_ent_loss_ratio": [100, 500, 1000, 5000, 10000, 100000]
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=1)


def call_freematch_cifar100_missing_labels_v12(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": [40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v13(version):
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
        "random_missing_labels_num": 10,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v14(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 25000,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 50,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 1.5,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": [0, 1],
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v15(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,
        "num_train_iter": 1048576,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 2500,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [10, 40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v16(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,
        "num_train_iter": 1048576,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 400,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [40],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 0,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v17(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 1024,
        "num_train_iter": 1048576,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 5000,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_8",
        "missing_labels": None,
        "random_missing_labels_num": [45],
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 1,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v18(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 15,
        "num_train_iter": 15360,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": 1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 1,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=1)


def call_freematch_cifar100_missing_labels_v19(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 100,
        "num_train_iter": 102400,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": [400, 2500],
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": 1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 5,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 1,
        "comment": "freematch_cifar100_missing_labels_v19",
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=2)


def call_freematch_cifar100_missing_labels_v20(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 100,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": 1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"test",
        "python_code_version": 6,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 1,
        "comment": 1120,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v21(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 100,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": 1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 6,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 0,
        "comment": 1121,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v22(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 100,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": 35,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 6,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 0,
        "comment": 1122,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v23(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 50,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": 1,
        "lambda_entropy": [0, 1.5],
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 6,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 0,
        "comment": 1123,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v24(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 5000,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "wrn_28_2",
        "missing_labels": None,
        "random_missing_labels_num": 45,
        "choose_random_labeled_training_set": -1,
        "lambda_entropy": 0,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 6,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 1,
        "comment": 1124,
    }
    main(parameters_dict_missing_labels, seed_start=0, seed_end=5)


def call_freematch_cifar100_missing_labels_v25(version):
    parameters_dict_missing_labels = {
        "load_path": "",
        "seed": "",
        "save_name": "",
        "epoch": 500,
        "num_train_iter": 512000,
        "num_eval_iter": 1024,
        "num_classes": 100,
        "num_labels": 100,
        "algorithm": "freematch",
        "save_dir": "./saved_models/classic_cv/tuning",
        "dataset": "cifar100",
        "slurm_job_id": None,
        "net_new": "None",
        "missing_labels": None,
        "random_missing_labels_num": -1,
        "choose_random_labeled_training_set": 1,
        "lambda_entropy": 1.5,
        "lb_imb_ratio": 1,
        "threshold": 0.95,
        "new_p_cutoff": -1,
        "weight_decay": 0.001,
        "lambda_datapoint_entropy": 0,
        "project_wandb": f"Flexmatch_{str(version)}_project_missing_labels",
        "python_code_version": 6,
        "ulb_loss_ratio": 1.0,
        "delete": 0,
        "lambda_entropy_with_labeled_data": 0,
        "lambda_entropy_with_labeled_data_v2": 0,
        "new_ent_loss_ratio": -1,
        "split_to_superclasses": 0,
        "comment": 2000,
    }
    main(parameters_dict_missing_labels, killable=False, seed_start=0, seed_end=5)

