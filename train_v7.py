# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import argparse
import time

import os
import logging
import random
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.imb_algorithms import get_imb_algorithm, name2imbalg
from semilearn.algorithms.utils import str2bool
from semilearn.core.utils import get_net_builder, get_logger, get_port, send_model_cuda, count_parameters, \
    over_write_args_from_file, TBLog
import wandb
from wandb_util import *
from log_wandb import log_data_dist_by_unique


class epochs_func(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        print("start epochs_func")
        new_args = []
        for value in values:
            if value.isnumeric():
                new_args.append(int(value))
            else:
                val, repetitions = value.split(':')
                new_args += [int(val)] * int(repetitions)
        setattr(namespace, option_string.strip('-'), new_args)


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description='Semi-Supervised Learning (USB)')

    '''
    Saving & loading of the model.
    '''
    parser.add_argument('--save_dir', type=str, default='./saved_models')
    parser.add_argument('-sn', '--save_name', type=str, default='fixmatch')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--load_path', type=str)
    parser.add_argument('-o', '--overwrite', action='store_true', default=True)
    parser.add_argument('--use_tensorboard', action='store_true',
                        help='Use tensorboard to plot and save curves, otherwise save the curves locally.')
    parser.add_argument('--use_wandb', action='store_true', help='Use wandb to plot and save curves')
    parser.add_argument('--use_aim', action='store_true', help='Use aim to plot and save curves')
    '''
    Training Configuration of FixMatch
    '''

    parser.add_argument('--epoch', type=int, default=1)
    parser.add_argument('--num_train_iter', type=int, default=20,
                        help='total number of training iterations')
    parser.add_argument('--num_warmup_iter', type=int, default=0,
                        help='cosine linear warmup iterations')
    parser.add_argument('--num_eval_iter', type=int, default=10,
                        help='evaluation frequency')
    parser.add_argument('--num_log_iter', type=int, default=5,
                        help='logging frequencu')
    parser.add_argument('-nl', '--num_labels', type=int, default=400)
    parser.add_argument('-bsz', '--batch_size', type=int, default=8)
    parser.add_argument('--uratio', type=int, default=1,
                        help='the ratio of unlabeled data to labeld data in each mini-batch')
    parser.add_argument('--eval_batch_size', type=int, default=16,
                        help='batch size of evaluation data loader (it does not affect the accuracy)')
    parser.add_argument('--ema_m', type=float, default=0.999, help='ema momentum for eval_model')
    parser.add_argument('--ulb_loss_ratio', type=float, default=1.0)

    '''
    Optimizer configurations
    '''
    parser.add_argument('--optim', type=str, default='SGD')
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--layer_decay', type=float, default=1.0,
                        help='layer-wise learning rate decay, default to 1.0 which means no layer decay')

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_new', type=str, default='')
    parser.add_argument('--net_from_name', type=str2bool, default=False)
    parser.add_argument('--use_pretrain', default=False, type=str2bool)
    parser.add_argument('--pretrain_path', default='', type=str)

    '''
    Algorithms Configurations
    '''

    ## core algorithm setting
    parser.add_argument('-alg', '--algorithm', type=str, default='fixmatch', help='ssl algorithm')
    parser.add_argument('--use_cat', type=str2bool, default=True, help='use cat operation in algorithms')
    parser.add_argument('--amp', type=str2bool, default=False, help='use mixed precision training or not')
    parser.add_argument('--clip_grad', type=float, default=0)

    ## imbalance algorithm setting
    parser.add_argument('-imb_alg', '--imb_algorithm', type=str, default=None, help='imbalance ssl algorithm')

    '''
    Data Configurations
    '''

    ## standard setting configurations
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('-ds', '--dataset', type=str, default='cifar10')
    parser.add_argument('-nc', '--num_classes', type=int, default=10)
    parser.add_argument('--train_sampler', type=str, default='RandomSampler')
    parser.add_argument('--num_workers', type=int, default=1)
    parser.add_argument('--include_lb_to_ulb', type=str2bool, default='True',
                        help='flag of including labeled data into unlabeled data, default to True')

    ## imbalanced setting arguments
    parser.add_argument('--lb_imb_ratio', type=int, default=1, help="imbalance ratio of labeled data, default to 1")
    parser.add_argument('--ulb_imb_ratio', type=int, default=1, help="imbalance ratio of unlabeled data, default to 1")
    parser.add_argument('--ulb_num_labels', type=int, default=None,
                        help="number of labels for unlabeled data, used for determining the maximum number of labels in imbalanced setting")

    ## cv dataset arguments
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=float, default=0.875)

    ## nlp dataset arguments 
    parser.add_argument('--max_length', type=int, default=512)

    ## speech dataset algorithms
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)

    '''
    multi-GPUs & Distrbitued Training
    '''

    ## args for distributed training (from https://github.com/pytorch/examples/blob/master/imagenet/main.py)
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='**node rank** for distributed training')
    parser.add_argument('-du', '--dist-url', default='tcp://127.0.0.1:11111', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--seed', default=1, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', type=str2bool, default=False,
                        help='Use multi-processing distributed training to launch '
                             'N processes per node, which has N GPUs. This is the '
                             'fastest way to use PyTorch for either single node or '
                             'multi node data parallel training')
    # config file
    parser.add_argument('--c', type=str, default='')

    parser.add_argument('--project_wandb', type=str, default='test',
                        help="wandb project")
    parser.add_argument('--slurm_job_id', type=str, default='-23',
                        help="slurm_job_id")
    parser.add_argument('--log_info', type=int, default=1
                        , help="log [2022-11-21 11:02:06,308 INFO] 230780 iteration, USE_EMA: True....")

    parser.add_argument('--missing_labels', nargs='+', default=[], type=int,
                        help='labels we don\'t have labels in the dataset')
    parser.add_argument('--missing_labels_1234', nargs='+', default=[], type=int,
                        help='labels we don\'t have labels in the dataset')

    parser.add_argument('--random_missing_labels_num', type=int, default=-1,
                        help='the number of missing labels to gerate')
    parser.add_argument('--choose_random_labeled_training_set', type=int, default=-1,
                        help='if to choose random labeled dataset - number of labels = args.num_labels')
    parser.add_argument('--choose_random_labeled_training_set_duplicate', type=int, default=-1,
                        help='to dupliaate the labeled dataset so for every class there will be the same number of samples (can be duplicated)')
    parser.add_argument('--choose_last_classes_as_unseen', type=int, default=-1,
                        help='use the random_missing_labels_num to choose the last classes as unseen (for example, if random_missing_labels_num=2, then the last 98,99 classes will be unseen)')

    parser.add_argument('--lambda_entropy', default=0, type=float,
                        help='the entropy loss coefficient')
    parser.add_argument('--MNAR_round_type', default="ceil", type=str,
                        help='ceil or floor the number of labeled training data')
    parser.add_argument('--shuffle_MNAR_classes', default=0, type=int,
                        help='shuffle_MNAR_classes')
    parser.add_argument('--threshold', default=0.95, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--new_p_cutoff', default=-1, type=float,
                        help='pseudo label threshold')
    parser.add_argument('--lambda_datapoint_entropy', type=float, default=0,
                        help='the datapoint entropy loss coefficient')
    parser.add_argument('--shell_code_version', type=float, default=0,
                        help='version')
    parser.add_argument('--python_code_version', type=float, default=4,
                        help='version')

    parser.add_argument('--lambda_entropy_with_labeled_data', type=float, default=0,
                        help='Robust Semi-Supervised Learning when Not All Classes have Labels')
    parser.add_argument('--lambda_entropy_with_labeled_data_v2', type=float, default=0,
                        help='Robust Semi-Supervised Learning when Not All Classes have Labels - by code')

    parser.add_argument('--split_to_superclasses', type=float, default=0,
                        help='Robust Semi-Supervised Learning when Not All Classes have Labels - by code')
    parser.add_argument('--new_ent_loss_ratio', type=float, default=-1,
                        help='Robust Semi-Supervised Learning when Not All Classes have Labels - by code')

    parser.add_argument('--delete', type=int, default=0,
                        help='version')
    parser.add_argument('--comment', type=int, default=0,
                        help='anything')
    parser.add_argument('--commit_hash', type=str, default="9288db0d34ea027a65596d6b7010c952b68b5a6b",
                        help='anything')

    # add algorithm specific parameters
    args = parser.parse_args()

    print("args.c is", args.c)
    print("args.gpu", args.gpu)
    print("args.ulb_loss_ratio 0", args.ulb_loss_ratio)
    print(f"start1 args.dataset is {args.dataset}")
    gpu_id = args.gpu
    over_write_args_from_file(args, args.c)
    print("args.ulb_loss_ratio 1", args.ulb_loss_ratio)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    print("args.ulb_loss_ratio 2", args.ulb_loss_ratio)
    # add imbalanced algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    print("check")

    if args.imb_algorithm is not None:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(argument.name, type=argument.type, default=argument.default, help=argument.help)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.new_p_cutoff != -1:
        args.p_cutoff = args.new_p_cutoff
    if args.new_ent_loss_ratio != -1:
        args.ent_loss_ratio = args.new_ent_loss_ratio
    if args.net_new != "" and args.net_new != "None":
        args.net = args.net_new
    print("args.ulb_loss_ratio 3", args.ulb_loss_ratio)
    args.iter_per_train_epoch = args.num_train_iter / args.epoch

    args.gpu = gpu_id

    print("finish config!!! - train v2")
    print("args.save_name1", args.save_name)
    print("args.load_path1", args.load_path)
    args.save_name = os.path.join(args.save_name, str(args.slurm_job_id))
    args.load_path = os.path.join(os.path.join(args.save_dir, args.save_name), "latest_model.pth")
    print("args.save_name2", args.save_name)
    print("args.load_path2", args.load_path)
    print("args.ulb_loss_ratio finish", args.ulb_loss_ratio)
    print(f"end args.dataset is {args.dataset}")

    return args


def main(args):
    '''
    For (Distributed)DataParallelism,
    main(args) spawn each process (main_worker) to each GPU.
    '''
    assert args.num_train_iter % args.epoch == 0, \
        f"# total training iter. {args.num_train_iter} is not divisible by # epochs {args.epoch}"

    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume == False:
        import shutil
        shutil.rmtree(save_path, ignore_errors=True)
    if os.path.exists(save_path) and not args.overwrite:
        raise Exception('already existing model: {}'.format(save_path))
    if args.resume:
        if args.load_path is None:
            raise Exception('Resume of training requires --load_path in the args')
        if os.path.abspath(save_path) == os.path.abspath(args.load_path) and not args.overwrite:
            raise Exception('Saving & Loading pathes are same. \
                            If you want over-write, give --overwrite in the argument.')

    if args.seed is not None:
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu == 'None':
        args.gpu = None
    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # distributed: true if manually selected or if world_size > 1
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed
    ngpus_per_node = torch.cuda.device_count()  # number of gpus of each node

    print("args.multiprocessing_distributed", args.multiprocessing_distributed)
    print("ngpus_per_node", ngpus_per_node)
    print("args.distributed", args.distributed)
    print("args.gpu", args.gpu)
    print("args.ulb_num_labels", args.ulb_num_labels)
    if args.multiprocessing_distributed:
        # now, args.world_size means num of total processes in all nodes
        args.world_size = ngpus_per_node * args.world_size

        # args=(,) means the arguments of main_worker
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        main_worker(args.gpu, ngpus_per_node, args)


def set_random(args):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    print("add torch.cuda.manual_seed_all(args.seed)")
    print("cudnn.benchmark", cudnn.benchmark)


def create_missing_classes_by_superclasses(args):
    missing_classes = []
    coarse_labels = np.array([4, 1, 14, 8, 0, 6, 7, 7, 18, 3,
                              3, 14, 9, 18, 7, 11, 3, 9, 7, 11,
                              6, 11, 5, 10, 7, 6, 13, 15, 3, 15,
                              0, 11, 1, 10, 12, 14, 16, 9, 11, 5,
                              5, 19, 8, 8, 15, 13, 14, 17, 18, 10,
                              16, 4, 17, 4, 2, 0, 17, 4, 18, 17,
                              10, 3, 2, 12, 12, 16, 12, 1, 9, 19,
                              2, 10, 0, 1, 16, 12, 9, 13, 15, 13,
                              16, 19, 2, 4, 6, 19, 5, 5, 8, 19,
                              18, 1, 2, 15, 6, 0, 17, 8, 14, 13])
    missing_superclasses = random.sample(range(20), k=int(args.random_missing_labels_num / 5))
    print("missing_superclasses", missing_superclasses)
    for superclass in missing_superclasses:
        missing_classes.extend(np.where(coarse_labels == superclass)[0])
    print("missing_classes", missing_classes)
    return missing_classes


def main_worker(gpu, ngpus_per_node, args):
    '''
    main_worker is conducted on each GPU.
    '''

    global best_acc1
    args.gpu = gpu
    print("\nargs.gpu", args.gpu)
    # random seed has to be set for the syncronization of labeled data sampling in each process.
    assert args.seed is not None
    set_random(args)
    print('os.environ["WANDB_CACHE_DIR"]', os.environ["WANDB_CACHE_DIR"])
    if args.random_missing_labels_num != -1:

        if args.split_to_superclasses > 0:
            print("args.split_to_superclasses!", args.split_to_superclasses)
            args.missing_labels = create_missing_classes_by_superclasses(args)
            args.missing_labels.sort()
        elif args.choose_last_classes_as_unseen > 0:
            args.missing_labels = np.array(range(args.num_classes - args.random_missing_labels_num, args.num_classes))
            args.missing_labels.sort()
        else:
            args.missing_labels = random.sample(range(args.num_classes), k=args.random_missing_labels_num)
            args.missing_labels.sort()
    elif args.missing_labels is not None and len(args.missing_labels) > 0:
        args.missing_labels.sort()
        args.random_missing_labels_num = len(args.missing_labels)
    print("args.missing_labels", args.missing_labels)
    wandb.config.update({'missing_labels': args.missing_labels}, allow_val_change=True)
    if args.lb_imb_ratio > 1:
        args.num_labels = args.lb_imb_ratio
    args.choose_random_labeled_training_set_unique, args.choose_random_labeled_training_set_counts = [], []
    if args.choose_random_labeled_training_set != -1:
        choosing_labels = [random.randint(0, args.num_classes - 1) for i in range(args.num_labels)]
        args.choose_random_labeled_training_set_unique, args.choose_random_labeled_training_set_counts = np.unique(
            choosing_labels, return_counts=True)
        log_data_dist_by_unique(args, log_string="lb_count/dataset")

    # SET UP FOR DISTRIBUTED TRAINING
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])

        if args.multiprocessing_distributed:
            args.rank = args.rank * ngpus_per_node + gpu  # compute global rank

        # set distributed group:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    # SET save_path and logger
    save_path = os.path.join(args.save_dir, args.save_name)
    logger_level = "WARNING"
    tb_log = None

    if args.rank % ngpus_per_node == 0:
        tb_log = TBLog(save_path, 'tensorboard', use_tensorboard=args.use_tensorboard)
        logger_level = "INFO"

    logger = get_logger(args.save_name, save_path, logger_level)
    print("print in logget that we use", args.gpu, "gpus")
    logger.info(f"Use GPU: {args.gpu} for training")

    _net_builder = get_net_builder(args.net, args.net_from_name)
    # optimizer, scheduler, datasets, dataloaders with be set in algorithms
    if args.imb_algorithm is not None:
        model = get_imb_algorithm(args, _net_builder, tb_log, logger)
    else:
        model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f'Number of Trainable Params: {count_parameters(model.model)}')

    # SET Devices for (Distributed) DataParallel
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # If args.resume, load checkpoints from args.load_path
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    if hasattr(model, 'warmup'):
        logger.info(("Warmup stage"))
        model.warmup()

    # START TRAINING of FixMatch
    logger.info("Model training")
    model.train()

    # print validation (and test results)
    for key, item in model.results_dict.items():
        logger.info(f"Model result - {key} : {item}")

    if hasattr(model, 'finetune'):
        logger.info("Finetune stage")
        model.finetune()

    if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
        model.save_model('latest_model.pth', save_path)

    logging.warning(f"GPU {args.rank} training is FINISHED")


def print_important_args(args):
    print("\n\n\nstart big print - train v2")
    print("algorithm:", args.algorithm)
    print("save_dir:", args.save_dir)
    print("save_name:", args.save_name)
    print("resume:", args.resume)
    print("load_path:", args.load_path)
    print("epoch:", args.epoch)
    print("num_train_iter:", args.num_train_iter)
    print("num_log_iter:", args.num_log_iter)

    print("data_dir:", args.data_dir)
    print("dataset:", args.dataset)
    print("num_classes:", args.num_classes)
    print("num_workers:", args.num_workers)
    print("seed:", args.seed)
    print("project_wandb:", args.project_wandb)
    print("slurm_job_id:", args.slurm_job_id)

    print("missing_labels:", type(args.missing_labels), args.missing_labels)
    print("random_missing_labels_num:", args.random_missing_labels_num)
    print("choose_random_labeled_training_set:", args.choose_random_labeled_training_set)
    print("lambda_entropy:", args.lambda_entropy)
    print("lb_imb_ratio:", args.lb_imb_ratio)
    print("MNAR_round_type:", args.MNAR_round_type)
    print("slurm_job_id:", args.slurm_job_id)
    print("threshold:", args.threshold)
    print("lambda_datapoint_entropy:", args.lambda_datapoint_entropy)
    print("num_labels:", args.num_labels)
    print("weight_decay:", args.weight_decay)
    if hasattr(args, 'thresh_warmup'):
        print("thresh_warmup:", args.thresh_warmup)
    else:
        print("thresh_warmup: non existing")
    print("end big print\n\n\n")


if __name__ == "__main__":
    print("check")
    start_time = time.time()
    args = get_config()
    print_important_args(args)
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    init_wandb(args)
    main(args)
    finish_wandb()
    print("final run time", time.time() - start_time)