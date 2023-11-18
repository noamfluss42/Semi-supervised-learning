# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import csv

import os
import json
import torchvision
import numpy as np
import math
import wandb
from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data,get_cifar_lt_indices,create_labeled_data
from log_wandb import log_data_dist, log_labeled_dist_config_main, log_count_data_dist_by_array

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def remove_missing_labels(args, lb_data, lb_targets):
    if args.choose_random_labeled_training_set == -1 and len(
            args.missing_labels) > 0 or args.random_missing_labels_num > 0:
        return lb_data[np.invert(np.isin(lb_targets, args.missing_labels))], lb_targets[
            np.invert(np.isin(lb_targets, args.missing_labels))]
    if args.choose_random_labeled_training_set != -1:
        choose_random_labeled_training_set_counts_temp = args.choose_random_labeled_training_set_counts.copy()
        result_bool = []
        class_to_indexes = {}
        for target_index in range(len(lb_targets)):
            if lb_targets[target_index] not in args.choose_random_labeled_training_set_unique:
                result_bool.append(False)
                continue
            count_index = np.argwhere(args.choose_random_labeled_training_set_unique == lb_targets[target_index])[0][0]
            if choose_random_labeled_training_set_counts_temp[count_index] == 0:
                result_bool.append(False)
            else:
                result_bool.append(True)
                choose_random_labeled_training_set_counts_temp[count_index] -= 1
                if hasattr(args, "choose_random_labeled_training_set_duplicate"):
                    if args.choose_random_labeled_training_set_duplicate == 1:
                        if lb_targets[target_index] not in class_to_indexes:
                            class_to_indexes[lb_targets[target_index]] = []
                        class_to_indexes[lb_targets[target_index]].append(target_index)
        if hasattr(args, "choose_random_labeled_training_set_duplicate"):
            if args.choose_random_labeled_training_set_duplicate == 1:
                result_indexes = []
                max_per_class = args.choose_random_labeled_training_set_counts.max()
                for class_index in range(len(args.choose_random_labeled_training_set_unique)):
                    current_count = args.choose_random_labeled_training_set_counts[class_index]
                    class_indexes = class_to_indexes[args.choose_random_labeled_training_set_unique[class_index]]
                    new_class_indexes = class_indexes * int(math.floor(max_per_class / current_count))
                    new_class_indexes = np.concatenate(
                        (new_class_indexes, class_indexes[:max_per_class % len(new_class_indexes)]))
                    result_indexes = np.concatenate((result_indexes, new_class_indexes))
                np.random.shuffle(result_indexes)
                # convert result_indexes to int
                result_indexes = result_indexes.astype(int)
                return lb_data[result_indexes], lb_targets[result_indexes], lb_data[result_bool], lb_targets[
                    result_bool]
        return lb_data[result_bool], lb_targets[result_bool]
    return lb_data, lb_targets



def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True,
              is_transductive=False):
    eval = hasattr(args, 'is_eval')
    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=True, download=True)
    data, targets = dset.data, dset.targets

    crop_size = args.img_size
    crop_ratio = args.crop_ratio

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop(crop_size, padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        RandAugment(3, 5),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_val = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name], )
    ])

    if args.choose_random_labeled_training_set != -1:
        lb_num_labels_before_removing = args.num_classes * args.choose_random_labeled_training_set_counts.max()
    else:
        lb_num_labels_before_removing = num_labels
    print(
        f"lb_num_labels={num_labels},ulb_num_labels={args.ulb_num_labels},"
        f"lb_num_labels_before_removing={lb_num_labels_before_removing},"
        f"lb_imbalance_ratio={args.lb_imb_ratio},ulb_imbalance_ratio={args.ulb_imb_ratio}")

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes,
                                                                lb_num_labels=lb_num_labels_before_removing,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)


    if args.python_code_version >= 11:
        if args.lt_ratio != 1:
            lb_data, lb_targets = create_labeled_data(args,ulb_data, ulb_targets)
    if hasattr(args, "choose_random_labeled_training_set_duplicate"):
        if args.choose_random_labeled_training_set_duplicate == 1:

            lb_data, lb_targets, lb_data_original, lb_targets_original = remove_missing_labels(args, lb_data,
                                                                                               lb_targets)
            log_count_data_dist_by_array(args, "lb_count/dataset after duplicate count", lb_targets)
        else:
            lb_data, lb_targets = remove_missing_labels(args, lb_data, lb_targets)
    else:
        lb_data, lb_targets = remove_missing_labels(args, lb_data, lb_targets)

    print("lb_targets", len(lb_targets))

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1

    if args.choose_random_labeled_training_set == 1:
        for class_index in range(len(args.choose_random_labeled_training_set_unique)):
            lb_count[args.choose_random_labeled_training_set_unique[class_index]] = \
                args.choose_random_labeled_training_set_counts[class_index]
        log_labeled_dist_config_main(args, np.array(lb_count))
        args.missing_labels = np.argwhere(np.array(lb_count) == 0).flatten()
        print(f"missing_labels when choose_random_labeled_training_set ={args.missing_labels}")
        args.random_missing_labels_num = len(args.missing_labels)

    if not eval:
        print("lb count: {}".format(lb_count))
        print("ulb count: {}".format(ulb_count))

        print(f"my lb count: {len(lb_count)} - {lb_count}")
        print(f"my ulb count: {len(ulb_count)} - {ulb_count}")
        log_data_dist(args, lb_count, "dataset/labeled training data dist")
        log_data_dist(args, ulb_count, "dataset/unlabeled training data dist")

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=is_transductive, download=True)
    test_data, test_targets = dset.data, dset.targets
    if args.python_code_version >= 11:
        if args.lt_ratio != 1:
            test_indices = get_cifar_lt_indices(args,"test")
            test_data, test_targets = test_data[test_indices], list(np.array(test_targets)[test_indices])

    unique, counts = np.unique(test_targets,
                               return_counts=True)

    unique_args_sort = np.argsort(unique)
    print("debug args.lt_ratio")
    print("unique", unique)
    print("cound",counts)
    if not eval:
        log_data_dist(args, counts[unique_args_sort], "dataset/test data dist")

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
