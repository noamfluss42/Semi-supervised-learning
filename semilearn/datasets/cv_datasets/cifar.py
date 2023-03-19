# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math

from torchvision import transforms
from .datasetbase import BasicDataset
from semilearn.datasets.augmentation import RandAugment, RandomResizedCropAndInterpolation
from semilearn.datasets.utils import split_ssl_data
from log_wandb import log_data_dist

mean, std = {}, {}
mean['cifar10'] = [0.485, 0.456, 0.406]
mean['cifar100'] = [x / 255 for x in [129.3, 124.1, 112.4]]

std['cifar10'] = [0.229, 0.224, 0.225]
std['cifar100'] = [x / 255 for x in [68.2, 65.4, 70.4]]


def remove_missing_labels(args, lb_data, lb_targets):
    if len(args.missing_labels) > 0 or args.random_missing_labels_num != -1:
        return lb_data[np.invert(np.isin(lb_targets, args.missing_labels))], lb_targets[
            np.invert(np.isin(lb_targets, args.missing_labels))]
    if args.choose_random_labeled_training_set != -1:
        choose_random_labeled_training_set_counts_temp = args.choose_random_labeled_training_set_counts.copy()
        result_bool = []
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
        return lb_data[result_bool], lb_targets[result_bool]
    return lb_data, lb_targets

def get_cifar(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=True):
    
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
        transforms.Normalize(mean[name], std[name],)
    ])

    if args.choose_random_labeled_training_set != -1:
        num_labels = args.num_classes * args.choose_random_labeled_training_set_counts.max()
    print(
        f"lb_num_labels={num_labels},ulb_num_labels={args.ulb_num_labels},"
        f"lb_imbalance_ratio={args.lb_imb_ratio},ulb_imbalance_ratio={args.ulb_imb_ratio}")

    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, data, targets, num_classes, 
                                                                lb_num_labels=num_labels,
                                                                ulb_num_labels=args.ulb_num_labels,
                                                                lb_imbalance_ratio=args.lb_imb_ratio,
                                                                ulb_imbalance_ratio=args.ulb_imb_ratio,
                                                                include_lb_to_ulb=include_lb_to_ulb)

    lb_data, lb_targets = remove_missing_labels(args, lb_data, lb_targets)
    print("lb_targets",len(lb_targets),lb_targets)

    lb_count = [0 for _ in range(num_classes)]
    ulb_count = [0 for _ in range(num_classes)]
    for c in lb_targets:
        lb_count[c] += 1
    for c in ulb_targets:
        ulb_count[c] += 1
    print("lb count: {}".format(lb_count))
    print("ulb count: {}".format(ulb_count))

    print(f"my lb count: {len(lb_count)} - {lb_count}")
    print(f"my ulb count: {len(ulb_count)} - {ulb_count}")
    log_data_dist(args, lb_count, "dataset/labeled training data dist")
    log_data_dist(args, ulb_count, "dataset/unlabeled training data dist")
    # lb_count = lb_count / lb_count.sum()
    # ulb_count = ulb_count / ulb_count.sum()
    # args.lb_class_dist = lb_count
    # args.ulb_class_dist = ulb_count

    if alg == 'fullysupervised':
        lb_data = data
        lb_targets = targets
        # if len(ulb_data) == len(data):
        #     lb_data = ulb_data 
        #     lb_targets = ulb_targets
        # else:
        #     lb_data = np.concatenate([lb_data, ulb_data], axis=0)
        #     lb_targets = np.concatenate([lb_targets, ulb_targets], axis=0)
    
    # output the distribution of labeled data for remixmatch
    # count = [0 for _ in range(num_classes)]
    # for c in lb_targets:
    #     count[c] += 1
    # dist = np.array(count, dtype=float)
    # dist = dist / dist.sum()
    # dist = dist.tolist()
    # out = {"distribution": dist}
    # output_file = r"./data_statistics/"
    # output_path = output_file + str(name) + '_' + str(num_labels) + '.json'
    # if not os.path.exists(output_file):
    #     os.makedirs(output_file, exist_ok=True)
    # with open(output_path, 'w') as w:
    #     json.dump(out, w)

    lb_dset = BasicDataset(alg, lb_data, lb_targets, num_classes, transform_weak, False, None, False)

    ulb_dset = BasicDataset(alg, ulb_data, ulb_targets, num_classes, transform_weak, True, transform_strong, False)

    dset = getattr(torchvision.datasets, name.upper())
    dset = dset(data_dir, train=False, download=True)
    test_data, test_targets = dset.data, dset.targets

    unique, counts = np.unique(test_targets,
                               return_counts=True)
    print("unique_test_targets", unique)
    print("counts_test_targets", counts)
    unique_args_sort = np.argsort(unique)
    print("unique_args_sort", unique_args_sort)

    print("counts_test_targets_sum", counts.sum())
    log_data_dist(args, counts[unique_args_sort], "dataset/test data dist")

    eval_dset = BasicDataset(alg, test_data, test_targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
