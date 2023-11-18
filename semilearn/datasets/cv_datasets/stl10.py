# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json
import torchvision
import numpy as np
import math 
from torchvision import transforms

from .datasetbase import BasicDataset
from semilearn.datasets.utils import sample_labeled_unlabeled_data, split_ssl_data
from semilearn.datasets.augmentation import RandAugment


from log_wandb import log_data_dist, log_labeled_dist_config_main,log_count_data_dist_by_array

mean, std = {}, {}
mean['stl10'] = [x / 255 for x in [112.4, 109.1, 98.6]]
std['stl10'] = [x / 255 for x in [68.4, 66.6, 68.5]]
img_size = 96

def get_transform(mean, std, crop_size, train=True, crop_ratio=0.95):
    img_size = int(img_size / crop_ratio)

    if train:
        return transforms.Compose([transforms.RandomHorizontalFlip(),
                                   transforms.Resize(img_size),
                                   transforms.RandomCrop(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])
    else:
        return transforms.Compose([transforms.Resize(crop_size),
                                   transforms.ToTensor(),
                                   transforms.Normalize(mean, std)])



def remove_missing_labels(args, lb_data, lb_targets):
    if args.choose_random_labeled_training_set == -1 and len(args.missing_labels) > 0 or args.random_missing_labels_num > 0:
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
                if hasattr(args,"choose_random_labeled_training_set_duplicate"):
                    if args.choose_random_labeled_training_set_duplicate == 1:
                        if lb_targets[target_index] not in class_to_indexes:
                            class_to_indexes[lb_targets[target_index]] = []
                        class_to_indexes[lb_targets[target_index]].append(target_index)
        if hasattr(args,"choose_random_labeled_training_set_duplicate"):
            if args.choose_random_labeled_training_set_duplicate == 1:
                result_indexes = []
                max_per_class = args.choose_random_labeled_training_set_counts.max()
                for class_index in range(len(args.choose_random_labeled_training_set_unique)):
                    current_count = args.choose_random_labeled_training_set_counts[class_index]
                    class_indexes = class_to_indexes[args.choose_random_labeled_training_set_unique[class_index]]
                    new_class_indexes = class_indexes*int(math.floor(max_per_class/current_count))
                    new_class_indexes = np.concatenate((new_class_indexes,class_indexes[:max_per_class%len(new_class_indexes)]))
                    result_indexes = np.concatenate((result_indexes,new_class_indexes))
                np.random.shuffle(result_indexes)
                # convert result_indexes to int
                result_indexes = result_indexes.astype(int)
                return lb_data[result_indexes], lb_targets[result_indexes], lb_data[result_bool], lb_targets[result_bool]
        return lb_data[result_bool], lb_targets[result_bool]
    return lb_data, lb_targets



def get_stl10(args, alg, name, num_labels, num_classes, data_dir='./data', include_lb_to_ulb=False,is_transductive=False):
    
    crop_size = args.img_size
    crop_ratio = args.crop_ratio
    img_size = int(math.floor(crop_size / crop_ratio))

    transform_weak = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean[name], std[name])
    ])

    transform_strong = transforms.Compose([
        transforms.Resize(crop_size),
        transforms.RandomCrop((crop_size, crop_size), padding=int(crop_size * (1 - crop_ratio)), padding_mode='reflect'),
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

    data_dir = os.path.join(data_dir, name.lower())
    dset = getattr(torchvision.datasets, name.upper())

    dset_lb = dset(data_dir, split='train', download=True)
    dset_ulb = dset(data_dir, split='unlabeled', download=True)
    lb_data, lb_targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    #ulb_data = dset_ulb.data.transpose([0, 2, 3, 1])

    # Note this data can have imbalanced labeled set, and with unknown unlabeled set
    # ulb_data = np.concatenate([ulb_data, lb_data], axis=0)
    #ulb_data = lb_data.copy()
    # lb_idx, ulb_idx = sample_labeled_unlabeled_data(args, lb_data, lb_targets, num_classes,
    #                                           lb_num_labels=num_labels,
    #                                           ulb_num_labels=args.ulb_num_labels,
    #                                           lb_imbalance_ratio=args.lb_imb_ratio,
    #                                           ulb_imbalance_ratio=args.ulb_imb_ratio,
    #                                           load_exist=True)
    #ulb_targets = np.ones((ulb_data.shape[0], )) * -1
    # lb_data, lb_targets = lb_data[lb_idx], lb_targets[lb_idx]
    # if include_lb_to_ulb:
    #     ulb_data = np.concatenate([lb_data, ulb_data], axis=0)
    #     ulb_targets = np.concatenate([lb_targets, np.ones((ulb_data.shape[0] - lb_data.shape[0], )) * -1], axis=0)
    # ulb_targets = ulb_targets.astype(np.int64)
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(args, lb_data, lb_targets, num_classes,
                                              lb_num_labels=num_labels,
                                              ulb_num_labels=args.ulb_num_labels,
                                              lb_imbalance_ratio=args.lb_imb_ratio,
                                              ulb_imbalance_ratio=args.ulb_imb_ratio)

    # output the distribution of labeled data for remixmatch

    if hasattr(args, "choose_random_labeled_training_set_duplicate"):
        if args.choose_random_labeled_training_set_duplicate == 1:

            lb_data, lb_targets,lb_data_original, lb_targets_original = remove_missing_labels(args, lb_data, lb_targets)
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
    dset_lb = dset(data_dir, split='test', download=True)
    data, targets = dset_lb.data.transpose([0, 2, 3, 1]), dset_lb.labels.astype(np.int64)
    eval_dset = BasicDataset(alg, data, targets, num_classes, transform_val, False, None, False)

    return lb_dset, ulb_dset, eval_dset
