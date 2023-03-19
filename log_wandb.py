import os

import numpy as np
import torch
import wandb
from scipy.optimize import linear_sum_assignment
from matplotlib import pyplot as plt
import PIL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shutil
from log_wandb_dir.log_wandb_accuracy import log_missing_labels_metrics_v2
from log_wandb_dir.log_flexmatch import log_flexmatch
from log_wandb_dir.log_cm import *


def log_confidence(args, all_logits, epoch, max_mask_percentage, max_mask_percentage_per_class, threshold_all_classes,
                   threshold_per_class):
    pseudo_label = torch.softmax(all_logits, dim=-1)
    max_probs, targets_u = torch.max(pseudo_label, dim=-1)
    mask = max_probs.ge(args.threshold).float()

    mask_percentage = mask.mean().item()

    max_prob_values = torch.mean(max_probs)

    topk_softmax = torch.topk(pseudo_label, 2, dim=1)[0]
    margin_softmax = topk_softmax[:, 0] - topk_softmax[:, 1]
    confidence_margin_softmax = torch.mean(margin_softmax).item()

    topk_logits = torch.topk(all_logits, 2, dim=1)[0]
    margin_logits = topk_logits[:, 0] - topk_logits[:, 1]
    confidence_margin_logits = torch.mean(margin_logits).item()

    entropy_softmax = torch.mean(torch.sum(pseudo_label * torch.log(pseudo_label), dim=1)).item()

    wandb.log({"confidence/train confidence margin softmax": confidence_margin_softmax}, step=epoch)

    wandb.log({"confidence/train confidence margin logits": confidence_margin_logits}, step=epoch)

    wandb.log({"confidence/train confidence entropy softmax": entropy_softmax}, step=epoch)

    wandb.log({"confidence/train confidence max prob softmax": max_prob_values}, step=epoch)

    wandb.log({"confidence/train mask percentage": mask_percentage}, step=epoch)
    wandb.log({"confidence/train mask count": mask.shape[0] * mask_percentage}, step=epoch)

    if mask_percentage > max_mask_percentage:
        max_mask_percentage = mask_percentage

    if args.refix_mask_all_classes_threshold_max_percentage != -1 and mask_percentage < max_mask_percentage * (
            1 - args.refix_mask_all_classes_threshold_max_percentage / 100):
        threshold_all_classes = torch.quantile(max_probs, 1 - max_mask_percentage).item()
        wandb.log({"new mask/new mask threshold all classes": threshold_all_classes}, step=epoch)

    for i in range(len(max_mask_percentage_per_class)):
        class_mask_percentage = mask[targets_u == i].mean().item()
        if class_mask_percentage > max_mask_percentage_per_class[i]:
            max_mask_percentage_per_class[i] = class_mask_percentage

        if args.refix_mask_each_class_threshold_max_percentage != -1 and class_mask_percentage < \
                max_mask_percentage_per_class[i] * (1 - args.refix_mask_each_class_threshold_max_percentage / 100):
            threshold_per_class[i] = torch.quantile(max_probs[targets_u == i],
                                                    1 - max_mask_percentage_per_class[i]).item()
            wandb.log({"new mask per class/new mask threshold class " + str(i): threshold_per_class[i]}, step=epoch)
        wandb.log({"new mask per class/train class " + str(i) + " mask percentage": class_mask_percentage}, step=epoch)
        wandb.log({"new mask per class/train class " + str(i) + " mask count": mask[targets_u == i].shape[
                                                                                   0] * class_mask_percentage},
                  step=epoch)

    return max_mask_percentage, threshold_all_classes


def log_data_dist(args, count_data, log_string, epoch=0):
    print("log_string", log_string)
    print("count_data", len(count_data), count_data)
    print("args.num_classes", args.num_classes)
    fig, ax = plt.subplots()
    ax.bar(list(range(args.num_classes)), count_data)  # counts_sort, unique_sort)
    wandb.log({log_string: fig}, step=epoch)
    plt.close()


def wandb_log_best_permutation_acc(args, cm, epoch, eval_dict,
                                   iteration_validation_clustering_accuracy_with_missing_labels=None):
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    wandb_im2, ax = plt.subplots()
    ax.bar(row_ind, col_ind)

    wandb.log({"permutation/iteration validation clustering permutation": wandb_im2}, step=epoch)
    plt.close()
    res_max = cm[row_ind, col_ind].sum()
    print("wandb_log_best_permutation_acc - epoch -", epoch, "with permutation", col_ind)
    wandb.log(
        {"permutation/iteration validation clustering accuracy": 100 * res_max / cm.sum()},
        step=epoch)
    wandb.log({"permutation/clustering - top1_acc": 100 * (res_max / cm.sum() - eval_dict["eval/top-1-acc"])},
              step=epoch)
    if iteration_validation_clustering_accuracy_with_missing_labels is not None:
        wandb.log({"permutation/clustering - missing_labels_clustering": 100 * (
                res_max / cm.sum()) - iteration_validation_clustering_accuracy_with_missing_labels}, step=epoch)

    wandb_log_iteration_validation_accuracy_labels_hist(args, cm, epoch, use_perm=True, perm=col_ind)


def wandb_log_iteration_validation_accuracy_labels_hist(args, cm, epoch, use_perm=False, perm=None):
    if not use_perm:
        perm = list(range(args.num_classes))
    accuracy = [0 for i in range(args.num_classes)]
    for i in range(args.num_classes):
        accuracy[i] = 100 * cm[i, perm[i]] / (cm.sum() / args.num_classes)
    wandb_im2, ax = plt.subplots()
    ax.bar(range(len(accuracy)), accuracy)

    if not use_perm:
        wandb.log({"media_graph/iteration validation accuracy labels": wandb_im2}, step=epoch)
    else:
        wandb.log({"media_graph/iteration validation accuracy labels - best permutation": wandb_im2}, step=epoch)
    plt.close()


def wandb_log_iteration_validation_prediction_count_labels_hist(args, cm, epoch):
    prediction_count = [0 for i in range(args.num_classes)]
    for i in range(args.num_classes):
        prediction_count[i] = cm[:, i].sum()
    wandb_im2, ax = plt.subplots()
    ax.bar(range(len(prediction_count)), prediction_count)
    wandb.log({"media_graph/iteration validation prediction_count labels": wandb_im2}, step=epoch)
    plt.close()


def log_wandb_cm(args, epoch, y_true, y_pred, eval_dict):
    print("start log cm with epoch -", epoch)

    cm = confusion_matrix(y_true, y_pred)

    print("shape cm", cm.shape)
    if len(args.missing_labels) > 0:
        if args.python_code_version == 2:
            iteration_validation_clustering_accuracy_with_missing_labels = log_missing_labels_metrics_v2(args, cm,
                                                                                                         epoch, y_true,
                                                                                                         y_pred)
        wandb_log_best_permutation_acc(args, cm, epoch, eval_dict,
                                       iteration_validation_clustering_accuracy_with_missing_labels)
    if len(args.missing_labels) == 0:
        wandb_log_best_permutation_acc(args, cm, epoch, eval_dict)
    if args.num_classes <= 10:
        if args.num_classes <= 10:
            wandb_log_cm_img(args, cm, epoch)
            wandb_log_cm_last_instance(y_true=y_true,
                                       preds=y_pred,
                                       class_names=[i for i in range(args.num_classes)], epoch=epoch)


def log_loss(args, epoch, loss_train_epoch):
    wandb.log({"loss/entropy_loss": loss_train_epoch["entropy_loss"]}, step=epoch)
    wandb.log({"loss/datapoint_entropy_loss": loss_train_epoch["datapoint_entropy_loss"]}, step=epoch)
    wandb.log({"loss/total_loss": loss_train_epoch["total_loss"]}, step=epoch)
    wandb.log({"loss/sup_loss": loss_train_epoch["sup_loss"]}, step=epoch)
    wandb.log({"loss/unsup_loss": loss_train_epoch["unsup_loss"]}, step=epoch)

    wandb.log({"loss_in_total_loss/entropy_loss": args.lambda_entropy * loss_train_epoch["entropy_loss"]}, step=epoch)
    wandb.log({"loss_in_total_loss/datapoint_entropy_loss": args.lambda_datapoint_entropy * loss_train_epoch[
        "datapoint_entropy_loss"]}, step=epoch)
    wandb.log({"loss_in_total_loss/total_loss": loss_train_epoch["total_loss"]}, step=epoch)
    wandb.log({"loss_in_total_loss/sup_loss": loss_train_epoch["sup_loss"]}, step=epoch)
    wandb.log({"loss_in_total_loss/unsup_loss": args.ulb_loss_ratio * loss_train_epoch["unsup_loss"]}, step=epoch)


def main_log_wandb(args, epoch, y_true, y_pred, eval_dict, loss_train_epoch):
    log_loss(args, epoch, loss_train_epoch)
    print_cm(y_true, y_pred)
    wandb.log(eval_dict, step=epoch)
    print("\neval_dict", eval_dict, "\n", "current epoch", epoch)
    log_wandb_cm(args, epoch, y_true, y_pred, eval_dict)
