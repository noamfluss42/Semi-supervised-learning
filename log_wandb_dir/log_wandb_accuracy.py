import numpy as np
import wandb
from scipy.optimize import linear_sum_assignment

def log_appearing_lables_matrices_v2(args, cm, epoch):
    iteration_validation_accuracy_appearing_lables_count = 0
    for label in range(args.num_classes):
        if label not in args.missing_labels:
            iteration_validation_accuracy_appearing_lables_count += cm[label, label]
    wandb.log(
        {
            "accuracy/iteration validation accuracy appearing lables": 100 * iteration_validation_accuracy_appearing_lables_count / (
                    cm.sum() * (1 - len(args.missing_labels) / args.num_classes))},
        step=epoch)
    wandb.log(
        {
            "accuracy/iteration validation count appearing lables": iteration_validation_accuracy_appearing_lables_count},
        step=epoch)
    return iteration_validation_accuracy_appearing_lables_count


def log_accuracy_section_v2(args, cm, epoch, iteration_missing_labels_tp, iteration_missing_labels_tp_plus_fp,
                            iteration_missing_labels_tp_plus_fn):
    iteration_validation_accuracy_appearing_lables_count = log_appearing_lables_matrices_v2(args, cm, epoch)

    if iteration_missing_labels_tp_plus_fp == 0:
        iteration_missing_labels_tp_plus_fp = 1
    if iteration_missing_labels_tp_plus_fn == 0:
        iteration_missing_labels_tp_plus_fn = 1
    wandb.log(
        {
            "accuracy/iteration validation accuracy missing labels precision": 100 * iteration_missing_labels_tp / iteration_missing_labels_tp_plus_fp},
        step=epoch)
    wandb.log(
        {
            "accuracy/iteration validation accuracy missing labels recall": 100 * iteration_missing_labels_tp / iteration_missing_labels_tp_plus_fn},
        step=epoch)

    wandb.log(
        {
            "accuracy/iteration validation clustering accuracy with missing labels": 100 * (
                    iteration_validation_accuracy_appearing_lables_count + iteration_missing_labels_tp) / cm.sum()},
        step=epoch)
    return iteration_validation_accuracy_appearing_lables_count


def log_confusion_metrics_summary_v2(args, cm, epoch):
    cm_missing_labels = np.transpose(cm[args.missing_labels, [[i] * len(args.missing_labels) for i in
                                                              args.missing_labels]])  # rows = true, col = pred index (0,1) = true 0, pred 1
    row_ind, col_ind = linear_sum_assignment(cm_missing_labels, maximize=True)
    iteration_missing_labels_tp = cm_missing_labels[row_ind, col_ind].sum()
    iteration_missing_labels_tp_plus_fp = cm[:, args.missing_labels].sum()
    iteration_missing_labels_tp_plus_fn = cm.sum() * len(args.missing_labels) / args.num_classes
    print("confusion_matrix_summary/iteration_missing_labels_tp", iteration_missing_labels_tp)
    print("confusion_matrix_summary/iteration_missing_labels_tp_plus_fp", iteration_missing_labels_tp_plus_fp)
    print("confusion_matrix_summary/iteration_missing_labels_tp_plus_fn", iteration_missing_labels_tp_plus_fn)

    wandb.log({"confusion_matrix_summary/iteration validation missing labels tp": iteration_missing_labels_tp},
              step=epoch)
    wandb.log(
        {"confusion_matrix_summary/iteration validation missing labels tp+fp": iteration_missing_labels_tp_plus_fp},
        step=epoch)
    wandb.log(
        {"confusion_matrix_summary/iteration validation missing labels tp+fn": iteration_missing_labels_tp_plus_fn},
        step=epoch)
    wandb.log(
        {
            "confusion_matrix_summary/iteration validation missing labels fp": iteration_missing_labels_tp_plus_fp - iteration_missing_labels_tp},
        step=epoch)
    wandb.log(
        {
            "confusion_matrix_summary/iteration validation missing labels fn": iteration_missing_labels_tp_plus_fn - iteration_missing_labels_tp},
        step=epoch)
    return iteration_missing_labels_tp, iteration_missing_labels_tp_plus_fp, iteration_missing_labels_tp_plus_fn


def log_missing_labels_metrics_v2(args, cm, epoch, y_true, y_pred):
    iteration_missing_labels_tp, iteration_missing_labels_tp_plus_fp, iteration_missing_labels_tp_plus_fn = log_confusion_metrics_summary_v2(
        args, cm, epoch)
    iteration_validation_accuracy_appearing_lables_count = log_accuracy_section_v2(args, cm, epoch,
                                                                                   iteration_missing_labels_tp,
                                                                                   iteration_missing_labels_tp_plus_fp,
                                                                                   iteration_missing_labels_tp_plus_fn)
    return 100 * (iteration_validation_accuracy_appearing_lables_count + iteration_missing_labels_tp) / cm.sum()

