import math

import numpy as np
import wandb
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix


def get_iteration_validation_accuracy_appearing_lables_count(args, cm):
    iteration_validation_accuracy_appearing_lables_count = 0
    for label in range(args.num_classes):
        if label not in args.missing_labels:
            iteration_validation_accuracy_appearing_lables_count += cm[label, label]
    return iteration_validation_accuracy_appearing_lables_count


def get_iteration_missing_labels_tp(args, cm):
    cm_missing_labels = np.transpose(cm[args.missing_labels, [[i] * len(args.missing_labels) for i in
                                                              args.missing_labels]])  # rows = true, col = pred index (0,1) = true 0, pred 1
    row_ind, col_ind = linear_sum_assignment(cm_missing_labels, maximize=True)
    iteration_missing_labels_tp = cm_missing_labels[row_ind, col_ind].sum()
    iteration_missing_labels_tp_plus_fp = cm[:, args.missing_labels].sum()
    iteration_missing_labels_tp_plus_fn = cm[args.missing_labels, :].sum()
    print("\n\n\n\n\n\n\n\nmay be a problem in iteration_missing_labels_tp_plus_fn")
    print("original", cm.sum() * len(args.missing_labels) / args.num_classes, "new",
          iteration_missing_labels_tp_plus_fn)
    return iteration_missing_labels_tp, iteration_missing_labels_tp_plus_fp, iteration_missing_labels_tp_plus_fn


def main_calc_accuracy(args, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    iteration_validation_accuracy_appearing_lables_count = get_iteration_validation_accuracy_appearing_lables_count(
        args, cm)
    iteration_missing_labels_tp, iteration_missing_labels_tp_plus_fp, iteration_missing_labels_tp_plus_fn = get_iteration_missing_labels_tp(
        args, cm)
    clustering_accuracy_with_permutation = 100 * (
            iteration_validation_accuracy_appearing_lables_count + iteration_missing_labels_tp) / cm.sum()

    iteration_validation_accuracy_appearing_lables = 100 * iteration_validation_accuracy_appearing_lables_count / \
                                                     cm[[i for i in range(args.num_classes) if
                                                         i not in args.missing_labels], :].sum()
    if iteration_missing_labels_tp_plus_fp == 0:
        iteration_missing_labels_tp_plus_fp = 1
    if iteration_missing_labels_tp_plus_fn == 0:
        iteration_missing_labels_tp_plus_fn = 1
    iteration_validation_accuracy_recall = 100 * iteration_missing_labels_tp / iteration_missing_labels_tp_plus_fn
    iteration_validation_accuracy_precision = 100 * iteration_missing_labels_tp / iteration_missing_labels_tp_plus_fp
    return clustering_accuracy_with_permutation, \
           iteration_validation_accuracy_appearing_lables, \
           iteration_validation_accuracy_recall, \
           iteration_validation_accuracy_precision


def get_clustering_accuracy(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    row_ind, col_ind = linear_sum_assignment(cm, maximize=True)
    res_max = cm[row_ind, col_ind].sum()
    return 100 * res_max / cm.sum()


def clac_threshold_score(args, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)  # lines = true, col = pred
    seen_classes = [i for i in range(args.num_classes) if i not in args.missing_labels]
    P = cm[seen_classes, :].sum()
    N = cm.sum() - P
    TP = cm[seen_classes, :][:, seen_classes].sum()
    TN = cm[args.missing_labels, :][:, args.missing_labels].sum()
    sensitivity = TP / P  # TPR
    specificity = TN / N  # TNR
    return math.sqrt(sensitivity * specificity)


def calc_balanced_accuracy_multiclass_classification(args, y_true, y_pred):
    sensitivity_per_class = []  # recall for each class = sensitivity
    sensitivity_per_class_unseen = []
    sensitivity_per_class_seen = []
    cm = confusion_matrix(y_true, y_pred)  # lines = true, col = pred
    cm_missing_labels = cm[args.missing_labels, :][:,
                        args.missing_labels]  # rows = true, col = pred index (0,1) = true 0, pred 1
    row_ind, col_ind = linear_sum_assignment(cm_missing_labels, maximize=True)

    for i in range(args.num_classes):
        current_col = i
        if i in args.missing_labels:
            index = np.where(args.missing_labels == i)[0][0]
            current_col = col_ind[index]
        current_P = cm[i, :].sum()
        current_TP = cm[i, current_col].sum()
        current_sensitivity = current_TP / current_P  # TPR
        if i in args.missing_labels:
            sensitivity_per_class_unseen.append(current_sensitivity)
        else:
            sensitivity_per_class_seen.append(current_sensitivity)
        sensitivity_per_class.append(current_sensitivity)
    return 100 * np.mean(sensitivity_per_class),100*np.mean(sensitivity_per_class_seen),100*np.mean(sensitivity_per_class_unseen)


def calc_balanced_precision_multiclass_classification(args, y_true, y_pred):
    precision_per_class = []  # precision for each class
    precision_per_class_seen = []
    precision_per_class_unseen = []
    cm = confusion_matrix(y_true, y_pred)  # lines = true, col = pred
    cm_missing_labels = cm[args.missing_labels, :][:,
                        args.missing_labels]  # rows = true, col = pred index (0,1) = true 0, pred 1
    row_ind, col_ind = linear_sum_assignment(cm_missing_labels, maximize=True)
    for i in range(args.num_classes):
        current_col = i
        if i in args.missing_labels:
            index = np.where(args.missing_labels == i)[0][0]
            current_col = col_ind[index]
        current_TP_FP = cm[:, current_col].sum()
        current_TP = cm[i, current_col].sum()
        if current_TP_FP == 0:
            current_precision = 0
        else:
            current_precision = current_TP / current_TP_FP  # TPR
        if i in args.missing_labels:
            precision_per_class_unseen.append(current_precision)
        else:
            precision_per_class_seen.append(current_precision)
        precision_per_class.append(current_precision)
    return 100 * np.mean(precision_per_class),100 * np.mean(precision_per_class_seen),100 * np.mean(precision_per_class_unseen)


#    pass

def calc_F1_score(balanced_accuracy_multiclass_classification, balanced_precision_multiclass_classification):
    return 2 * balanced_accuracy_multiclass_classification * balanced_precision_multiclass_classification / \
           (balanced_accuracy_multiclass_classification + balanced_precision_multiclass_classification)
