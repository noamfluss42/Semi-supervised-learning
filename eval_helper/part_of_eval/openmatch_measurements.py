import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score


def calc_unk_acc(args, test_labels_openmatch_format, test_preds_openmatch_format):
    eq_unseen = \
        np.where(
            (test_labels_openmatch_format == args.num_classes) & (test_preds_openmatch_format == args.num_classes))[
            0].shape[0]
    count_unseen = np.where(test_labels_openmatch_format == args.num_classes)[0].shape[0]
    return eq_unseen / count_unseen


def calc_closed_acc(args, test_labels_openmatch_format, test_preds_openmatch_format):
    eq_seen = np.where((test_labels_openmatch_format != args.num_classes) & (
            test_preds_openmatch_format == test_labels_openmatch_format))[0].shape[0]
    count_seen = np.where(test_labels_openmatch_format != args.num_classes)[0].shape[0]
    return eq_seen / count_seen


def get_openmatch_settings(args,test_labels,test_preds):
    test_labels_openmatch_format = test_labels.copy()
    test_preds_openmatch_format = test_preds.copy()
    for label in range(args.num_classes):
        if label in args.missing_labels:
            test_labels_openmatch_format[test_labels_openmatch_format == label] = args.num_classes
            test_preds_openmatch_format[test_preds_openmatch_format == label] = args.num_classes
    return test_labels_openmatch_format,test_preds_openmatch_format

def calc_openmatch_measurements(args, test_labels, test_probs, test_preds, test_feats, test_preds_only_seen):
    test_labels_openmatch_format,test_preds_openmatch_format = get_openmatch_settings(args,test_labels,test_preds)
    overall_acc = 100*accuracy_score(test_labels_openmatch_format, test_preds_openmatch_format)
    unk_acc = 100*calc_unk_acc(args, test_labels_openmatch_format, test_preds_openmatch_format)
    closed_acc = 100*calc_closed_acc(args, test_labels_openmatch_format, test_preds_only_seen)

    return overall_acc, unk_acc, closed_acc


def compute_roc(args, unseen_scores, test_labels):
    Y_test = np.zeros(unseen_scores.shape[0])
    unk_pos = np.where(np.isin(test_labels, args.missing_labels))[0]
    Y_test[unk_pos] = 1
    return roc_auc_score(Y_test, unseen_scores)



def calc_openmatch_measurements_roc(args, test_labels, test_probs, test_preds, test_feats):
    test_labels_openmatch_format, test_preds_openmatch_format = get_openmatch_settings(args, test_labels, test_preds)
    unseen_scores = test_probs[:, args.missing_labels].max(axis=1)
    seen_labels = np.setdiff1d(range(args.num_classes), args.missing_labels)
    seen_scores = test_probs[:, seen_labels].max(axis=1)
    roc = 100*compute_roc(args, unseen_scores, test_labels)
    roc_soft = 100*compute_roc(args, -seen_scores, test_labels)
    return roc, roc_soft

def calc_rejection_acc(args,test_labels, test_probs, test_preds, test_feats):
    test_labels_openmatch_format, test_preds_openmatch_format = get_openmatch_settings(args, test_labels, test_preds)
    unseen_indices = np.where(test_labels_openmatch_format == args.num_classes)[0]
    rejection_acc = accuracy_score(test_labels_openmatch_format[unseen_indices], test_preds_openmatch_format[unseen_indices])
    return rejection_acc