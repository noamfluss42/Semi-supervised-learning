# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import pprint

import os
import torch

from torch.utils.data import DataLoader
from semilearn.core.utils import get_net_builder, get_dataset

from eval_helper.part_of_eval.calc_accuracy import *

from eval_helper.part_of_eval.eval_kmeans import create_k_means, calc_accuracy_with_k_means_optimal, \
    run_threshold_range, save_in_file_threshold, calc_accuracy_with_k_means_specific_threshold, run_basic_algorithm
from eval_helper.part_of_eval.eval_validation_compare import test_val
from eval_helper.part_of_eval.utils import get_missing_labels_by_path, get_out_path_by_slurm_job_id_remote, \
    get_parameter_by_path, get_out_path_by_slurm_job_id_local, get_slurm_job_id
from eval_helper.part_of_eval.openmatch_measurements import calc_openmatch_measurements, \
    calc_openmatch_measurements_roc, calc_rejection_acc
from eval_helper.part_of_eval.eval_openmatch_measurements import save_in_file_threshold_openmatch_measurments

slurm_job_id_to_stop_epoch = {
    # 10 - 400
    #     17177970: 404,
    #     17177969: 425,
    #     17177967: 467,
    #     17177968: 427,
    #     17160396: 450,
    17166716: 352,
    17166712: 348,
    17166715: 367,
    17166714: 389,
    17160265: 320,
    # 10 - 700
    # 17188014: 449,
    # 17188013: 320,
    # 17188012: 449,
    # 17188011: 423,
    # 17182998: 437,
    17195785: 449,
    17195783: 437,
    17195784: 422,
    17195787: 449,
    17195786: 461,
    # 100 - 400
    17171959: 476,
    17171958: 412,
    17171957: 79,
    17171956: 432,
    17166713: 342,
    # 100 - 1500
    # 17187998: 430,
    # 17187999: 461,
    # 17187997: 453,
    # 17188000: 456,
    # 17182987: 464
    17195790: 499,
    17195793: 444,
    17195792: 499,
    17195791: 499,
    17195789: 464,
    # 50 - 1500
    17195858: 440,
    17195852: 405,
    17195854: 406,
    17195850: 446,
    17195856: 454,

}


def create_mean_confidence_graph(test_probs, test_labels, test_preds, missing_labels):
    test_max_prob = test_probs.max(axis=1)
    mean_threshold = []
    mean_threshold_wrong = []
    missing_labels_mean_threshold = []
    for class_index in range(100):
        mean_threshold.append(test_max_prob[test_labels == class_index].mean())
        mean_threshold_wrong.append(test_max_prob[(test_labels == class_index) & (test_preds != class_index)].mean())
    for class_index in missing_labels:
        missing_labels_mean_threshold.append(test_max_prob[test_labels == class_index].mean())

    mean_threshold = np.array(mean_threshold)
    mean_threshold_wrong = np.array(mean_threshold_wrong)
    missing_labels_mean_threshold = np.array(missing_labels_mean_threshold)
    print([round(i, 2) for i in missing_labels_mean_threshold])
    print("mean_threshold", mean_threshold.mean())
    lowest_mean_threshold = np.sort((mean_threshold.argsort()[:len(missing_labels)]))
    print(f"min {len(missing_labels)} confidence", np.sort((mean_threshold.argsort()[:len(missing_labels)])))
    print(f"min {len(missing_labels)} confidence wrong",
          np.sort((mean_threshold_wrong.argsort()[:len(missing_labels)])))
    count_missing_classes_with_low_confidence = sum(
        [int(class_index in lowest_mean_threshold) for class_index in missing_labels])
    print("count_missing_classes_with_low_confidence", count_missing_classes_with_low_confidence)
    up = test_max_prob[test_max_prob > mean_threshold.mean()]


def main():
    print("start")
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--load_path', type=str, required=True)

    '''
    Backbone Net Configurations
    '''
    parser.add_argument('--net', type=str, default='wrn_28_2')
    parser.add_argument('--net_from_name', type=bool, default=False)

    '''
    Data Configurations
    '''
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--data_dir', type=str, default='./data')
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--img_size', type=int, default=32)
    parser.add_argument('--crop_ratio', type=int, default=0.875)
    parser.add_argument('--max_length', type=int, default=512)
    parser.add_argument('--max_length_seconds', type=float, default=4.0)
    parser.add_argument('--sample_rate', type=int, default=16000)
    parser.add_argument('--choose_random_labeled_training_set', type=int, default=-1)
    parser.add_argument('--random_missing_labels_num', type=int, default=-1)
    parser.add_argument('--missing_labels', nargs='+', default=[], type=int,
                        help='labels we don\'t have labels in the dataset')
    # ['15573937' '15573941' '15573945' '16312936' '16312937']
    parser.add_argument('--is_eval', type=int, default=1)
    parser.add_argument('--use_k_means_optimal', type=int, default=0)
    parser.add_argument('--save_test_val', type=int, default=0)
    parser.add_argument('--run_threshold_range', type=int, default=0)
    parser.add_argument('--run_threshold_range_get_same_unseen_count', type=int, default=0)
    parser.add_argument('--run_threshold_validation_optimum_range', type=int, default=0)
    parser.add_argument('--version', type=int, default=0)
    parser.add_argument('--add_threshold_pass_by_ours', type=int, default=0)
    # prev - my_algorithm_slurm_job_id
    parser.add_argument('--my_algorithm_unseen_count', type=int, default=-1)
    parser.add_argument('--is_transductive', type=int, default=0)
    parser.add_argument('--run_simple_eval', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--add_openmatch_measurements', type=int, default=0)
    parser.add_argument('--threshold_kmeans', type=float, default=-1)
    parser.add_argument('--python_code_version', type=float, default=11,
                        help='version')
    parser.add_argument('--lt_ratio', type=int, default=1,
                        help='anything')
    parser.add_argument('--lt_choose_unseen_classes_randomly', type=int, default=1,
                        help='anything')
    parser.add_argument('--lt_choose_labeled_data_randomly', type=int, default=1,
                        help='anything')
    parser.add_argument('--lambda_kl_divergence', default=0, type=float,
                        help='the entropy loss coefficient')
    parser.add_argument('--remote', default=1, type=float,
                        help='-1 = local')
    parser.add_argument('--use_specific_step', default=-1, type=float,
                        help='-1 = local')
    parser.add_argument('--use_specific_step_change', default=0, type=int,
                        help='-1 = local')
    parser.add_argument('--use_best', default=-1, type=float,
                        help='-1 = local')
    parser.add_argument('--save_results', default=1, type=float,
                        help='')
    parser.add_argument('--choose_unseen_count_by_test', default=-1, type=float,
                        help='')

    # parser.add_argument('--seed', default=-1, type=int,
    #                     help='-1 = local')
    # parser.add_argument('--algorithm', default=-1, type=int,
    #                     help='-1 = local')

    args = parser.parse_args()
    print("start eval args.load_path", args.load_path)
    print("args.python_code_version", args.python_code_version)
    print("args.lt_ratio", args.lt_ratio)
    print("args.lt_choose_unseen_classes_randomly", args.lt_choose_unseen_classes_randomly)
    print("args.lt_choose_labeled_data_randomly", args.lt_choose_labeled_data_randomly)
    print("args.lambda_kl_divergence", args.lambda_kl_divergence)
    if "cs" not in args.load_path:
        if args.remote == 1:
            model_search_path = "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/saved_models/classic_cv"
            args.load_path = get_out_path_by_slurm_job_id_remote(args.load_path, model_search_path, in_name=False)
        if args.remote == -1:
            model_search_path = r"C:\Users\noamf\Documents\thesis\models_from_server"
            args.load_path = get_out_path_by_slurm_job_id_local(args.load_path, model_search_path)
    slurm_job_id = get_slurm_job_id(args)
    print("before change - args.load_path", args.load_path)
    if args.use_specific_step == 1 and int(slurm_job_id) in slurm_job_id_to_stop_epoch.keys():
        args.load_path = args.load_path.replace("latest_model",
                                                f"model_{max(50, slurm_job_id_to_stop_epoch[int(slurm_job_id)] + args.use_specific_step_change)}_")
        print("change to model number:",
              max(50, slurm_job_id_to_stop_epoch[int(slurm_job_id)] + args.use_specific_step_change))
    elif args.use_best == 1:
        args.load_path = args.load_path.replace("latest_model", f"model_best")

    print("chosen load_path", args.load_path)
    checkpoint_path = os.path.join(args.load_path)
    if args.remote == 1:
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    load_model = checkpoint['ema_model']
    load_state_dict = {}
    for key, item in load_model.items():
        if key.startswith('module'):
            new_key = '.'.join(key.split('.')[1:])
            load_state_dict[new_key] = item
        else:
            load_state_dict[key] = item
    save_dir = '/'.join(checkpoint_path.split('/')[:-1])
    args.save_dir = save_dir
    args.save_name = ''
    print("load_state_dict", args.load_path)
    try:
        net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
        keys = net.load_state_dict(load_state_dict, strict=False)
    except:
        try:
            args.net = "wrn_28_8"
            net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
            keys = net.load_state_dict(load_state_dict, strict=False)
        except:
            args.net = "wrn_var_37_2"
            args.num_classes = 10
            print("num_classes", args.num_classes)
            net = get_net_builder(args.net, args.net_from_name)(num_classes=args.num_classes)
            args.dataset = "stl10"
            print("dataset changed to stl10")
            keys = net.load_state_dict(load_state_dict, strict=False)
    if args.net == "wrn_var_37_2":
        args.img_size = 96
    if torch.cuda.is_available():
        net.cuda()
    net.eval()

    # specify these arguments manually
    args.num_labels = 400
    args.ulb_num_labels = 49600
    args.lb_imb_ratio = 1
    args.ulb_imb_ratio = 1
    args.seed = 0
    args.epoch = 1
    args.num_train_iter = 1024
    print("args.dataset", args.dataset)
    if args.remote == 1:
        args.lt_ratio = int(get_parameter_by_path(args, "lt_ratio"))
        args.lambda_kl_divergence = get_parameter_by_path(args, "kl_divergence")

    dataset_dict = get_dataset(args, 'flexmatch', args.dataset, args.num_labels, args.num_classes, args.data_dir,
                               False, is_transductive=args.is_transductive == 1)
    eval_dset = dataset_dict['eval']

    eval_loader = DataLoader(eval_dset, batch_size=args.batch_size, drop_last=False, shuffle=False,
                             num_workers=args.num_workers)
    if args.remote == 1:
        args.missing_labels = get_missing_labels_by_path(args)
    elif args.lt_ratio != 1:
        args.missing_labels = np.array(list(range(args.random_missing_labels_num)))
    else:
        exit("error - finding missing labels only for lt_ratio != 1")
    acc = 0.0
    test_feats = []
    test_preds = []
    test_probs = []
    test_labels = []
    with torch.no_grad():
        for data in eval_loader:
            image = data['x_lb']
            target = data['y_lb']
            if torch.cuda.is_available():
                image = image.type(torch.FloatTensor).cuda()
            else:
                image = image.type(torch.FloatTensor)
            feat = net(image, only_feat=True)
            logit = net(feat, only_fc=True)
            prob = logit.softmax(dim=-1)
            pred = prob.argmax(1)
            acc += pred.cpu().eq(target).numpy().sum()

            test_feats.append(feat.cpu().numpy())
            test_preds.append(pred.cpu().numpy())
            test_probs.append(prob.cpu().numpy())
            test_labels.append(target.cpu().numpy())

    test_feats = np.concatenate(test_feats)
    test_preds = np.concatenate(test_preds)
    test_probs = np.concatenate(test_probs)
    test_labels = np.concatenate(test_labels)
    threshold_optimal = -1
    if args.save_results == 1:
        path_to_save = "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/saved_result"
        path_to_save = os.path.join(path_to_save, slurm_job_id)
        if not os.path.isdir(path_to_save):
            os.mkdir(path_to_save)

        np.savez(os.path.join(path_to_save, args.load_path[args.load_path.rfind(os.sep) + 1:]), test_feats=test_feats,
                 test_preds=test_preds, test_probs=test_probs, test_labels=test_labels)
    if args.is_transductive == 1 or args.run_simple_eval == 1:
        print("simple eval:")
        clustering_accuracy_with_permutation, \
        iteration_validation_accuracy_appearing_lables, \
        iteration_validation_accuracy_recall, \
        iteration_validation_accuracy_precision = \
            main_calc_accuracy(args, test_labels, test_preds)
        print("Combine score - accuracy of seen + clustering of unseen",
              clustering_accuracy_with_permutation)
        print("seen classes accuracy", iteration_validation_accuracy_appearing_lables)
        print("unseen classes recall", iteration_validation_accuracy_recall)
        print("unseen classes precision", iteration_validation_accuracy_precision)
        print("missing classes", args.missing_labels)
        print("balanced acc")
        print(calc_balanced_accuracy_multiclass_classification(args, test_labels, test_preds))
        balanced_accuracy_multiclass_classification, balanced_accuracy_multiclass_classification_seen, \
        balanced_accuracy_multiclass_classification_unseen = \
            calc_balanced_accuracy_multiclass_classification(args, test_labels, test_preds)
        balanced_precision_multiclass_classification, balanced_precision_multiclass_classification_seen, \
        balanced_precision_multiclass_classification_unseen = \
            calc_balanced_precision_multiclass_classification(args, test_labels, test_preds)
        F1_score = calc_F1_score(balanced_accuracy_multiclass_classification,
                                 balanced_precision_multiclass_classification)
        F1_score_seen = calc_F1_score(balanced_accuracy_multiclass_classification_seen,
                                      balanced_precision_multiclass_classification_seen)
        F1_score_unseen = calc_F1_score(balanced_accuracy_multiclass_classification_unseen,
                                        balanced_precision_multiclass_classification_unseen)
        print("F1_score", F1_score)
        print("F1_score_seen", F1_score_seen)
        print("F1_score_unseen", F1_score_unseen)
        print("balanced_accuracy_multiclass_classification", balanced_accuracy_multiclass_classification)
        print("balanced_accuracy_multiclass_classification_seen", balanced_accuracy_multiclass_classification_seen)
        print("balanced_accuracy_multiclass_classification_unseen", balanced_accuracy_multiclass_classification_unseen)
        print("balanced_precision_multiclass_classification", balanced_precision_multiclass_classification)
        print("balanced_precision_multiclass_classification_seen", balanced_precision_multiclass_classification_seen)
        print("balanced_precision_multiclass_classification_unseen",
              balanced_precision_multiclass_classification_unseen)

    if args.use_k_means_optimal == 1:
        cluster_ids_x, cluster_centers = create_k_means(test_feats.copy(), args.num_classes)
        print("k means clustering accuracy", get_clustering_accuracy(test_labels.copy(), cluster_ids_x))
        calc_accuracy_with_k_means_optimal(args, test_labels.copy(), test_preds.copy(), test_feats.copy())

    if args.save_test_val == 1:
        test_val(args, test_labels.copy(), test_preds.copy())

    if args.run_threshold_range_get_same_unseen_count == 1:
        result_dict, threshold_unseen_count = run_threshold_range(args, test_labels.copy(), test_probs.copy(),
                                                                  test_preds.copy(),
                                                                  test_feats.copy(), first_half=0)
        save_in_file_threshold(args, result_dict, name="same_unseen_count_threshold",
                               chosen_threshold=threshold_unseen_count)
    if args.run_threshold_range == 1 and args.lambda_kl_divergence != 0:
        result_dict = run_basic_algorithm(args, test_labels.copy(), test_probs.copy(),
                                          test_preds.copy(),
                                          test_feats.copy())
        save_in_file_threshold(args, result_dict, name="optimal_threshold", chosen_threshold=threshold_optimal)
    elif args.run_threshold_range == 1 or (args.add_openmatch_measurements == 1 and args.threshold_kmeans == 1):
        result_dict, threshold_optimal = run_threshold_range(args, test_labels.copy(), test_probs.copy(),
                                                             test_preds.copy(),
                                                             test_feats.copy(), first_half=0)
        # calc_accuracy_with_k_means_specific_threshold(args, test_labels, test_probs, test_preds, test_feats, threshold_optimal)

        save_in_file_threshold(args, result_dict, name="optimal_threshold", chosen_threshold=threshold_optimal)

    if args.run_threshold_validation_optimum_range == 1:
        result_dict_first_half, threshold_optimal_first_half = run_threshold_range(args, test_labels.copy()[:5000],
                                                                                   test_probs.copy()[:5000],
                                                                                   test_preds.copy()[:5000],
                                                                                   test_feats.copy()[:5000],
                                                                                   first_half=1)
        result_dict_second_half = calc_accuracy_with_k_means_specific_threshold(args, test_labels.copy()[5000:],
                                                                                test_probs.copy()[5000:],
                                                                                test_preds.copy()[5000:],
                                                                                test_feats.copy()[5000:],
                                                                                threshold_optimal_first_half)
        save_in_file_threshold(args, result_dict_second_half, name="validation_threshold")
    if args.add_openmatch_measurements == 1:
        # TODO - it will not work for clean flexmatch
        test_preds_only_seen = test_preds.copy()
        test_probs_only_seen = test_probs.copy()
        if args.threshold_kmeans == 1:
            test_preds_new = test_preds.copy()
            # write code to set test_preds_new to the c
            test_preds_new[test_probs.max(axis=1) < threshold_optimal] = args.num_classes
            print("not ours - threshold_optimal", threshold_optimal)
        else:
            test_preds_new = test_preds.copy()
            pred_unseen_indices = np.isin(test_preds_only_seen, args.missing_labels)
            test_probs_unseen_0 = test_probs[pred_unseen_indices].copy()
            test_probs_unseen_0[:, args.missing_labels] = 0
            test_preds_only_seen[pred_unseen_indices] = test_probs_unseen_0.argmax(axis=1)
            test_probs_only_seen[:, args.missing_labels] = 0
            print("ours algorithm")
        test_probs_only_seen = test_probs_only_seen.max(axis=1)

        overall_acc, unk_acc, closed_acc = calc_openmatch_measurements(args, test_labels.copy(), test_probs.copy(),
                                                                       test_preds_new.copy(),
                                                                       test_feats.copy(), test_preds_only_seen.copy())
        roc, roc_soft = calc_openmatch_measurements_roc(args, test_labels.copy(), test_probs.copy(),
                                                        test_preds_new.copy(), test_feats.copy())

        rejection_acc = calc_rejection_acc(args, test_labels.copy(), test_probs.copy(), test_preds_new.copy(),
                                           test_feats.copy())
        save_in_file_threshold_openmatch_measurments(args, overall_acc, unk_acc, closed_acc, roc, roc_soft,
                                                     rejection_acc)
        print("overall_acc, unk_acc, closed_acc", overall_acc, unk_acc, closed_acc)
        print("rejection_acc", rejection_acc)
        print("roc, roc_soft", roc, roc_soft)


if __name__ == "__main__":
    main()
    print("end eval")

# 400 - 40
# /cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/saved_models/classic_cv/tuning/flexmatch_cifar100_2/15616880/latest_model.pth


# 2500 - 4
# /cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/saved_models/classic_cv/tuning/flexmatch_cifar100_1/15615211/latest_model.pth
