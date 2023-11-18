import pprint
import sys

import numpy as np
import os
import torch
from matplotlib import pyplot as plt
from kmeans_pytorch import kmeans
from .utils import get_slurm_job_id, get_out_path_by_slurm_job_id_remote, get_parameter_from_out_path, \
    get_unseen_count_by_slurm_job_id
# from eval_helper.part_of_eval.utils import get_slurm_job_id, get_out_path_by_slurm_job_id, get_parameter_from_out_path, \
#     get_unseen_count_by_slurm_job_id
from .calc_accuracy import get_clustering_accuracy, main_calc_accuracy, clac_threshold_score, \
    calc_balanced_accuracy_multiclass_classification, calc_balanced_precision_multiclass_classification, calc_F1_score

# from eval_helper.part_of_eval.calc_accuracy import get_clustering_accuracy, main_calc_accuracy, clac_threshold_score, \
#     calc_balanced_accuracy_multiclass_classification
threshold_range_big = [0.01 + i / 100 for i in range(80)] + [0.81 + i / 200 for i in range(20)] + [0.9 + i / 300
                                                                                                   for i in
                                                                                                   range(30)] + [0.9999,
                                                                                                                 1.1]
threshold_range_big1 = [0.01 + i / 100 for i in range(80)] + [0.81 + i / 200 for i in range(20)] + [0.9 + i / 500
                                                                                                   for i in
                                                                                                   range(50)] + [0.9999,
                                                                                                                 1.1]
threshold_range_small = [0.01 + i / 10 for i in range(8)] + [0.81 + i / 20 for i in range(2)] + [0.9 + i / 30
                                                                                                 for i in range(3)] + [
                            0.9999, 1.1]

threshold_range = threshold_range_big1

FIGURE_OUTPUT_DIR_DICT_REMOTE_LOCAL = {1: "/cs/labs/daphna/noam.fluss/project/create_graphs/figure_output",
                                       -1: r"C:\Users\noamf\Documents\thesis\code\create_graphs\figure_output"}
TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL = {
    1: f"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper/result_output",
    -1: r"C:\Users\noamf\Documents\thesis\code\SSL_Benchmark\new_forked_semi_supervised_learning\Semi-supervised-learning\eval_helper\result_output"}


# [0.01 + i / 50 for i in range(40)] + [0.81 + i / 100 for i in range(10)] + [0.9 + i / 400 for i in
#                                                                             range(40)]


def reindex_after_k_means(args, labels_after_k_means):
    labels_after_k_means_filter_list = []
    for missing_class_index in range(len(args.missing_labels)):
        labels_after_k_means_filter_list.append(labels_after_k_means == missing_class_index)
    for missing_class_index in range(len(args.missing_labels)):
        labels_after_k_means[labels_after_k_means_filter_list[missing_class_index]] = args.missing_labels[
            missing_class_index]
    return labels_after_k_means


def create_k_means(train_feature, num_clusters):
    train_feature = torch.from_numpy(train_feature)
    # TODO num_clusters=min(num_clusters,len(train_feature))
    print("train_feature.shape", train_feature.shape)
    print("num_clusters", min(num_clusters, len(train_feature)))
    if min(num_clusters, len(train_feature)) == 1:
        cluster_ids_x = np.zeros(len(train_feature))
        cluster_centers = train_feature.mean(dim=0).unsqueeze(dim=0)
    else:
        if torch.cuda.is_available():
            cluster_ids_x, cluster_centers = kmeans(
                X=train_feature, num_clusters=min(num_clusters, len(train_feature)), distance='euclidean',
                device=torch.device('cuda:0')
            )
        else:
            cluster_ids_x, cluster_centers = kmeans(
                X=train_feature, num_clusters=min(num_clusters, len(train_feature)), distance='euclidean',
                device=torch.device('cpu')
            )
    # cluster_ids_x, cluster_centers = kmeans(
    #     X=train_feature, num_clusters=min(num_clusters, len(train_feature)), distance='euclidean',
    #     device=torch.device('cuda:0')
    # )
    return cluster_ids_x, cluster_centers


def save_in_file_threshold(args, result_dict_chosen_threshold, name="optimal_threshold", chosen_threshold=-1):
    print("print result_dict_chosen_threshold")
    pprint.pprint(result_dict_chosen_threshold)

    with open(os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[args.remote],
                           f"tmp_res_{name}_{get_slurm_job_id(args)}.out"), "w") as f:
        for key, value in result_dict_chosen_threshold.items():
            f.write(f"{key} {round(value,2)}\n")
        # f.write(
        #     f"Combine score - accuracy of seen + clustering of unseen "
        #     f"{round(result_dict_chosen_threshold['Combine score - accuracy of seen + clustering of unseen'], 2)}\n")
        # f.write(
        #     f"seen classes accuracy "
        #     f"{round(result_dict_chosen_threshold['seen classes accuracy'], 2)}\n")
        # f.write(
        #     f"unseen classes recall "
        #     f"{round(result_dict_chosen_threshold['unseen classes recall'], 2)}\n")
        # f.write(
        #     f"unseen classes precision "
        #     f"{round(result_dict_chosen_threshold['unseen classes precision'], 2)}\n")
        # f.write(
        #     f"chosen_threshold "
        #     f"{round(chosen_threshold, 3)}\n")
        # f.write(
        #     f"balanced accuracy "
        #     f"{round(result_dict_chosen_threshold['balanced_accuracy_multiclass_classification'], 2)}\n")
        # f.write(
        #     f"balanced precision "
        #     f"{round(result_dict_chosen_threshold['balanced_precision_multiclass_classification'], 2)}\n")
        # f.write(
        #     f"F1 score "
        #     f"{round(result_dict_chosen_threshold['F1_score'], 2)}\n")
        # f.write(
        #     f"F1 score seen "
        #     f"{round(result_dict_chosen_threshold['F1_score_seen'], 2)}\n")
        # f.write(
        #     f"F1 score unseen "
        #     f"{round(result_dict_chosen_threshold['F1_score_unseen'], 2)}\n")


def get_seed_algorithm_by_slurm_job_id(args):
    if args.remote == -1:
        return 999, "flexmatch"
    searched_path = "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/" \
                    "new_forked_semi_supervised_learning/Semi-supervised-learning/"

    out_path = get_out_path_by_slurm_job_id_remote(get_slurm_job_id(args), searched_path, in_name=False, out_file=True)
    seed = get_parameter_from_out_path(out_path, "seed: ")
    algorithm = get_parameter_from_out_path(out_path, "algorithm: ")
    return seed, algorithm


def save_optimal_threshold_graph_search(args, dir_path, seed, algorithm):
    algorithm_dir_path = os.path.join(dir_path, algorithm)
    if not os.path.isdir(algorithm_dir_path):
        os.mkdir(algorithm_dir_path)
    plt.savefig(os.path.join(algorithm_dir_path, f"optimal_threshold_graph_search_{seed}_{get_slurm_job_id(args)}.png"))


def create_figure_temp_save_dir(args, dir_name):
    dir_path = os.path.join(FIGURE_OUTPUT_DIR_DICT_REMOTE_LOCAL[args.remote], str(args.version))
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    dir_path = os.path.join(dir_path, dir_name)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)
    return dir_path


def save_optimal_threshold_figure(args, threshold_range, result_dict, first_half):
    dir_path = create_figure_temp_save_dir(args, "optimal_threshold")
    seed, algorithm = get_seed_algorithm_by_slurm_job_id(args)
    plt.clf()
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.figure(figsize=(10, 10))
    plt.scatter(threshold_range, result_dict["Combine score - accuracy of seen + clustering of unseen"])
    title = f"\nalgorithm {algorithm}, seed {seed}\n" \
            f"max Combine score - accuracy of seen + clustering of unseen is " \
            f"{max(result_dict['Combine score - accuracy of seen + clustering of unseen'])}\n" \
            f"threshold is {threshold_range[np.array(result_dict['Combine score - accuracy of seen + clustering of unseen']).argmax()]}"
    plt.title(title)
    if first_half == 1:
        plt.savefig(os.path.join(dir_path, "optimal_threshold_graph_search_first_half.png"))
    if first_half == 2:
        plt.savefig(os.path.join(dir_path, "optimal_threshold_graph_search_second_half.png"))
    else:  # first_half == 0
        # plt.savefig(f"{dir_path}/optimal_threshold_graph_search_{get_slurm_job_id(args)}.png")
        save_optimal_threshold_graph_search(args, dir_path, seed, algorithm)


def save_threshold_pass_graph_search(args, dir_path, seed, algorithm):
    algorithm_dir_path = os.path.join(dir_path, algorithm)
    if not os.path.isdir(algorithm_dir_path):
        os.mkdir(algorithm_dir_path)
    plt.savefig(os.path.join(algorithm_dir_path, "threshold_pass_graph_search_{seed}_{get_slurm_job_id(args)}.png"))


def save_general_threshold_range_figure(args, threshold_range, values, name):
    dir_path = create_figure_temp_save_dir(args, name.replace(" ", "_"))
    seed, algorithm = get_seed_algorithm_by_slurm_job_id(args)
    plt.clf()
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.figure(figsize=(10, 10))
    plt.scatter(threshold_range, values)
    title = f"\nalgorithm {algorithm}, seed {seed}\n{name}"
    plt.title(title)
    save_optimal_threshold_graph_search(args, dir_path, seed, algorithm)


def save_threshold_pass_figure(args, threshold_range, mask_pass_unseen_classes_all):
    dir_path = create_figure_temp_save_dir(args, "threshold_pass")
    seed, algorithm = get_seed_algorithm_by_slurm_job_id(args)
    plt.clf()
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.figure(figsize=(10, 10))
    plt.scatter(threshold_range, mask_pass_unseen_classes_all)
    title = f"\nalgorithm {algorithm}, seed {seed}\n" \
            f"mask pass unseen classes count "
    plt.title(title)
    save_optimal_threshold_graph_search(args, dir_path, seed, algorithm)


def save_threshold_score_figure(args, threshold_range, threshold_score_all):
    dir_path = create_figure_temp_save_dir(args, "threshold_score")
    seed, algorithm = get_seed_algorithm_by_slurm_job_id(args)
    plt.clf()
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.figure(figsize=(10, 10))
    plt.scatter(threshold_range, threshold_score_all)
    title = f"\nalgorithm {algorithm}, seed {seed}\n" \
            f"threshold score "
    plt.title(title)
    save_optimal_threshold_graph_search(args, dir_path, seed, algorithm)


def save_balanced_accuracy_multiclass_classification_figure(args, threshold_range, balanced_accuracy_all):
    dir_path = create_figure_temp_save_dir(args, "balanced_accuracy")
    seed, algorithm = get_seed_algorithm_by_slurm_job_id(args)
    plt.clf()
    plt.grid(True)
    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)
    plt.figure(figsize=(10, 10))
    plt.scatter(threshold_range, balanced_accuracy_all)
    title = f"\nalgorithm {algorithm}, seed {seed}\n" \
            f"balanced accuracy "
    plt.title(title)
    save_optimal_threshold_graph_search(args, dir_path, seed, algorithm)


def calc_accuracy_with_k_means_optimal(args, test_labels, test_preds, test_feats):
    filter_unseen_classes = np.array([label in args.missing_labels for label in test_labels])
    test_feats_unseen_classes = test_feats[filter_unseen_classes]

    test_preds[filter_unseen_classes] = reindex_after_k_means(args, create_k_means(test_feats_unseen_classes,
                                                                                   len(args.missing_labels))[0])
    print("clustering accuracy on unseen after k means",
          get_clustering_accuracy(test_labels[filter_unseen_classes], test_preds[filter_unseen_classes]))
    print("Combine score - accuracy of seen + clustering of unseen optimal k means",
          main_calc_accuracy(args, test_labels, test_preds)[0])
    print("threshold score", clac_threshold_score(args, test_labels, test_preds))


def calc_accuracy_with_k_means_specific_threshold(args, test_labels, test_probs, test_preds, test_feats, threshold):
    test_max_prob = test_probs.max(axis=1)

    print(f"\n\nstart calc_accuracy_with_k_means_optimal - {threshold}")
    filter_unseen_classes = np.array([max_prob < threshold for max_prob in test_max_prob])
    print(f"filter_unseen_classes count - {sum(filter_unseen_classes)}")
    test_feats_unseen_classes = test_feats[filter_unseen_classes]
    if len(test_feats_unseen_classes) > 1:
        test_preds[filter_unseen_classes] = reindex_after_k_means(args, create_k_means(test_feats_unseen_classes,
                                                                                       len(args.missing_labels))[0])
    print("clustering accuracy on unseen after k means",
          get_clustering_accuracy(test_labels[filter_unseen_classes], test_preds[filter_unseen_classes]))
    clustering_accuracy_with_permutation, \
    iteration_validation_accuracy_appearing_lables, \
    iteration_validation_accuracy_recall, \
    iteration_validation_accuracy_precision = \
        main_calc_accuracy(args, test_labels, test_preds)
    threshold_score = clac_threshold_score(args, test_labels, test_preds)
    balanced_accuracy_multiclass_classification, balanced_accuracy_multiclass_classification_seen, \
    balanced_accuracy_multiclass_classification_unseen = \
        calc_balanced_accuracy_multiclass_classification(args, test_labels, test_preds)
    balanced_precision_multiclass_classification, balanced_precision_multiclass_classification_seen, \
    balanced_precision_multiclass_classification_unseen = \
        calc_balanced_precision_multiclass_classification(args, test_labels, test_preds)
    F1_score = calc_F1_score(balanced_accuracy_multiclass_classification, balanced_precision_multiclass_classification)
    F1_score_seen = calc_F1_score(balanced_accuracy_multiclass_classification_seen,
                                  balanced_precision_multiclass_classification_seen)
    F1_score_unseen = calc_F1_score(balanced_accuracy_multiclass_classification_unseen,
                                    balanced_precision_multiclass_classification_unseen)
    return {"Combine score - accuracy of seen + clustering of unseen": clustering_accuracy_with_permutation,
            "seen classes accuracy": iteration_validation_accuracy_appearing_lables,
            "unseen classes recall": iteration_validation_accuracy_recall,
            "unseen classes precision": iteration_validation_accuracy_precision,
            "balanced_accuracy_multiclass_classification": balanced_accuracy_multiclass_classification,
            "balanced_precision_multiclass_classification": balanced_precision_multiclass_classification,
            "F1_score": F1_score,
            "F1_score_seen": F1_score_seen,
            "F1_score_unseen": F1_score_unseen}, sum(
        filter_unseen_classes), threshold_score, balanced_accuracy_multiclass_classification, \
           balanced_precision_multiclass_classification, F1_score, F1_score_seen, F1_score_unseen


def get_index_same_unseen_count(args, mask_pass_unseen_classes_all,unseen_count):

    #                                 args.random_missing_labels_num * 100)
    print("start get_index_same_unseen_count unseen_count", unseen_count)
    print("mask_pass_unseen_classes_all", mask_pass_unseen_classes_all)
    print("np.abs(mask_pass_unseen_classes_all - unseen_count)", np.abs(mask_pass_unseen_classes_all - unseen_count))
    return np.argmin(np.abs(mask_pass_unseen_classes_all - unseen_count)), np.abs(
        mask_pass_unseen_classes_all - unseen_count)


def run_threshold_range(args, test_labels, test_probs, test_preds, test_feats, first_half=0):
    # threshold_range = [0.9 + i / 100 for i in range(5)]
    result_dict = {"Combine score - accuracy of seen + clustering of unseen": [],
                   "seen classes accuracy": [],
                   "unseen classes recall": [],
                   "unseen classes precision": [],
                   "balanced_accuracy_multiclass_classification": [],
                   "balanced_precision_multiclass_classification": [],
                   "F1_score": [], "F1_score_seen": [],"F1_score_unseen": []
                   }
    result_dict_all_optimal_threshold = {"Combine score - accuracy of seen + clustering of unseen": [],
                                         "seen classes accuracy": [],
                                         "unseen classes recall": [],
                                         "unseen classes precision": [],
                                         "balanced_accuracy_multiclass_classification": [],
                                         "balanced_precision_multiclass_classification": [],
                                         "F1_score": [], "F1_score_seen": [],"F1_score_unseen": []}
    mask_pass_unseen_classes_all = []
    threshold_score_all = []
    balanced_accuracy_multiclass_classification_all = []
    balanced_precision_multiclass_classification_all = []
    F1_score_all = []
    F1_score_seen_all = []
    F1_score_unseen_all = []
    for threshold in threshold_range:
        if threshold == 0.982:
            print("!")
        current_result, mask_pass_unseen_classes, threshold_score, balanced_accuracy_multiclass_classification, \
        balanced_precision_multiclass_classification, F1_score,F1_score_seen, F1_score_unseen = calc_accuracy_with_k_means_specific_threshold(
            args, test_labels,
            test_probs, test_preds,
            test_feats,
            threshold)
        for k, v in current_result.items():
            result_dict_all_optimal_threshold[k].append(v)
        mask_pass_unseen_classes_all.append(mask_pass_unseen_classes)
        threshold_score_all.append(threshold_score)
        balanced_accuracy_multiclass_classification_all.append(balanced_accuracy_multiclass_classification)
        balanced_precision_multiclass_classification_all.append(balanced_precision_multiclass_classification)
        F1_score_all.append(F1_score)
        F1_score_seen_all.append(F1_score)
        F1_score_unseen_all.append(F1_score)
    save_optimal_threshold_figure(args, threshold_range, result_dict_all_optimal_threshold, first_half)

    save_general_threshold_range_figure(args, threshold_range, mask_pass_unseen_classes_all,
                                        "mask pass unseen classes count")
    save_general_threshold_range_figure(args, threshold_range, threshold_score_all, "threshold score")
    save_general_threshold_range_figure(args, threshold_range, balanced_accuracy_multiclass_classification_all,
                                        "balanced accuracy")
    save_general_threshold_range_figure(args, threshold_range, balanced_precision_multiclass_classification_all,
                                        "balanced precision")
    save_general_threshold_range_figure(args, threshold_range, F1_score_all, "F1 score")
    save_general_threshold_range_figure(args, threshold_range, F1_score_seen_all, "F1 score seen")
    save_general_threshold_range_figure(args, threshold_range, F1_score_unseen_all, "F1 score unseen")
    if args.choose_unseen_count_by_test == 1:
        test_unseen_count = len([label for label in test_labels if label in args.missing_labels])
        print("test_unseen_count", test_unseen_count)
        print(get_index_same_unseen_count(args, np.array(mask_pass_unseen_classes_all),test_unseen_count))
        threshold_chosen_index = get_index_same_unseen_count(args, np.array(mask_pass_unseen_classes_all),test_unseen_count)[0]
    elif args.lt_ratio > 1:
        print("np.argmax(threshold_score_all)", np.argmax(threshold_score_all))
        print("threshold_range[np.argmax(threshold_score_all)]", threshold_range[np.argmax(threshold_score_all)])
        threshold_chosen_index = np.argmax(threshold_score_all)
    elif args.run_threshold_range_get_same_unseen_count == 0:
        threshold_chosen_index = np.argmax(
            result_dict_all_optimal_threshold['Combine score - accuracy of seen + clustering of unseen'])
    else:
        threshold_chosen_index = get_index_same_unseen_count(args, np.array(mask_pass_unseen_classes_all),args.my_algorithm_unseen_count)[0]
    threshold_chosen = threshold_range[threshold_chosen_index]
    print(
        f"threshold_chosen {threshold_chosen},run_threshold_range_get_same_unseen_count {args.run_threshold_range_get_same_unseen_count}")
    for k, v in result_dict_all_optimal_threshold.items():
        result_dict[k] = v[threshold_chosen_index]
    pprint.pprint(result_dict)
    return result_dict, threshold_chosen


def run_basic_algorithm(args, test_labels, test_probs, test_preds, test_feats):
    result_dict = {"Combine score - accuracy of seen + clustering of unseen": None,
                   "seen classes accuracy": None,
                   "unseen classes recall": None,
                   "unseen classes precision": None,
                   "balanced_accuracy_multiclass_classification": None,
                   "balanced_precision_multiclass_classification": None,
                   "F1_score": None, "F1_score_seen": None, "F1_score_unseen": None
                   }
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
    balanced_accuracy_multiclass_classification, balanced_accuracy_multiclass_classification_seen, \
    balanced_accuracy_multiclass_classification_unseen = \
        calc_balanced_accuracy_multiclass_classification(args, test_labels, test_preds)
    balanced_precision_multiclass_classification, balanced_precision_multiclass_classification_seen, \
    balanced_precision_multiclass_classification_unseen = \
        calc_balanced_precision_multiclass_classification(args, test_labels, test_preds)
    F1_score = calc_F1_score(balanced_accuracy_multiclass_classification, balanced_precision_multiclass_classification)
    F1_score_seen = calc_F1_score(balanced_accuracy_multiclass_classification_seen,
                                  balanced_precision_multiclass_classification_seen)
    F1_score_unseen = calc_F1_score(balanced_accuracy_multiclass_classification_unseen,
                                    balanced_precision_multiclass_classification_unseen)
    result_dict["Combine score - accuracy of seen + clustering of unseen"] = clustering_accuracy_with_permutation
    result_dict["seen classes accuracy"] = iteration_validation_accuracy_appearing_lables
    result_dict["unseen classes recall"] = iteration_validation_accuracy_recall
    result_dict["unseen classes precision"] = iteration_validation_accuracy_precision
    result_dict["balanced_accuracy_multiclass_classification"] = balanced_accuracy_multiclass_classification
    result_dict["balanced_precision_multiclass_classification"] = balanced_precision_multiclass_classification
    result_dict["F1_score"] = F1_score
    result_dict["F1_score_seen"] = F1_score_seen
    result_dict["F1_score_unseen"] = F1_score_unseen
    return result_dict
