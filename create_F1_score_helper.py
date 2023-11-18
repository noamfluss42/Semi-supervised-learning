import os
import time

import pandas as pd

lt_ratio_num_labels_flexmatch_dict = {
    (10, 400): [17160274, 17171962, 17171964, 17171966, 17171968],
    # (10, 700): [17182999, 17183001, 17183003, 17183005, 17183007],
    # (100, 400): [17160278, 17171965, 17171963, 17171967, 17171969],
    # (100, 1500): [17182988, 17182990, 17182992, 17182994, 17182996]
}
# original step of
slurm_job_id_to_stop_epoch = {
    (10, 400): {
        # 17177970: 404,
        # 17177969: 425,
        # 17177967: 467,
        # 17177968: 427,
        # 17160396: 450
        17166716, 352,
        17166712, 348,
        17166715, 367,
        17166714, 389,
        17160265, 320
    },
    (10, 700): {
        17188014: 449,
        17188013: 320,
        17188012: 449,
        17188011: 423,
        17182998: 437,
    },
    (100, 400): {
        17171959: 476,
        17171958: 412,
        17171957: 79,
        17171956: 432,
        17166713: 342,
    },
    (100, 1500): {
        17187998: 430,
        17187999: 461,
        17187997: 453,
        17188000: 456,
        17182987: 464
    }
}
lt_ratio_num_labels_our_algorithm_dict_specific_step = {
    (10, 400): [17160265, 17166712, 17166714, 17166715, 17166716],
    # (10, 700): [17182998, 17188011, 17188012, 17188013, 17188014],
    (100, 400): [17166713, 17171956, 17171957, 17171958, 17171959],
    # (100, 1500): [17182987, 17187997, 17187998, 17187999, 17188000]
}

lt_ratio_num_labels_our_algorithm_dict = {
    (10, 400): [17160265, 17166712, 17166714, 17166715, 17166716],
    (10, 700): [17195783, 17195784, 17195785, 17195786, 17195787],
    # (10, 700): old - [17182998, 17188011, 17188012, 17188013, 17188014],
    (100, 400): [17166713, 17171956, 17171957, 17171958, 17171959],
    (100, 1500): [17195789, 17195790, 17195791, 17195792, 17195793]
    # (100, 1500): old - [17187999, 17188000]  # 17187997,17187998 - corrupted,17182987 - is finished
}

lt_ratio_num_labels_our_algorithm_dict_not_finished_run = {
    (10, 700): [17195783, 17195784, 17195785, 17195786, 17195787],

    (100, 1500): [17195789, 17195790, 17195791, 17195792, 17195793]
}
lt_ratio_num_labels_our_algorithm_dict_50_1100 = {
    (50, 1100): [17195850, 17195852, 17195854, 17195856, 17195858],
}

lt_ratio_num_labels_flexmatch_dict_50_1100 = {
    (50, 1100): [17195849, 17195851, 17195853, 17195855, 17195857],
}



REMOTE = 1
PYTHON_PATH_LOCAL = r"C:\Users\noamf\anaconda3\envs\usb_v1\python.exe"
PYTHON_PATH_REMOTE = "/cs/labs/daphna/noam.fluss/usb_new_venv/bin/python"
PYTHON_PATH_DICT = {-1: PYTHON_PATH_LOCAL, 1: PYTHON_PATH_REMOTE}
EVAL_PATH_LOCAL = r"C:\Users\noamf\Documents\thesis\code\SSL_Benchmark\new_forked_semi_supervised_learning\Semi-supervised-learning\eval.py"
EVAL_PATH_REMOTE = r"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval.py"
EVAL_PATH_DICT = {-1: EVAL_PATH_LOCAL, 1: EVAL_PATH_REMOTE}
PARAMETERS = "--dataset cifar100 --num_classes 100 --load_path {} --num_workers 0 --run_threshold_range 1 --remote {} --lt_ratio {} --random_missing_labels_num 40 --use_specific_step {} --use_specific_step_change {}"
TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL = {
    1: f"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper/result_output",
    -1: r"C:\Users\noamf\Documents\thesis\code\SSL_Benchmark\new_forked_semi_supervised_learning\Semi-supervised-learning\eval_helper\result_output"}
COLUMNS = ["accuracy/iteration validation clustering accuracy with missing labels",
           "accuracy/iteration validation accuracy appearing lables",
           "accuracy/iteration validation accuracy missing labels recall",
           "accuracy/iteration validation accuracy missing labels precision",
           "accuracy/iteration validation balanced accuracy",
           "accuracy/iteration validation balanced precision",
           "accuracy/iteration validation F1 score", "accuracy/iteration validation F1_score_seen",
           "accuracy/iteration validation F1_score_unseen"]


def get_eval_results(slurm_job_id):
    path_out_tmp_file = os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[REMOTE],
                                     f"tmp_res_optimal_threshold_{slurm_job_id}.out")
    result = []
    with open(path_out_tmp_file, "r") as f:
        lines = f.readlines()
        for current_line in lines:
            value = float(current_line[current_line.rfind(" "):])
            print(current_line[:current_line.rfind(" ")], value)
            result.append(value)
    # with open(path_out_tmp_file, "r") as f:
    #     clustering_accuracy_with_permutation = float(f.readline()[56:])
    #     seen_classes_accuracy = float(f.readline()[22:])
    #     unseen_classes_recall = float(f.readline()[22:])
    #     unseen_classes_precision = float(f.readline()[25:])
    #     # chosen_threshold = float(f.readline()[17:])
    #     balanced_accuracy = float(f.readline()[18:])
    #     balanced_precision = float(f.readline()[len(f"balanced precision "):])
    #     f1_score = float(f.readline()[len(f"F1 score "):])
    #     f1_score_seen = float(f.readline()[len(f"F1 score seen "):])
    #     f1_score_unseen = float(f.readline()[len(f"F1 score unseen "):])
    print("result", result)
    os.remove(path_out_tmp_file)
    return result
    # return [clustering_accuracy_with_permutation, seen_classes_accuracy, unseen_classes_recall,
    #         unseen_classes_precision, balanced_accuracy, balanced_precision, f1_score, f1_score_seen, f1_score_unseen]


def save_in_csv_file(result_dict, lt_ratio, num_labels, our_algorithm, use_specific_step,
                     use_specific_step_change, use_best, choose_unseen_count_by_test):
    algorithm_name = "ours" if our_algorithm else "flexmatch"

    if our_algorithm and use_specific_step == 1:
        path_csv_file = os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[REMOTE],
                                     f"result_output_{algorithm_name}_{lt_ratio}_{num_labels}_"
                                     f"specific_step_with_change_{use_specific_step_change}.csv")
    elif our_algorithm and use_best == 1:
        path_csv_file = os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[REMOTE],
                                     f"result_output_{algorithm_name}_{lt_ratio}_{num_labels}_best.csv")
    elif our_algorithm:
        path_csv_file = os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[REMOTE],
                                     f"result_output_{algorithm_name}_{lt_ratio}_{num_labels}_last.csv")
    elif not our_algorithm and choose_unseen_count_by_test == 1:
        path_csv_file = os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[REMOTE],
                                     f"result_output_{algorithm_name}_{lt_ratio}_{num_labels}_choose_unseen_count_by_test.csv")
    else:
        path_csv_file = os.path.join(TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL[REMOTE],
                                     f"result_output_{algorithm_name}_{lt_ratio}_{num_labels}.csv")
    # create data frame
    result_df = pd.DataFrame(columns=COLUMNS)
    for slurm_job_id in result_dict.keys():
        current_dict = {}
        for i, column in enumerate(COLUMNS):
            current_dict[column] = result_dict[slurm_job_id][i]
        result_df = result_df.append(current_dict, ignore_index=True)
    result_df.to_csv(path_csv_file, index=False)
    create_mean_std_csv_file(result_df, path_csv_file, our_algorithm=our_algorithm)


def create_mean_std_csv_file(result_df, path_csv_file, our_algorithm=False):
    new_df = pd.DataFrame()
    path_csv_file = path_csv_file.replace("result_output_", "result_mean_std_error_output_")
    # Iterate over columns in the original DataFrame
    for col in result_df.columns:
        # Calculate mean and standard error for each column
        mean_val = result_df[col].mean()
        std_err_val = result_df[col].sem()  # sem() calculates standard error of the mean

        # Create new columns in the new DataFrame
        new_df[f"{col} - mean"] = [mean_val]
        new_df[f"{col} - standard error"] = [std_err_val]
    new_df.to_csv(path_csv_file, index=False)


def create_row_from_values(lt_ratio, num_labels, current_slurm_job_id_dict, our_algorithm=False, use_specific_step=-1,
                           use_specific_step_change=0, use_best=-1, choose_unseen_count_by_test=-1):
    result_dict = {}

    for slurm_job_id in current_slurm_job_id_dict[(lt_ratio, num_labels)]:
        command_run_eval = f"{PYTHON_PATH_DICT[REMOTE]} {EVAL_PATH_DICT[REMOTE]} " \
                           f"{PARAMETERS.format(slurm_job_id, REMOTE, lt_ratio, use_specific_step, use_specific_step_change)}"
        if our_algorithm and use_specific_step == -1 and use_best == 1:
            command_run_eval += " --use_best 1"
        if not our_algorithm and choose_unseen_count_by_test == 1:
            command_run_eval += " --choose_unseen_count_by_test 1"
        os.system(command_run_eval)
        time.sleep(2)  # wait for the eval to finish
        print("end run command")
        result_dict[slurm_job_id] = get_eval_results(slurm_job_id)
    print("!")
    save_in_csv_file(result_dict, lt_ratio, num_labels, our_algorithm=our_algorithm,
                     use_specific_step=use_specific_step, use_specific_step_change=use_specific_step_change,
                     use_best=use_best, choose_unseen_count_by_test=choose_unseen_count_by_test)


def main():
    current_slurm_job_id_dict = lt_ratio_num_labels_our_algorithm_dict
    # specific_step_change_options = [0, -20, -40, -60, -80]
    specific_step_change_options = [-150]
    use_specific_step = 1
    choose_unseen_count_by_test = -1
    our_algorithm = True
    for use_best in [-1]:
        for use_specific_step_change in specific_step_change_options:
            for lt_ratio, num_labels in current_slurm_job_id_dict.keys():
                create_row_from_values(lt_ratio, num_labels, current_slurm_job_id_dict, our_algorithm=our_algorithm,
                                       use_specific_step=use_specific_step,
                                       use_specific_step_change=use_specific_step_change, use_best=use_best,
                                       choose_unseen_count_by_test=choose_unseen_count_by_test)


if __name__ == "__main__":
    main()
    print("end eval")
