import os
import time

lt_ratio_num_labels_our_algorithm_dict = {
    (10, 400): [17160265, 17166712, 17166714, 17166715, 17166716],
    (10, 700): [17182998, 17188011, 17188012, 17188013, 17188014],
    (100, 400): [17166713, 17171956, 17171957, 17171958, 17171959],
    (100, 1500): [17182987, 17187997, 17187998, 17187999, 17188000]
}
lt_ratio_num_labels_dict = {
    (10, 400): [17160274, 17171962, 17171964, 17171966, 17171968],
    (10, 700): [17182999, 17183001, 17183003, 17183005, 17183007],
    (100, 400): [17160278, 17171965, 17171963, 17171967, 17171969],
    (100, 1500): [17182988, 17182990, 17182992, 17182994, 17182996]
}
lt_ratio_num_labels_our_algorithm_dict_new = {
    (10, 700): [17195783, 17195784, 17195785, 17195786, 17195787],
    # (10, 700): old - [17182998, 17188011, 17188012, 17188013, 17188014],

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
EVAL_PATH = r"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval.py"
PYTHON_PATH = "/cs/labs/daphna/noam.fluss/usb_new_venv/bin/python"
PARAMETERS = "--dataset cifar100 --num_classes 100 --load_path {} --num_workers 0 --lt_ratio {} --random_missing_labels_num 40 --run_threshold_range 1"


def main():
    for lt_ratio, num_labels in lt_ratio_num_labels_flexmatch_dict_50_1100:
        for slurm_job_id in lt_ratio_num_labels_flexmatch_dict_50_1100[(lt_ratio, num_labels)]:
            command_run_eval = f"{PYTHON_PATH} {EVAL_PATH} {PARAMETERS.format(slurm_job_id, lt_ratio)}"
            command_run_eval += " --choose_unseen_count_by_test 1"
            os.system(command_run_eval)
            time.sleep(1)
            # command_run_eval = f"{PYTHON_PATH} {EVAL_PATH} --use_specific_step 1 {PARAMETERS.format(slurm_job_id, lt_ratio)}"
            #
            # os.system(command_run_eval)
            # command_run_eval = f"{PYTHON_PATH} {EVAL_PATH} --use_best 1 {PARAMETERS.format(slurm_job_id, lt_ratio)}"
            #
            # os.system(command_run_eval)
            #
            # time.sleep(1)



if __name__ == "__main__":
    main()
    print("end eval")
