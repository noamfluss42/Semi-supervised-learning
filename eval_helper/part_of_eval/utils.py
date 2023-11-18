import numpy as np
import os

EVAL_PATH = fr"/cs/labs/daphna/noam.fluss/usb_new_venv/bin/python /cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval.py"
path_out_tmp_file = "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper/result_output/tmp_res_"


def get_slurm_job_id(args):
    slurm_job_id = args.load_path[:args.load_path.rfind(os.sep)]
    slurm_job_id = slurm_job_id[slurm_job_id.rfind(os.sep) + 1:]
    return slurm_job_id


def get_out_path_by_slurm_job_id_remote(slurm_job_id, searched_path, in_name=False, out_file=False):
    print(f"out_file {out_file}, in_name {in_name}")
    for path, subdirs, files in os.walk(searched_path):
        for name in files:
            if out_file and slurm_job_id in name and name.endswith(".out"):
                return os.path.join(path, name)
            elif in_name and slurm_job_id in name and not name.endswith(".sh"):
                return os.path.join(path, name)
            elif not in_name and slurm_job_id in path and name.endswith("latest_model.pth"):
                return os.path.join(path, name)


def get_out_path_by_slurm_job_id_local(slurm_job_id,searched_path):
    # Walk through the directory structure
    for root, dirs, files in os.walk(searched_path):
        # Check if the slurm_job_id directory is in the current directory
        if str(slurm_job_id) in dirs:
            base_path = os.path.join(root, str(slurm_job_id))

            # Check if the desired file is in the slurm_job_id directory
            if "latest_model.pth" in os.listdir(base_path):
                # Return the full path to the file
                return os.path.join(base_path, "latest_model.pth")
    # If the file is not found
    print(f"The file 'latest_model.pth' was not found in the specified directory structure for slurm_job_id {slurm_job_id}.")
    return None

def get_missing_labels_by_path(args):
    slurm_job_id = get_slurm_job_id(args)
    if slurm_job_id == "16654751":
        return [0, 2, 6, 9]

    out_path = get_out_path_by_slurm_job_id_remote(slurm_job_id, "/cs/labs/daphna/noam.fluss/project/"
                                                          "SSL_Benchmark/new_forked_semi_supervised_learning/"
                                                          "Semi-supervised-learning/my_scripts/run_hyper_parameter_tuning_v2",
                                                   in_name=True, out_file=True)
    print("out_path", out_path)
    with open(out_path, "r") as f:
        lines = f.readlines()

    lb_count_line = [line for line in lines if "lb count:" in line]
    if len(lb_count_line) > 0:
        lb_count_line = lb_count_line[0]
        lb_count_str = lb_count_line[11:-2].split(", ")
        lb_count = np.array([int(i) for i in lb_count_str])
        return np.where(lb_count == 0)[0]
    else:
        title_line = [line for line in lines if "title epoch:" in line][0]
        missing_labels_from_title = title_line[
                                    title_line.find("--missing labels:") + len("--missing labels:"):title_line.find(
                                        "  --seed:")].split(",")
        missing_labels_from_title = [int(i) for i in missing_labels_from_title]
        print("missing_labels_from_title", missing_labels_from_title)
        return missing_labels_from_title



def get_parameter_by_path(args,parameter):
    slurm_job_id = get_slurm_job_id(args)

    out_path = get_out_path_by_slurm_job_id_remote(slurm_job_id, "/cs/labs/daphna/noam.fluss/project/"
                                                          "SSL_Benchmark/new_forked_semi_supervised_learning/"
                                                          "Semi-supervised-learning/my_scripts/run_hyper_parameter_tuning_v2",
                                                   in_name=True, out_file=True)
    print("out_path", out_path)
    with open(out_path, "r") as f:
        lines = f.readlines()

    lb_count_line = [line for line in lines if f"{parameter}=" in line]
    if len(lb_count_line) > 0:
        lb_count_line = lb_count_line[0]
        start = lb_count_line.find(f"{parameter}=") + len(f"{parameter}=")
        end = lb_count_line.find(",", start)
        print(f"{parameter} - lb_count_line[start:end]", lb_count_line[start:end])
        return float(lb_count_line[start:end])
    else:
        exit(f"{parameter} not found")


def get_value_of_measure_by_out_files_lines(lines, measure):
    searched_line = [line for line in lines if measure in line][-1]
    return float(searched_line[searched_line.rfind(" ") + 1:])


# get_unseen_count_by_path
def get_unseen_count_by_slurm_job_id(slurm_job_id, unseen_classes_data_in_test):
    print("start get_unseen_count_by_slurm_job_id")
    print("slurm_job_id", slurm_job_id)
    out_path = get_out_path_by_slurm_job_id_remote(slurm_job_id, "/cs/labs/daphna/noam.fluss/project/"
                                                          "SSL_Benchmark/new_forked_semi_supervised_learning/"
                                                          "Semi-supervised-learning/my_scripts/run_hyper_parameter_tuning_v2",
                                                   in_name=True, out_file=True)
    with open(out_path, "r") as f:
        lines = f.readlines()

    unseen_classes_recall = get_value_of_measure_by_out_files_lines(lines,
                                                                    "iteration validation accuracy missing labels recall")
    unseen_classes_precision = get_value_of_measure_by_out_files_lines(lines,
                                                                       "iteration validation accuracy missing labels precision")
    print("calc unseeen count - alt:", unseen_classes_data_in_test * unseen_classes_recall / unseen_classes_precision)
    print("calc unseeen count - original:", get_value_of_measure_by_out_files_lines(lines,
                                                                                    "confusion_matrix_summary/iteration_missing_labels_tp_plus_fp"))
    return int(unseen_classes_data_in_test * unseen_classes_recall / unseen_classes_precision)


def get_parameter_from_out_path(path, parameter):
    current_line = ""
    with open(path, "r") as f:
        while not (current_line.startswith(f"{parameter}") and "?" not in current_line):
            current_line = f.readline()
    return current_line[len(f"{parameter}"):-1]


if __name__ == '__main__':
    get_unseen_count_by_slurm_job_id("15573937", 4000)
