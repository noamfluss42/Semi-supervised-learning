# from eval_helper.part_of_eval.utils import get_slurm_job_id, get_out_path_by_slurm_job_id, get_parameter_from_out_path, \
#     get_unseen_count_by_slurm_job_id
from .utils import get_slurm_job_id
def save_in_file_threshold_openmatch_measurments(args, overall_acc, unk_acc, closed_acc, roc, roc_soft, rejection_acc,
                                                 name="openmatch_measurments"):
    with open(f"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
              f"new_forked_semi_supervised_learning/Semi-supervised-learning/"
              f"eval_helper/result_output/tmp_res_{name}_{get_slurm_job_id(args)}.out", "w") as f:
        f.write(
            f"overall_acc "
            f"{round(overall_acc, 2)}\n")
        f.write(
            f"unk_acc "
            f"{round(unk_acc, 2)}\n")
        f.write(
            f"closed_acc "
            f"{round(closed_acc, 2)}\n")
        f.write(
            f"roc "
            f"{round(roc, 2)}\n")
        f.write(
            f"roc_soft "
            f"{round(roc_soft, 2)}\n")
        f.write(
            f"rejection_acc "
            f"{round(rejection_acc, 2)}")
    print("overall_acc", overall_acc)
    print("unk_acc", unk_acc)
    print("closed_acc", closed_acc)
    print("roc", roc)
    print("roc_soft", roc_soft)
    print("rejection_acc", rejection_acc)
