TEMP_RESULT_OUTPUT_DIR_DICT_REMOTE_LOCAL = {
    1: f"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/eval_helper/result_output",
    -1: r"C:\Users\noamf\Documents\thesis\code\SSL_Benchmark\new_forked_semi_supervised_learning\Semi-supervised-learning\eval_helper\result_output"}

def test_val(args, test_labels, test_preds):
    print("\n\n\nstart first half")
    first_half_accuracy = main_calc_accuracy(args, test_labels[:5000], test_preds[:5000])[0]
    print("\n\n\nstart second half")
    second_half_accuracy = main_calc_accuracy(args, test_labels[5000:], test_preds[5000:])[0]
    print("first_half Combine score - accuracy of seen + clustering of unseen", first_half_accuracy)
    print("second_half Combine score - accuracy of seen + clustering of unseen", second_half_accuracy)
    print("diff between test half's", second_half_accuracy - first_half_accuracy)

    with open(f"/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/"
              f"new_forked_semi_supervised_learning/Semi-supervised-learning/"
              f"eval_helper/result_output/tmp_res_{get_slurm_job_id(args)}.out", "w") as f:
        f.write(f"first_half_accuracy {round(first_half_accuracy, 2)}\n")
        f.write(f"second_half_accuracy {round(second_half_accuracy, 2)}\n")
        f.write(f"diff {abs(round(second_half_accuracy - first_half_accuracy, 2))}")