import numpy as np
import torch
import wandb
from matplotlib import pyplot as plt
import PIL
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import shutil
import os


def print_cm(y_true, y_pred):
    cf_mat = confusion_matrix(y_true, y_pred, normalize='true')
    cf_mat_not_norm = confusion_matrix(y_true, y_pred)
    print('confusion matrix - normalize:\n' + np.array_str(cf_mat))
    print('confusion matrix - not normalize:\n' + np.array_str(cf_mat_not_norm))


def wandb_log_cm_last_instance(y_true, preds, class_names, epoch):
    print("start wandb_log_cm_last_instance - new")
    wandb.log({"confusion_matrix/confusion_matrix_instance_new": wandb.plot.confusion_matrix(probs=None,
                                                                                             y_true=y_true, preds=preds,
                                                                                             class_names=class_names
                                                                                             )}, step=epoch)
    print("end wandb_log_cm_last_instance - new")


def get_title(args, epoch):
    title = "epoch: " + str(epoch) + "  --lambda entropy: " + str(args.lambda_entropy)

    if args.lb_imb_ratio == 1 and len(args.missing_labels) > 0:
        title += "  --missing labels: " + ",".join([str(i) for i in args.missing_labels])
    elif args.lb_imb_ratio == 1 and len(args.missing_labels) == 0:
        title += "  --missing labels: None"
    else:
        title += "  --lb_imb_ratio: " + str(args.lb_imb_ratio)
    title += "  --seed: " + str(args.seed)
    title += " --threshold:  " + str(args.threshold)
    title += " --lambda_datapoint_entropy:  " + str(args.lambda_datapoint_entropy)
    title += " --algorithm:  " + str(args.algorithm)
    print("title", title)
    return title


def wandb_log_cm_img(args, cm, epoch):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[str(i) for i in range(args.num_classes)])
    dir_path = f'/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/debug_img/{str(args.slurm_job_id)}'
    os.mkdir(dir_path)
    current_img_path = f'{dir_path}/current_img_{str(epoch)}_.png'
    disp.plot().figure_.savefig(current_img_path)
    im = PIL.Image.open(current_img_path)
    title = get_title(args, epoch)
    wandb_im = wandb.Image(im, caption=title)
    wandb.log({"confusion_matrix/confusion_matrix_image": wandb_im}, step=epoch)
    plt.close()
    shutil.rmtree(dir_path)


