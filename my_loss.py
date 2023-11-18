import numpy as np
import torch
import torch.nn.functional as F
from log_wandb import log_loss
import os
# PATH_TRAIN_RELATIVE = "/cs/labs/daphna/noam.fluss/project/SSL_Benchmark/new_forked_semi_supervised_learning/Semi-supervised-learning/semilearn/datasets"
PATH_TRAIN_RELATIVE = r"C:\Users\noamf\Documents\thesis\code\SSL_Benchmark\new_forked_semi_supervised_learning\Semi-supervised-learning"
TRAIN_RELATIVE_10 = np.loadtxt(os.path.join(PATH_TRAIN_RELATIVE,"train_relative_10.csv"), delimiter=',')
TRAIN_RELATIVE_50 = np.loadtxt(os.path.join(PATH_TRAIN_RELATIVE,"train_relative_50.csv"), delimiter=',')
TRAIN_RELATIVE_100 = np.loadtxt(os.path.join(PATH_TRAIN_RELATIVE,"train_relative_100.csv"), delimiter=',')
TRAIN_RELATIVE_DICT = {10: TRAIN_RELATIVE_10, 50: TRAIN_RELATIVE_50, 100: TRAIN_RELATIVE_100}


def get_entropy_loss(args, logits_u_w):
    pseudo_label = torch.softmax(logits_u_w / args.T, dim=-1)
    avg_prob = torch.mean(pseudo_label, dim=0)
    return torch.sum(avg_prob * torch.log(avg_prob))


def get_datapoint_entropy_loss(args, logits_u_w):
    x = logits_u_w / args.T
    b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
    return -b.sum(dim=1).mean()


def main_get_extra_loss(args, logits_u_w):
    entropy_loss = get_entropy_loss(args, logits_u_w)
    datapoint_entropy_loss = get_datapoint_entropy_loss(args, logits_u_w)
    return entropy_loss, datapoint_entropy_loss


def main_get_entropy_with_labeled_data_loss(args, logits_u_w, logits_x_lb):
    pseudo_label_u_w = torch.softmax(logits_u_w / args.T, dim=-1)
    avg_prob_u_w = torch.mean(pseudo_label_u_w, dim=0)
    pseudo_label_x_lb = torch.softmax(logits_x_lb / args.T, dim=-1)
    avg_prob_x_lb = torch.mean(pseudo_label_x_lb, dim=0)

    avg_prob_total = avg_prob_u_w + avg_prob_x_lb

    return torch.sum(avg_prob_total * torch.log(avg_prob_total))


def main_get_entropy_with_labeled_data_loss_v2(args, logits_u_w, logits_x_lb):
    all_logits = torch.cat([logits_u_w, logits_x_lb], 0)
    all_pseudo_label = torch.softmax(all_logits / args.T, dim=1)
    all_avg_prob = torch.mean(all_pseudo_label, dim=0)
    # if args.project_wandb == "test":
    #     print("all_logits.shape", all_logits.shape)
    #
    #     print("all_pseudo_label.shape",all_pseudo_label.shape)
    #     print("all_avg_prob.shape",all_avg_prob.shape)
    #
    #     print("v1",main_get_entropy_with_labeled_data_loss(args, logits_u_w, logits_x_lb))
    EPS = 1e-8
    all_avg_prob_clamp = torch.clamp(all_avg_prob, min=EPS)
    b = all_avg_prob_clamp * torch.log(all_avg_prob_clamp)
    if len(b.size()) == 2:  # Sample-wise entropy
        if args.project_wandb == "test":
            print("Sample-wise entropy")
        return - b.sum(dim=1).mean()
    elif len(b.size()) == 1:  # Distribution-wise entropy
        # if args.project_wandb == "test":
        #     print("Distribution-wise entropy")
        #     print("b.shape",b.shape)
        #     print("v2",- b.sum())
        return - b.sum()


def get_kl_divergence_loss(args, logits_u_w):
    # calc KL divergence: D_KL(P || Q)
    # P = the model current output
    # Q = the target distribution = long tail distribution

    pseudo_label = torch.softmax(logits_u_w / args.T, dim=-1)
    avg_prob = torch.mean(pseudo_label, dim=0)  # P

    train_relative = torch.tensor(TRAIN_RELATIVE_DICT[args.lt_ratio]).cuda() # Q

    return torch.sum(avg_prob * torch.log(avg_prob / train_relative))
