import torch
import torch.nn.functional as F
from log_wandb import log_loss


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
    if args.project_wandb == "test":
        print("all_logits.shape", all_logits.shape)

        print("all_pseudo_label.shape",all_pseudo_label.shape)
        print("all_avg_prob.shape",all_avg_prob.shape)

        print("v1",main_get_entropy_with_labeled_data_loss(args, logits_u_w, logits_x_lb))
    EPS = 1e-8
    all_avg_prob_clamp = torch.clamp(all_avg_prob, min=EPS)
    b = all_avg_prob_clamp * torch.log(all_avg_prob_clamp)
    if len(b.size()) == 2: # Sample-wise entropy
        if args.project_wandb == "test":
            print("Sample-wise entropy")
        return - b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        if args.project_wandb == "test":
            print("Distribution-wise entropy")
            print("b.shape",b.shape)
            print("v2",- b.sum())
        return - b.sum()
