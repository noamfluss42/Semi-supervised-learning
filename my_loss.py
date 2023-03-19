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
