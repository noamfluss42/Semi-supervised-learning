import numpy as np
import wandb


def log_labeled_dist_config_main(args, lb_count: np.array):
    wandb.config.update({'lb_count_max': lb_count.max()},allow_val_change=True)
    wandb.config.update({'lb_count_min': lb_count.min()},allow_val_change=True)
    wandb.config.update({'lb_count_mean': lb_count.mean()},allow_val_change=True)
    wandb.config.update({'lb_count_std': lb_count.std()},allow_val_change=True)
    wandb.config.update({'lb_count_unseen': np.count_nonzero(lb_count == 0)},allow_val_change=True)
    wandb.log({'lb_count/lb_count_max': lb_count.max()}, step=0)
    wandb.log({'lb_count/lb_count_min': lb_count.min()}, step=0)
    wandb.log({'lb_count/lb_count_mean': lb_count.mean()}, step=0)
    wandb.log({'lb_count/lb_count_std': lb_count.std()}, step=0)
    wandb.log({'lb_count/lb_count_unseen': np.count_nonzero(lb_count == 0)}, step=0)
