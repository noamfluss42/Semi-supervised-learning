import numpy as np
import wandb


def log_all_pseudo_label_after_mask(args, current_pseudo_label_after_mask, label, steps_per_epoch, epoch):
    wandb.log({f"{label}/mean": current_pseudo_label_after_mask.mean().item()},
              step=epoch)
    wandb.log({f"{label}/count": current_pseudo_label_after_mask.sum().item()},
              step=epoch)
    wandb.log({f"{label}/count_per_step": current_pseudo_label_after_mask.sum().item() / steps_per_epoch},
              step=epoch)
    wandb.log({f"{label}/std": current_pseudo_label_after_mask.std().item()},
              step=epoch)
    wandb.log({f"{label}/max": current_pseudo_label_after_mask.max().item()},
              step=epoch)
    wandb.log({f"{label}/min": current_pseudo_label_after_mask.min().item()},
              step=epoch)
    wandb.log({f"{label}/len": len(current_pseudo_label_after_mask)},
              step=epoch)


def log_flexmatch_pseudo_label(args, all_pseudo_label_after_mask, ulb_mimi_batch_size, steps_per_epoch, epoch):
    for c in range(args.num_classes):
        wandb.log(
            {f"Pseudo_label_after_mask_count/count_per_epoch({c})": all_pseudo_label_after_mask[c] / steps_per_epoch},
            step=epoch)
        wandb.log(
            {f"Pseudo_label_after_mask_percentage_out_of_all_logits/percentage_out_of_all_logits({c})": 100 *
                                                                                                        all_pseudo_label_after_mask[
                                                                                                            c] / (
                                                                                                                ulb_mimi_batch_size * steps_per_epoch)},
            step=epoch)
        wandb.log({
            f"Pseudo_label_after_mask_percentage_out_of_all_pseudo_labels/percentage_out_of_all_pseudo_labels({c})": 100 *
                                                                                                                     all_pseudo_label_after_mask[
                                                                                                                         c] / all_pseudo_label_after_mask.sum()},
            step=epoch)

    all_pseudo_label_after_mask_missing_labels = all_pseudo_label_after_mask[args.missing_labels]
    all_pseudo_label_after_mask_appearing_labels = all_pseudo_label_after_mask[
        [i for i in range(args.num_classes) if i not in args.missing_labels]]
    if len(all_pseudo_label_after_mask_missing_labels) > 0:
        log_all_pseudo_label_after_mask(args=args,
                                        current_pseudo_label_after_mask=all_pseudo_label_after_mask_missing_labels,
                                        label="pseudo_label_after_mask_missing_labels", steps_per_epoch=steps_per_epoch,
                                        epoch=epoch)
    log_all_pseudo_label_after_mask(args=args,
                                    current_pseudo_label_after_mask=all_pseudo_label_after_mask_appearing_labels,
                                    label="pseudo_label_after_mask_appearing_labels", steps_per_epoch=steps_per_epoch,
                                    epoch=epoch)
    log_all_pseudo_label_after_mask(args=args,
                                    current_pseudo_label_after_mask=all_pseudo_label_after_mask,
                                    label="pseudo_label_after_mask_all_labels", steps_per_epoch=steps_per_epoch,
                                    epoch=epoch)


def log_flexmatch_classwise_acc(args, classwise_acc, p_cutoff, steps_per_epoch, epoch):
    classwise_acc = classwise_acc.reshape(steps_per_epoch, args.num_classes)
    classwise_acc = classwise_acc.mean(axis=0)
    gama_t = p_cutoff * (classwise_acc / (2 - classwise_acc))
    wandb.log({f"Flexmatch_mask_parameters/beta_std": np.std(classwise_acc).item()}, step=epoch)
    wandb.log({f"Flexmatch_mask_parameters/beta_mean": np.mean(classwise_acc).item()}, step=epoch)
    wandb.log({f"Flexmatch_mask_parameters/gama_std": np.std(gama_t).item()}, step=epoch)
    wandb.log({f"Flexmatch_mask_parameters/gama_mean": np.mean(gama_t).item()}, step=epoch)
    for c in range(args.num_classes):
        wandb.log({f"Flexmatch_mask_parameters_beta/beta_t({c})": classwise_acc[c]}, step=epoch)
        wandb.log({f"Flexmatch_mask_parameters_gama/gama_t({c})": p_cutoff * (classwise_acc[c] / (2 - classwise_acc[c]))},
                  step=epoch)


def log_flexmatch_mask(args, pass_mask, steps_per_epoch, epoch):
    wandb.log({"Flexmatch_pass_mask/pass mask percentage per epoch": 100*pass_mask.mean().item()}, step=epoch)
    wandb.log({"Flexmatch_pass_mask/pass mask count per epoch": pass_mask.sum().item()}, step=epoch)
    wandb.log({"Flexmatch_pass_mask/pass mask count per step": pass_mask.sum().item() / steps_per_epoch}, step=epoch)
    wandb.log({"Flexmatch_pass_mask/pass mask len per epoch": len(pass_mask)}, step=epoch)


def log_flexmatch_max_prob_before_mask(args, max_prob_values, steps_per_epoch, epoch):

    wandb.log({"Flexmatch_max_prob_before_mask/max_prob_values mean": 100*max_prob_values.mean().item()},
              step=epoch)
    wandb.log({"Flexmatch_max_prob_before_mask/max_prob_values std": max_prob_values.std().item()}, step=epoch)


def log_flexmatch_max_prob_after_mask(args, max_prob_values, steps_per_epoch, epoch):
    if max_prob_values.std().item() == 0:
        print("max_prob_values std is 0","max_prob_values[:20]", max_prob_values[:20])
    wandb.log({"Flexmatch_max_prob_after_mask/max_prob_values mean": 100*max_prob_values.mean().item()}, step=epoch)
    wandb.log({"Flexmatch_max_prob_after_mask/max_prob_values std": max_prob_values.std().item()}, step=epoch)


def log_flexmatch_confidence(args, pass_mask, max_probs_values, classwise_acc, p_cutoff, steps_per_epoch, epoch):
    log_flexmatch_mask(args, pass_mask, steps_per_epoch, epoch)
    log_flexmatch_max_prob_before_mask(args, max_probs_values, steps_per_epoch, epoch)
    log_flexmatch_max_prob_after_mask(args, max_probs_values[pass_mask], steps_per_epoch, epoch)
    log_flexmatch_classwise_acc(args, classwise_acc, p_cutoff, steps_per_epoch, epoch)


def log_flexmatch(args, pass_mask, max_probs_values, classwise_acc, p_cutoff, all_pseudo_label_after_mask, epoch):
    """

    Args:
        args: dict
        pass_mask:
        max_probs_values: for every logit - the max prob of being some class, even below threshold
        classwise_acc: the acc of every class (sigmat(c)), for every iter -> len = #number of classes * #iter_per_epoch, beta value in flexmatch alg
        p_cutoff:
        all_pseudo_label_after_mask: len = #number of classes, the number of pseudo labels per class for all iter of epoch
        epoch: int, step number
    """

    pass_mask = pass_mask.astype(int)
    ulb_mimi_batch_size = int(args.batch_size * args.uratio)
    steps_per_epoch = int(args.num_train_iter / args.epoch)
    print("args.num_train_iter / args.epoch",args.num_train_iter / args.epoch)
    log_flexmatch_confidence(args, pass_mask, max_probs_values, classwise_acc,
                             p_cutoff, steps_per_epoch, epoch)
    log_flexmatch_pseudo_label(args, all_pseudo_label_after_mask, ulb_mimi_batch_size, steps_per_epoch, epoch)
