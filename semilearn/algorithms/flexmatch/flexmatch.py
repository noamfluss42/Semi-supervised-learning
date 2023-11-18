# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch

from .utils import FlexMatchThresholdingHook

from semilearn.core import AlgorithmBase
from semilearn.core.utils import ALGORITHMS
from semilearn.algorithms.hooks import PseudoLabelingHook
from semilearn.algorithms.utils import SSL_Argument, str2bool
import sys

sys.path.append("....")
from my_loss import main_get_extra_loss, log_loss, main_get_entropy_with_labeled_data_loss, \
    main_get_entropy_with_labeled_data_loss_v2, get_kl_divergence_loss


@ALGORITHMS.register('flexmatch')
class FlexMatch(AlgorithmBase):
    """
        FlexMatch algorithm (https://arxiv.org/abs/2110.08263).

        Args:
            - args (`argparse`):
                algorithm arguments
            - net_builder (`callable`):
                network loading function
            - tb_log (`TBLog`):
                tensorboard logger
            - logger (`logging.Logger`):
                logger to use
            - T (`float`):
                Temperature for pseudo-label sharpening
            - p_cutoff(`float`):
                Confidence threshold for generating pseudo-labels
            - hard_label (`bool`, *optional*, default to `False`):
                If True, targets have [Batch size] shape with int values. If False, the target is vector
            - ulb_dest_len (`int`):
                Length of unlabeled data
            - thresh_warmup (`bool`, *optional*, default to `True`):
                If True, warmup the confidence threshold, so that at the beginning of the training, all estimated
                learning effects gradually rise from 0 until the number of unused unlabeled data is no longer
                predominant

        """

    def __init__(self, args, net_builder, tb_log=None, logger=None):
        super().__init__(args, net_builder, tb_log, logger)
        # flexmatch specified arguments
        self.init(T=args.T, p_cutoff=args.p_cutoff, hard_label=args.hard_label, thresh_warmup=args.thresh_warmup)

    def init(self, T, p_cutoff, hard_label=True, thresh_warmup=True):
        self.T = T
        self.p_cutoff = p_cutoff
        self.use_hard_label = hard_label
        self.thresh_warmup = thresh_warmup

    def set_hooks(self):
        self.register_hook(PseudoLabelingHook(), "PseudoLabelingHook")
        self.register_hook(FlexMatchThresholdingHook(ulb_dest_len=self.args.ulb_dest_len, num_classes=self.num_classes,
                                                     thresh_warmup=self.args.thresh_warmup), "MaskingHook")

        super().set_hooks()

    def get_new_lambda_after_collapse_epoch(self):
        lambda_entropy_to_add = 0
        lambda_datapoint_entropy_to_add = 0
        if self.args.lambda_entropy_decay_value != -1 or self.args.lambda_entropy_decay_rate != -1:
            lambda_entropy_to_add = self.args.lambda_entropy
            lambda_datapoint_entropy_to_add = self.args.lambda_datapoint_entropy
            if self.args.lambda_entropy_decay_value != -1:
                lambda_entropy_to_add -= self.args.lambda_entropy_decay_value
                lambda_datapoint_entropy_to_add -= self.args.lambda_entropy_decay_value
            if self.args.lambda_entropy_decay_rate != -1:
                lambda_entropy_to_add -= (self.epoch - self.args.lambda_entropy_stop_epoch) * \
                                         self.args.lambda_entropy_decay_rate
                lambda_datapoint_entropy_to_add -= (self.epoch - self.args.lambda_entropy_stop_epoch) * \
                                                   self.args.lambda_entropy_decay_rate
            lambda_entropy_to_add = max(lambda_entropy_to_add, 0)
            lambda_datapoint_entropy_to_add = max(lambda_datapoint_entropy_to_add, 0)
        return lambda_entropy_to_add, lambda_datapoint_entropy_to_add

    def train_step(self, x_lb, y_lb, idx_ulb, x_ulb_w, x_ulb_s):
        num_lb = y_lb.shape[0]
        super().set_p_cutoff(self.p_cutoff)
        # inference and calculate sup/unsup losses
        with self.amp_cm():
            if self.use_cat:
                inputs = torch.cat((x_lb, x_ulb_w, x_ulb_s))
                outputs = self.model(inputs)
                logits_x_lb = outputs['logits'][:num_lb]
                logits_x_ulb_w, logits_x_ulb_s = outputs['logits'][num_lb:].chunk(2)
                feats_x_lb = outputs['feat'][:num_lb]
                feats_x_ulb_w, feats_x_ulb_s = outputs['feat'][num_lb:].chunk(2)
            else:
                outs_x_lb = self.model(x_lb)
                logits_x_lb = outs_x_lb['logits']
                feats_x_lb = outs_x_lb['feat']
                outs_x_ulb_s = self.model(x_ulb_s)
                logits_x_ulb_s = outs_x_ulb_s['logits']
                feats_x_ulb_s = outs_x_ulb_s['feat']
                with torch.no_grad():
                    outs_x_ulb_w = self.model(x_ulb_w)
                    logits_x_ulb_w = outs_x_ulb_w['logits']
                    feats_x_ulb_w = outs_x_ulb_w['feat']
            feat_dict = {'x_lb': feats_x_lb, 'x_ulb_w': feats_x_ulb_w, 'x_ulb_s': feats_x_ulb_s}

            sup_loss = self.ce_loss(logits_x_lb, y_lb, reduction='mean')

            # probs_x_ulb_w = torch.softmax(logits_x_ulb_w, dim=-1)
            probs_x_ulb_w = self.compute_prob(logits_x_ulb_w.detach())

            # if distribution alignment hook is registered, call it 
            # this is implemented for imbalanced algorithm - CReST
            if self.registered_hook("DistAlignHook"):
                probs_x_ulb_w = self.call_hook("dist_align", "DistAlignHook", probs_x_ulb=probs_x_ulb_w.detach())

            # compute mask
            mask = self.call_hook("masking", "MaskingHook", logits_x_ulb=probs_x_ulb_w, softmax_x_ulb=False,
                                  idx_ulb=idx_ulb)

            # generate unlabeled targets using pseudo label hook
            pseudo_label = self.call_hook("gen_ulb_targets", "PseudoLabelingHook",
                                          logits=probs_x_ulb_w,
                                          use_hard_label=self.use_hard_label,
                                          T=self.T,
                                          softmax=False)

            self.update_pseudo_label_flexmatch(pseudo_label, mask)

            unsup_loss = self.consistency_loss(logits_x_ulb_s,
                                               pseudo_label,
                                               'ce',
                                               mask=mask)

            total_loss = sup_loss + self.lambda_u * unsup_loss

            entropy_loss, datapoint_entropy_loss = main_get_extra_loss(self.args, logits_x_ulb_w)
            if hasattr(self.args, 'lambda_entropy_with_labeled_data'):
                entropy_with_labeled_data_loss = main_get_entropy_with_labeled_data_loss(self.args, logits_x_ulb_w,
                                                                                         logits_x_lb)
                total_loss += self.args.lambda_entropy_with_labeled_data * entropy_with_labeled_data_loss

            if hasattr(self.args, 'lambda_entropy_with_labeled_data_v2'):
                entropy_with_labeled_data_loss = main_get_entropy_with_labeled_data_loss_v2(self.args, logits_x_ulb_w,
                                                                                            logits_x_lb)
                total_loss += self.args.lambda_entropy_with_labeled_data_v2 * entropy_with_labeled_data_loss

            total_loss += self.args.lambda_entropy * entropy_loss + self.args.lambda_datapoint_entropy * datapoint_entropy_loss

            # if self.args.python_code_version >= 9:
            #     if self.epoch > self.args.lambda_entropy_stop_epoch and self.args.lambda_entropy_stop_epoch != -1:
            #         print("remove lambda entropy")
            #         total_loss -= self.args.lambda_entropy * entropy_loss + self.args.lambda_datapoint_entropy * datapoint_entropy_loss
            if self.args.python_code_version >= 10:
                if self.epoch > self.args.lambda_entropy_stop_epoch and self.args.lambda_entropy_stop_epoch != -1:
                    print("change lambda entropy")
                    total_loss -= self.args.lambda_entropy * entropy_loss + self.args.lambda_datapoint_entropy * datapoint_entropy_loss
                    lambda_entropy_to_add, lambda_datapoint_entropy_to_add = self.get_new_lambda_after_collapse_epoch()
                    total_loss += lambda_entropy_to_add * entropy_loss + lambda_datapoint_entropy_to_add * \
                                  datapoint_entropy_loss
            if self.args.python_code_version >= 11:
                if self.args.lambda_kl_divergence != 0:
                    kl_divergence_loss = get_kl_divergence_loss(self.args, logits_x_ulb_w)
                else:
                    kl_divergence_loss = 0
                total_loss += self.args.lambda_kl_divergence * kl_divergence_loss
                super().update_loss_train_epoch(total_loss, sup_loss, unsup_loss, entropy_loss,
                                                datapoint_entropy_loss, kl_divergence_loss=kl_divergence_loss)
            else:
                super().update_loss_train_epoch(total_loss, sup_loss, unsup_loss, entropy_loss,
                                                datapoint_entropy_loss)

        out_dict = self.process_out_dict(loss=total_loss, feat=feat_dict)
        log_dict = self.process_log_dict(sup_loss=sup_loss.item(),
                                         unsup_loss=unsup_loss.item(),
                                         total_loss=total_loss.item(),
                                         util_ratio=mask.float().mean().item())
        return out_dict, log_dict

    def get_save_dict(self):
        save_dict = super().get_save_dict()
        # additional saving arguments
        save_dict['classwise_acc'] = self.hooks_dict['MaskingHook'].classwise_acc.cpu()
        save_dict['selected_label'] = self.hooks_dict['MaskingHook'].selected_label.cpu()
        return save_dict

    def load_model(self, load_path):
        checkpoint = super().load_model(load_path)
        self.hooks_dict['MaskingHook'].classwise_acc = checkpoint['classwise_acc'].cuda(self.gpu)
        self.hooks_dict['MaskingHook'].selected_label = checkpoint['selected_label'].cuda(self.gpu)
        self.print_fn("additional parameter loaded")
        return checkpoint

    @staticmethod
    def get_argument():
        return [
            SSL_Argument('--hard_label', str2bool, True),
            SSL_Argument('--T', float, 0.5),
            SSL_Argument('--p_cutoff', float, 0.95),
            SSL_Argument('--thresh_warmup', str2bool, True),
        ]

    def update_confidence_flexmatch(self, current_pass_mask, current_max_prob_values, current_classwise_acc):
        super().update_confidence_flexmatch(current_pass_mask=current_pass_mask,
                                            current_max_prob_values=current_max_prob_values,
                                            current_classwise_acc=current_classwise_acc)

    def update_pseudo_label_flexmatch(self, current_pseudo_label, mask):
        super().update_pseudo_label_flexmatch(current_pseudo_label=current_pseudo_label, current_mask=mask)
