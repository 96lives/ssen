import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch import nn as nn
from abc import ABC, abstractmethod
from models.unet import BACKBONES
from utils.util import pairwise_distance, compute_ious
from utils.solvers import build_lr_scheduler, build_optimizer
from collections import defaultdict
from time import time


# ==========
# Model ABCs
# ==========


class Model(nn.Module, ABC):
    def __init__(self, config, writer: SummaryWriter):
        nn.Module.__init__(self)
        self.config = config
        self.device = config['device']
        self.writer = writer
        self.backbone = None  # initialize on concrete model
        self.step_time = time()
        self.loss = None
        self.scalar_summaries = defaultdict(list)

    def forward(self, x):
        return self.backbone(x)

    def init_pretrained(self, pretrained_path, strict):
        state_dict = torch.load(pretrained_path)
        if state_dict.get('state_dict') is not None:
            state_dict = state_dict['state_dict']
        # remove final
        if not strict:
            for k in list(state_dict.keys()):
                if 'final' in k:
                    state_dict.pop(k)
        self.backbone.load_state_dict(state_dict, strict=strict)

    def _clip_grad_value(self, clip_value):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_value_(group['params'], clip_value)

    def _clip_grad_norm(self, max_norm, norm_type=2):
        for group in self.optimizer.param_groups:
            nn.utils.clip_grad_norm_(group['params'], max_norm, norm_type)

    def clip_grad(self):
        clip_grad_config = self.config['clip_grad']
        if clip_grad_config['type'] == 'value':
            self._clip_grad_value(**clip_grad_config['options'])
        elif clip_grad_config['type'] == 'norm':
            self._clip_grad_norm(**clip_grad_config['options'])
        else:
            raise ValueError('Invalid clip_grad type: {}'
                             .format(clip_grad_config.type))

    def get_lr(self):
        for param_group in self.optimizer.param_groups:
            return param_group['lr']

    def learn(self, x, y, step):

        x, y = x.to(self.device), y.to(self.device)
        y_hat = self.forward(x)

        loss, scalar_summaries, list_summaries = self.compute_loss(x, y, y_hat, step)
        # accumulate batch
        loss = loss / self.config['acc_batch_step']

        for (k, v) in scalar_summaries.items():
            self.scalar_summaries[k].append(v)

        loss.backward()

        if (step + 1) % self.config['acc_batch_step'] == 0:
            self.clip_grad()
            self.optimizer.step()
            self.zero_grad()

        if (step + 1) % self.config['summary_step'] == 0:
            # write all the averaged summaries
            for (k, v) in self.scalar_summaries.items():
                v = np.array(v).mean().item()
                self.writer.add_scalar(k, v, step)

            for (k, v) in list_summaries.items():
                v = np.array(v)
                if k[:4] == 'mIoU':
                    num_classes = self.config['y_c']
                    confusion_matrix = v.reshape(-1, num_classes ** 2)
                    confusion_matrix = confusion_matrix.sum(axis=0) \
                        .reshape(num_classes, num_classes)
                    mious = []
                    for i in range(num_classes):
                        true_positive = confusion_matrix[i, i].item()
                        false_positive = (confusion_matrix[i, :].sum() - true_positive).item()
                        false_negative = (confusion_matrix[:, i].sum() - true_positive).item()
                        denom = true_positive + false_positive + false_negative
                        mious.append(0 if denom == 0 else float(true_positive) / denom)
                        if hasattr(self, 'class_id2label'):
                            self.writer.add_scalar(k + self.class_id2label[i], mious[-1], step)
                    self.writer.add_scalar(k + 'mIoU/overall', sum(mious) / len(mious), step)
                else:
                    v = v[v <= 1.]
                    bins = np.linspace(0., 1.05, num=22)
                    counts, limits = np.histogram(v, bins=bins)
                    sum_sq = v.dot(v)
                    self.writer.add_histogram_raw(
                        tag=k,
                        min=v.min(), max=v.max(),
                        num=len(v), sum=v.sum(),
                        sum_squares=sum_sq,
                        bucket_limits=limits[1:].tolist(),
                        bucket_counts=counts.tolist(),
                        global_step=step
                    )

            # reset summaries
            self.scalar_summaries = defaultdict(list)

            # write current learning rate
            self.writer.add_scalar('lr', self.get_lr(), step)

            # write time elapsed since summary_step
            self.writer.add_scalar(
                'time_per_step',
                (time() - self.step_time) / self.config['summary_step'], step
            )
            self.step_time = time()

        return loss.item()

    def compute_loss(self, x, y, y_hat, step) \
            -> (torch.Tensor, defaultdict, defaultdict):
        raise NotImplementedError()


class SemanticSegModel(Model):
    def __init__(self, config, writer: SummaryWriter):
        Model.__init__(self, config, writer)
        self.backbone = BACKBONES[config['backbone']['name']](
            config['backbone']['in_channels'],
            config['y_c'],
            config['backbone']['conv1_kernel_size']
        )
        pretrained_config = self.config['backbone']['init_pretrained']
        if self.config['backbone']['init_pretrained'] is not None:
            self.init_pretrained(pretrained_config['path'], pretrained_config['strict'])
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean', ignore_index=-1)
        self.optimizer = build_optimizer(
            self.config['optimizer'], self.parameters()
        )
        self.lr_scheduler = build_lr_scheduler(
            self.config['lr_scheduler'], self.optimizer
        )

    def compute_loss(self, x, y, y_hat, step) -> (torch.Tensor, defaultdict, defaultdict):
        scalar_summaries = defaultdict(float)
        list_summaries = defaultdict(list)

        loss = self.ce_loss(y_hat, y)

        # summary prefix
        mode = 'train' if self.training else 'val'
        loss_prefix = 'loss/{}/'.format(mode)
        acc_prefix = 'acc/{}/'.format(mode)
        iou_prefix = 'mIoU/{}/'.format(mode)

        # loss
        scalar_summaries[loss_prefix + 'total'] += loss.item()

        # prediction
        pred = y_hat.max(dim=1).indices
        acc = (pred == y).sum().item() / pred.shape[0]
        scalar_summaries[acc_prefix + 'total'] += acc

        # bg prediction
        if self.config.get('eval_bg_acc'):
            bg_pred = (pred != 0) & (pred != 1)
            bg_y = (y != 0) & (y != 1) & (y != -1)
            bg_acc = (bg_pred == bg_y).sum().item() / bg_pred.shape[0]
            scalar_summaries[acc_prefix + 'bg'] += bg_acc

        # iou
        num_classes = self.config['y_c']
        pred = pred[y >= 0]
        y = y[y >= 0]
        confusion_matrix = torch.bincount(
            pred + num_classes * y, minlength=num_classes ** 2).cpu().tolist()
        list_summaries[iou_prefix] += confusion_matrix

        return loss, scalar_summaries, list_summaries


class InstanceSegModel(Model):
    def __init__(self, config, writer: SummaryWriter):
        Model.__init__(self, config, writer)
        self.backbone = BACKBONES[config['backbone']['name']](
            config['backbone']['in_channels'],
            config['backbone']['emb_dim'],
            config['backbone']['conv1_kernel_size']
        )
        pretrained_config = self.config['backbone']['init_pretrained']
        if self.config['backbone']['init_pretrained'] is not None:
            self.init_pretrained(pretrained_config['path'], pretrained_config['strict'])
        self.class_id2label = {
            i: label for (i, label) in enumerate(self.config['class_labels'])
        }
        self.optimizer = build_optimizer(
            self.config['optimizer'], self.parameters()
        )
        self.lr_scheduler = build_lr_scheduler(
            self.config['lr_scheduler'], self.optimizer
        )

    def compute_loss(self, x, y, y_hat, step) -> (torch.Tensor, defaultdict, defaultdict):
        loss_config = self.config['loss']

        total_losses = []
        scalar_summaries = defaultdict(float)
        list_summaries = defaultdict(list)
        batch_size = self.config['batch_size'] \
            if self.training else self.config['eval_batch_size']

        for batch_idx in range(batch_size):
            # fix scene
            scene_idxs = (x.C[:, 0] == batch_idx) \
                .nonzero().squeeze(dim=1)
            if scene_idxs.shape[0] <= 0:
                continue

            # unravel objects
            embs = y_hat[scene_idxs]
            # only backgrounds
            gt_insts = y[scene_idxs, 1]
            if gt_insts.sum() <= 0.:
                continue
            num_insts = gt_insts.max()

            inst_emb_means = []
            inst_idxs = []
            inter_losses = []
            dist2mean = []  # compute average distance to instance mean

            for inst in range(1, num_insts + 1):
                # fix instance in data
                single_inst_idxs = (gt_insts == inst)
                # no instance
                if single_inst_idxs.sum() == 0:
                    continue
                inst_idxs.append(single_inst_idxs)
                inst_embs = embs[single_inst_idxs]  # Tensor of N x D
                inst_emb_mean = inst_embs.mean(dim=0)
                inst_emb_means.append(inst_emb_mean)

                # compute inter_loss
                inst_dists = torch.norm(
                    inst_embs - inst_emb_mean.unsqueeze(dim=0),
                    dim=1
                )  # Tensor of N
                inter_losses.append(torch.relu(inst_dists - loss_config['delta_inter']).mean())
                dist2mean.append(inst_dists)

            # inter loss
            inter_losses = torch.stack(inter_losses)
            num_inst_points = torch.tensor([x.shape[0] for x in inst_idxs])

            # weight loss by p, s.t. 0 <= p <= 1
            # if p == 0, equal weighting to each instance
            # if p == 1, equal weighting to per point
            # exclude bg from inter losses
            inter_loss_weight = num_inst_points.float() \
                .pow(loss_config['inter_chill']).to(self.device)
            inter_loss_weight = inter_loss_weight / inter_loss_weight.sum()
            inter_loss = torch.dot(inter_losses, inter_loss_weight)

            # intra_loss
            inst_emb_means = torch.stack(inst_emb_means, dim=0)
            pair_dist_mean = pairwise_distance(inst_emb_means)
            pair_dist_mean = torch.sqrt(torch.relu(pair_dist_mean))  # relu assures positiveness
            henge_dist_pair = torch.relu(2 * loss_config['delta_intra'] - pair_dist_mean)
            intra_loss = henge_dist_pair.sum() - torch.diag(henge_dist_pair).sum()

            # delete for memory efficiency
            del pair_dist_mean
            del henge_dist_pair

            # if background alone or one instance
            intra_loss /= 2 * num_insts * (num_insts - 1)
            if num_insts <= 1:
                intra_loss = torch.tensor(0.).to(self.device)

            # reg_loss
            reg_loss = torch.norm(inst_emb_means, dim=1).mean()

            # sum all the losses
            eff_inter_loss = loss_config['gamma_inter'] * inter_loss
            eff_intra_loss = loss_config['gamma_intra'] * intra_loss
            eff_reg_loss = loss_config['gamma_reg'] * reg_loss
            total_loss = eff_inter_loss + eff_intra_loss + eff_reg_loss
            total_losses.append(total_loss)

            if torch.isnan(total_loss):
                __import__('pdb').set_trace()

            # add losses to summaries
            mode = 'train' if self.training else 'val'
            loss_prefix = 'loss/{}/'.format(mode)
            raw_prefix = loss_prefix + 'raw/'
            eff_prefix = loss_prefix + 'eff/'
            ratio_prefix = loss_prefix + 'ratio/'
            iou_prefix = 'iou/{}/'.format(mode)
            dist_prefix = 'dist/{}/'.format(mode)

            # add total loss
            scalar_summaries[loss_prefix + 'total'] += total_loss.item()

            # add raw loss
            scalar_summaries[raw_prefix + 'inter_loss'] += inter_loss.item()
            scalar_summaries[raw_prefix + 'intra_loss'] += intra_loss.item()
            scalar_summaries[raw_prefix + 'reg_loss'] += reg_loss.item()

            # add eff loss
            scalar_summaries[eff_prefix + 'inter_loss'] += eff_inter_loss.item()
            scalar_summaries[eff_prefix + 'intra_loss'] += eff_intra_loss.item()
            scalar_summaries[eff_prefix + 'reg_loss'] += eff_reg_loss.item()

            # add loss ratio
            if total_loss.item() != 0:
                scalar_summaries[ratio_prefix + 'inter_loss'] \
                    += eff_inter_loss.item() / total_loss.item()
                scalar_summaries[ratio_prefix + 'intra_loss'] \
                    += eff_intra_loss.item() / total_loss.item()
                scalar_summaries[ratio_prefix + 'reg_loss'] \
                    += eff_reg_loss.item() / total_loss.item()

            # add dist2mean
            dist2mean = torch.cat(dist2mean)  # Tensor of shape N
            scalar_summaries[dist_prefix + 'dist_to_mean'] \
                += dist2mean.mean().item()  # without bg
            eff_dist2mean = dist2mean.clone()
            eff_dist2mean = eff_dist2mean[dist2mean > loss_config['delta_inter']]
            scalar_summaries[dist_prefix + 'eff_dist_to_mean'] \
                += eff_dist2mean.mean().item()

            # bg dist2_emb_mean
            bg_embs = embs[gt_insts == 0]
            if bg_embs.nelement() != 0:
                bg_dist2emb_mean = pairwise_distance(bg_embs, inst_emb_means)
                bg_dist2emb_mean = bg_dist2emb_mean.min(dim=1).values  # bg to nearest embedding mean
                scalar_summaries[dist_prefix + 'bg_dist_to_emb_mean'] \
                    += bg_dist2emb_mean.mean().item()
                list_summaries[dist_prefix + 'bg_dist_to_emb_mean_hist'] += bg_dist2emb_mean.cpu().tolist()

            if (not self.training) or (self.training and ((step + 1) % self.config['summary_step'] == 0)):
                list_summaries[loss_prefix + 'inter_loss_weight'] += inter_loss_weight.tolist()

                # use ground truth mean for debugging
                if num_insts > 1:
                    inst_mean_seeds = [(inst_emb_means[i], i) for i in range(inst_emb_means.shape[0])]
                    ious = compute_ious(inst_mean_seeds, embs, inst_idxs, self.config['emb_thres'])
                    scalar_summaries[iou_prefix + 'mean_sample/mean'] += sum(ious) / float(len(ious))
                    scalar_summaries[iou_prefix + 'mean_sample/max'] += max(ious)
                    scalar_summaries[iou_prefix + 'mean_sample/min'] += min(ious)
                    list_summaries[iou_prefix + 'mean_sample'] += ious

        scalar_summaries = {k: (float(v) / batch_size) for (k, v) in scalar_summaries.items()}
        loss = torch.stack(total_losses).mean()
        return loss, scalar_summaries, list_summaries
