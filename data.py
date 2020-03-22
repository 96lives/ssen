from abc import ABC, abstractmethod
from collections import Iterator, defaultdict
import torch
import open3d
from torch.utils.data import (
	Sampler,
	Dataset,
	DataLoader,
)
from torch.utils.tensorboard import SummaryWriter
from MinkowskiEngine.utils import sparse_collate
from MinkowskiEngine.utils import sparse_quantize
from MinkowskiEngine import SparseTensor
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import List
from models.base import Model
from utils import data_augmentation, visualization

# scannet evaluator
open3d.set_verbosity_level(open3d.VerbosityLevel.Error)

# =====================
# Base Classes and ABCs
# =====================


class RepetitiveSampler(Sampler):
	def __init__(self, data_source, iter_cnt):
		self.data_source = data_source
		self.iter_cnt = iter_cnt

	def __iter__(self):
		return iter([int(i / self.iter_cnt) for i in range(self.iter_cnt * len(self.data_source))])

	def __len__(self):
		return self.iter_cnt * len(self.data_source)


class InfiniteRandomSampler(Sampler):
	def __init__(self, dataset, max_samples=None):
		super().__init__(dataset)
		self.dataset = dataset
		self.len_data = len(self.dataset)
		self.iterator = iter(torch.randperm(self.len_data).tolist())
		self.max_samples = max_samples
		self.sample_num = 0

	def __next__(self):
		if self.max_samples is not None and \
				self.sample_num >= self.max_samples:
			raise StopIteration

		self.sample_num += 1

		try:
			idx = next(self.iterator)
		except StopIteration:
			self.iterator = iter(torch.randperm(self.len_data).tolist())
			idx = next(self.iterator)

		return idx

	def __iter__(self):
		return self

	def __len__(self):
		if self.max_samples is not None:
			return self.max_samples
		else:
			return len(self.dataset)


class DataScheduler(Iterator):
	def __init__(self, config):
		self.config = config
		self.dataset = DATASET[self.config['dataset']](config, train=True)
		self.eval_dataset = DATASET[self.config['dataset']](config, train=False)
		self.total_epoch = self.config['epoch']
		self.step_cnt = 0
		self.epoch_cnt = 0
		self._remainder = len(self.dataset)
		self.sampler = InfiniteRandomSampler(
			self.dataset,
			max_samples=int(self.config['epoch'] * len(self.dataset))
		)
		self.data_loader = iter(DataLoader(
			self.dataset,
			batch_size=self.config['batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.dataset.collate_fn,
			sampler=self.sampler,
			drop_last=True,
		))
		self._check_vis = {}

	def __next__(self):
		if self.data_loader is None:
			raise StopIteration
		data = next(self.data_loader)
		self.step_cnt += 1
		self._remainder -= self.config['batch_size']
		if self._remainder < self.config['batch_size']:
			self._remainder = len(self.dataset)
			self.epoch_cnt += 1

		# Get next data
		return data[0], data[1], self.epoch_cnt

	def __len__(self):
		return len(self.sampler)

	def check_eval_step(self, step):
		return ((step + 1) % self.config['eval_step'] == 0) \
			   or self.config['debug_eval']

	def check_vis_step(self, step):
		vis = False
		vis_config = self.config['vis']
		for (k, v) in vis_config.items():
			if not isinstance(v, dict):
				continue
			if ((step + 1) % v['step'] == 0) or (self.config['debug_vis']):
				self._check_vis[k] = True
				vis = True
			else:
				self._check_vis[k] = False
		return vis

	def eval(self, model, writer, step):
		self.eval_dataset.eval(model, writer, step)

	def visualize(self, model, writer, step):
		dataset = self.eval_dataset
		if self.config['overfit_one_ex']:
			dataset = self.dataset  # train dataset

		# find options to visualize in this step
		options = []
		for (k, v) in self._check_vis.items():
			if not v:
				continue
			if k == 'embs':
				if self.config['backbone']['emb_dim'] == 3:
					options.append(k)
			else:
				options.append(k)
		assert len(options) > 0, \
			'Visualization Error!, maybe dim != 3 and try to visualize embs?'
		dataset.visualize(options, model, writer, step)

		# reset _check_vis
		self._check_vis = {}


class BaseDataset(Dataset, ABC):
	name = 'base'

	def __init__(self, config, train=True):
		self.config = config
		self.train = train
		self.sparse_transform = self.build_transform()

	def build_transform(self):
		transform_config = self.config['transform']
		if transform_config is None:
			return None
		return data_augmentation.Compose([
			getattr(data_augmentation, config['type'])(**config['options'])
			if config.get('options') else getattr(data_augmentation, config['type'])()
			for config in transform_config
		])

	def collate_fn(self, batch):
		coords, features, labels = list(zip(*batch))
		coords, features, labels = sparse_collate(coords, features, labels)
		return SparseTensor(features, coords=coords), labels

	def quantize_data(self, coords, feats, labels):
		# Create SparseTensor
		coords = torch.floor(coords / self.config['voxel_size']).cpu()
		coords = coords - coords.min(0).values
		idxs = sparse_quantize(
			coords.numpy(),
			return_index=True,
			quantization_size=1
		)
		return coords[idxs], feats[idxs], labels[idxs]

	def sample_data(self, coords, feats, labels):
		max_sample = self.config['max_train_sample']
		if (max_sample is not None) and (coords.shape[0] > max_sample):
			perm = torch.randperm(coords.shape[0])
			coords = coords[perm[:max_sample]]
			feats = feats[perm[:max_sample]]
			labels = labels[perm[:max_sample]]
		return coords, feats, labels

	def eval(self, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()
		scalar_summaries = defaultdict(list)
		list_summaries = defaultdict(list)
		data_loader = DataLoader(
			self,
			batch_size=self.config['eval_batch_size'],
			num_workers=self.config['num_workers'],
			collate_fn=self.collate_fn,
			drop_last=True,
		)

		print('')
		for eval_step, data in enumerate(data_loader):
			x, y = data[0], data[1]
			x, y = x.to(self.config['device']), y.to(self.config['device'])
			with torch.no_grad():
				y_hat = model(x)
			loss, scalar_summary, list_summary = model.compute_loss(x, y, y_hat, step)
			print('\r[Evaluating, Step {:7}, Loss {:5}]'.format(
				eval_step, '%.3f' %loss), end=''
			)

			for (k, v) in scalar_summary.items():
				scalar_summaries[k].append(v)

			for (k, v) in list_summary.items():
				list_summaries[k] += v

		# write summaries
		for (k, v) in scalar_summaries.items():
			v = np.array(v).mean().item()
			writer.add_scalar(k, v, step)

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
						writer.add_scalar(k + self.class_id2label[i], mious[-1], step)
				writer.add_scalar(k + 'mIoU/overall', sum(mious) / len(mious), step)
			else:
				bins = np.linspace(0., 1.1, num=12)
				counts, limits = np.histogram(v, bins=bins)
				sum_sq = v.dot(v)

				writer.add_histogram_raw(
					tag=k,
					min=v.min(), max=v.max(),
					num=len(v), sum=v.sum(),
					sum_squares=sum_sq,
					bucket_limits=limits[1:].tolist(),
					bucket_counts=counts.tolist(),
					global_step=step
				)

		model.train(training)

	def visualize(self, model: Model, writer: SummaryWriter, epoch, step):
		training = model.training
		model.eval()

		vis_indices = self.config['vis_indices']
		if isinstance(self.config['vis_indices'], int):
			# sample k data points from n data points with equal interval
			n = len(self)
			k = self.config['vis_indices']
			vis_indices = torch.linspace(0, n - 1, k) \
				.type(torch.IntTensor).tolist()

		self.visualize_data(
			model, writer, self,
			vis_indices, 'val_pc', step
		)
		model.train(training)

	def visualize_data(
			self, model: Model, writer: SummaryWriter,
			dataset: Dataset, indices: List, tag, step
	):
		# visualize one data
		batch = [dataset[i] for i in indices]
		coords, feats, label, _ = list(zip(*batch))
		coords, feats, = sparse_collate(coords, feats)
		x = SparseTensor(feats, coords)

		x = x.to(model.device)
		with torch.no_grad():
			y = model(x)
		pred = y['pred']
		pred_choices = pred.max(dim=1).indices

		for i in range(len(indices)):
			# get indices with specific indices
			data_indices = (y.C[:, 3] == i).nonzero().squeeze(1)
			coord = coords[data_indices, :3].type(torch.FloatTensor)
			coord = coord * self.config['voxel_size']
			coord = torch.stack([coord, coord])  # Tensor of 2 x N x 3
			pred_choice = pred_choices[data_indices]

			# add color for prediction
			pred_color = torch.stack(
				[self.cmap[point] for point in pred_choice],
				dim=0
			)  # Tensor of N x 3 (1 for batch)
			gt_color = torch.stack(
				[self.cmap[point] for point in label[i]],
				dim=0
			)  # Tensor of N x 3 (1 for batch)
			color = torch.stack([pred_color, gt_color], dim=0)  # Tensor of 2 x N x 3
			color = (color * 255).type(torch.IntTensor)

			max_sample = self.config['max_vis_sample']
			if coord.shape[1] > max_sample:
				perm = np.random.RandomState(0).permutation(coord.shape[1])
				coord = coord[:, perm[:max_sample], :]
				color = color[:, perm[:max_sample], :]

			writer.add_mesh(
				tag=tag + '/vis_%d' % i, vertices=coord,
				colors=color, global_step=step
			)


# ================
# Generic Datasets
# ================

class InstanceSegmentDataset(BaseDataset, ABC):
	targets = NotImplemented

	def __init__(self, config, train=True):
		BaseDataset.__init__(self, config, train)


class SemanticSegmentDataset(BaseDataset, ABC):

	def __init__(self, config, train=True):
		BaseDataset.__init__(self, config, train)
		self.cmap = build_cmap(self.config['color_map'], self.config['y_c'])



# =================
# Concrete Datasets
# =================


class ScanNet(BaseDataset, ABC):
	name = 'scannet'

	def __init__(self, config, train=True):
		BaseDataset.__init__(self, config, train)
		self.root_dir = config['train_root'] if train else config['val_root']
		self.data_list_path = config['train_list'] if train else config['val_list']
		self.data_list = open(self.data_list_path).read().split('\n')
		self.idx2dir = {
			i: x
			for (i, x) in enumerate(self.data_list)
		}
		self.dir2idx = {v: k for (k, v) in self.idx2dir.items()}
		self.tensor_postfix = config['tensor_postfix']
		self.class_id2label = {i: label for (i, label) in enumerate(self.config['class_labels'])}
		self.check_config()

	def __getitem__(self, index):
		'''
		Args:
			index (int): Index
		Returns:
			coords: Tensor of N x 3
			feats: Tensor of N x 3
			labels: Tensor of N x 2 (classes, objects)
		'''

		# get coords, feats
		dir_name = self.idx2dir[index]
		if self.config['overfit_one_ex']:
			dir_name = self.config['overfit_one_ex']
		file_path = os.path.join(self.root_dir, dir_name, dir_name + self.tensor_postfix)
		data = torch.load(file_path, map_location=self.config['device'])
		coords, feats, labels = data[:, :3], data[:, 3:6], data[:, 6:8]  # tensor of N x 3, N x 3, N x 2
		bg = data[:, 8]

		if self.config['remove_bg_with_pretrained']:
			# add coords that has been predicted as floor or wall but really not
			rand_num = torch.rand(1).item()
			if rand_num < 0.33 and self.train:
				idxs = (bg != 0) | (labels[:, 1] != 0)
			elif rand_num < 0.66 and self.train:
				# use gt
				idxs = (labels[:, 1] != 0)
			else:
				# use prediction
				idxs = (bg != 0)

			# if empty
			if idxs.sum() == 0:
				idxs = (labels[:, 1] != 0)

			coords = coords[idxs, :]
			feats = feats[idxs, :]
			labels = labels[idxs, :]
		labels = labels.long()  # for labels

		# Perform transform
		if (self.sparse_transform is not None) and self.train:
			try:
				coords, feats = self.sparse_transform(coords, feats)
			except RuntimeError:
				__import__('pdb').set_trace()

		# parse data for training
		try:
			coords, feats, labels = self.quantize_data(coords, feats, labels)
		except RuntimeError:
			__import__('pdb').set_trace()

		if self.train:
			coords, feats, labels = self.sample_data(coords, feats, labels)
		if self.config['zero_mean_normalize']:
			feats = feats - 0.5

		return coords, feats, labels, index

	def __len__(self):
		return len(self.idx2dir)

	def collate_fn(self, batch):
		coords, features, labels, indices = list(zip(*batch))
		coords, features, labels = sparse_collate(coords, features, labels)

		if self.train:
			return SparseTensor(features, coords=coords), labels

		dir_names = [self.idx2dir[idx] for idx in indices]
		return SparseTensor(features, coords=coords), labels, dir_names

	def visualize(self, options, model: Model, writer: SummaryWriter, step):
		training = model.training
		model.eval()

		vis_config = self.config['vis']

		if vis_config.get('num_scene_samples'):
			# sample k data points from n data points with equal interval
			n = len(self)
			k = vis_config.get('num_scene_samples')
			vis_indices = torch.linspace(0, n - 1, k) \
				.type(torch.IntTensor).tolist()
		else:
			vis_indices = [self.dir2idx[i] for i in vis_config.get('scene_names')]

		if self.config['overfit_one_ex']:
			vis_scene = self.config['overfit_one_ex']
			vis_indices = [self.dir2idx[vis_scene]]
			vis_indices = list(set(vis_indices))

		for i in vis_indices:
			coords, feats, labels, _ = self[i]
			coords, feats, = sparse_collate([coords], [feats])
			x = SparseTensor(feats, coords)

			x = x.to(model.device)
			with torch.no_grad():
				y_hat = model(x)

			embs = y_hat
			insts = labels[:, 1]

			for option in options:
				# visualize tsne
				if option == 'tsne':
					tsne_img = visualization.visualize_tsne(
						embs.cpu(), insts.cpu(),
						config=self.config['vis']['tsne']
					)
					writer.add_image('tsne/{}'.format(self.idx2dir[i]), tsne_img, step)

				elif option == 'embs':
					vis_config = self.config['vis']['embs']

					# visualize embs with background
					emb_imgs, axis_range = visualization.visualize_embs(
						embs.cpu(), insts.cpu(),
						remove_bg=False, max_sample=vis_config['max_sample'],
						num_view=vis_config['num_view']
					)
					for view_num, img in enumerate(emb_imgs):
						writer.add_image(
							'emb/with_bg/{}_{}'.format(self.idx2dir[i], view_num),
							img, step
						)

					# visualize embs without background
					not_bg_emb_imgs, _ = visualization.visualize_embs(
						embs.cpu(), insts.cpu(),
						remove_bg=True, max_sample=vis_config['max_sample'],
						num_view=vis_config['num_view'], axis_range=axis_range
					)
					for view_num, img in enumerate(not_bg_emb_imgs):
						writer.add_image(
							'emb/no_bg/{}_{}'.format(self.idx2dir[i], view_num),
							img, step
						)

			model.train(training)

	# check config is correct during initialization
	def check_config(self):

		# check if either num_scene_samples or scene_names is in the config
		vis_config = self.config['vis']
		assert vis_config.get('num_scene_samples') != vis_config.get('scene_names'),\
			'exactly one of num_scene_samples or scene_names must be in the config'

		# check if vis_names exist in validation set
		if (not self.train) and vis_config.get('scene_names'):
			for scene in vis_config.get('scene_names'):
				assert self.dir2idx.get(scene) is not None, \
					'{} is not in the validation dataset, check scene_names in config'.format(scene)


class ScanNetSemantic(ScanNet, SemanticSegmentDataset):
	name = 'scannet-ss'

	def __init__(self, config, train=True):
		SemanticSegmentDataset.__init__(self, config, train)
		ScanNet.__init__(self, config, train)

	def __getitem__(self, index):
		coords, feats, targets, indices = ScanNet.__getitem__(self, index)
		targets = targets[:, 0]
		return coords, feats, targets, indices


class ScanNetInstance(ScanNet, InstanceSegmentDataset):
	name = 'scannet-is'

	def __init__(self, config, train=True):
		InstanceSegmentDataset.__init__(self, config, train)
		ScanNet.__init__(self, config, train)


DATASET = {
	ScanNetSemantic.name: ScanNetSemantic,
	ScanNetInstance.name: ScanNetInstance,
}
