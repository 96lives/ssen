import os
import open3d
import torch
import numpy as np
import argparse
import yaml
from tqdm import tqdm
from models.unet import Res16UNet34C
from data import DataScheduler
from MinkowskiEngine.utils.collation import sparse_collate
from MinkowskiEngine import SparseTensor

open3d.set_verbosity_level(open3d.VerbosityLevel.Error)

parser = argparse.ArgumentParser()
parser.add_argument(
	'--config', default='configs/scannet-is-high_dim.yaml'
)

args = parser.parse_args()
config = yaml.load(open(args.config), Loader=yaml.FullLoader)
config['train_list'] = './data/scannet/scannet_combined.txt'
config['y_c'] = 20
DATA_DIR = './data/scannet/scans'
OUT_POSTFIX = config['tensor_postfix']


def raw_getitem(dataset, i):
	scene_name = dataset.idx2dir[i]
	file_path = os.path.join(dataset.root_dir, scene_name, scene_name + dataset.tensor_postfix)
	data = torch.load(file_path)
	coords, feats, targets = data[:, :3], data[:, 3:6], data[:, 6:8]  # tensor of N x 3, N x 3, N x 2
	return coords, feats, targets, scene_name


def vox2point(vcoords, coords_voxel):
	voxel_all = np.vstack((vcoords, coords_voxel)).astype(int)
	voxel_min = voxel_all.min(0)
	voxel_dim = voxel_all.max(0) - voxel_all.min(0) + 1
	input_key = _hash_coords(vcoords.int().numpy(), voxel_min, voxel_dim)
	original_key = _hash_coords(coords_voxel, voxel_min, voxel_dim)
	key2idx = dict(zip(input_key, range(len(input_key))))
	vox_idx = np.array([key2idx.get(i, -1) for i in original_key])
	return vox_idx


def _hash_coords(coords, coords_min, coords_dim):
	return np.ravel_multi_index((coords - coords_min).T, coords_dim)


def main():
	dataset = DataScheduler(config).dataset
	model = Res16UNet34C(3, config['y_c'], config['semantic_model']['conv1_kernel_size']) \
		.to(config['device'])
	state_dict = torch.load(config['semantic_model']['path'])
	if 'state_dict' in state_dict:
		state_dict = state_dict['state_dict']
	model.load_state_dict(state_dict)
	model.eval()

	for i in tqdm(range(len(dataset))):
		coords, feats, targets, scene_name = raw_getitem(dataset, i)
		coords_voxel = torch.floor(coords / config['voxel_size']).int().numpy()
		vcoords, vfeats, vtargets = dataset.quantize_data(coords, feats, targets)
		vox_idx = vox2point(vcoords, coords_voxel)

		vcoords, vfeats, = sparse_collate([vcoords], [vfeats])
		x = SparseTensor(vfeats, vcoords).to(config['device'])
		y = vtargets

		with torch.no_grad():
			y_hat = model(x)
		semantic_pred = y_hat.max(dim=1).indices[vox_idx]
		mask = ((semantic_pred != 0) & (semantic_pred != 1)).float().cpu()
		outfile = os.path.join(DATA_DIR, scene_name, scene_name + OUT_POSTFIX)
		preprocessed_data = torch.cat(
			[coords, feats, targets, mask.unsqueeze(dim=1)],
			dim=1
		)
		torch.save(preprocessed_data, outfile)


if __name__ == "__main__":
	main()
