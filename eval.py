import yaml
import torch
import open3d as o3d
import hdbscan
import numpy as np
from argparse import ArgumentParser
from open3d import Vector3dVector as vector
from MinkowskiEngine.utils import sparse_collate, sparse_quantize
from MinkowskiEngine import SparseTensor
from utils.visualization import build_cmap
from models import MODEL

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument(
        '--config', '-c',
        default='configs/scannet-is-high_dim-eval.yaml'
    )
    args = parser.parse_args()

    # Load config
    config_path = args.config
    config = yaml.load(open(config_path), Loader=yaml.FullLoader)

    # Load data
    raw_data = torch.load('data/example_scene.pt')
    coords, feats = raw_data[:, :3], raw_data[:, 3:6]
    feats = feats - 0.5
    coords = torch.floor(coords / config['voxel_size']).cpu()
    idxs = sparse_quantize(
        coords.numpy(),
        return_index=True,
        quantization_size=1
    )
    # coords, feats = coords[idxs], feats[idxs]
    coords, feats = sparse_collate([coords[idxs]], [feats[idxs]])
    x = SparseTensor(feats, coords.int()).to(config['device'])

    # Load semantic segmentation model
    semantic_model = MODEL['semantic-segmentation-model'](config, None)
    state_dict = torch.load(config['semantic_model']['path'])
    semantic_model.load_state_dict(state_dict)
    semantic_model.to(config['device'])
    semantic_model.eval()

    # Forward pass the semantic model
    with torch.no_grad():
        semantic_labels = semantic_model(x)
    semantic_labels = semantic_labels.max(dim=1).indices  # Tensor of N

    # remove labels predicted as wall and floor
    not_bg_idxs = (semantic_labels != 0) & (semantic_labels != 1)
    not_bg_coords, not_bg_feats = coords[not_bg_idxs], feats[not_bg_idxs]

    x = SparseTensor(not_bg_feats, not_bg_coords).to(config['device'])

    # Load instance segmentation model
    instance_model = MODEL['instance-segmentation-model'](config, None)
    state_dict = torch.load(config['backbone']['init_pretrained']['path'])
    instance_model.load_state_dict(state_dict)
    instance_model.to(config['device'])
    instance_model.eval()

    # do forward pass
    with torch.no_grad():
        embs = instance_model(x)

    # run hdbscan clustering
    cluster = hdbscan.HDBSCAN(
        min_cluster_size=config['clustering']['min_cluster_size'],
        allow_single_cluster=True,
    ).fit(embs.cpu())
    instance_labels = cluster.labels_
    cmap = build_cmap(config['color_map'], instance_labels.max().item() + 1).tolist()

    # visualize ground truth and instance labels
    gt_pcd = o3d.PointCloud()
    gt_pcd.points = vector(coords[:, 1: 4].numpy() * config['voxel_size'])
    gt_pcd.colors = vector(feats.numpy() + 0.5)

    trans = np.array([0., 10., 0.])
    pred_pcd = o3d.PointCloud()
    pred_pcd.points = vector(not_bg_coords[:, 1: 4].numpy() * config['voxel_size'] + trans)
    pred_pcd.colors = vector(np.array([cmap[p] for p in instance_labels]))

    o3d.draw_geometries([gt_pcd, pred_pcd])

