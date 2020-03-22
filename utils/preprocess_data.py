'''
To run, from the repo root, run `python -m utils.preprocess_data`
'''

import os
import open3d
import torch
import numpy as np
import argparse
import yaml
from scipy import stats
from tqdm import tqdm
from utils.export_train_mesh import read_segmentation, read_aggregation
from utils.scannet_utils import read_label_mapping

open3d.set_verbosity_level(open3d.VerbosityLevel.Error)

parser = argparse.ArgumentParser()
parser.add_argument(
    '--config', default='configs/scannet-is-high_dim.yaml'
)

args = parser.parse_args()
config = yaml.load(open(args.config), Loader=yaml.FullLoader)

DATA_DIR = './data/scannet/'
TRAIN_DIR = os.path.join(DATA_DIR, 'scans')
LABEL_MAP_PATH = os.path.join(DATA_DIR, 'scannetv2-labels.combined.tsv')
PLY_POSTFIX = '_vh_clean_2.ply'
AGG_POSTFIX = '.aggregation.json'
SEG_POSTFIX = '_vh_clean_2.0.010000.segs.json'
OUT_POSTFIX = config['tensor_postfix']


def main():
    label_map = read_label_mapping(
        LABEL_MAP_PATH,
        label_from='raw_category', label_to='nyu40id'
    )
    scan_list = os.listdir(TRAIN_DIR)
    num_points = np.zeros(len(scan_list))
    cnt = 0

    valid_class_ids = config['valid_class_ids']
    scannet_id2id = {scan_id: i for (i, scan_id) in enumerate(valid_class_ids)}

    for scan_name in tqdm(scan_list):
        # read data from aggregation file and segmentation file
        scan_dir = os.path.join(TRAIN_DIR, scan_name)
        ply_path = os.path.join(scan_dir, scan_name + PLY_POSTFIX)
        agg_path = os.path.join(scan_dir, scan_name + AGG_POSTFIX)
        seg_path = os.path.join(scan_dir, scan_name + SEG_POSTFIX)

        # read point cloud scene
        feat_pcd = open3d.read_point_cloud(ply_path)
        coords = torch.as_tensor(feat_pcd.points, dtype=torch.float)
        feats = torch.as_tensor(feat_pcd.colors, dtype=torch.float)

        obj2segs, label2segs = read_aggregation(agg_path)
        segs2verts, num_verts = read_segmentation(seg_path)

        label_ids = torch.zeros(num_verts)
        instance_ids = torch.zeros(num_verts)

        # class ids
        for label, segs in label2segs.items():
            label_id = label_map[label]
            for seg in segs:
                verts = segs2verts[seg]
                label = scannet_id2id.get(label_id)
                # set invalid label as -1 label
                # later processed with ignore_index
                label_ids[verts] = label if label is not None else -1

        # object instances
        valid_object_cnt = 1
        for object_id, segs in obj2segs.items():
            verts = segs2verts[segs[0]]
            new_object_id = 0
            obj_label = label_ids[verts[0]]

            # set instance id to 0 if either unlabeled wall or floor
            if obj_label not in [-1, 0, 1]:
                new_object_id = valid_object_cnt
                valid_object_cnt += 1

            for seg in segs:
                verts = segs2verts[seg]
                instance_ids[verts] = new_object_id

        out_path = os.path.join(scan_dir, scan_name + OUT_POSTFIX)
        preprocessed_data = torch.cat(
            [coords, feats, label_ids.unsqueeze(dim=1), instance_ids.unsqueeze(dim=1)],
            dim=1
        )  # tensor of N x 8 (3 + 3 + 1 + 1)
        torch.save(preprocessed_data, out_path)
        num_points[cnt] = num_verts
        cnt += 1
    print("Preprocessing ended")
    print("Stats of number of point clouds: \n" + str(stats.describe(num_points)))


if __name__ == '__main__':
    main()