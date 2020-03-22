# Example of the output format for evaluation for 3d semantic label and instance prediction.
# Exports a train scan in the evaluation format using:
#   - the *_vh_clean_2.ply mesh
#   - the labels defined by the *.aggregation.json and *_vh_clean_2.0.010000.segs.json files
#
# example usage: export_train_mesh.py --scan_path [path to scan data] --output_file [output file] --type label
# Note: technically does not need to load in the ply file, since the ScanNet annotations are defined against the mesh vertices, but we load it in here as an example.

# For testing, run python -m utils.export_train_mesh

# python imports
import math
import os, sys, argparse
import inspect
import json
import numpy as np

import utils.scannet_utils as util
import utils.util_3d as util_3d

'''
curr_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(curr_dir)
sys.path.insert(0, parent_dir)

TASK_TYPES = {'label', 'instance'}
data_dir = os.path.join('/data1/dszhang/metric-segmentation', 'data', 'scannet')

parser = argparse.ArgumentParser()
parser.add_argument(
    '--scan_path',
    default=os.path.join(data_dir, 'scans', 'scene0000_00'),
    help='path to scannet scene (e.g., data/ScanNet/v2/scene0000_00'
)
parser.add_argument(
    '--output_file',
    default=os.path.join('/data1/dszhang/metric-segmentation/temp', 'temp'),
    help='output file'
)
parser.add_argument(
    '--label_map_file',
    default=os.path.join(data_dir, 'scannetv2-labels.combined.tsv'),
    help='path to scannetv2-labels.combined.tsv'
)
parser.add_argument(
    '--type', default='instance',
    help='task type [label or instance]'
)
opt = parser.parse_args()
assert opt.type in TASK_TYPES
'''
LABEL_MAP_FILE = 'scannetv2-labels.combined.tsv'

def read_aggregation(filename):
    assert os.path.isfile(filename)
    object_id_to_segs = {}
    label_to_segs = {}
    with open(filename) as f:
        data = json.load(f)
        num_objects = len(data['segGroups'])
        for i in range(num_objects):
            object_id = data['segGroups'][i]['objectId'] + 1 # instance ids should be 1-indexed
            label = data['segGroups'][i]['label']
            segs = data['segGroups'][i]['segments']
            object_id_to_segs[object_id] = segs
            if label in label_to_segs:
                label_to_segs[label].extend(segs)
            else:
                label_to_segs[label] = segs
    return object_id_to_segs, label_to_segs


def read_segmentation(filename):
    assert os.path.isfile(filename)
    seg_to_verts = {}
    with open(filename) as f:
        data = json.load(f)
        num_verts = len(data['segIndices'])
        for i in range(num_verts):
            seg_id = data['segIndices'][i]
            if seg_id in seg_to_verts:
                seg_to_verts[seg_id].append(i)
            else:
                seg_to_verts[seg_id] = [i]
    return seg_to_verts, num_verts


def export(mesh_file, agg_file, seg_file, label_map_file, type, output_file):
    label_map = util.read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to='nyu40id')
    mesh_vertices = util_3d.read_mesh_vertices(mesh_file)
    object_id_to_segs, label_to_segs = read_aggregation(agg_file)
    seg_to_verts, num_verts = read_segmentation(seg_file)
    label_ids = np.zeros(shape=(num_verts), dtype=np.uint32)     # 0: unannotated
    for label, segs in label_to_segs.items():
        label_id = label_map[label]
        for seg in segs:
            verts = seg_to_verts[seg]
            label_ids[verts] = label_id
    if type == 'label':
        util_3d.export_ids(output_file, label_ids)
    elif type == 'instance':
        instance_ids = np.zeros(shape=(num_verts), dtype=np.uint32)  # 0: unannotated
        for object_id, segs in object_id_to_segs.items():
            for seg in segs:
                verts = seg_to_verts[seg]
                instance_ids[verts] = object_id
        util_3d.export_instance_ids_for_eval(output_file, label_ids, instance_ids)
    else:
        raise


'''
def main():
    scan_name = os.path.split(opt.scan_path)[-1]
    mesh_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.ply')
    agg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean.aggregation.json') # '.aggregation.json')
    seg_file = os.path.join(opt.scan_path, scan_name + '_vh_clean_2.0.010000.segs.json')
    export(mesh_file, agg_file, seg_file, opt.label_map_file, opt.type, opt.output_file)


if __name__ == '__main__':
    main()
'''
