import open3d as o3d
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
	'--scene_path', help="ScanNet ply files path"
)
args = parser.parse_args()

pcd = o3d.read_point_cloud(args.scene_path)
o3d.draw_geometries([pcd])