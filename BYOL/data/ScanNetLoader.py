import os
import os.path as osp
import shlex
import shutil
import subprocess
import copy

import pickle
import numpy as np
import torch
import random
import sys
from PIL import Image
from data.data_utils import points_sampler


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
try:
    from plyfile import PlyData, PlyElement
except:
    print("Please install the module 'plyfile' for PLY i/o, e.g.")
    print("pip install plyfile")
    sys.exit(-1)

def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    el = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([el], text=text).write(filename)

class ScannetWholeSceneContrast():
    def __init__(self, num_points, transforms=None, train=True):
        self.npoints = num_points
        self.transforms = transforms
        self.root = os.path.join(BASE_DIR, "scannet")
        if train:
            self.split = "train"
        else:
            self.split = "test"
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(self.split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding="bytes")


    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        point_set_ = point_set.copy()

        if self.transforms is not None:
            point_set = self.transforms(point_set)
            point_set_ = self.transforms(point_set_)

        point_set = points_sampler(point_set, self.npoints)
        point_set_ = points_sampler(point_set_, self.npoints)

        return point_set, point_set_

    def __len__(self):
        return len(self.scene_points_list)


class ScannetWholeSceneContrastHeight():
    def __init__(self, num_points, transforms_1=None, transforms_2=None, train=True, no_height=True):
        self.npoints = num_points
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2
        self.no_height = no_height
        self.root = os.path.join(BASE_DIR, "scannet")
        if train:
            self.split = "train"
        else:
            self.split = "test"
        self.data_filename = os.path.join(self.root, 'scannet_%s.pickle'%(self.split))
        with open(self.data_filename,'rb') as fp:
            self.scene_points_list = pickle.load(fp, encoding="bytes")


    def __getitem__(self, index):
        point_set = self.scene_points_list[index]
        point_set_ = point_set.copy()

        if self.transforms_1 is not None:
            point_set = self.transforms_1(point_set)
            point_set_ = self.transforms_1(point_set_)

        point_set = point_set.numpy()
        point_set_ = point_set_.numpy()
        if not self.no_height:
            floor = np.percentile(point_set[:, 2], 0.99)
            floor_ = np.percentile(point_set_[:, 2], 0.99)

        if self.transforms_2 is not None:
            point_set = self.transforms_2(point_set)
            point_set_ = self.transforms_2(point_set_)

        point_set = points_sampler(point_set, self.npoints)
        point_set_ = points_sampler(point_set_, self.npoints)

        if not self.no_height:
            height = point_set[:, 2] - floor
            height_ = point_set_[:, 2] - floor_

            height = torch.unsqueeze(height, 1)
            height_ = torch.unsqueeze(height_, 1)
            point_set = torch.cat([point_set, height], 1)
            point_set_ = torch.cat([point_set_, height_], 1)

        return point_set, point_set_

    def __len__(self):
        return len(self.scene_points_list)


class ScanNetFrameContrast():
    def __init__(self, num_points, transforms_1=None, transforms_2=None, no_height=True, mode="temporal"):
        self.npoints = num_points
        self.transforms_1 = transforms_1
        self.transforms_2 = transforms_2
        self.no_height = no_height
        self.root_path = os.path.join(BASE_DIR, "scannet", "scannet_frames_25k")
        self.mode = mode
        assert mode in ["spatial", "temporal", "both"]

        self.load_filenames()

    def load_filenames(self):
        self.scenes = os.listdir(self.root_path)
        self.scenes.sort()
        self.depth_files = {}  # {scene: [depth_files]}
        self.cam_pose_files = {}  # {scene: [cam_pose_files]}
        self.frame_num = {}  # {scene: frame_num}
        self.total_num = 0
        self.frame_idx = []  # [(scene, id_in_scene)]

        for scene in self.scenes:
            scene_dir = os.path.join(self.root_path, scene)
            depth_dir = os.path.join(scene_dir, "depth")
            self.depth_files[scene] = os.listdir(depth_dir)  # 000100.png, 000200.png...
            self.depth_files[scene].sort(key=lambda f: int(f.split('.')[0]))
            cam_pose_dir = os.path.join(scene_dir, "pose")
            self.cam_pose_files[scene] = os.listdir(cam_pose_dir)  # 000100.txt, 000200.txt...
            self.cam_pose_files[scene].sort(key=lambda f: int(f.split('.')[0]))
            self.frame_num[scene] = len(self.depth_files[scene])
            self.total_num += self.frame_num[scene]

            for idx in range(self.frame_num[scene]):
                self.frame_idx.append((scene, idx))

    def get_adjacent(self, scene, frameidx, index, both=False):
        if self.frame_num[scene] == 1:
            scene_adj, frameidx_adj = self.frame_idx[index]
        elif frameidx == self.frame_num[scene] - 1:
            if both:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index-1], self.frame_idx[index]])
            else:
                scene_adj, frameidx_adj = self.frame_idx[index-1]
        elif frameidx == 0:
            if both:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index+1], self.frame_idx[index]])
            else:
                scene_adj, frameidx_adj = self.frame_idx[index+1]
        else:
            if both:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index - 1], self.frame_idx[index + 1], self.frame_idx[index]])
            else:
                scene_adj, frameidx_adj = random.choice([self.frame_idx[index-1], self.frame_idx[index+1]])
        return scene_adj, frameidx_adj

    def __len__(self):
        return self.total_num

    def __getitem__(self, index):
        scene, idx = self.frame_idx[index]

        cam_pose_file = self.cam_pose_files[scene][idx]
        cam_pose_file = os.path.join(self.root_path, scene, "pose", cam_pose_file)
        cam_pose = np.loadtxt(cam_pose_file)
        global_translation = cam_pose[:3, 3]

        if self.mode == "temporal":
            scene_adj, idx_adj = self.get_adjacent(scene, idx, index)
            point_set = self.get_point_cloud(scene, idx, global_translation=global_translation)
            point_set_ = self.get_point_cloud(scene_adj, idx_adj, global_translation=global_translation)
        elif self.mode == "spatial":
            point_set = self.get_point_cloud(scene, idx, global_translation=global_translation)
            point_set_ = copy.deepcopy(point_set)
        else:
            assert self.mode == "both"
            scene_adj, idx_adj = self.get_adjacent(scene, idx, index, both=True)
            point_set = self.get_point_cloud(scene, idx, global_translation=global_translation)
            point_set_ = self.get_point_cloud(scene_adj, idx_adj, global_translation=global_translation)

        if self.transforms_1 is not None:
            point_set = self.transforms_1(point_set)
            point_set_ = self.transforms_1(point_set_)

        point_set = point_set.numpy()
        point_set_ = point_set_.numpy()

        if not self.no_height:
            floor = np.percentile(point_set[:, 2], 0.99)
            floor_ = np.percentile(point_set_[:, 2], 0.99)

        if self.transforms_2 is not None:
            point_set = self.transforms_2(point_set)
            point_set_ = self.transforms_2(point_set_)

        point_set = points_sampler(point_set, self.npoints)
        point_set_ = points_sampler(point_set_, self.npoints)

        if not self.no_height:
            height = point_set[:, 2] - floor
            height_ = point_set_[:, 2] - floor_

            height = torch.unsqueeze(height, 1)
            height_ = torch.unsqueeze(height_, 1)
            point_set = torch.cat([point_set, height], 1)
            point_set_ = torch.cat([point_set_, height_], 1)
        return point_set, point_set_

    def get_point_cloud(self, scene, frameidx, global_translation, depth_scale=1000):
        depth_file = self.depth_files[scene][frameidx]
        depth_file = os.path.join(self.root_path, scene, "depth", depth_file)
        cam_pose_file = self.cam_pose_files[scene][frameidx]
        cam_pose_file = os.path.join(self.root_path, scene, "pose", cam_pose_file)
        intrinsics_file = os.path.join(self.root_path, scene, "intrinsics_depth.txt")
        depth_map = np.asarray(Image.open(depth_file))
        # depth_map = cv2.imread(depth_file, cv2.IMREAD_GRAYSCALE)
        cam_pose = np.loadtxt(cam_pose_file)
        # cam_pose[:3, :]
        cam_pose[:3, 3] -= global_translation
        depth_cam_matrix = np.loadtxt(intrinsics_file)

        fx, fy = depth_cam_matrix[0, 0], depth_cam_matrix[1, 1]
        cx, cy = depth_cam_matrix[0, 2], depth_cam_matrix[1, 2]
        h, w = np.mgrid[0:depth_map.shape[0], 0:depth_map.shape[1]]
        z = depth_map / depth_scale
        x = (w - cx) * z / fx
        y = (h - cy) * z / fy
        xyz = np.dstack((x, y, z))
        pad = np.ones((xyz.shape[0], xyz.shape[1], 1))
        xyz = np.concatenate([xyz, pad], 2)
        height, width, _ = xyz.shape
        xyz = xyz.transpose([2, 0, 1])

        xyz = xyz.reshape(4, -1)
        xyz = np.matmul(cam_pose, xyz)
        # xyz = np.matmul(cam_pose_inverse, xyz)
        xyz = xyz.reshape(4, height, width)

        xyz = xyz.transpose([1, 2, 0])

        pc = xyz[:, :, :3]
        pc = pc.reshape(-1, 3)
        return pc


if __name__ == "__main__":
    from torchvision import transforms
    import data_utils as d_utils

    train_transforms_1 = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            # d_utils.PointcloudUpSampling(max_num_points=4096 * 2, centroid="random"),
            # d_utils.PointcloudRandomCutout(p=0.5, min_num_points=4096),
            # d_utils.PointcloudScale(p=1),
            # d_utils.PointcloudRotate(p=1, axis=np.array([0.0, 0.0, 1.0])),
            # d_utils.PointcloudRotatePerturbation(p=1),
            # d_utils.PointcloudTranslate(p=1),
            # d_utils.PointcloudJitter(p=1),

        ]
    )

    train_transforms_2 = transforms.Compose(
        [
            d_utils.PointcloudToTensor(),
            # d_utils.PointcloudRandomCrop(p=0.5, min_num_points=4096),
            # d_utils.PointcloudRandomCutout(p=0.5, min_num_points=4096),
            d_utils.PointcloudRandomInputDropout(p=1),
            d_utils.PointcloudSample(num_pt=100000)
        ]
    )
    # dset = ScanNetFrameContrast(4096, transforms_1=train_transforms_1, transforms_2=train_transforms_2)
    dset = ScanNetFrameWhole(4096, transforms_1=train_transforms_1, transforms_2=train_transforms_2, train=True, root_path='/media/siyuan/2f9b2b54-148c-456e-912c-24c692a0a092/home/siyuan/Dataset/scannet/scans')
    dloader = torch.utils.data.DataLoader(dset, batch_size=1, shuffle=False)

    print(len(dloader))
    for i, data in enumerate(dloader):
        p1, p2 = data
        # write_ply(p1[0].cpu().numpy(), '/home/siyuan/Downloads/test/test1.ply')
        # write_ply(p2[0].cpu().numpy(), '/home/siyuan/Downloads/test/test2.ply')
        print(p1.shape)
        # break

