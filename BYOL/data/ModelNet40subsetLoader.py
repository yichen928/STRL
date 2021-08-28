
import os
import sys
import glob
import h5py
import random
import numpy as np
from torch.utils.data import Dataset


def download():
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def load_data(partition):
    download()
    DATA_DIR = os.path.dirname(os.path.abspath(__file__))
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class ModelNet40Subset(Dataset):
<<<<<<< HEAD
    def __init__(self, num_points, train=True, normalize=True, transforms=None, xyz_only=True, percent=1.0):
=======
    def __init__(self, num_points, train=True, normalize=False, transforms=None, xyz_only=False, percent=1.0):
>>>>>>> 4b87dfb36a172807370dae6dff8a62b2892ad944
        partition = "train" if train else "test"
        self.data, self.label = load_data(partition)
        self.num_points = num_points
        self.partition = partition
        self.normalize = normalize
        self.percent = percent
        self.transforms = transforms
        self.xyz_only = xyz_only

        if self.percent < 1:
            self.data, self.label = self.sample_data()

    def sample_data(self):
        data_by_label = {}
        for i in range(len(self.data)):
            label = self.label[i][0]
            if label not in data_by_label:
                data_by_label[label] = [self.data[i]]
            else:
                data_by_label[label].append(self.data[i])
        chosen_data = []
        chosen_label = []
        all_data = []
        all_label = []
        for label in data_by_label:
            idx = list(range(len(data_by_label[label])))
            cidx = np.random.choice(idx)
            chosen_data.append(data_by_label[label][cidx])
            chosen_label.append(label)
            del data_by_label[label][cidx]
            all_data.extend(data_by_label[label])
            all_label.extend([label]*len(data_by_label[label]))
        remain_num = int(round(len(self.data) * self.percent)) - len(chosen_data)
        idx = list(range(len(all_data)))
        cidx = random.sample(idx, remain_num)
        chosen_data = np.array(chosen_data)
        chosen_label = np.array(chosen_label)
        all_data = np.array(all_data)
        all_label = np.array(all_label)
        chosen_data = np.concatenate([chosen_data, all_data[cidx]], 0)
        chosen_label = np.concatenate([chosen_label, all_label[cidx]], 0)
        return chosen_data, chosen_label

    def __getitem__(self, item):
        point_set = self.data[item][:self.num_points]
        label = self.label[item]
        if self.normalize:
            point_set[:, :3] = pc_normalize(point_set[:, :3])

        if self.transforms is not None:
            point_set = self.transforms(point_set)

        if self.xyz_only:
            point_set = point_set[:, :3]

        return point_set, label

    def __len__(self):
        return self.data.shape[0]


if __name__ == '__main__':
    train = ModelNet40Subset(1024, percent=0.01)
    test = ModelNet40Subset(1024, False)
    print(len(train), len(test))
    for data, label in train:
        print(data.shape)
        print(label.shape)
        break
