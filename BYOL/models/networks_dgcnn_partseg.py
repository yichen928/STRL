import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import torch.nn.init as init

import BYOL.data.data_utils as d_utils
from BYOL.data.ModelNet40Loader import ModelNet40Cls


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2,
                    1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature



class Transform_Net(nn.Module):
    def __init__(self, hparams):
        super(Transform_Net, self).__init__()
        self.hparams = hparams
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3*3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)                       # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)                       # (batch_size, 64, num_points, k) -> (batch_size, 128, num_points, k)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 128, num_points, k) -> (batch_size, 128, num_points)

        x = self.conv3(x)                       # (batch_size, 128, num_points) -> (batch_size, 1024, num_points)
        x = x.max(dim=-1, keepdim=False)[0]     # (batch_size, 1024, num_points) -> (batch_size, 1024)

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)     # (batch_size, 1024) -> (batch_size, 512)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)     # (batch_size, 512) -> (batch_size, 256)

        x = self.transform(x)                   # (batch_size, 256) -> (batch_size, 3*3)
        x = x.view(batch_size, 3, 3)            # (batch_size, 3*3) -> (batch_size, 3, 3)

        return x


class DGCNN_partseg(nn.Module):
    def __init__(self, hparams):
        super(DGCNN_partseg, self).__init__()
        self.hparams = hparams
        self.k = self.hparams["k"]
        self.transform_net = Transform_Net(hparams)

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(64)
        self.bn5 = nn.BatchNorm2d(64)
        self.bn6 = nn.BatchNorm1d(self.hparams["emb_dims"])
        self.bn7 = nn.BatchNorm1d(64)
        self.bn8 = nn.BatchNorm1d(256)
        self.bn9 = nn.BatchNorm1d(256)
        self.bn10 = nn.BatchNorm1d(128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, self.hparams["emb_dims"], kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
        #                            self.bn7,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
        #                            self.bn8,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp1 = nn.Dropout(p=args.dropout)
        # self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
        #                            self.bn9,
        #                            nn.LeakyReLU(negative_slope=0.2))
        # self.dp2 = nn.Dropout(p=args.dropout)
        # self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
        #                             self.bn10,
        #                             nn.LeakyReLU(negative_slope=0.2))
        # self.conv11 = nn.Conv1d(128, self.seg_num_all, kernel_size=1, bias=False)

    def forward(self, x):
        batch_size = x.size(0)
        num_points = x.size(2)

        x = x[:, :3, :]
        x0 = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        t = self.transform_net(x0)  # (batch_size, 3, 3)
        x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        x = x.squeeze()
        # l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        # l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        #
        # x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        # x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)
        #
        # x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)
        #
        # x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        # x = self.dp1(x)
        # x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        # x = self.dp2(x)
        # x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        # x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        return x


class TargetNetwork_DGCNN_Partseg(DGCNN_partseg):
    def __init__(self, hparams):
        super(TargetNetwork_DGCNN_Partseg,self).__init__(hparams)
        self.hparams = hparams
        self.emb_dims = hparams["emb_dims"]

        self.build_target_network()

    def build_target_network(self):
        """
            add a projector MLP to original netwrok
        """
        self.projector = nn.Sequential(
            nn.Linear(self.emb_dims, self.hparams["mlp_hidden_size"], bias=False),
            nn.BatchNorm1d(self.hparams["mlp_hidden_size"]),
            nn.ReLU(True),
            nn.Linear(self.hparams["mlp_hidden_size"], self.hparams["projection_size"], bias=False)
        )

    def forward(self, pointcloud):
        """

        :param pointcloud: input point cloud
        :return:
        y: representation
        z: projection
        """
        y = super(TargetNetwork_DGCNN_Partseg, self).forward(pointcloud)
        z = self.projector(y)
        return y, z


class OnlineNetwork_DGCNN_Partseg(TargetNetwork_DGCNN_Partseg):
    def __init__(self, hparams):
        super(OnlineNetwork_DGCNN_Partseg, self).__init__(hparams)
        self.hparams = hparams

        self.build_online_network()

    def build_online_network(self):
        """
            add a predictor MLP to target netwrok
        """
        self.predictor = nn.Sequential(
            nn.Linear(self.hparams["projection_size"], self.hparams["mlp_hidden_size"], bias=False),
            nn.BatchNorm1d(self.hparams["mlp_hidden_size"]),
            nn.ReLU(True),
            nn.Linear(self.hparams["mlp_hidden_size"], self.hparams["projection_size"], bias=False)
        )

    def forward(self, pointcloud):
        """

        :param pointcloud: input point cloud
        :return:
        y: representation
        z: projection
        qz: prediction of target network's projection
        """
        y, z = super(OnlineNetwork_DGCNN_Partseg, self).forward(pointcloud)
        qz = self.predictor(z)
        return y, z, qz



class Projector(nn.Module):
    def __init__(self, hparams):
        super(Projector, self).__init__()

        self.hparams = hparams
        self.emb_dims = hparams["emb_dims"]

        self.projector = nn.Sequential(
            nn.Linear(self.emb_dims, self.hparams["mlp_hidden_size"], bias=False),
            nn.BatchNorm1d(self.hparams["mlp_hidden_size"]),
            nn.ReLU(True),
            nn.Linear(self.hparams["mlp_hidden_size"], self.hparams["projection_size"], bias=False)
        )

    def forward(self, rep):
        """

        :param rep: representation vector
        :return: projection
        """

        proj = self.projector(rep)
        return proj

