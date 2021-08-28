import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

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


class DGCNN(nn.Module):
    def __init__(self, hparams):
        super(DGCNN, self).__init__()
        self.hparams = hparams
        self.k = hparams["k"]
        self.emb_dims = hparams["emb_dims"]
        self.dropout_rate = hparams["dropout"]

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(self.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128 * 2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, self.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        # self.linear1 = nn.Linear(self.emb_dims * 2, 512, bias=False)
        # self.bn6 = nn.BatchNorm1d(512)
        # self.dp1 = nn.Dropout(p=self.dropout_rate)
        # self.linear2 = nn.Linear(512, 256)
        # self.bn7 = nn.BatchNorm1d(256)
        # self.dp2 = nn.Dropout(p=self.dropout_rate)
        # self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)
        x = get_graph_feature(x, k=self.k)
        x = self.conv1(x)
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)
        x = self.conv2(x)
        x2 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x2, k=self.k)
        x = self.conv3(x)
        x3 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x3, k=self.k)
        x = self.conv4(x)
        x4 = x.max(dim=-1, keepdim=False)[0]

        x = torch.cat((x1, x2, x3, x4), dim=1)

        x = self.conv5(x)
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        x = torch.cat((x1, x2), 1)

        # x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)
        # x = self.dp1(x)
        # x = self.linear2(x)
        # x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2)
        # x = self.dp2(x)
        # x = self.linear3(x)
        return x


class TargetNetwork_DGCNN(DGCNN):
    def __init__(self, hparams):
        super(TargetNetwork_DGCNN,self).__init__(hparams)
        self.hparams = hparams
        self.emb_dims = hparams["emb_dims"]

        self.build_target_network()

    def build_target_network(self):
        """
            add a projector MLP to original netwrok
        """
        self.projector = nn.Sequential(
            nn.Linear(self.emb_dims*2, self.hparams["mlp_hidden_size"], bias=False),
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
        y = super(TargetNetwork_DGCNN, self).forward(pointcloud)
        z = self.projector(y)
        return y, z


class OnlineNetwork_DGCNN(TargetNetwork_DGCNN):
    def __init__(self, hparams):
        super(OnlineNetwork_DGCNN, self).__init__(hparams)
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
        y, z = super(OnlineNetwork_DGCNN, self).forward(pointcloud)
        qz = self.predictor(z)
        return y, z, qz
