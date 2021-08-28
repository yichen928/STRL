import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms

import BYOL.data.data_utils as d_utils
from BYOL.data.ModelNet40Loader import ModelNet40Cls

import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)

sys.path.append(os.path.join(ROOT_DIR,  "pointnet2_vote"))
from pointnet2_modules import PointnetFPModule, PointnetSAModuleVotes, PointnetSAModule


class VoteNet(nn.Module):
    def __init__(self, hparams):
        super(VoteNet, self).__init__()
        self.hparams = hparams

        self._build_model_wo_cls()

    def _build_model_wo_cls(self):
        if self.hparams["with_height"]:
            input_feature_dim = 1
        else:
            input_feature_dim = 0

        self.sa1 = PointnetSAModuleVotes(
                npoint=2048,
                radius=0.2,
                nsample=64,
                mlp=[input_feature_dim, 64, 64, 128],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa2 = PointnetSAModuleVotes(
                npoint=1024,
                radius=0.4,
                nsample=32,
                mlp=[128, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa3 = PointnetSAModuleVotes(
                npoint=512,
                radius=0.8,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.sa4 = PointnetSAModuleVotes(
                npoint=256,
                radius=1.2,
                nsample=16,
                mlp=[256, 128, 128, 256],
                use_xyz=True,
                normalize_xyz=True
            )

        self.fp1 = PointnetFPModule(mlp=[256+256,256,256])
        self.fp2 = PointnetFPModule(mlp=[256+256,256,256])

        self.max_pooling = nn.MaxPool1d(kernel_size=1024)

        # self.sa_aggre = PointnetSAModule(
        #         mlp=[256, 256, 512, 1024],
        #         use_xyz=True,
        #     )
        #
        #
        # self.fc_layer = nn.Sequential(
        #     nn.Linear(1024, 512, bias=False),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(True),
        #     nn.Linear(512, 256, bias=False),
        #     # nn.BatchNorm1d(256),
        #     # nn.ReLU(True),
        #     # nn.Dropout(0.5),
        #     # nn.Linear(256, 40),
        # )

    # def forward(self, pointcloud):
    #     r"""
    #         Forward pass of the network
    #
    #         Parameters
    #         ----------
    #         pointcloud: Variable(torch.cuda.FloatTensor)
    #             (B, N, 3 + input_channels) tensor
    #             Point cloud to run predicts on
    #             Each point in the point-cloud MUST
    #             be formated as (x, y, z, features...)
    #     """
    #     xyz, features = self._break_up_pc(pointcloud)
    #
    #     xyz, features, _ = self.sa1(xyz, features)
    #
    #     xyz, features, _ = self.sa2(xyz, features)
    #
    #     xyz, features, _ = self.sa3(xyz, features)
    #
    #     xyz, features, _ = self.sa4(xyz, features)
    #
    #     xyz, features = self.sa_aggre(xyz, features)
    #
    #     return self.fc_layer(features.squeeze(-1))


    def forward(self, pointcloud, end_points=None):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_feature_dim) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)

            Returns
            ----------
            end_points: {XXX_xyz, XXX_features, XXX_inds}
                XXX_xyz: float32 Tensor of shape (B,K,3)
                XXX_features: float32 Tensor of shape (B,K,D)
                XXX-inds: int64 Tensor of shape (B,K) values in [0,N-1]
        """
        if not end_points:
            end_points = {}
        batch_size = pointcloud.shape[0]

        xyz, features = self._break_up_pc(pointcloud)

        # --------- 4 SET ABSTRACTION LAYERS ---------
        xyz, features, fps_inds = self.sa1(xyz, features)
        end_points['sa1_inds'] = fps_inds
        end_points['sa1_xyz'] = xyz
        end_points['sa1_features'] = features

        xyz, features, fps_inds = self.sa2(xyz, features) # this fps_inds is just 0,1,...,1023
        end_points['sa2_inds'] = fps_inds
        end_points['sa2_xyz'] = xyz
        end_points['sa2_features'] = features

        xyz, features, fps_inds = self.sa3(xyz, features) # this fps_inds is just 0,1,...,511
        end_points['sa3_xyz'] = xyz
        end_points['sa3_features'] = features

        xyz, features, fps_inds = self.sa4(xyz, features) # this fps_inds is just 0,1,...,255
        end_points['sa4_xyz'] = xyz
        end_points['sa4_features'] = features

        # --------- 2 FEATURE UPSAMPLING LAYERS --------
        features = self.fp1(end_points['sa3_xyz'], end_points['sa4_xyz'], end_points['sa3_features'], end_points['sa4_features'])
        features = self.fp2(end_points['sa2_xyz'], end_points['sa3_xyz'], end_points['sa2_features'], features)
        # end_points['fp2_features'] = features
        # end_points['fp2_xyz'] = end_points['sa2_xyz']
        # num_seed = end_points['fp2_xyz'].shape[1]
        # end_points['fp2_inds'] = end_points['sa1_inds'][:,0:num_seed] # indices among the entire input point clouds

        features_pooling = self.max_pooling(features)
        features_pooling = features_pooling.reshape(features_pooling.shape[0], features_pooling.shape[1])

        return features_pooling

    def _break_up_pc(self, pc):
        xyz = pc[..., 0:3].contiguous()
        features = pc[..., 3:].transpose(1, 2).contiguous() if pc.size(-1) > 3 else None

        return xyz, features


class TargetNetwork_VoteNet(VoteNet):
    def __init__(self, hparams):
        super(TargetNetwork_VoteNet,self).__init__(hparams)
        self.hparams = hparams

        self.build_target_network()

    def build_target_network(self):
        """
            add a projector MLP to original netwrok
        """
        self.projector = nn.Sequential(
            nn.Linear(256, self.hparams["mlp_hidden_size"], bias=False),
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
        y = super(TargetNetwork_VoteNet, self).forward(pointcloud)
        z = self.projector(y)
        return y, z


class OnlineNetwork_VoteNet(TargetNetwork_VoteNet):
    def __init__(self, hparams):
        super(OnlineNetwork_VoteNet, self).__init__(hparams)
        self.hparams = hparams

        self.build_online_network()

    def build_online_network(self):
        """
            add a predictor MLP to target netwrok
        """
        self.predictor = nn.Sequential(
            nn.Linear(256, self.hparams["mlp_hidden_size"], bias=False),
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
        y, z = super(OnlineNetwork_VoteNet, self).forward(pointcloud)
        qz = self.predictor(z)
        return y, z, qz

