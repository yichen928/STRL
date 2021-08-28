import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_sched
# from pointnet2_ops.pointnet2_modules import PointnetFPModule, PointnetSAModule
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import transforms
import math

import BYOL.data.data_utils as d_utils
from BYOL.data.ModelNet40Loader import ModelNet40ClsContrast
from BYOL.data.ShapeNetLoader import PartNormalDatasetContrast, WholeNormalDatasetContrast
from BYOL.data.ScanNetLoader import ScannetWholeSceneContrast, ScannetWholeSceneContrastHeight, ScanNetFrameContrast, ScanNetFrameWhole

from BYOL.models.lars_scheduling import LARSWrapper

# from BYOL.models.networks import TargetNetwork, OnlineNetwork
from BYOL.models.networks_dgcnn import TargetNetwork_DGCNN, OnlineNetwork_DGCNN
from BYOL.models.networks_dgcnn_semseg import TargetNetwork_DGCNN_Semseg, OnlineNetwork_DGCNN_Semseg
from BYOL.models.networks_dgcnn_partseg import TargetNetwork_DGCNN_Partseg, OnlineNetwork_DGCNN_Partseg
from BYOL.models.networks_votenet import TargetNetwork_VoteNet, OnlineNetwork_VoteNet

from BYOL.models.networks_PointNet import TargetNetwork_PointNet, OnlineNetwork_PointNet


class BasicalModel(pl.LightningModule):
    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams

        assert self.hparams["network"] in ["DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg", "votenet"]
        if self.hparams["network"] == "DGCNN":
            print("Network: DGCNN\n\n\n")
            self.target_network = TargetNetwork_DGCNN(hparams)
            self.online_network = OnlineNetwork_DGCNN(hparams)
        elif self.hparams["network"] == "PointNet":
            print("Network: PointNet\n\n\n")
            self.target_network = TargetNetwork_PointNet(hparams)
            self.online_network = OnlineNetwork_PointNet(hparams)
        elif self.hparams["network"] == "DGCNN-Semseg":
            print("Network: DGCNN for Semseg")
            self.target_network = TargetNetwork_DGCNN_Semseg(hparams)
            self.online_network = OnlineNetwork_DGCNN_Semseg(hparams)
        elif self.hparams["network"] == "DGCNN-Partseg":
            print("Network: DGCNN for Partseg")
            self.target_network = TargetNetwork_DGCNN_Partseg(hparams)
            self.online_network = OnlineNetwork_DGCNN_Partseg(hparams)
        elif self.hparams["network"] == "votenet":
            print("Network: VoteNet for detection")
            self.target_network = TargetNetwork_VoteNet(hparams)
            self.online_network = OnlineNetwork_VoteNet(hparams)

        self.update_module(self.target_network, self.online_network, decay_rate=0)
        self.tau = self.hparams["decay_rate"]

    def update_module(self, target_module, online_module, decay_rate):
        online_dict = online_module.state_dict()
        target_dict = target_module.state_dict()
        for key in target_dict:
            target_dict[key] = decay_rate * target_dict[key] + (1 - decay_rate) * online_dict[key]
        target_module.load_state_dict(target_dict)

    def forward(self, pointcloud1, pointcloud2):
        r"""
            Forward pass of the network

            Parameters
            ----------
            pointcloud: Variable(torch.cuda.FloatTensor)
                (B, N, 3 + input_channels) tensor
                Point cloud to run predicts on
                Each point in the point-cloud MUST
                be formated as (x, y, z, features...)
        """
        y1_online, z1_online, qz1_online = self.online_network(pointcloud1)
        y2_online, z2_online, qz2_online = self.online_network(pointcloud2)

        with torch.no_grad():
            y1_target, z1_target = self.target_network(pointcloud1)
            y2_target, z2_target = self.target_network(pointcloud2)

        return y1_online, qz1_online, y2_online, qz2_online, y1_target, z1_target, y2_target, z2_target

    def regression_loss(self, x, y):
        norm_x = F.normalize(x, dim=1)
        norm_y = F.normalize(y, dim=1)
        loss = 2 - 2 * (norm_x * norm_y).sum() / x.size(0)
        return loss

    def get_current_decay_rate(self, base_tau):
        tau = 1 - (1 - base_tau) * (math.cos(math.pi * self.global_step / (self.epoch_steps * self.hparams["epochs"])) + 1) / 2
        return tau

    def training_step_end(self, batch_parts_outputs):
        # Add callback for user automatically since it's key to BYOL weight update
        self.tau = self.get_current_decay_rate(self.hparams["decay_rate"])
        self.update_module(self.target_network, self.online_network, decay_rate=self.tau)
        return batch_parts_outputs

    def training_step(self, batch, batch_idx):
        pc_aug1, pc_aug2 = batch

        if self.hparams["network"] in {"DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg"}:
            pc_aug1 = pc_aug1.permute(0, 2, 1)
            pc_aug2 = pc_aug2.permute(0, 2, 1)

        y1_online, qz1_online, y2_online, qz2_online, y1_target, z1_target, y2_target, z2_target \
            = self.forward(pc_aug1, pc_aug2)
        loss = self.regression_loss(qz1_online, z2_target)
        loss += self.regression_loss(qz2_online, z1_target)

        log = dict(train_loss=loss)

        return dict(loss=loss, log=log, progress_bar=dict(train_loss=loss))

    def validation_step(self, batch, batch_idx):
        pc_aug1, pc_aug2 = batch

        if self.hparams["network"] in {"DGCNN", "PointNet", "DGCNN-Semseg", "DGCNN-Partseg"}:
            pc_aug1 = pc_aug1.permute(0, 2, 1)
            pc_aug2 = pc_aug2.permute(0, 2, 1)

        y1_online, qz1_online, y2_online, qz2_online, y1_target, z1_target, y2_target, z2_target \
            = self.forward(pc_aug1, pc_aug2)
        loss = self.regression_loss(qz1_online, z2_target)
        loss += self.regression_loss(qz2_online, z1_target)

        return dict(val_loss=loss)

    def validation_epoch_end(self, outputs):
        reduced_outputs = dict()
        reduced_outputs['val_loss'] = torch.stack([output['val_loss'] for output in outputs]).mean()

        reduced_outputs.update(
            dict(log=reduced_outputs.copy(), progress_bar=reduced_outputs.copy())
        )

        return reduced_outputs

    def configure_optimizers(self):
        if self.hparams["optimizer.type"] == "adam":
            print("Adam optimizer")
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            optimizer = LARSWrapper(optimizer)
        elif self.hparams["optimizer.type"] == "adamw":
            print("AdamW optimizer")
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"]
            )
        elif self.hparams["optimizer.type"] == "sgd":
            print("SGD optimizer")
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
                momentum=0.9
            )
        else:
            print("LARS optimizer")
            base_optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams["optimizer.lr"],
                weight_decay=self.hparams["optimizer.weight_decay"],
            )
            optimizer = LARSWrapper(base_optimizer)

        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.hparams["epochs"], eta_min=0,
                                                                  last_epoch=-1)
        return [optimizer], [lr_scheduler]

    def prepare_data(self):
        train_transforms = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=self.hparams["num_points"] * 2, centroid="random"),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=[0.0, 0.0, 1.0]),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),
                d_utils.PointcloudRandomInputDropout(p=1),
                # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
            ]
        )

        eval_transforms = train_transforms

        train_transforms_scannet_1 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudUpSampling(max_num_points=self.hparams["num_points"] * 2, centroid="random"),
                d_utils.PointcloudNormalize(),
                d_utils.PointcloudScale(p=1),
                # d_utils.PointcloudRotate(p=1, axis=np.array([0.0, 0.0, 1.0])),
                d_utils.PointcloudRotatePerturbation(p=1),
                d_utils.PointcloudTranslate(p=1),
                d_utils.PointcloudJitter(p=1),

            ]
        )

        train_transforms_scannet_2 = transforms.Compose(
            [
                d_utils.PointcloudToTensor(),
                d_utils.PointcloudRandomCrop(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudRandomCutout(p=0.5, min_num_points=self.hparams["num_points"]),
                d_utils.PointcloudRandomInputDropout(p=1),
                # d_utils.PointcloudSample(num_pt=self.hparams["num_points"])
            ]
        )

        eval_transforms_scannet_1 = train_transforms_scannet_1
        eval_transforms_scannet_2 = train_transforms_scannet_2

        if self.hparams["dataset"] == "ModelNet40":
            print("Dataset: ModelNet40")
            self.train_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=train_transforms, train=True
            )

            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False
            )
        elif self.hparams["dataset"] == "ShapeNetPart":
            print("Dataset: ShapeNetPart")
            self.train_dset = PartNormalDatasetContrast(
                self.hparams["num_points"], transforms=train_transforms, split="trainval", normal_channel=True
            )

            self.val_dset = PartNormalDatasetContrast(
                self.hparams["num_points"], transforms=eval_transforms, split="test", normal_channel=True
            )

        elif self.hparams["dataset"] == "ShapeNet":
            print("Dataset: ShapeNet")
            self.train_dset = WholeNormalDatasetContrast(
                self.hparams["num_points"], transforms=train_transforms
            )

            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False, xyz_only=True
            )
        elif self.hparams["dataset"] == "ScanNet":
            print("Dataset: ScanNet")
            self.train_dset = ScannetWholeSceneContrast(
                self.hparams["num_points"], transforms=train_transforms, train=True
            )
            self.val_dset = ModelNet40ClsContrast(
                self.hparams["num_points"], transforms=eval_transforms, train=False, xyz_only=True
            )

        elif self.hparams["dataset"] == "ScanNetFrames":
            print("Dataset: ScanNetFrames")
            self.train_dset = ScanNetFrameContrast(
                self.hparams["num_points"], transforms_1=train_transforms_scannet_1, transforms_2=train_transforms_scannet_2,
                no_height=True, mode=self.hparams["transform_mode"])
            self.val_dset = ScannetWholeSceneContrastHeight(
                self.hparams["num_points"], transforms_1=eval_transforms_scannet_1, transforms_2=eval_transforms_scannet_2, train=False,
                no_height=True)
        elif self.hparams["dataset"] == "ScanNetWhole":
            print("Dataset: ScanNetWhole")
            self.train_dset = ScanNetFrameWhole(
                self.hparams["num_points"], transforms_1=train_transforms_scannet_1, transforms_2=train_transforms_scannet_2,
                no_height=True, train=True, root_path=self.hparams["root_path"], mode=self.hparams["transform_mode"], k=self.hparams["window_length"])
            self.val_dset = ScanNetFrameWhole(
                self.hparams["num_points"], transforms_1=train_transforms_scannet_1, transforms_2=train_transforms_scannet_2,
                no_height=True, train=False, root_path=self.hparams["root_path"], mode=self.hparams["transform_mode"], k=self.hparams["window_length"])

    def _build_dataloader(self, dset, mode, batch_size=None):
        if batch_size is None:
            batch_size = self.hparams["batch_size"]
        return DataLoader(
            dset,
            batch_size=batch_size,
            shuffle=mode == "train",
            num_workers=self.hparams["num_workers"],
            pin_memory=True,
            drop_last=True,
        )

    def train_dataloader(self):
        train_loader = self._build_dataloader(self.train_dset, mode="train")
        self.epoch_steps = len(train_loader)
        return train_loader

    def val_dataloader(self):
        return self._build_dataloader(self.val_dset, mode="val")
