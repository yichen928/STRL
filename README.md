# Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds



This is the official code implementation for the paper "Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds" (ICCV 2021) [paper](https://arxiv.org/abs/2109.00179)

## Checklist

### Self-supervised Pre-training Framework

+ [x] BYOL
+ [ ] SimCLR

### Downstream Tasks

+ [x] Shape Classification
+ [x] Semantic Segmentation
+ [x] Indoor Object Detection
+ [ ] Outdoor Object Detection

## Installation 

The code was tested with the following environment: Ubuntu 18.04, python 3.7, pytorch 1.7.1, torchvision 0.8.2 and CUDA 11.1.

For self-supervised pre-training, run the following command:

```
git clone https://github.com/yichen928/STRL.git
cd STRL
pip install -r requirements.txt
```

For downstream tasks, please refer to the `Downstream Tasks` section.

## Datasets

Please download the used dataset with the following links:

+ ShapeNet: https://drive.google.com/uc?id=1sJd5bdCg9eOo3-FYtchUVlwDgpVdsbXB

+ ModelNet40: https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip
+ ScanNet (subset): Please follow the instruction in their official [website](http://www.scan-net.org/). The 25k frames subset is enough for our model.

Make sure to put the files in the following structure:

```
|-- ROOT
|	|-- BYOL
|		|-- data
|			|-- modelnet40_normal_resampled_cache
|			|-- shapenet57448xyzonly.npz
|			|-- scannet
|				|-- scannet_frames_25k
```

## Pre-training

### BYOL framework

Please run the following command:

```
python BYOL/train.py
```

You need to edit the config file `BYOL/config/config.yaml` to switch different backbone architectures (currently including `BYOL-pointnet-cls, BYOL-dgcnn-cls, BYOL-dgcnn-semseg, BYOL-votenet-detection`).

### Pre-trained Models

You can find the checkpoints of the pre-training and downstream tasks in our [Google Drive](https://drive.google.com/drive/folders/1uip_uZtyoVdTbwUM4QUZmEcXROpV2-8n?usp=sharing).

## Linear Evaluation

For PointNet or DGCNN classification backbones, you may evaluate the learnt representation with linear SVM classifier by running the following command:

For PointNet:

```
python BYOL/evaluate_pointnet.py -w /path/to/your/pre-trained/checkpoints
```

For DGCNN:

```
python BYOL/evaluate_dgcnn.py -w /path/to/your/pre-trained/checkpoints
```

## Downstream Tasks

### Checkpoints Transformation

You can transform the pre-trained checkpoints to different downstream tasks by running:

For VoteNet:

```
python BYOL/transform_ckpt_votenet.py --input_path /path/to/your/pre-trained/checkpoints --output_path /path/to/the/transformed/checkpoints
```

For other backbones:

```
python BYOL/transform_ckpt.py --input_path /path/to/your/pre-trained/checkpoints --output_path /path/to/the/transformed/checkpoints
```

### Fine-tuning and Evaluation for Downstream Tasks

For the fine-tuning and evaluation of downstream tasks, please refer to other corresponding repos. We sincerely thank all these authors for their nice work!

+ Classification: [WangYueFt/dgcnn](https://github.com/WangYueFt/dgcnn)
+ Semantic Segmentation: [AnTao97/*dgcnn*.pytorch](https://github.com/AnTao97/dgcnn.pytorch)
+ Indoor Object Detection: [facebookresearch/*votenet*](https://github.com/facebookresearch/votenet)

## Citation

If you found our paper or code useful for your research, please cite the following paper:

```
@article{huang2021spatio,
  title={Spatio-temporal Self-Supervised Representation Learning for 3D Point Clouds},
  author={Huang, Siyuan and Xie, Yichen and Zhu, Song-Chun and Zhu, Yixin},
  journal={arXiv preprint arXiv:2109.00179},
  year={2021}
}
```
