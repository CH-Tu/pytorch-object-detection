import torch
from easydict import EasyDict

"""Configuration.

Attributes:
    output_name (str): Name for output checkpoints, loss files and loss plots.
    path (str): Path of the dataset.
    batch_size (int): Batch size.
    num_workers (int): How many subprocesses to use for data loading.
    pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    pretrained_weights (Weights, str): Pretrained Weights.
    lr (float): Learning rate.
    epochs (int): Training epochs.
    checkpoint_epochs (int): How many epochs to save a checkpoint.
    device (torch.device): The device on which tensor will be allocated.
    classes (list): Class names.
    colors (list): Bounding box colors.
    score_threshold (float): Score threshold.
"""

# Demo

cfg = EasyDict()
cfg.output_name = 'fasterrcnn'
cfg.path = r'C:\Datasets\PennFudanPed'
cfg.batch_size = 4
cfg.num_workers = 1
cfg.pin_memory = True
cfg.pretrained_weights = 'DEFAULT'
cfg.lr = 0.00001
cfg.epochs = 100
cfg.checkpoint_epochs = 10
cfg.device = torch.device('cuda')
cfg.classes = ['pedestrian']
cfg.colors = ['white']
cfg.score_threshold = 0.6
