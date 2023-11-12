import torch
from easydict import EasyDict
from torchvision.transforms import v2

"""Configuration.

Attributes:
    dataset (str): The class name of the dataset in datasets.py.
    root (str): The root directory of the dataset.
    transforms (torchvision.transforms.v2): Training data transformation.
    batch_size (int): Batch size.
    num_workers (int): How many subprocesses to use for data loading.
    pin_memory (bool): If True, the data loader will copy Tensors into CUDA pinned memory before returning them.
    pretrained_weights (Weights, str): The pretrained weights for the model.
    lr (float): Learning rate.
    epochs (int): Training epochs.
    mask_threshold (float): The probability threshold for predicted mask.
    score_threshold (float): The score threshold for detection.
    classes (list): Class names.
    colors (list): The colors of bounding box and mask.
    
    train_dataset_args (dict): The arguments of training dataset.
    test_dataset_args (dict): The arguments of testing dataset.
"""

cfg = EasyDict()
cfg.dataset = 'PennFudan'
cfg.root = r'C:\Datasets\PennFudanPed'
cfg.transforms = v2.ToDtype(torch.float32, scale=True)
cfg.train_batch_size = 4
cfg.pred_batch_size = 4
cfg.train_num_workers = 1
cfg.test_num_workers = 1
cfg.pred_num_workers = 1
cfg.pin_memory = True
cfg.pretrained_weights = 'DEFAULT'
cfg.lr = 1e-4
cfg.epochs = 20
cfg.mask_threshold = 0.5
cfg.score_threshold = 0.6
cfg.classes = ['pedestrian']
cfg.colors = ['white']

cfg.train_dataset_args = {
    'root': cfg.root,
    'train': True,
    'transform': cfg.transforms
}
cfg.test_dataset_args = {
    'root': cfg.root,
    'train': False
}
