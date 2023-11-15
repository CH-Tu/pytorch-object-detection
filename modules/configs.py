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

penn_fudan = EasyDict()
penn_fudan.dataset = 'PennFudan'
penn_fudan.root = r'C:\Datasets\PennFudanPed'
penn_fudan.transforms = v2.Compose([
    v2.ToDtype(torch.float32, scale=True)
])
penn_fudan.train_batch_size = 4
penn_fudan.pred_batch_size = 4
penn_fudan.train_num_workers = 1
penn_fudan.test_num_workers = 1
penn_fudan.pred_num_workers = 1
penn_fudan.pin_memory = True
penn_fudan.pretrained_weights = 'DEFAULT'
penn_fudan.lr = 1e-4
penn_fudan.epochs = 20
penn_fudan.mask_threshold = 0.5
penn_fudan.score_threshold = 0.6
penn_fudan.classes = ['pedestrian']
penn_fudan.colors = ['white']

penn_fudan.train_dataset_args = {
    'root': penn_fudan.root,
    'train': True,
    'transform': penn_fudan.transforms
}
penn_fudan.test_dataset_args = {
    'root': penn_fudan.root,
    'train': False
}
