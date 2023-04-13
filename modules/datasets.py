import os
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class PennFudanDataset(Dataset):
    """PennFudan Dataset.

    Attributes:
        image_names (list): Image names.
        images (list): Images.
        masks (list): Segmentation masks for each objects.
        boxes (list): Bounding boxes for each objects in [x_min, y_min, x_max, y_max] format.
        labels (list): Labels for each objects. There is only one class: pedestrian.
    """

    def __init__(self, path, transform=transforms.ToTensor()):
        """
        Args:
            path (str): Path of PennFudan dataset.
            transform (torchvision.transforms): Transform for image.
        """

        # Get the paths of all images and masks

        image_names = os.listdir(os.path.join(path, 'PNGImages'))
        image_paths = [os.path.join(path, 'PNGImages', name) for name in image_names]
        mask_names = [f'{os.path.splitext(name)[0]}_mask.png' for name in image_names]
        mask_paths = [os.path.join(path, 'PedMasks', name) for name in mask_names]
        self.image_names = [os.path.splitext(name)[0] for name in image_names]

        # Load data

        self.images = []
        self.masks = []
        self.boxes = []
        self.labels = []
        for image_path, mask_path in zip(image_paths, mask_paths):

            # Load image

            image = Image.open(image_path).convert('RGB')
            image = transform(image)
            self.images.append(image)

            # Load mask

            mask = np.array(Image.open(mask_path))
            object_ids = np.unique(mask)
            object_ids = object_ids[1:]
            masks = mask == object_ids[:, None, None] # (len(object_ids), height, width)
            self.masks.append(torch.tensor(masks, dtype=torch.uint8))

            # Calculate bounding boxes

            boxes = []
            for mask in masks:
                y, x = np.where(mask)
                boxes.append([np.min(x), np.min(y), np.max(x), np.max(y)])
            self.boxes.append(torch.tensor(boxes, dtype=torch.float32))

            # Generate labels

            self.labels.append(torch.ones(len(object_ids), dtype=torch.int64))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Input index.

        Returns:
            image (torch.Tensor): An image.
            target (dict): {'masks' (torch.Tensor): Segmentation masks for each objects,
                            'boxes' (torch.Tensor): Bounding boxes for each objects,
                            'labels' (torch.Tensor): Labels for each objects}.

        Shape:
            image: (C, H, W).
            target: {'masks': (N, H, W),
                     'boxes': (N, 4),
                     'labels': (N)}.
        """
        image = self.images[idx]
        target = {'masks': self.masks[idx], 'boxes': self.boxes[idx], 'labels': self.labels[idx]}
        return image, target

    def get_num_classes(self):
        return 2
