import os
import torch
from torch.utils.data import Dataset
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2

class PennFudan(Dataset):
    """PennFudan Dataset.

    Attributes:
        image_names (list): Image names.
        images (list): Images.
        masks (list): The segmentation masks for each object.
        boxes (list): The bounding boxes for each object in [x_min, y_min, x_max, y_max] format.
        labels (list): The labels for each object. There is only one class: pedestrian.
    """

    def __init__(self, root, train=True, transform=v2.ToDtype(torch.float32, scale=True)):
        """
        Args:
            root (str): The root directory of PennFudan dataset.
            train (bool): If True, return training data; else, return testing data.
            transform (torchvision.transforms): Image transformation. (not implemented)
        """

        # Get the paths of all images and masks

        if train:
            image_names = os.listdir(os.path.join(root, 'PNGImages'))[:-50]
        else:
            image_names = os.listdir(os.path.join(root, 'PNGImages'))[-50:]
        image_paths = [os.path.join(root, 'PNGImages', name) for name in image_names]
        mask_names = [f'{os.path.splitext(name)[0]}_mask.png' for name in image_names]
        mask_paths = [os.path.join(root, 'PedMasks', name) for name in mask_names]
        self.image_names = [os.path.splitext(name)[0] for name in image_names]

        # Load data

        self.images = []
        self.masks = []
        self.boxes = []
        self.labels = []
        for image_path, mask_path in zip(image_paths, mask_paths):

            # Load image

            image = read_image(image_path)
            image = transform(image)
            self.images.append(image)

            # Load mask

            mask = read_image(mask_path)
            object_ids = torch.unique(mask)
            object_ids = object_ids[1:]
            masks = (mask == object_ids[:, None, None]).to(torch.uint8) # (len(object_ids), height, width)
            self.masks.append(masks)

            # Get bounding boxes

            boxes = masks_to_boxes(masks)
            self.boxes.append(boxes)

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
            target (dict): {'masks' (torch.Tensor): The segmentation masks for each object,
                            'boxes' (torch.Tensor): The bounding boxes for each object,
                            'labels' (torch.Tensor): The labels for each object}.

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
