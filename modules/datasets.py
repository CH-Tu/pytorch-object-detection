import os
import torch
from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.io import read_image
from torchvision.ops.boxes import masks_to_boxes
from torchvision.transforms import v2

class PennFudan(Dataset):
    """PennFudan Dataset.

    Attributes:
        image_names (list): Image names.
        images (list): Images.
        target (list): The dictionaries of ground truth.
            {'boxes' (torchvision.tv_tensors.BoundingBoxes): The bounding boxes for each object in [x_min, y_min, x_max, y_max] format,
             'labels' (torch.Tensor): The labels for each object. There is only one class: pedestrian,
             'masks' (torchvision.tv_tensors.Mask): The segmentation masks for each object}
    """

    def __init__(self, root, train=True, transform=v2.ToDtype(torch.float32, scale=True)):
        """
        Args:
            root (str): The root directory of PennFudan dataset.
            train (bool): If True, return training data; else, return testing data.
            transform (torchvision.transforms.v2): Data transformation.
        """
        self._transform = transform

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
        self.targets = []
        for image_path, mask_path in zip(image_paths, mask_paths):

            # Load image

            image = read_image(image_path)
            image = tv_tensors.Image(image)

            # Load mask

            mask = read_image(mask_path)
            object_ids = torch.unique(mask)
            object_ids = object_ids[1:]
            masks = (mask == object_ids[:, None, None]).to(torch.uint8) # (len(object_ids), height, width)
            masks = tv_tensors.Mask(masks)

            # Get bounding boxes

            boxes = masks_to_boxes(masks)
            boxes = tv_tensors.BoundingBoxes(boxes, format='XYXY', canvas_size=image.shape[-2:])

            # Generate labels

            labels = torch.ones(len(object_ids), dtype=torch.int64)

            # Append data

            target = {'boxes': boxes, 'labels': labels, 'masks': masks}
            self.images.append(image)
            self.targets.append(target)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Input index.

        Returns:
            image (torchvision.tv_tensors.Image): An image.
            target (dict): {'boxes' (torchvision.tv_tensors.BoundingBoxes): The bounding boxes for each object,
                            'labels' (torch.Tensor): The labels for each object,
                            'masks' (torchvision.tv_tensors.Mask): The segmentation masks for each object}.

        Shape:
            image: (C, H, W).
            target: {'boxes': (N, 4),
                     'labels': (N),
                     'masks': (N, H, W)}.
        """
        image, target = self._transform(self.images[idx], self.targets[idx])
        return image, target

    def get_num_classes(self):
        return 2
