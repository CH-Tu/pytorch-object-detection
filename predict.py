import argparse
import importlib
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image, write_jpeg
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2

class Images(Dataset):
    """Load the images in the directory.

    Attributes:
        image_names (list): Image names.
        images (list): Images.
    """

    def __init__(self, root):
        """
        Args:
            root (str): The directory of images.
        """

        # Get the paths of all images

        image_names = os.listdir(root)
        image_paths = [os.path.join(root, name) for name in image_names]
        self.image_names = [os.path.splitext(name)[0] for name in image_names]

        # Load images

        to_float = v2.ToDtype(torch.float32, scale=True)
        self.images = []
        for image_path in image_paths:
            image = read_image(image_path)
            image = to_float(image)
            self.images.append(image)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """
        Args:
            idx (int): Input index.

        Returns:
            image (torch.Tensor): An image.

        Shape:
            image: (C, H, W).
        """
        image = self.images[idx]
        return image

def collate_fn(batch):
    """Customized collate_fn for dataloader.

    Args:
        batch (list): [dataset[0], dataset[1], ...].
    """
    return batch

def main():
    """
    Args:
        config (str): The configuration name in config.py.
        name (str): Output name.
        model (str): Model pt file.
        path (str): The directory of images to be detected.
        mask: Use Mask R-CNN.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the configuration name in config.py',
                        metavar='str', required=True)
    parser.add_argument('-n', '--name', help='output name',
                        metavar='str', required=True)
    parser.add_argument('-m', '--model', help='model pt file',
                        metavar='str', required=True)
    parser.add_argument('-p', '--path', help='the directory of images to be detected',
                        metavar='str', required=True)
    parser.add_argument('--mask', help='use Mask R-CNN', action='store_true')
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directory

    root_detection = os.path.join('detections', args.name)
    if not os.path.isdir('detections'):
        os.mkdir('detections')
    if not os.path.isdir(root_detection):
        os.mkdir(root_detection)

    # Load images

    print('Load the images...')
    datasets = Images(args.path)
    dataloader = DataLoader(datasets, batch_size=cfg.pred_batch_size, shuffle=False,
                            num_workers=cfg.pred_num_workers, collate_fn=collate_fn,
                            pin_memory=cfg.pin_memory)

    # Load model

    model_state_dict = torch.load(args.model, map_location=device)
    out_features = model_state_dict['roi_heads.box_predictor.cls_score.weight'].shape[0]
    if args.mask:

        # Mask R-CNN

        model = maskrcnn_resnet50_fpn_v2()
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, out_features)
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, out_features)
    else:

        # Faster R-CNN

        model = fasterrcnn_resnet50_fpn_v2()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, out_features)
    model.to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Predict

    preds = []
    print(f'Start predicting {args.name}...')
    with torch.no_grad():
        for images in dataloader:
            images = [image.to(device) for image in images]
            predictions = model(images)
            preds = preds + predictions

    # Output detection results

    to_uint8 = v2.ToDtype(torch.uint8, scale=True)
    for image, image_name, pred in zip(datasets.images, datasets.image_names, preds):
        image = to_uint8(image)
        for detection in zip(*pred.values()):
            box = detection[0].unsqueeze(0)
            label = detection[1]
            score = detection[2]
            if score > cfg.score_threshold:
                image = draw_bounding_boxes(image, box, labels=[cfg.classes[label-1]],
                                            colors=[cfg.colors[label-1]])
                if args.mask:
                    mask = detection[3] > cfg.mask_threshold
                    image = draw_segmentation_masks(image, mask, colors=[cfg.colors[label-1]])
        write_jpeg(image, os.path.join(root_detection, f'{image_name}.jpg'))

    # Save predictions

    preds = [{'image_name': image_name, **pred} for pred, image_name in zip(preds, datasets.image_names)]
    torch.save(preds, os.path.join(root_detection, 'preds.pt'))

if __name__ == '__main__':
    main()
