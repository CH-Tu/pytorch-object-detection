import argparse
import importlib
import os
import json
import numpy as np
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from modules import utils
from modules.datasets import PennFudanDataset

def main():
    """
    Args:
        config (str): configuration name in config.py.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='configuration name in config.py', metavar='str', default='cfg')
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)

    # Create directory

    pt = f'{cfg.output_name}.pt-{cfg.epochs}'
    dir_path = os.path.join('results', pt)
    if not os.path.isdir(dir_path):
        os.mkdir(dir_path)

    # Load data

    dataset = PennFudanDataset(cfg.path)
    testset = Subset(dataset, range(len(dataset))[-50:])
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=utils.collate_fn,
                            pin_memory=cfg.pin_memory)

    # Load checkpoint

    fasterrcnn = fasterrcnn_resnet50_fpn_v2()
    in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.get_num_classes())
    fasterrcnn.to(cfg.device)
    checkpoint = torch.load(f'checkpoints/{pt}', map_location=cfg.device)
    fasterrcnn.load_state_dict(checkpoint['model'])

    # Start testing

    image_names = np.array(dataset.image_names)[range(len(dataset))[-50:]]
    preds = []
    fasterrcnn.eval()
    with torch.no_grad():
        for (image, target), image_name in zip(testloader, image_names):

            # Test

            image = [image[0].to(cfg.device)]
            target =  [{key: value.to(cfg.device) for key, value in target[0].items()}]

            output = fasterrcnn(image)
            preds = preds + output

            # Output predicted image

            image_path = os.path.join(dir_path, f'{image_name}.pred.jpg')
            image = (image[0]*255).to(torch.uint8)
            utils.output_pred_image(image_path, image, output[0], cfg.classes, cfg.colors,
                                    threshold=cfg.score_threshold)

    # Output json file

    preds = [{key: value.tolist() for key, value in pred.items()} for pred in preds]
    preds = [{'image_name': image_name, **pred} for pred, image_name in zip(preds, image_names)]
    preds_json_path = os.path.join(dir_path, 'preds.json')
    with open(preds_json_path, 'w') as f:
        json.dump(preds, f)

if __name__ == '__main__':
    main()
