import argparse
import importlib
import os
import json
import torch
from torch.utils.data import Subset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from modules import utils
from modules.datasets import PennFudanDataset

def main():
    """
    Args:
        config (str): configuration name in config.py.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration name in config.py',
                        metavar='str', default='cfg')
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)

    # Create directory

    if not os.path.isdir('results'):
        os.mkdir('results')

    # Load data

    dataset = PennFudanDataset(cfg.path)
    testset = Subset(dataset, range(len(dataset))[-50:])
    testloader = DataLoader(testset, batch_size=1, shuffle=False, num_workers=cfg.num_workers,
                            collate_fn=utils.collate_fn, pin_memory=cfg.pin_memory)

    # Start testing

    maps_list = []
    total_inference_time = 0
    count_inference_time = 0
    for epoch in range(0, cfg.epochs, cfg.checkpoint_epochs):

        # Load checkpoint

        fasterrcnn = fasterrcnn_resnet50_fpn_v2()
        in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
        fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features,
                                                               dataset.get_num_classes())
        fasterrcnn.to(cfg.device)
        pt = f'{cfg.output_name}.pt-{epoch+cfg.checkpoint_epochs}'
        checkpoint = torch.load(f'checkpoints/{pt}', map_location=cfg.device)
        fasterrcnn.load_state_dict(checkpoint['model'])
        fasterrcnn.eval()

        # Test

        preds = []
        targets = []
        print(f'Start testing {pt}...')
        with torch.no_grad():
            for image, target in testloader:
                image = [image[0].to(cfg.device)]
                target =  [{key: value.to(cfg.device) for key, value in target[0].items()}]

                start_event = torch.cuda.Event(enable_timing=True)
                end_event = torch.cuda.Event(enable_timing=True)
                start_event.record()
                output = fasterrcnn(image)
                end_event.record()
                torch.cuda.synchronize()
                inference_time = start_event.elapsed_time(end_event)
                total_inference_time += inference_time
                count_inference_time += 1

                preds = preds + output
                targets = targets + target

        # Calculate mAPs

        metric = MeanAveragePrecision()
        metric.update(preds, targets)
        maps = metric.compute()
        maps_list.append(maps)
        print(f'mAP = {maps["map"]:.3f}')

    # Output json file

    maps_list = [{key: value.item() for key, value in maps.items()} for maps in maps_list]
    results = [{'epoch': cfg.checkpoint_epochs*(i+1), **maps} for i, maps in enumerate(maps_list)]
    json_path = os.path.join('results', f'{cfg.output_name}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f)

    print(f'Inference time = {total_inference_time/count_inference_time:.4f} ms')

if __name__ == '__main__':
    main()
