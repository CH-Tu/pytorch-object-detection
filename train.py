import argparse
import importlib
import os
import time
import datetime
import numpy as np
import torch
from torch import optim
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
    t_start = time.time()

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='configuration name in config.py',
                        metavar='str', default='cfg')
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)

    # Create directory

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')
    if not os.path.isdir('plots'):
        os.mkdir('plots')

    # Load data

    dataset = PennFudanDataset(cfg.path)
    trainset = Subset(dataset, range(len(dataset))[:-50])
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
                             num_workers=cfg.num_workers, collate_fn=utils.collate_fn,
                             pin_memory=cfg.pin_memory, drop_last=True)

    # Faster R-CNN

    fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights=cfg.pretrained_weights)
    in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.get_num_classes())
    fasterrcnn.to(cfg.device)

    # Training settings

    optimizer = optim.Adam(fasterrcnn.parameters(), lr=cfg.lr)
    metric = MeanAveragePrecision()

    # Start training

    loss_history = []
    map_history = []
    print(f'Start training {cfg.output_name}...')
    for i in range(cfg.epochs):
        print(f'Epoch {i+1}/{cfg.epochs}')
        for j, (images, targets) in enumerate(trainloader):
            images = [image.to(cfg.device) for image in images]
            targets =  [{key: value.to(cfg.device) for key, value in target.items()}
                        for target in targets]

            # Train

            fasterrcnn.train()
            optimizer.zero_grad()
            loss_dict = fasterrcnn(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            loss_history.append([value.item() for value in loss_dict.values()])
            utils.print_loss(j+1, len(trainloader), loss)

            # Predict

            fasterrcnn.eval()
            with torch.no_grad():
                preds = fasterrcnn(images)
                metric.update(preds, targets)

        # Calculate mAP

        maps = metric.compute()
        map_history.append(maps['map'].item())
        print(f'mAP: {maps["map"]:.3f}')
        metric.reset()

        # Save checkpoint

        if (i+1)%cfg.checkpoint_epochs == 0:
            torch.save({
                'model': fasterrcnn.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'checkpoints/{cfg.output_name}.pt-{i+1}')

    # Output plots

    utils.output_plot(f'{cfg.output_name}.loss', np.sum(loss_history, axis=1), ylabel='loss')
    utils.output_plot(f'{cfg.output_name}.map', np.array(map_history), xlabel= 'epoch',
                      ylabel='mAP', ylim=[0, 1.05])
    for i, loss_name in enumerate(['loss_classifier', 'loss_box_reg',
                                   'loss_objectness', 'loss_rpn_box_reg']):
        utils.output_plot(f'{cfg.output_name}.{loss_name}', np.array(loss_history)[:, i],
                          ylabel=loss_name)

    # Record training time

    t_end = time.time()
    t = t_end - t_start
    with open(f'checkpoints/{cfg.output_name}.time.txt', 'w') as f:
        f.write(str(datetime.timedelta(seconds=t)))

if __name__ == '__main__':
    main()
