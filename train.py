import argparse
import importlib
import os
import time
import datetime
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from modules import utils

def main():
    """
    Args:
        config (str): The configuration name in config.py.
        name (str): The name for output files.
        mask: Use Mask R-CNN.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the configuration name in config.py',
                        metavar='str', required=True)
    parser.add_argument('-n', '--name', help='the name for output files',
                        metavar='str', required=True)
    parser.add_argument('--mask', help='use Mask R-CNN', action='store_true')
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)
    output_name = args.name
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Create directories

    root = os.path.join('results', output_name)
    if not os.path.isdir('results'):
        os.mkdir('results')
    if not os.path.isdir(root):
        os.mkdir(root)

    # Load training data

    print('Load training data...')
    datasets = importlib.import_module('modules.datasets')
    Dataset = getattr(datasets, cfg.dataset)
    trainset = Dataset(**cfg.train_dataset_args)
    trainloader = DataLoader(trainset, batch_size=cfg.train_batch_size, shuffle=True,
                             num_workers=cfg.train_num_workers, collate_fn=utils.collate_fn,
                             pin_memory=cfg.pin_memory, drop_last=True)

    # Load model

    if args.mask:

        # Mask R-CNN

        model = maskrcnn_resnet50_fpn_v2(weights=cfg.pretrained_weights)
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, trainset.get_num_classes())
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
                                                           trainset.get_num_classes())
    else:

        # Faster R-CNN

        model = fasterrcnn_resnet50_fpn_v2(weights=cfg.pretrained_weights)
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, trainset.get_num_classes())
    model.to(device)

    # Training settings

    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=len(trainloader)*cfg.epochs)
    metric = MeanAveragePrecision(iou_type='segm' if args.mask else 'bbox')

    # Start training

    loss_history = []
    map_history = []
    train_start = time.time()
    print(f'Start training {output_name}...')
    for i in range(cfg.epochs):
        print(f'Epoch {i+1}/{cfg.epochs}')
        for j, (images, targets) in enumerate(trainloader):
            images = [image.to(device) for image in images]
            targets =  [{key: value.to(device) for key, value in target.items()} for target in targets]

            # Train

            model.train()
            optimizer.zero_grad()
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Calculate mAP

            model.eval()
            with torch.no_grad():
                preds = model(images)
                if args.mask:
                    preds = [{key: (value > cfg.mask_threshold).squeeze(1) if key == 'masks' else value
                             for key, value in pred.items()} for pred in preds]
                metric.update(preds, targets)
                maps = metric.compute()
                metric.reset()

            # Record loss and mAP

            loss_history.append([value.item() for value in loss_dict.values()])
            map_history.append(maps['map'].item())
            utils.print_loss_and_map(j+1, len(trainloader), loss, maps['map'])
    train_end = time.time()

    # Record training time

    train_time = train_end - train_start
    with open(os.path.join(root, f'{output_name}.train_time.txt'), 'w') as f:
        f.write(str(datetime.timedelta(seconds=train_time)))

    # Save model

    torch.save(model.state_dict(), os.path.join(root, f'{output_name}.pt'))

    # Output plots

    utils.output_plot(os.path.join(root, f'{output_name}.loss'), np.sum(loss_history, axis=1),
                      ylabel='Loss', linewidth=0.5)
    utils.output_plot(os.path.join(root, f'{output_name}.map'), map_history,
                      ylabel='mAP', ylim=[0, 1.05], linewidth=0.5)
    for i, loss_name in enumerate(loss_dict.keys()):
        utils.output_plot(os.path.join(root, f'{output_name}.{loss_name}'), np.array(loss_history)[:, i],
                          ylabel=loss_name, linewidth=0.5)

if __name__ == '__main__':
    main()
