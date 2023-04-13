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
    parser.add_argument('--config', help='configuration name in config.py', metavar='str', default='cfg')
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
    testset = Subset(dataset, range(len(dataset))[-50:])
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            num_workers=cfg.num_workers, collate_fn=utils.collate_fn,
                            pin_memory=cfg.pin_memory)

    # Faster R-CNN

    fasterrcnn = fasterrcnn_resnet50_fpn_v2(weights=cfg.pretrained_weights)
    in_features = fasterrcnn.roi_heads.box_predictor.cls_score.in_features
    fasterrcnn.roi_heads.box_predictor = FastRCNNPredictor(in_features, dataset.get_num_classes())
    fasterrcnn.to(cfg.device)

    # Training settings

    optimizer = optim.Adam(fasterrcnn.parameters(), lr=cfg.lr)

    # Start training

    train_loss_history = []
    test_loss_history = []
    print(f'Start training {cfg.output_name}...')
    for i in range(cfg.epochs):
        print(f'Epoch {i+1}/{cfg.epochs}')

        # Train

        fasterrcnn.train()
        for j, (images, targets) in enumerate(trainloader):
            images = [image.to(cfg.device) for image in images]
            targets =  [{key: value.to(cfg.device) for key, value in target.items()} for target in targets]

            optimizer.zero_grad()
            loss_dict = fasterrcnn(images, targets)
            loss = sum(loss_dict.values())
            loss.backward()
            optimizer.step()

            train_loss_history.append([value.item() for value in loss_dict.values()])
            utils.print_loss(j+1, len(trainloader), loss)

        # Calculate testing loss

        losses = []
        with torch.no_grad():
            for image, target in testloader:
                image = [image[0].to(cfg.device)]
                target =  [{key: value.to(cfg.device) for key, value in target[0].items()}]
                loss_dict = fasterrcnn(image, target)
                losses.append([value.item() for value in loss_dict.values()])
        test_loss_history.append(np.mean(losses, axis=0))
        print(f'Testing loss: {np.sum(test_loss_history[-1]):.4f}')

        # Save checkpoint

        if (i+1)%cfg.checkpoint_epochs == 0:
            torch.save({
                'model': fasterrcnn.state_dict(),
                'optimizer': optimizer.state_dict()
            }, f'checkpoints/{cfg.output_name}.pt-{i+1}')

    # Output loss files

    utils.output_loss(cfg.output_name, np.sum(train_loss_history, axis=1),
                      np.sum(test_loss_history, axis=1), len(trainloader), cfg.epochs)
    for i, loss_name in enumerate(['loss_classifier', 'loss_box_reg', 'loss_objectness', 'loss_rpn_box_reg']):
        utils.output_loss(cfg.output_name, np.array(train_loss_history)[:, i],
                          np.array(test_loss_history)[:, i], len(trainloader), cfg.epochs, loss_name=loss_name)

    # Record training time

    t_end = time.time()
    t = t_end - t_start
    with open(f'checkpoints/{cfg.output_name}.time.txt', 'w') as f:
        f.write(str(datetime.timedelta(seconds=t)))

if __name__ == '__main__':
    main()
