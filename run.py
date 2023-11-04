import argparse
import importlib
import os
import time
import datetime
import json
import numpy as np
import torch
from torch import optim
from torch.utils.data import DataLoader
from torchvision.io import write_jpeg
from torchvision.utils import draw_bounding_boxes, draw_segmentation_masks
from torchvision.models.detection import fasterrcnn_resnet50_fpn_v2, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2
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
    parser.add_argument('-m', '--mask', help='use Mask R-CNN', action='store_true')
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
    root_detection = os.path.join('detections', output_name)
    if not os.path.isdir('detections'):
        os.mkdir('detections')
    if not os.path.isdir(root_detection):
        os.mkdir(root_detection)

    # Load training data

    print('Load training data...')
    datasets = importlib.import_module('modules.datasets')
    Dataset = getattr(datasets, cfg.dataset)
    trainset = Dataset(**cfg.train_dataset_args)
    trainloader = DataLoader(trainset, batch_size=cfg.batch_size, shuffle=True,
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
        model.to(device)

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

    torch.save(model.state_dict(), os.path.join(root, f'{output_name}.model_dict.pt'))

    # Output plots

    utils.output_plot(os.path.join(root, f'{output_name}.loss'), np.sum(loss_history, axis=1),
                      ylabel='Loss', linewidth=0.5)
    utils.output_plot(os.path.join(root, f'{output_name}.map'), map_history,
                      ylabel='mAP', ylim=[0, 1.05], linewidth=0.5)
    for i, loss_name in enumerate(loss_dict.keys()):
        utils.output_plot(os.path.join(root, f'{output_name}.{loss_name}'), np.array(loss_history)[:, i],
                          ylabel=loss_name, linewidth=0.5)

    # Load testing data

    print('Load testing data...')
    testset = Dataset(**cfg.test_dataset_args)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            num_workers=cfg.test_num_workers, collate_fn=utils.collate_fn,
                            pin_memory=cfg.pin_memory)

    # Start testing

    preds = []
    targets = []
    total_inference_time = 0
    to_uint8 = v2.ToDtype(torch.uint8, scale=True)
    print(f'Start testing {output_name}...')
    with torch.no_grad():
        for (image, target), image_name in zip(testloader, testset.image_names):
            image = [image[0].to(device)]
            target =  [{key: value.to(device) for key, value in target[0].items()}]

            # Test

            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            pred = model(image)
            end_event.record()
            torch.cuda.synchronize()
            inference_time = start_event.elapsed_time(end_event)
            total_inference_time += inference_time

            # Output detection result

            image = to_uint8(image[0])
            for detection in zip(*pred[0].values()):
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

            # Record prediction and target

            pred = [{'image_name': image_name, **pred[0]}]
            preds = preds + pred
            targets = targets + target

    # Save predictions

    torch.save(preds, os.path.join(root, f'{output_name}.preds.pt'))

    # Calculate mAPs

    if args.mask:
        preds = [{key: (value > cfg.mask_threshold).squeeze(1) if key == 'masks' else value
                 for key, value in pred.items()} for pred in preds]
    metric.update(preds, targets)
    maps = metric.compute()

    # Output testing results

    maps = {key: value.item() for key, value in maps.items()}
    with open(os.path.join(root, f'{output_name}.maps.json'), 'w') as f:
        json.dump(maps, f)
    inference_time = total_inference_time / len(testloader)
    with open(os.path.join(root, f'{output_name}.infer_time.txt'), 'w') as f:
        f.write(f'{inference_time:.3f} ms')
    print(f'mAP = {maps["map"]:.3f}')
    print(f'Inference time: {inference_time:.3f} ms')

if __name__ == '__main__':
    main()
