import argparse
import importlib
import os
import json
import torch
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
        model (str): Model pt file.
        mask: Use Mask R-CNN.
    """

    # Import configuration

    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', help='the configuration name in config.py',
                        metavar='str', required=True)
    parser.add_argument('-m', '--model', help='model pt file',
                        metavar='str', required=True)
    parser.add_argument('--mask', help='use Mask R-CNN', action='store_true')
    args = parser.parse_args()
    configs = importlib.import_module('modules.configs')
    cfg = getattr(configs, args.config)
    output_name = os.path.splitext(os.path.basename(args.model))[0]
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

    # Load testing data

    print('Load testing data...')
    datasets = importlib.import_module('modules.datasets')
    Dataset = getattr(datasets, cfg.dataset)
    testset = Dataset(**cfg.test_dataset_args)
    testloader = DataLoader(testset, batch_size=1, shuffle=False,
                            num_workers=cfg.test_num_workers, collate_fn=utils.collate_fn,
                            pin_memory=cfg.pin_memory)

    # Load model

    model_state_dict = torch.load(args.model, map_location=device)
    if args.mask:

        # Mask R-CNN

        model = maskrcnn_resnet50_fpn_v2()
        in_features_box = model.roi_heads.box_predictor.cls_score.in_features
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = model.roi_heads.mask_predictor.conv5_mask.out_channels
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features_box, testset.get_num_classes())
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer,
                                                           testset.get_num_classes())
    else:

        # Faster R-CNN

        model = fasterrcnn_resnet50_fpn_v2()
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, testset.get_num_classes())
    model.to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Start testing

    preds = []
    targets = []
    total_inference_time = 0
    to_uint8 = v2.ToDtype(torch.uint8, scale=True)
    metric = MeanAveragePrecision(iou_type='segm' if args.mask else 'bbox')
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

    torch.save(preds, os.path.join(root_detection, 'preds.pt'))

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
