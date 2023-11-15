# An Example of Object Detection in PyTorch
An implementation of Faster R-CNN and Mask R-CNN on [Penn-Fudan dataset](https://www.cis.upenn.edu/~jshi/ped_html/).  
[Official Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)  

## Packages
```
torch        2.1.0
torchvision  0.16.0
torchmetrics 1.2.0
matplotlib
easydict
pycocotools
```

## Configuration
```
dataset             the class name of the dataset in datasets.py
root                the root directory of the dataset
transforms          training data transformation
batch_size          batch size
num_workers         how many subprocesses to use for data loading
pin_memory          if True, the data loader will copy Tensors into CUDA pinned memory before returning them
pretrained_weights  the pretrained weights for the model
lr                  learning rate
epochs              training epochs
mask_threshold      the probability threshold for predicted mask
score_threshold     the score threshold for detection
classes             class names
colors              the colors of bounding box and mask

train_dataset_args  the arguments of training dataset
test_dataset_args   the arguments of testing dataset
```

## Train
```shell
python train.py --config penn_fudan --name fasterrcnn      # Faster R-CNN
python train.py --config penn_fudan --name maskrcnn --mask # Mask R-CNN
```

## Test
```shell
python test.py --config penn_fudan --model results\fasterrcnn\fasterrcnn.pt    # Faster R-CNN
python test.py --config penn_fudan --model results\maskrcnn\maskrcnn.pt --mask # Mask R-CNN
```

## Predict
```shell
python predict.py --config penn_fudan --name images_fasterrcnn --model results\fasterrcnn\fasterrcnn.pt --path docs\images  # Faster R-CNN
python predict.py --config penn_fudan --name images_maskrcnn --model results\maskrcnn\maskrcnn.pt --path docs\images --mask # Mask R-CNN
```