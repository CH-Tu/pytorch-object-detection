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

## Train and Test
```shell
python run.py --config cfg --name fasterrcnn      # Faster R-CNN
python run.py --config cfg --name maskrcnn --mask # Mask R-CNN
```
