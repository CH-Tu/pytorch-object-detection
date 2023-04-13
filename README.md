# An Example of Object Detection in PyTorch
An implementation of Faster R-CNN on PennFudan Dataset.  
[Official Tutorial](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html)


## Quick Start

1. Download code

2. Download [dataset](https://www.cis.upenn.edu/~jshi/ped_html/)

3. Install packages
```
torch
torchvision
torchmetrics
matplotlib
easydict
```

4. Modify [configs.py](https://github.com/CH-Tu/pytorch-object-detection/blob/main/modules/configs.py)

5. Run
```shell
python train.py --config cfg      # train
python test.py --config cfg       # test
python visualize.py --config cfg  # visualize testing results
```
