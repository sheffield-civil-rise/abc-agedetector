### ABC WP17 Age Detection Base Code

#### Create environment
`conda create env -f environment.yml`

Assumes [CUDA 11.6](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html) and [cudnn libs](https://docs.nvidia.com/deeplearning/cudnn/install-guide/index.html) all installed, but these can be changed by manually installing libs as versioning not hugely important.


#### Data folder format
```
path/to/dataset
|- classes.json
`- images
   |- img0001.jpg
   |- img0002.jpg
   |  ...
   `- img000N.jpg
```

##### Classes format
`classes.json`
```
{'img0001.jpg': 'label0',
 'img0002.jpg': 'label1',
 ...}
```

#### `train.py`
```
usage: train.py [-h] [-e NB_EPOCHS] [-b BATCH_SIZE] [-p NB_PATCHES] [-s PATCH_SIZE] [-c CROP_SIZE]
                [--seed SEED] [-w WEIGHTS] [--backbone {resnet18,alexnet,inception_v3,mobilenet_v2}]
                [--pretrained_backbone] [--fix_backbone] [--cpu] [--debug] [--verbose] [--quiet]
                base_dir out_path

Trains an age detection model

positional arguments:
  base_dir              directory containing images to train/test and class.json
  out_path              path to save model weights to

options:
  -h, --help            show this help message and exit
  -e NB_EPOCHS, --nb_epochs NB_EPOCHS
                        number of epochs for training
  -b BATCH_SIZE, --batch_size BATCH_SIZE
                        minibatch size for training
  -p NB_PATCHES, --nb_patches NB_PATCHES
                        [parameter] number of patches to extract from images
  -s PATCH_SIZE, --patch_size PATCH_SIZE
                        [parameter] patch size [sz X sz]
  -c CROP_SIZE, --crop_size CROP_SIZE
                        [parameter] centre crop size [cs X cs]
  --seed SEED           random seed (for repeatability)
  -w WEIGHTS, --weights WEIGHTS
                        path to weights for using pretrained network
  --backbone {resnet18,alexnet,inception_v3,mobilenet_v2}
                        choice of backbone neural network
  --pretrained_backbone
                        use pretrained weights for backbone
  --fix_backbone        fix weights in backbone (will assume pretrained backbone even if not specified)
  --cpu                 force use of cpu
  --debug               log state debug
  --verbose             log state verbose
  --quiet               log state quiet
```

#### Example
This will train on a dataset at `path/to/dataset` for 100 epochs using a batch size of 10. The main backbone will be resnet18 and it will be initialised with the imagenet pretrained weights, but these will not be fixed during the training. Model weights will be exported to `path/out/model_weights.pt`, which can be used for loading in in future.
```
$ python train.py path/to/dataset path/out/model_weights.pt -e 100 -b 10 --backbone resnet18 --pretrained_backbone
```
