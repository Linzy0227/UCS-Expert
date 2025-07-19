# UCS-Expert(We will release the complete code after the paper is accepted.)
This is the official repository for UCS-Expert: High-Quality Segmentation for Any Underwater Coral Imagery.

## We provide a simple tool for interactively segment coral images
<img src="sample/ucs.gif" alt="UCS-Expert Demo" width="600">

### Enjoy various beautiful corals
<img src="sample/fig1.png" alt="Various beautiful corals" width="600">

## Getting Started

Download the [model checkpoint](https://drive.google.com/xxx) and place it at e.g., `work_dir/UCS-Expert/ucs_vit_b`

### Environmental Setups
Our code is developed on Ubuntu 20.04 using Python 3.8 and torch=2.5+cu121.

```bash

git clone https://github.com/Linzy0227/UCS-Expert.git
cd UCS-Expert
conda env create -f environment.yml
conda activate ucs
```

We provide two ways to quickly test the model on your images

1. Command line

```bash
bash test.sh # segment the demo image
```

2. GUI

Install `PyQt5` with [pip](https://pypi.org/project/PyQt5/): `pip install PyQt5 ` or [conda](https://anaconda.org/anaconda/pyqt): `conda install -c anaconda pyqt`

```bash
python gui.py
```

Load the image to the GUI and specify segmentation targets by drawing bounding boxes.


### Preparing Dataset and checkpoint of SAM
To validate the performance of coral segmentation, we have provided the [coralscape](https://huggingface.co/datasets/EPFL-ECEO/coralscapes), [corals](https://github.com/YcShentu/CoralSegmentation) and [coralmask](https://docs.google.com/forms/d/e/1FAIpQLSc8qHFBwhsJS_46hqS42NHN-3OqD5GSwvv4Sb36njdrb3LI7g/viewform) datasets. 
Please create a new folder named "dataset", download and unzip the three datasets into the folder.

```bash
mkdir dataset
# download and move the zip files into the folder
```
Please donwload ViT-B SAM checkpoint from this [link](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) and place it into the "sam_ckp" folder.
```bash
mkdir sam_ckp
cd sam_ckp
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```
Finally the file structure is organized as:
```
UCS-Expert
├── dataset
│   ├── CoralMask
│   |   ├── train
│   |   |   ├── images
│   |   |   ├── masks
│   |   ├── test
│   |   |   ├── images
│   |   |   ├── masks
│   ├── CoralS
│   |   ├── test
│   |   |   ├── images
│   |   |   ├── masks
│   ├── CoralScape
│   |   ├── test
│   |   |   ├── images
│   |   |   ├── masks
├── sam_ckp
│   ├── sam_vit_b_01ec64.pth
└── other codes...
```

### Training

```bash
# for single GPU
bash train.sh
```
Then you can find checkpoints and training logs into the folder "work_dir".


## Acknowledgements
- We highly appreciate all the dataset owners for providing the public dataset to the community.
- We thank Meta AI for making the source code of [segment anything](https://github.com/facebookresearch/segment-anything) publicly available.


## Reference

```

```
