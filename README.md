# IAA: Intra-class Adaptive Augmentation with Neighbor Correctionfor Deep Metric Learning

The PyTorch implementation of our T-MM 2022 paper.



## Requirements

- torch 
- torchvision 
- tqdm
- numpy
- scipy
- Pillow
- matplotlib



## Preparing Datasets

1. Download these datasets.
   - [CUB-200-2011](http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz)
   - Cars-196 ([Img](http://imagenet.stanford.edu/internal/car196/car_ims.tgz), [Annotation](http://imagenet.stanford.edu/internal/car196/cars_annos.mat))
   - [Stanford Online Products](ftp://cs.stanford.edu/cs/cvgl/Stanford_Online_Products.zip)
   - [In-Shop Clothes Retrieval](https://drive.google.com/open?id=0B7EVK8r0v71pQ2FuZ0k0QnhBQnc)

2. Extract the compressed file (tgz or zip) into `MyDataset/`, e.g., for Cars-196, put the files in the `MyDataset/Cars196`. Other naming ways can see the python files of realted datatsets in  `src/utils/dataset`, or modify these names in your way.



## Training

The folder `MyDataset` is the root path of datasets using in this paper (including CUB, CARS, SOP), which can be customized by the argparse parameter `--data`. 

The folder `./results` is the log path to record corresponding models and results of training, which can be customized by the argparse parameter `--save-dir`. 

The folder `./weights_models` is the path to put the weighting parameters of pretrained backbone networks, which can be customized by the argparse parameter `--weight_path`. 


### CUB-200-2011
python run.py --dataset cub200 --backbone googlenet --loss MS --intra_lamda 0.8 --aug_num 3
python run.py --dataset cub200 --backbone googlenet --loss Contrastive  --lr 3e-5  --intra_lamda 0.8 --aug_num 3


### Cars-196
python run.py --dataset cars196 --backbone googlenet --loss MS --intra_lamda 0.8 --aug_num 3
python run.py --dataset cars196 --backbone googlenet --loss Contrastive --intra_lamda 0.8 --aug_num 3


### Stanford Online Products
python run.py --dataset stanford --backbone googlenet --batch 180 --lr 1e-4 --loss MS --intra_lamda 0.6 --aug_num 3
python run.py --dataset stanford --backbone googlenet --batch 180 --lr 1e-4 --loss Contrastive --intra_lamda 0.6 --aug_num 2

python run.py --dataset stanford --backbone bninception --batch 256 --lr 1e-4 --loss MS -intra_lamda 0.5 --aug_num 3
python run.py --dataset stanford --backbone resnet50 --batch 256 --lr 1e-4 --loss MS -intra_lamda 0.5 --aug_num 3


### In-Shop Clothes Retrieval
python run.py --dataset inshop --backbone googlenet --batch 180 --lr 1e-4 --loss MS --intra_lamda 0.6 --aug_num 3
python run.py --dataset inshop --backbone googlenet --batch 180 --lr 1e-4 --loss Contrastive --intra_lamda 0.6 --aug_num 2

python run.py --dataset inshop --backbone bninception --batch 256 --lr 1e-4 --loss MS --intra_lamda 0.5 --aug_num 3
python run.py --dataset inshop --backbone resnet50 --batch 256 --lr 1e-4 --loss MS --intra_lamda 0.5 --aug_num 3
