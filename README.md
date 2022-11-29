# IAA: Intra-class Adaptive Augmentation with Neighbor Correctionfor Deep Metric Learning

The PyTorch codes for IAA described in the paper "[Intra-class Adaptive Augmentation with Neighbor Correctionfor Deep Metric Learning](https://github.com/darkpromise98/IAA/blob/main/pdf/Intra-class%20Adaptive%20Augmentation%20with%20Neighbor%20Correction%20for%20Deep%20Metric%20Learning.pdf)".
The paper is accepted by the IEEE Transactions on Multimedia, 2022.




## Requirements

We recommended the following dependencies.

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
   - [Stanford Online Products](https://cvgl.stanford.edu/projects/lifted_struct/)
   - [In-Shop Clothes Retrieval](https://drive.google.com/open?id=0B7EVK8r0v71pQ2FuZ0k0QnhBQnc)

2. Extract the compressed file (tgz or zip) into `MyDataset/`, e.g., for Cars-196, put the files in the `MyDataset/Cars196`. Other naming ways can see the python files of realted datatsets in  `utils/dataset`, or modify these names in your way.



## Training

1. Set up the arguments.
   - The folder `MyDataset` is the root path of datasets using in this paper (including CUB, CARS, SOP), which can be customized by the argparse parameter `--data`. 

   - The folder `results` is the log path to record corresponding models and results of training, which can be customized by the argparse parameter `--save-dir`. 

   - The folder `weights_models` is the path to put the weighting parameters of pretrained backbone networks, which can be customized by the argparse parameter `--weight_path`. 

2. Run `train.py` for different datasets.

#### CUB-200-2011
```bash
python run.py --dataset cub200 --backbone googlenet --loss MS --intra_lamda 0.8 --aug_num 3
python run.py --dataset cub200 --backbone googlenet --loss Contrastive  --lr 3e-5  --intra_lamda 0.8 --aug_num 3
```

#### Cars-196
```bash
python run.py --dataset cars196 --backbone googlenet --loss MS --intra_lamda 0.8 --aug_num 3
python run.py --dataset cars196 --backbone googlenet --loss Contrastive --intra_lamda 0.8 --aug_num 3
```

#### Stanford Online Products
```bash
python run.py --dataset stanford --backbone googlenet --batch 180 --lr 1e-4 --loss MS --intra_lamda 0.6 --aug_num 3
python run.py --dataset stanford --backbone googlenet --batch 180 --lr 1e-4 --loss Contrastive --intra_lamda 0.6 --aug_num 2
```
```bash
python run.py --dataset stanford --backbone bninception --batch 256 --lr 1e-4 --loss MS -intra_lamda 0.5 --aug_num 3
python run.py --dataset stanford --backbone resnet50 --batch 256 --lr 1e-4 --loss MS -intra_lamda 0.5 --aug_num 3
```

#### In-Shop Clothes Retrieval
```bash
python run.py --dataset inshop --backbone googlenet --batch 180 --lr 1e-4 --loss MS --intra_lamda 0.6 --aug_num 3
python run.py --dataset inshop --backbone googlenet --batch 180 --lr 1e-4 --loss Contrastive --intra_lamda 0.6 --aug_num 2
```

```bash
python run.py --dataset inshop --backbone bninception --batch 256 --lr 1e-4 --loss MS --intra_lamda 0.5 --aug_num 3
python run.py --dataset inshop --backbone resnet50 --batch 256 --lr 1e-4 --loss MS --intra_lamda 0.5 --aug_num 3
```


## Reference

If you found this code useful, please cite the following paper:

```
@article{Fu2022IAA,
author = {Zheren Fu and Zhendong Mao and Bo Hu and An-an Liu and Yongdong Zhang},
title = {Intra-class Adaptive Augmentation with Neighbor Correctionfor Deep Metric Learning},
year = {2022},
journal = {IEEE Transactions on Multimedia},
}
```


## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)

