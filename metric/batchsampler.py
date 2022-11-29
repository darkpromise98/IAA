import random
import numpy as np
from torchvision.datasets import ImageFolder


def index_dataset(dataset: ImageFolder):
    kv = [(cls_ind, idx) for idx, (_, cls_ind) in enumerate(dataset.imgs)]
    cls_to_ind = {}

    for k, v in kv:
        if k in cls_to_ind:
            cls_to_ind[k].append(v)
        else:
            cls_to_ind[k] = [v]

    return cls_to_ind


class MImagesPerClassSampler:
    def __init__(self, data_source: ImageFolder, batch_size, m=5, iter_per_epoch=100):
        self.m = m
        self.batch_size = batch_size
        self.n_batch = iter_per_epoch
        self.class_idx = list(data_source.class_to_idx.values())
        self.images_by_class = index_dataset(data_source)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        for _ in range(self.n_batch):
            # selected_class = random.sample(self.class_idx, k=len(self.class_idx))
            selected_class = np.random.choice(self.class_idx, size=len(self.class_idx), replace=False)
            example_indices = []

            for c in selected_class:
                img_ind_of_cls = self.images_by_class[c]

                # maybe not satisfied P*K 
                # new_ind = random.sample(img_ind_of_cls, k=min(self.m, len(img_ind_of_cls)))
                new_ind =  np.random.choice(img_ind_of_cls, size=min(self.m, len(img_ind_of_cls)), replace=False).tolist()
                example_indices += new_ind

                if len(example_indices) >= self.batch_size:
                    break

            yield example_indices[: self.batch_size]
