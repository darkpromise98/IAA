import torch.utils.data as data
import os
from PIL import Image


class FashionInshop(data.Dataset):

    num_training_classes = 3997

    def __init__(
        self,
        root="data/",
        split="train",
        transform=None,
        return_text=False,
        return_index=False,
    ):
        self.root = os.path.join(root, "In_Shop")
        self.transform = transform
        self.split = split
        self.data_dict = {}
        self.data_labels = []
        self.return_text = return_text
        self.return_index = return_index

        self.img_path = []
        self.labels = []
        self.read_train_test()

        self.class_to_idx = {
            self.data_labels[i]: i for i in range(len(self.data_labels))
        }
        self.imgs = [(None, self.data_labels.index(l)) for l in self.labels]
        self.classes = self.data_labels

    def read_lines(self, path):
        with open(path) as fin:
            lines = fin.readlines()[2:]
            lines = list(filter(lambda x: len(x) > 0, lines))
            pairs = list(map(lambda x: x.strip().split(), lines))
        return pairs

    def read_train_test(self):
        lines = self.read_lines(os.path.join(self.root, "list_eval_partition.txt"))
        # valid_lines = list(filter(lambda x: x[0] in self.cloth, lines))
        # line 0 : img path
        # line 1 : img label (id000~)
        # line 2 : train / query / gallery

        for line in lines:
            if line[2] == self.split:

                self.img_path.append(line[0])
                self.labels.append(line[1])

                if line[1] not in self.data_dict:
                    self.data_dict[line[1]] = [line[0]]
                else:
                    self.data_dict[line[1]].append(line[0])

        self.data_labels = list(self.data_dict.keys())

    def process_img(self, img_path):
        img_full_path = os.path.join(self.root, img_path)
        with open(img_full_path, "rb") as f:
            with Image.open(f) as img:
                img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, index):
        img = self.process_img(self.img_path[index])
        label_text = self.labels[index]

        if self.return_text:
            label = label_text
        else:
            label = self.data_labels.index(label_text)

        if self.return_index:
            return img, label, index
        else:
            return img, label
