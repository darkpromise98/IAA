import torch
import os
import random
import numpy as np
import torch.nn.functional as F
import torchvision.transforms as transforms
import model.backbone as backbone
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from tqdm import tqdm
from PIL import Image


BIG_NUMBER = 1e5


def recall_inshop(args, query_embeddings, query_labels, gallery_embeddings, gallery_labels, K=[]):
  
    cos_sim = torch.mm(query_embeddings, gallery_embeddings.t())
    class_set, _, counts = torch.unique(gallery_labels, sorted=True, return_inverse=True, return_counts=True)
    knn_inds = cos_sim.topk(max(max(counts), max(K)))[1]

    assert knn_inds.shape[0] == query_labels.shape[0]

    selected_labels = gallery_labels[knn_inds]
    correct_labels = query_labels.unsqueeze(1) == selected_labels

    MLRC = (0, 0)
    if args.map:
        mAP = []
        RP = []    
        evaluation_iter = tqdm(range(correct_labels.shape[0]), ncols=80)
        evaluation_iter.set_description("Measuring MAP and RP")
        for i in evaluation_iter:

            cnt = counts[query_labels[i]-class_set[0]] 
            l = correct_labels[i, 0:cnt].float()
            rp = l.sum()/ cnt

            intersect_size = 0
            ap = 0
            for j in range(len(l)):

                if l[j]:
                    intersect_size += 1
                    precision = intersect_size / (j+1)
                    ap += precision / cnt

            RP.append(rp)
            mAP.append(ap)

        RP = sum(RP) / len(RP)
        mAP = sum(mAP) / len(mAP)

        MLRC = (100 * mAP.item(), 100 * RP.item())    

    recall_k = []
    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)

    return recall_k, MLRC


# compute recall@K
def recall(args, embeddings, labels, K=[]):
    knn_inds = []
    M = 10000

    class_set, inverse_indices, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)

    evaluation_iter = tqdm(range(embeddings.shape[0] // M + 1), ncols=80)
    evaluation_iter.set_description("Measuring recall...")
    for i in evaluation_iter:
        s = i * M
        e = min((i+1) * M, embeddings.shape[0])
        # print(s, e)

        embeddings_select = embeddings[s:e]
        cos_sim = F.linear(embeddings_select, embeddings)
        cos_sim[range(0, e-s), range(s, e)] = BIG_NUMBER
        knn_ind = cos_sim.topk(1 + max(max(counts), max(K)))[1][:, 1:]
        knn_inds.append(knn_ind)
    
    knn_inds = torch.cat(knn_inds, dim=0)

    selected_labels = labels[knn_inds]
    correct_labels = labels.unsqueeze(1) == selected_labels

    MLRC = (0, 0)
    if args.map:
        mAP = []
        RP = []    
        evaluation_iter = tqdm(range(labels.shape[0]), ncols=80)
        evaluation_iter.set_description("Measuring MAP and RP")
        for i in evaluation_iter:

            cnt = counts[labels[i]-class_set[0]] -1
            l = correct_labels[i, 0:cnt].float()
            rp = l.sum()/ cnt

            intersect_size = 0
            ap = 0
            for j in range(len(l)):

                if l[j]:
                    intersect_size += 1
                    precision = intersect_size / (j+1)
                    ap += precision / cnt

            RP.append(rp)
            mAP.append(ap)

        RP = sum(RP) / len(RP)
        mAP = sum(mAP) / len(mAP)

        MLRC = (100 * mAP.item(), 100 * RP.item())

    recall_k = []
    for k in K:
        correct_k = 100 * (correct_labels[:, :k].sum(dim=1) > 0).float().mean().item()
        recall_k.append(correct_k)

    return recall_k, MLRC


def fix_batchnorm(net):
    for m in net.modules():
        if (
            isinstance(m, torch.nn.BatchNorm1d)
            or isinstance(m, torch.nn.BatchNorm2d)
            or isinstance(m, torch.nn.BatchNorm3d)
        ):
            m.eval()


def build_transform(opts, model):
    if isinstance(model, backbone.BNInception):
        normalize = transforms.Compose(
            [
                transforms.Lambda(lambda x: x[[2, 1, 0], ...] * 255.0),
                transforms.Normalize(mean=[104, 117, 128], std=[1, 1, 1]),
            ]
        )
    else:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    if str(opts.dataset).split('.')[-2] == 'cub200':
        flag = 0
    else:
        flag = 1

    train_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), interpolation=Image.LANCZOS) if flag else Identity(),
            transforms.RandomResizedCrop(scale=(0.16, 1), ratio=(0.75, 1.33), size=224, interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    test_transform = transforms.Compose(
        [

            transforms.Resize((256, 256), interpolation=Image.LANCZOS) if flag else transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )

    return train_transform, test_transform


def fix_seed(seed):
    
    torch.backends.cudnn.deterministic=True

    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def logs_path(args):
    
    dataset = str(args.dataset).split('.')[-2]
    backbone = str(args.backbone).split('.')[-2]
    test = args.test
    if test:
        logs = '{}/{}_{}'.format(dataset, backbone, 'test')
    else:
        logs =  '{}/{}_{}_{}_bs{}_instance_{}'.format(dataset, backbone, args.loss, args.embedding_size, 
                                                  args.batch, args.num_image_per_class)
    
    save_dir = args.save_dir
    LOG_DIR = os.path.join(save_dir, logs)

    if test:
        if not os.path.exists(LOG_DIR):
            os.makedirs(LOG_DIR)
    else:
        counter = 1
        checkfolder = LOG_DIR
        while os.path.exists(checkfolder):
            checkfolder = LOG_DIR + '_' + str(counter)
            counter += 1      
        LOG_DIR = checkfolder 
        os.makedirs(LOG_DIR)

    return LOG_DIR


def save_parameters(opt, save_path):

    varx = vars(opt)
    base_str = ''
    for key in varx:
        base_str += str(key)
        if isinstance(varx[key], dict):
            for sub_key, sub_item in varx[key].items():
                base_str += '\n\t'+str(sub_key)+': '+str(sub_item)
        else:
            base_str += '\n\t'+str(varx[key])
        base_str+='\n\n'
    
    with open(os.path.join(save_path, 'Parameters.txt'), 'w') as f:
        f.write(base_str)


def plot_recalls(args, recalls_list, losses_list):
    
    LOG_DIR = args.save_dir
    K = args.recall

    losses = np.array(losses_list)
    recalls = np.array(recalls_list)
        
    plt.figure(figsize=(16, 8))
    
    plt.subplot(1, 2, 1)
    plt.title('Loss Curve')
    plt.plot(losses, label='L_metric')
    plt.tick_params(labelsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(linestyle='--')
    
    plt.subplot(1, 2, 2)
    plt.title('Evaluation Results')      
    for i in range(len(K)):
        plt.plot(recalls[:, i], label='Recall@{}'.format(K[i]))       
    plt.legend(fontsize=10, loc='upper right')
    plt.tick_params(labelsize=20)
    plt.xlabel('Epochs')
    plt.ylabel('Recalls')
    plt.grid(linestyle='--')
    
    plt.savefig(os.path.join(LOG_DIR, 'losses_and_recalls.png'))
    plt.close()

    
class Identity(): 
    def __call__(self, im):
        return im

