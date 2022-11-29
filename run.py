import os
import sys
import time

import argparse
import random
import numpy as np

import torch
import torch.optim as optim

from utils import dataset
import model.backbone as backbone


from tqdm import tqdm
from torch.utils.data import DataLoader

import torch
import torch.nn.functional as F

from metric.batchsampler import MImagesPerClassSampler
from model.embedding import Embedding
from utils.common import *
from intra import *
from losses import *


def train(opts, net, loader, optimizer, criterion, ep=0, estimator=None, loader_train_eval=None):
    net.train()
    fix_batchnorm(net)
     
    if (estimator is not None) and (ep > opts.start_epoch) and opts.global_update and ((ep-opts.start_epoch-1)%opts.update_every_M_epoch == 0):
        estimator.global_update(net, loader_train_eval)

    train_iter = tqdm(loader, ncols=80)
    loss_all = []

    # dramatic lamda
    lamda = lamda_epoch(opts.intra_lamda, opts.epochs, ep)

    for images, labels in train_iter:
        images, labels = images.cuda(), labels.cuda()
        embeddings = net(images)

        if (estimator is not None) and (ep > opts.start_epoch):
                     
            # warmup for construction of corvariance matrix
            if ep > (opts.start_epoch + opts.warmup_epoch):

                aug_embs, aug_labels = intra_synthetsis(opts, embeddings, labels, estimator, lamda=lamda, 
                                                        aug_num=opts.aug_num, diag=estimator.diag)
                
                embeddings = F.normalize(embeddings, dim=1, p=2)
                aug_embs = F.normalize(aug_embs, dim=1, p=2)

                all_embs = torch.cat((embeddings, aug_embs), dim=0)
                all_labels = torch.cat((labels, aug_labels), dim=0)

                loss_minibatch = criterion(all_embs, all_labels)
            
            else:
                embeddings = F.normalize(embeddings, dim=1, p=2)
                loss_minibatch = criterion(embeddings, labels)


        else: 
            embeddings = F.normalize(embeddings, dim=1, p=2)
            loss_minibatch = criterion(embeddings, labels)

        optimizer.zero_grad()

        loss = loss_minibatch
        loss.backward()
        optimizer.step()

        train_iter.set_description("[Train][Epoch %d] Loss: %.4f" % (ep, loss_minibatch.item()))

        loss_all.append(loss_minibatch.item())
 
    return np.mean(loss_all)


def eval_dml(args, net, loader, K=[1, 2, 4, 8], epoch=0):
    net.eval()
    test_iter = tqdm(loader, ncols=80)
    embeddings_all, labels_all = [], []

    test_iter.set_description("[Eval][Epoch %d]" % epoch)

    with torch.no_grad():
        for images, labels in test_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            embeddings_all.append(embedding.data)
            labels_all.append(labels.data)

        embeddings_all = torch.cat(embeddings_all)
        embeddings_all = F.normalize(embeddings_all, dim=1, p=2)
        labels_all = torch.cat(labels_all)

        rec, MLRC = recall(args, embeddings_all, labels_all, K=K)

        for k, r in zip(K, rec):
            print("[Epoch %d] Recall@%d: [%.4f]" % (epoch, k, r))
        
        if MLRC[0] > 0:
            print("[Epoch %d] MAP@R: [%.4f]" % (epoch, MLRC[0]))
            print("[Epoch %d] RP: [%.4f]" % (epoch, MLRC[1]))

    return rec[0], K, rec, MLRC


def eval_inshop(args, net, loader_query, loader_gallery, K=[1], ep=0):
    net.eval()
    query_iter = tqdm(loader_query, ncols=80)
    gallery_iter = tqdm(loader_gallery, ncols=80)

    query_embeddings_all, query_labels_all = [], []
    gallery_embeddings_all, gallery_labels_all = [], []

    with torch.no_grad():
        for images, labels in query_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            query_embeddings_all.append(embedding)
            query_labels_all.append(labels)

        query_embeddings_all = torch.cat(query_embeddings_all)
        query_embeddings_all = F.normalize(query_embeddings_all, dim=1, p=2)
        query_labels_all = torch.cat(query_labels_all)

        for images, labels in gallery_iter:
            images, labels = images.cuda(), labels.cuda()
            embedding = net(images)
            gallery_embeddings_all.append(embedding)
            gallery_labels_all.append(labels)

        gallery_embeddings_all = torch.cat(gallery_embeddings_all)
        gallery_embeddings_all = F.normalize(gallery_embeddings_all, dim=1, p=2)
        gallery_labels_all = torch.cat(gallery_labels_all)

        rec, MLRC = recall_inshop(args, query_embeddings_all, query_labels_all, gallery_embeddings_all, gallery_labels_all, K=K)    
        for k, r in zip(K, rec):
            print("[Epoch %d] Recall@%d: [%.4f]" % (ep, k, r))

        if MLRC[0] > 0:
            print("[Epoch %d] MAP@R: [%.4f]" % (epoch, MLRC[0]))
            print("[Epoch %d] RP: [%.4f]" % (epoch, MLRC[1]))

    return rec[0], K, rec, MLRC



def build_args():
    parser = argparse.ArgumentParser()
    LookupChoices = type(
        "",
        (argparse.Action,),
        dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])),
    )

    parser.add_argument("--mode", choices=["train", "eval"], default="train")
    parser.add_argument(
        "--dataset",
        choices=dict(
            cub200=dataset.CUB2011Metric,
            cars196=dataset.Cars196Metric,
            stanford=dataset.StanfordOnlineProductsMetric,
            inshop=dataset.FashionInshop,
        ),
        default=dataset.CUB2011Metric, action=LookupChoices, )

    parser.add_argument(
        "--backbone",
        choices=dict(
            bninception=backbone.BNInception,
            googlenet=backbone.GoogLeNet,
            resnet50=backbone.ResNet50,
        ),
        default=backbone.BNInception, action=LookupChoices, )

    parser.add_argument("--lr", default=5e-5, type=float)
    parser.add_argument("--lr-decay-epochs", type=int, default=[40, 60, 80], nargs="+")
    parser.add_argument("--lr-decay-gamma", default=0.2, type=float)  
    parser.add_argument('--weight_decay', default=1e-5, type=float)

    parser.add_argument("--embedding-size", type=int, default=512)
    parser.add_argument("--batch", default=128, type=int)
    parser.add_argument("--eval_batch", default=128, type=int)
    parser.add_argument('--workers', default=8, type=int)
    parser.add_argument("--num_image_per_class", default=4, type=int)
    parser.add_argument("--iter-per-epoch", default=300, type=int) 
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--recall", default=[1, 2, 4, 8], type=int, nargs="+")
    parser.add_argument("--map", default=0, type=int)

    parser.add_argument("--seed", default=random.randint(1, 1000), type=int)
    parser.add_argument("--data", default='./MyDataset', type=str)
    parser.add_argument("--weight_path", default='./weights_models', type=str)
    parser.add_argument("--save-dir", default="./results", type=str)
    parser.add_argument("--test", default=0, type=int)
    parser.add_argument('--save_models', default=False, action="store_true")

    parser.add_argument('--gpu-id', default=0, type=int)
    parser.add_argument('--loss', default='MS', type=str)

    parser.add_argument("--intra", default=1, type=int)
    parser.add_argument("--start_epoch", default=30, type=int)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--intra_lamda", default=0.8, type=float)
    parser.add_argument("--aug_num", default=3, type=int)

    parser.add_argument("--momentum", default=0.2, type=float) 

    parser.add_argument("--beta", default=0.1, type=float)  
    parser.add_argument("--gamma", default=0.1, type=float)  
    parser.add_argument("--num_neighbor", default=25, type=int)  

    parser.add_argument("--diag", default=1, type=int)
    parser.add_argument("--global_update", default=1, type=int)
    parser.add_argument("--update_every_M_epoch", default=2, type=int)


    opts = parser.parse_args()

    return opts


if __name__ == "__main__":
    opts = build_args()

    # preparing work
    torch.cuda.set_device(opts.gpu_id)
    fix_seed(opts.seed)
    opts.save_dir = logs_path(opts)
    save_parameters(opts, opts.save_dir)

    base_model = opts.backbone(weight_path=opts.weight_path, pretrained=True)
    model = Embedding(base_model, feature_size=base_model.output_size, embedding_size=opts.embedding_size,).cuda()

    train_transform, test_transform = build_transform(opts, base_model)

    if str(opts.dataset).split('.')[-2] == 'inshop':

        dataset_train = dataset.FashionInshop(opts.data, split="train", transform=train_transform)
        dataset_query = dataset.FashionInshop(opts.data, split="query", transform=test_transform)
        dataset_gallery = dataset.FashionInshop(opts.data, split="gallery", transform=test_transform)

        opts.iter_per_epoch = min(len(dataset_train) // opts.batch, opts.iter_per_epoch)
        opts.recall = [1, 10, 20, 30]
        opts.num_image_per_class = 3

        batch_sampler = MImagesPerClassSampler(
            dataset_train, opts.batch, m=opts.num_image_per_class, iter_per_epoch=opts.iter_per_epoch)

        loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler, pin_memory=True, num_workers=opts.workers)
        loader_train_eval = DataLoader(dataset_train, batch_size=opts.eval_batch, num_workers=opts.workers)
        loader_query = DataLoader(dataset_query, batch_size=opts.eval_batch, pin_memory=True, num_workers=opts.workers)
        loader_gallery = DataLoader(dataset_gallery, batch_size=opts.eval_batch, pin_memory=True, num_workers=opts.workers)

        print("Number of images in Training Set: %d" % len(dataset_train))
        print("Number of images in Query set: %d" % len(dataset_query))
        print("Number of images in Gallery set: %d" % len(dataset_gallery))

    else:
        dataset_train = opts.dataset(opts.data, train=True, transform=train_transform, download=False)
        dataset_eval = opts.dataset(opts.data, train=False, transform=test_transform, download=False)

        if len(dataset_train) > 1e4:
             opts.recall = [1, 10, 100, 1000]
             opts.num_image_per_class = 3

        opts.iter_per_epoch = min(len(dataset_train) // opts.batch, opts.iter_per_epoch)

        batch_sampler = MImagesPerClassSampler(
            dataset_train, opts.batch, m=opts.num_image_per_class, iter_per_epoch=opts.iter_per_epoch)

        loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler, pin_memory=True, num_workers=opts.workers)
        loader_train_eval = DataLoader(dataset_train, batch_size=opts.eval_batch, num_workers=opts.workers)
        loader_eval = DataLoader(dataset_eval, batch_size=opts.eval_batch, pin_memory=True, num_workers=opts.workers)

        print("Number of images in Training Set: %d" % len(dataset_train))
        print("Number of images in Test set: %d" % len(dataset_eval))

    if opts.loss == 'Contrastive':
        criterion = ContrastiveLoss(batch_size=opts.batch).cuda()
    elif opts.loss == 'MS':
        criterion = MultiSimilarityLoss(batch_size=opts.batch).cuda()    
    else:    
        raise Exception('loss function not found!')
    print(criterion)

    # intra-class corvariance matrices
    if opts.intra:
        estimator = Estimate_Covariance(opts=opts, class_num=dataset_train.num_training_classes, feature_num=opts.embedding_size,
                                        momentum=opts.momentum, diag=opts.diag).cuda()
    else:
        estimator = None

    
    param_groups = [{"lr": opts.lr, "params": model.parameters()},]
    optimizer = optim.Adam(param_groups, weight_decay=opts.weight_decay)
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)

    recalls_list = []
    losses_list = []
    best_recall = [0.]
    best_epoch = 0

    for epoch in range(1, opts.epochs + 1):

        loss_epoch = train(opts, model, loader_train, optimizer, criterion, epoch, estimator, loader_train_eval)     
        lr_scheduler.step()

        if str(opts.dataset).split('.')[-2] == 'inshop':
            val_recall, val_recall_K, val_recall_all, val_MLRC = eval_inshop(opts, model, loader_query, loader_gallery, opts.recall, epoch)
        else:
            val_recall, val_recall_K, val_recall_all, val_MLRC = eval_dml(opts, model, loader_eval, opts.recall, epoch)

        recalls_list.append(val_recall_all)
        losses_list.append(loss_epoch)
        plot_recalls(opts, recalls_list, losses_list)

        if best_recall[0] < val_recall_all[0]:

            best_recall = val_recall_all
            best_epoch = epoch
            best_MLRC = val_MLRC

            if opts.save_models:
                torch.save(model.state_dict(), os.path.join(opts.save_dir, 'best_model.pth'))

            with open(os.path.join(opts.save_dir, 'best_results.txt'), "w") as f:

                f.write('Best Epoch: {}\n'.format(best_epoch))
                for i, K in enumerate(opts.recall):
                    f.write("Best Recall@{}: {:.4f}\n".format(K, best_recall[i]))
                    
                if best_MLRC[0] > 0:
                    f.write("\nBest MAP@R: {:.4f}\n".format(best_MLRC[0]))
                    f.write("Best RP: {:.4f}\n".format(best_MLRC[1]))

        print("Best Recall@1: %.4f \n" % best_recall[0])


