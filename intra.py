
import torch
import torch.nn as nn
from tqdm import tqdm

BIG_NUMBER = 1e12


class Estimate_Covariance(nn.Module):

    def __init__(self, opts, class_num, feature_num=512, momentum=0.8, device=torch.device('cuda'), diag=True):
        super().__init__()

        self.class_num = class_num
        self.feature_num = feature_num
        self.momentum = momentum
        self.device = device
        self.diag = diag

        if self.class_num > 1000:
            self.large_dataset = True
            print('large_dataset')
        else:
            self.large_dataset = False
            
        self.register_buffer('covariance', torch.zeros(class_num, feature_num))
        self.register_buffer('mean', torch.zeros(class_num, feature_num))
        self.register_buffer('amount', torch.zeros(class_num))

        if self.large_dataset:
            self.register_buffer('global_covariance', torch.zeros(feature_num))
            self.register_buffer('neighbor_covariance', torch.zeros(class_num, feature_num))
            self.register_buffer('class_weight', torch.zeros(class_num))
            self.gamma = opts.gamma
            self.num_neighbor = opts.num_neighbor
            self.beta = opts.beta

    def update_diag(self, features, labels):

        features = features.detach()

        N = features.size(0)
        C = self.class_num
        A = features.size(1)

        NxCxFeatures = features.view(N, 1, A).expand(N, C, A)
        onehot = torch.zeros(N, C).to(self.device).scatter_(1, labels.view(-1, 1), 1)

        NxCxA_onehot = onehot.view(N, C, 1).expand(N, C, A)

        features_by_sort = NxCxFeatures.mul(NxCxA_onehot)

        Amount_CxA = NxCxA_onehot.sum(0)
        Amount_CxA[Amount_CxA == 0] = 1

        ave_CxA = features_by_sort.sum(0) / Amount_CxA

        var_temp = features_by_sort - ave_CxA.expand(N, C, A).mul(NxCxA_onehot)

        var_temp = var_temp.pow(2).sum(0).div(Amount_CxA)

        sum_weight_CV = onehot.sum(0).view(C, 1).expand(C, A)

        weight_CV = sum_weight_CV.div(sum_weight_CV + self.amount.view(C, 1).expand(C, A))
        weight_CV[weight_CV != weight_CV] = 0

        # keep momentum
        weight_CV = torch.clamp(weight_CV, min=1-self.momentum).mul((weight_CV > 0).float())

        additional_CV = weight_CV.mul(1 - weight_CV).mul((self.mean - ave_CxA).pow(2))

        self.covariance = self.covariance.mul(1 - weight_CV) + var_temp.mul(weight_CV) + additional_CV
        self.mean = self.mean.mul(1 - weight_CV) + ave_CxA.mul(weight_CV)
        self.amount += onehot.sum(0)

    def update(self, features, labels):
        self.update_diag(features, labels)

    def update_each_class(self, features, labels):

        assert self.diag == True
        class_set, inverse_indices, counts = torch.unique(labels, sorted=True, return_inverse=True, return_counts=True)
        assert self.class_num == len(class_set)

        class_iter = tqdm(range(len(class_set)), ncols=60)
        class_iter.set_description('Class Estimation')
        class_alone = 0

        num_neighbor = self.num_neighbor
        
        def weight_functions(counts):

            # weight function
            beta = self.beta
            weighting = 1 / (1 + torch.log(1 + beta*(counts-1)))
            return weighting * (counts < 40).float()

        self.class_weight = weight_functions(counts)

        for i in class_iter:
            
            label_index = class_set[i]
            assert label_index == i
            count = counts[i]

            # only one sample in these class
            if count == 1:
                class_alone += 1
            else:
                class_mask = (i == inverse_indices)
                class_features = features[class_mask]

                assert class_features.shape[0] == count

                ave_A = class_features.mean(dim=0)
                var_temp = (class_features - ave_A).pow(2).mean(dim=0)

                weight = count / (self.amount[label_index] + count)
                weight = torch.clamp(weight, min=1-self.momentum)

                additional_CV = weight * (1 - weight) * (self.mean[label_index] - ave_A).pow(2)
                self.covariance[label_index] = self.covariance[label_index] * (1 - weight) + var_temp * weight + additional_CV    
                self.mean[label_index] = self.mean[label_index] * (1 - weight) + ave_A * weight            
                         
            self.amount[label_index] += count
        
        if class_alone > 0:
            print('alone class:', class_alone)

        # global information,
        mask_temp = (counts != 1)
        self.global_covariance = (self.covariance[mask_temp] * counts[mask_temp].unsqueeze(1)).sum(0) / counts[mask_temp].sum()

        #neighbor information, 
        p = 2
        w_type = 2
        mean_dist_mat = torch.cdist(self.mean.pow(2), self.mean.pow(2), p=p) + \
        ( (~mask_temp).expand(self.class_num, -1) | torch.eye(self.class_num, dtype=torch.bool, device=labels.device) ).float() * BIG_NUMBER

        cov_dist_mat = torch.cdist(self.covariance, self.covariance, p=p)

        indices = torch.topk(mean_dist_mat, k=num_neighbor, dim=1, largest=False)[1]

        indices_iter = tqdm(enumerate(indices), ncols=60)
        indices_iter.set_description('Neighbor Estimation')

        for i, ind in indices_iter:
            sigma1 = 1
            sigma2 = 1
            d_mean = maxmin_norm(mean_dist_mat[i])[ind]
            d_cov = maxmin_norm(cov_dist_mat[i])[ind]
            w = torch.exp(-d_mean.pow(2)/(2*sigma1**2) -d_cov.pow(2)/(2*sigma2**2) ) * counts[ind] 

            self.neighbor_covariance[i] = (self.covariance[ind] * w.unsqueeze(1)).sum(0) / w.sum()

        torch.cuda.empty_cache()


    def distribution_calibration(self, cv_temp, labels):

        gamma = self.gamma

        weight_temp = self.class_weight[labels].unsqueeze(1)
        
        cv_temp = (1-weight_temp) * cv_temp  +  weight_temp * gamma * self.global_covariance.expand(labels.shape[0], -1)  +  weight_temp * (1-gamma) * self.neighbor_covariance[labels]
        
        return cv_temp

    def global_update(self, model, loader_train_eval):

        embeddings_all, labels_all = [], []        
        test_iter = tqdm(loader_train_eval, ncols=40)
        test_iter.set_description('Global Estimation')

        with torch.no_grad():
            for images, labels in test_iter:
                images, labels = images.cuda(), labels.cuda()
                embedding = model(images)
                embeddings_all.append(embedding.data)
                labels_all.append(labels.data)

            embeddings_all = torch.cat(embeddings_all)
            labels_all = torch.cat(labels_all)

            if self.large_dataset:
                print('large datasets')
                self.update_each_class(embeddings_all, labels_all)
            else:
                self.update(embeddings_all, labels_all)


def intra_synthetsis(opts, features, labels, estimator, lamda=0.2, aug_num=1, detach=False, diag=True):
    if detach:
        features = features.detach()

    features_aug = []
    cv_temp = estimator.covariance[labels]

    lamda = torch.tensor(lamda, device=cv_temp.device).sqrt()

    for i in range(aug_num):
        cv_temp = cv_temp.sqrt()
        aug = features + lamda * cv_temp.mul(torch.randn_like(features))
        features_aug.append(aug)

    features_aug = torch.cat(features_aug, dim=0)
    labels_aug = labels.repeat(aug_num)

    return features_aug, labels_aug


def lamda_epoch(lamda, epochs, ep):
    
    x1 = 0.4
    x2 = 0.6
    x3 = 0.8    

    y1 = 1.0
    y2 = 0.9
    y3 = 0.75
    y4 = 0.6

    radio = ep / epochs

    if radio <= x1:
        weight = y1
    elif radio > x1 and radio <= x2:
        weight = y2
    elif radio > x2 and radio <= x3:
        weight = y3
    else:
        weight = y4
    
    return weight * lamda


def maxmin_norm(x):

    return (x - x.min()) / (x.max() - x.min())

