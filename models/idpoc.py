import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.optim import SGD
from torch.utils.data import DataLoader

from backbone.MNISTMLP_OC import MNISTMLP_OC
from backbone.ResNet18_OC import resnet18_OC
from models.utils.continual_model import IncrementalModel

def get_backbone(args, embedding_dim, nf=64):
    if args.dataset == 'seq-mnist':
        return MNISTMLP_OC(28 * 28, embedding_dim, 10)
    else:
        if args.featureNet:
            return MNISTMLP_OC(1000, embedding_dim, 10, middle_size=[800, 500])
        else:
            return resnet18_OC(embedding_dim, 10, nf)


class IDPOC(IncrementalModel):
    NAME = 'IDPOC'
    COMPATIBILITY = ['class-il', 'task-il']

    def __init__(self, args):
        super(IDPOC, self).__init__(args)

        self.nets = []
        self.c = []
        self.stds = []
        self.svd_models = []

        self.nu = self.args.nu
        self.eta = self.args.eta
        self.eps = self.args.eps
        self.embedding_dim = self.args.embedding_dim
        self.weight_decay = self.args.weight_decay

        self.current_task = -1
        self.nc = None
        self.t_c_arr = []
        self.nf = self.args.nf

    def begin_il(self, dataset):

        self.t_c_arr = dataset.t_c_arr
        self.nc = dataset.nc

        for i in range(self.nc):
            self.nets.append(
                get_backbone(self.args, self.embedding_dim, self.nf).to(self.device)
            )
            self.c.append(
                torch.ones(self.embedding_dim, device=self.device)
            )
            self.stds.append(None)
            self.svd_models.append(None)


    def train_task(self, dataset, train_loader):
        self.current_task += 1
        plt.figure(figsize=(20, 12))

        categories = self.t_c_arr[self.current_task]
        print('==========\t task: %d\t categories:' % self.current_task, categories, '\t==========')
        for category in categories:
            losses = []

            for epoch in range(self.args.n_epochs):

                avg_loss, posdist, negdist, gloloss = self.train_category(train_loader, category, epoch)

                losses.append(avg_loss)
                if epoch == 0 or (epoch + 1) % 5 == 0:
                    print("epoch: %d\t task: %d \t category: %d \t loss: %f \t posloss: %f \t negloss: %f" % (
                        epoch + 1, self.current_task, category, avg_loss, np.mean(posdist), np.mean(negdist)))

            self.set_mean_var(train_loader, category)

    def train_category(self, data_loader, category: int, epoch_id):

        self.init_center_c(data_loader, category)
        c = self.c[category]

        network = self.nets[category].to(self.device)
        network.train()

        optimizer = SGD(network.parameters(), lr=self.args.lr, weight_decay=self.weight_decay)
        avg_loss = 0.0
        sample_num = 0

        posdist = []
        negdist = []
        gloloss = []

        for i, data in enumerate(data_loader):
            inputs = data[0].to(self.device)
            semi_targets = data[1].to(self.device)

            # Zero the network parameter gradients
            optimizer.zero_grad()

            outputs = network(inputs - 0.5)

            dists = torch.sum((outputs - c) ** 2, dim=1)

            loss_pos = dists
            loss_neg = self.eta * torch.relu(self.args.margin - dists)

            losses = torch.where(semi_targets == category, loss_pos, loss_neg)

            gloloss.append(losses.detach().cpu().data.numpy())

            loss = torch.mean(losses)

            loss.backward()
            optimizer.step()

            pos_dist = dists[semi_targets == category]
            posdist.append(pos_dist.detach().cpu().data.numpy())

            neg_dist = loss_neg[semi_targets != category]
            negdist.append(neg_dist.detach().cpu().data.numpy())

            avg_loss += loss.item()
            sample_num += inputs.shape[0]

        avg_loss /= sample_num
        posdist = np.hstack(posdist)
        negdist = np.hstack(negdist)
        gloloss = np.hstack(gloloss)
        return avg_loss, posdist, negdist, gloloss


    def set_mean_var(self, data_loader, category):
        network = self.nets[category].to(self.device)
        c = self.c[category].to(self.device)

        dists = []
        with torch.no_grad():
            for data in data_loader:
                inputs = data[0].to(self.device)
                semi_targets = data[1].to(self.device)

                outputs = network(inputs - 0.5)

                outputs = outputs[semi_targets == category]  
                dist = outputs - c

                dist = dist.to('cpu')
                dists.append(dist)

        dists = torch.cat(dists)

        def percentage2n(eigVals, percentage):
            sortArray = np.sort(eigVals) 
            sortArray = sortArray[-1::-1] 
            arraySum = sum(sortArray)
            tmpSum = 0
            num = 0
            for i in sortArray:
                tmpSum += i
                num += 1
                if tmpSum >= arraySum * percentage:
                    return num

        mean = torch.mean(dists, dim=0)
        dists_mean = dists - mean
        dists_np = dists_mean.detach().cpu().numpy()

        covMat = np.cov(dists_np, rowvar=False) 
        eigVals, eigVects = np.linalg.eig(np.mat(covMat))  
        n = percentage2n(eigVals, self.nu) 
        eigValIndice = np.argsort(eigVals) 
        n_eigValIndice = eigValIndice[-1:-(n + 1):-1]  
        n_eigVect = eigVects[:, n_eigValIndice]  
        self.svd_models[category] = torch.from_numpy(n_eigVect).float()

        self.stds[category] = torch.sqrt(torch.tensor(eigVals[n_eigValIndice]).float()).to(self.device)

    def init_center_c(self, train_loader: DataLoader, category):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = 0

        net = self.nets[category].to(self.device)

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                inputs = data[0].to(self.device)
                semi_targets = data[1].to(self.device)

                outputs = net(inputs - 0.5)

                outputs = outputs[semi_targets == category] 
                n_samples += outputs.shape[0]
                c += torch.sum(outputs, dim=0)

        c /= n_samples

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < self.eps) & (c < 0)] = -self.eps
        c[(abs(c) < self.eps) & (c > 0)] = self.eps
        self.c[category] = c.to(self.device)


    def get_score(self, dist, category):
        std = self.stds[category]

        # base-standlization
        dist_pca = torch.mm(dist, self.svd_models[category].to(self.device))

        # value-standlization
        dist_norm = dist_pca / std

        dist_l2 = torch.sum(dist_norm ** 2, dim=1)

        # score standardization
        degree = dist_pca.shape[1]
        dist_norm = (dist_l2 - degree) / np.sqrt(2 * degree)

        score = -dist_norm
        return score

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        categories = list(range(self.t_c_arr[self.current_task][-1] + 1))
        return self.predict(x, categories)[0]

    def predict(self, inputs: torch.Tensor, categories):
        inputs = inputs.to(self.device)
        outcome, dists = [], []
        with torch.no_grad():
            for i in categories:
                net = self.nets[i]
                net.to(self.device)
                net.eval()

                pred = net(inputs - 0.5)

                c = self.c[i].to(self.device)
                dist = pred - c

                scores = self.get_score(dist, i)

                outcome.append(scores.view(-1, 1))
                dists.append(dist.view(-1, 1))

        outcome = torch.cat(outcome, dim=1)
        dists = torch.cat(dists, dim=1)
        return outcome, dists

