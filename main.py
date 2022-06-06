
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'

from utils.args import get_args
from utils.training import train_il
import torch


def main():
    args = get_args()
    args.model = 'idpoc'
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.img_dir = 'img/idpoc/'

    # seq-mnist
    # args.dataset = 'seq-mnist'
    # args.lr = 5e-4
    # args.batch_size = 32
    # args.n_epochs = 10
    #
    # args.nu = 0.99
    # args.eta = 0.8
    # args.eps = 1
    # args.embedding_dim = 256
    # args.weight_decay = 0.01
    # args.nf = 64
    # args.margin = 20

    # seq-cifar10
    args.dataset = 'seq-cifar10'
    args.lr = 2e-3
    args.batch_size = 32
    args.n_epochs = 50

    args.nu = 0.999             # singular value retention ate
    args.eta = 10               # negative sample weight
    args.eps = 1
    args.embedding_dim = 1024
    args.weight_decay = 1e-2
    args.nf = 64
    args.margin = 10            # r

    # seq-tinyimg
    # args.dataset = 'seq-tinyimg'
    # args.lr = 5e-3
    # args.batch_size = 64
    # args.n_epochs = 100
    #
    # args.nu = 0.95
    # args.eta = 1.5
    # args.eps = 1
    # args.embedding_dim = 1024
    # args.weight_decay = 1e-4
    # args.nf = 32
    # args.margin = 10

    for conf in [1]:
        print("")
        print("=================================================================")
        print("==========================", "repeat", ":", conf, "==========================")
        print("=================================================================")
        print("")
        train_il(args)

if __name__ == '__main__':
    main()
