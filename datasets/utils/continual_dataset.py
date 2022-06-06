
import torch
torch.multiprocessing.set_sharing_strategy('file_system')
from abc import abstractmethod
from argparse import Namespace
import copy

from torch import nn as nn
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset
from typing import Tuple
from torchvision import datasets
import numpy as np
from pathlib import Path
import os

class ContinualDataset:
    """
    Continual learning evaluation setting.
    """
    NAME = None
    SETTING = None
    N_CLASSES_PER_TASK = None
    N_TASKS = None
    TRANSFORM = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass

    @staticmethod
    @abstractmethod
    def get_backbone() -> nn.Module:
        """
        Returns the backbone to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_transform() -> transforms:
        """
        Returns the transform to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_loss() -> nn.functional:
        """
        Returns the loss to be used for to the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_normalization_transform() -> transforms:
        """
        Returns the transform used for normalizing the current dataset.
        """
        pass

    @staticmethod
    @abstractmethod
    def get_denormalization_transform() -> transforms:
        """
        Returns the transform used for denormalizing the current dataset.
        """
        pass

class IncrementalDataset:
    NAME = None
    SETTING = None

    def __init__(self, args: Namespace) -> None:
        """
        Initializes the train and test lists of dataloaders.
        :param args: the arguments which contains the hyperparameters
        """
        self.train_loader = None
        self.test_loaders = []
        self.i = 0
        self.args = args
        self.nt = args.nt if args.nt else self.nt
        self.nc = self.nc

        self.t_c_arr = args.t_c_arr if args.t_c_arr else self.get_balance_classes()

    def get_balance_classes(self):
        class_arr = list(range(self.nc))
        cpt = self.nc // self.nt
        return [class_arr[i:i+cpt] for i in range(0, len(class_arr), cpt)]

    @abstractmethod
    def get_data_loaders(self) -> Tuple[DataLoader, DataLoader]:
        """
        Creates and returns the training and test loaders for the current task.
        The current training loader and all test loaders are stored in self.
        :return: the current training and test loaders
        """
        pass

    @abstractmethod
    def not_aug_dataloader(self, batch_size: int) -> DataLoader:
        """
        Returns the dataloader of the current task,
        not applying data augmentation.
        :param batch_size: the batch size of the loader
        :return: the current training loader
        """
        pass


def store_masked_loaders(train_dataset: datasets, test_dataset: datasets,
                    setting: ContinualDataset) -> Tuple[DataLoader, DataLoader]:
    """
    Divides the dataset into tasks.
    :param train_dataset: train dataset
    :param test_dataset: test dataset
    :param setting: continual learning setting
    :return: train and test loaders
    """
    c_arr = setting.t_c_arr[setting.i]
    train_mask = np.logical_and(np.array(train_dataset.targets) >= c_arr[0],
                                np.array(train_dataset.targets) <= c_arr[-1])
    test_mask = np.logical_and(np.array(test_dataset.targets) >= c_arr[0],
                               np.array(test_dataset.targets) <= c_arr[-1])

    train_dataset.data = train_dataset.data[train_mask]
    test_dataset.data = test_dataset.data[test_mask]

    train_dataset.targets = np.array(train_dataset.targets)[train_mask]
    test_dataset.targets = np.array(test_dataset.targets)[test_mask]

    train_loader = DataLoader(train_dataset,
                              batch_size=setting.args.batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset,
                             batch_size=setting.args.batch_size, shuffle=False, num_workers=4)
    setting.test_loaders.append(test_loader)
    setting.train_loader = train_loader

    setting.i += 1
    return train_loader, test_loader


def get_previous_train_loader(train_dataset: datasets, batch_size: int,
                              setting: ContinualDataset) -> DataLoader:
    """
    Creates a dataloader for the previous task.
    :param train_dataset: the entire training set
    :param batch_size: the desired batch size
    :param setting: the continual dataset at hand
    :return: a dataloader
    """
    train_mask = np.logical_and(np.array(train_dataset.targets) >=
        setting.i - setting.N_CLASSES_PER_TASK, np.array(train_dataset.targets)
        < setting.i - setting.N_CLASSES_PER_TASK + setting.N_CLASSES_PER_TASK)

    train_dataset.data = train_dataset.data[train_mask]
    train_dataset.targets = np.array(train_dataset.targets)[train_mask]

    return DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class ILDataset(Dataset):  # 继承Dataset
    def __init__(self, data, targets, transform=None, target_transform=None):  # __init__是初始化该类的一些基础参数
        self.data = data
        self.targets = targets
        self.attributes = ['data', 'targets']
        self.trans = [transform, target_transform]

    def __len__(self):
        return self.data.shape[0]

    def set_att(self, att_name, att_data, att_transform=None):
        self.attributes.append(att_name)
        self.trans.append(att_transform)
        setattr(self, att_name, att_data)

    def get_att_names(self):
        return self.attributes

    def __getitem__(self, index):
        ret_tuple = ()
        for i, att in enumerate(self.attributes):
            att_data = getattr(self, att)[index]

            trans = self.trans[i]
            if trans:
                att_data = trans(att_data)

            ret_tuple += (att_data,)
        return ret_tuple

def getfeature_loader(train_dataset: datasets, test_dataset: datasets, setting):

    if setting.args.featureNet:
        my_file = Path("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy")
        if my_file.exists():
            print("feature already extracted")
            train_data = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy", allow_pickle=True)
            train_label = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-label.npy", allow_pickle=True)
            test_data = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-data.npy", allow_pickle=True)
            test_label = np.load("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-label.npy", allow_pickle=True)
        else:
            print("feature file not found !!  extracting feature ...")
            train_data, train_label = get_feature_by_extractor(train_dataset, setting.extractor, setting)
            test_data, test_label = get_feature_by_extractor(test_dataset, setting.extractor, setting)

            np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-data.npy", train_data, allow_pickle=True)
            np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-train-label.npy", train_label, allow_pickle=True)
            np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-data.npy", test_data, allow_pickle=True)
            np.save("data/" + setting.args.dataset + "-" + setting.args.featureNet + "-test-label.npy", test_label, allow_pickle=True)

        train_dataset = ILDataset(train_data, train_label)
        test_dataset = ILDataset(test_data, test_label)

    train, test = store_masked_loaders(train_dataset, test_dataset, setting=setting)

    return train, test

def get_feature_by_extractor(train_dataset: datasets, extractor, setting: ContinualDataset):
    extractor = extractor.to(setting.args.device).eval()
    train_loader = DataLoader(train_dataset,
                              batch_size=256, shuffle=False, num_workers=4)
    features, labels = [], []
    for data in train_loader:
        # print(data)
        img = data[0]
        label = data[1]
        img = img.to(setting.args.device)
        with torch.no_grad():
            feature = extractor(img)

        feature = feature.to('cpu')
        img = img.to('cpu')

        features.append(feature)
        labels.append(label)

    feature = torch.cat(features).numpy()
    label = torch.cat(labels).numpy()

    return feature, label