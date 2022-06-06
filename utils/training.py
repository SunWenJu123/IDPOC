
import torch

from models import get_il_model
from utils.conf import set_random_seed
from utils.loggers import *
from argparse import Namespace
from models.utils.continual_model import ContinualModel
from datasets.utils.continual_dataset import ContinualDataset
from typing import Tuple
from datasets import get_dataset


def mask_classes(outputs: torch.Tensor, dataset: ContinualDataset, k: int) -> None:
    """
    Given the output tensor, the dataset at hand and the current task,
    masks the former by setting the responses for the other tasks at -inf.
    It is used to obtain the results for the task-il setting.
    :param outputs: the output tensor
    :param dataset: the continual dataset
    :param k: the task index
    """
    outputs[:, 0:k * dataset.N_CLASSES_PER_TASK] = -float('inf')
    outputs[:, (k + 1) * dataset.N_CLASSES_PER_TASK:
               dataset.N_TASKS * dataset.N_CLASSES_PER_TASK] = -float('inf')


def evaluate(model: ContinualModel, dataset: ContinualDataset, last=False) -> Tuple[list, list]:
    """
    Evaluates the accuracy of the model for each past task.
    :param model: the model to be evaluated
    :param dataset: the continual dataset at hand
    :return: a tuple of lists, containing the class-il
             and task-il accuracy for each task
    """
    # status = model.net.training
    # model.net.eval()
    accs, accs_mask_classes = [], []
    correct, correct_mask_classes, total = 0.0, 0.0, 0.0
    for k, test_loader in enumerate(dataset.test_loaders):
        if last and k < len(dataset.test_loaders) - 1:
            continue
        for data in test_loader:
            inputs = data[0]
            labels = data[1]
            inputs, labels = inputs.to(model.device), labels.to(model.device)
            if 'class-il' not in model.COMPATIBILITY:
                outputs = model(inputs, k)
            else:
                outputs = model(inputs)

            _, pred = torch.max(outputs.data, 1)
            correct += torch.sum(pred == labels).item()
            total += labels.shape[0]

            if dataset.SETTING == 'class-il':
                mask_classes(outputs, dataset, k)
                _, pred = torch.max(outputs.data, 1)
                correct_mask_classes += torch.sum(pred == labels).item()

    accs.append(correct / total * 100
                if 'class-il' in model.COMPATIBILITY else 0)
    accs_mask_classes.append(correct_mask_classes / total * 100)

    # model.net.train(status)
    return accs, accs_mask_classes


def train_il(args: Namespace) -> None:
    if args.seed is not None:
        set_random_seed(args.seed)
    if not os.path.exists(args.img_dir):
        os.makedirs(args.img_dir)

    dataset = get_dataset(args)
    model = get_il_model(args)

    model.begin_il(dataset)
    mean_accs = []
    for t in range(dataset.nt):

        train_loader, test_loader = dataset.get_data_loaders()

        model.train_task(dataset, train_loader)
        model.test_task(dataset, test_loader)
        accs = evaluate(model, dataset)

        mean_acc = np.mean(accs, axis=1)
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)
        mean_accs.append(mean_acc)

    model.end_il(dataset)

    for t in range(dataset.nt):
        mean_acc = mean_accs[t]
        print_mean_accuracy(mean_acc, t + 1, dataset.SETTING)