import gzip
import struct
from os import path
import numpy as np
import models
import torch
import torch.nn as nn
import random
from torch.utils.data import Dataset


def load_pretrained_cnn(cnn_id, n_classes=4, models_dir='trained-models/'):
    """
    Loads one of the pre-trained CNNs that will be used throughout the HW
    """
    if not isinstance(cnn_id, int) or cnn_id < 0 or cnn_id > 2:
        raise ValueError(f'Unknown cnn_id {id}')
    model = eval(f'models.SimpleCNN{cnn_id}(n_classes=n_classes)')
    fpath = path.join(models_dir, f'simple-cnn-{cnn_id}')
    model.load_state_dict(torch.load(fpath))
    return model


class TMLDataset(Dataset):
    """
    Used to load the dataset used throughout the HW
    """

    def __init__(self, fpath='dataset.npz', transform=None):
        with gzip.open(fpath, 'rb') as fin:
            self.data = np.load(fin, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        if self.transform:
            x = self.transform(x)
        return x, y


def compute_accuracy(model, data_loader, device):
    """
    Evaluates and returns the (benign) accuracy of the model 
    (a number in [0, 1]) on the labeled data returned by 
    data_loader.
    """
    correct_predictions = 0
    total_instances = len(data_loader.dataset)

    with torch.no_grad():
        for images, labels in data_loader:
            images, labels = images.to(device), labels.to(device)

            predictions = torch.argmax(model(images), dim=1)
            correct_predictions += int(sum(predictions == labels))

    return correct_predictions / total_instances


def run_whitebox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the white-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=c_x+randint(1, n_classes)%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    """
    target = None
    if targeted:
        # targeted_labels = torch.randint(n_classes, size=data_loader.batch_size)
        target = random.randint(0, n_classes)

    all_adversarials = None
    all_labels = None

    for images, labels in data_loader:
        if targeted:
            labels[:] = target

        images, labels = images.to(device), labels.to(device)

        adversarial = attack.execute(images, labels, targeted)
        all_adversarials = adversarial if all_adversarials is None else torch.cat((all_adversarials, adversarial), 0)
        all_labels = labels if all_labels is None else torch.cat((all_labels, labels), 0)

    return all_adversarials, all_labels


def run_blackbox_attack(attack, data_loader, targeted, device, n_classes=4):
    """
    Runs the black-box attack on the labeled data returned by
    data_loader. If targeted==True, runs targeted attacks, where
    targets are selected at random (t=(c_x+randint(1, n_classes))%n_classes).
    Otherwise, runs untargeted attacks. 
    The function returns:
    1- Adversarially perturbed sampels (one per input sample).
    2- True labels in case of untargeted attacks, and target labels in
       case of targeted attacks.
    3- The number of queries made to create each adversarial example.
    """
    target = None
    if targeted:
        # targeted_labels = torch.randint(n_classes, size=data_loader.batch_size)
        target = random.randint(0, n_classes)

    all_adversarials = None
    all_labels = None
    all_queries = None

    for images, labels in data_loader:
        if targeted:
            labels[:] = target

        images, labels = images.to(device), labels.to(device)

        adversarial, queries = attack.execute(images, labels, targeted)
        all_adversarials = adversarial if all_adversarials is None else torch.cat((all_adversarials, adversarial), 0)
        all_labels = labels if all_labels is None else torch.cat((all_labels, labels), 0)
        all_queries = queries if all_queries is None else torch.cat((all_queries, queries), 0)

    return all_adversarials, all_labels, all_queries


def compute_attack_success(model, x_adv, y, batch_size, targeted, device):
    """
    Returns the success rate (a float in [0, 1]) of targeted/untargeted
    attacks. y contains the true labels in case of untargeted attacks,
    and the target labels in case of targeted attacks.
    """
    x_batches = [x_adv[i:i + batch_size] for i in range(0, len(y), batch_size)]
    y_batches = [y[i:i + batch_size] for i in range(0, len(y), batch_size)]

    num_correct = 0

    for x_batch, y_batch in zip(x_batches, y_batches):
        predictions = torch.argmax(model(x_batch), dim=1)

        if targeted:
            correct = predictions == y_batch
        else:
            correct = predictions != y_batch

        num_correct += sum(correct)

    return num_correct / len(y)


def binary(num):
    """
    Given a float32, this function returns a string containing its
    binary representation (in big-endian, where the string only
    contains '0' and '1' characters).
    """
    pass  # FILL ME


def float32(binary):
    """
    This function inverts the "binary" function above. I.e., it converts 
    binary representations of float32 numbers into float32 and returns the
    result.
    """
    pass  # FILL ME


def random_bit_flip(w):
    """
    This functoin receives a weight in float32 format, picks a
    random bit to flip in it, flips the bit, and returns:
    1- The weight with the bit flipped
    2- The index of the flipped bit in {0, 1, ..., 31}
    """
    pass  # FILL ME
