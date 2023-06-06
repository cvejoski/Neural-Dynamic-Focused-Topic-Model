
import torch
import numpy as np
from torch.utils.data import SubsetRandomSampler
from collections import namedtuple

Vocab = namedtuple('Vocab', 'vocab, stoi, itos, word_count, vectors')
Time = namedtuple('Time', 'all_time, time2id, id2time')

def SimpleSplits(dataset,validation_split=.2,random_seed=87,shuffle_dataset=True):
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset:
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)
    return train_sampler, valid_sampler


def divide_data(counts, train_p=0.8, test_p=0.1):
    try:
        number_of_patients, number_of_dianosis = counts.shape
    except:
        number_of_patients = len(counts)

    max_train_index = int(number_of_patients * train_p)
    max_test_index = max_train_index + int(number_of_patients * test_p)

    train_counts = counts[:max_train_index]
    test_counts = counts[max_train_index:max_test_index]
    validation_counts = counts[max_test_index:]

    return train_counts, test_counts, validation_counts