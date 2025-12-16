from .transforms import get_dataset_transforms, MultiPatchTransforms, MultiPatchMAETransforms
import random
import numpy as np
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import ConcatDataset, Subset

from avalanche.benchmarks.classic import SplitCIFAR100, SplitCIFAR10, SplitImageNet
from torchvision.datasets import SVHN, StanfordCars
# from avalanche.benchmarks.classic.clear import CLEAR
from .clear_dataset import CLEAR

from .benchmark import Benchmark
def get_benchmark(dataset_name, dataset_root, num_exps=20, seed=42, val_ratio=0.1, evaluation_protocol_clear='iid', transforms='default', num_views=2, strategy_name=None):

    return_task_id = False
    shuffle = True

    # Get transformations to be applied to the dataset
    if transforms == 'default':
        # Default transformation is only normalization, the other can be applied online
        train_transform = get_dataset_transforms(dataset_name)
        eval_transform = get_dataset_transforms(dataset_name)
    elif transforms == 'multipatch':
        if strategy_name == 'mae_cmp':
            # Transforms for multipatch MAE training and evaluation
            print('Using MultiPatchMAETransforms for training and evaluation')
            train_transform = MultiPatchMAETransforms(num_patch=num_views, dataset_name=dataset_name)
            eval_transform = MultiPatchMAETransforms(num_patch=num_views, dataset_name=dataset_name)
        else:
            # Transforms for multipatch training and evaluation
            print('Using MultiPatchTransforms for training and evaluation')
            train_transform = MultiPatchTransforms(num_patch=num_views, dataset_name=dataset_name)
            eval_transform = MultiPatchTransforms(num_patch=num_views, dataset_name=dataset_name)
    

    if dataset_name == 'cifar100':
        benchmark = SplitCIFAR100(
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=train_transform,
                eval_transform=eval_transform,
            )
        image_size = 32
        
    elif dataset_name == 'cifar10':
        benchmark = SplitCIFAR10(
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=train_transform,
                eval_transform=eval_transform,
            )
        image_size = 32
        
    elif dataset_name == 'imagenet':
        benchmark = SplitImageNet(
                dataset_root=dataset_root,
                n_experiences=num_exps,
                seed=seed, # Fixed seed for reproducibility
                return_task_id=return_task_id,
                shuffle=shuffle,
                train_transform=train_transform,
                eval_transform=eval_transform,
            )
        image_size = 224
        
    elif dataset_name == 'imagenet100':
        # Select 100 random classes from Imagenet
        random.seed(seed) # Seed for getting always same classes
        classes = random.sample(range(0, 1000), 100)
        benchmark = SplitImageNet(
            dataset_root=dataset_root,
            n_experiences=num_exps,
            fixed_class_order = classes,
            return_task_id=return_task_id,
            shuffle=shuffle,
            train_transform=train_transform,
            eval_transform=eval_transform,
            # class_ids_from_zero_from_first_exp=True ## not allowed for Avalanche < 0.4.0
        )
        image_size = 224

        # Same code as in Avalanche 0.4.0 for enabling "class_ids_from_zero_from_first_exp=True"
        n_original_classes = max(benchmark.classes_order_original_ids) + 1
        benchmark.classes_order = list(range(0, benchmark.n_classes))
        benchmark.class_mapping = [-1] * n_original_classes
        for class_id in range(n_original_classes):
            # This check is needed because, when a fixed class order is
            # used, the user may have defined an amount of classes less than
            # the overall amount of classes in the dataset.
            if class_id in benchmark.classes_order_original_ids:
                benchmark.class_mapping[class_id] = (
                    benchmark.classes_order_original_ids.index(class_id)
                )

    elif dataset_name == 'clear100':

        benchmark = CLEAR(
            data_name='clear100',
            evaluation_protocol=evaluation_protocol_clear,
            feature_type=None,
            seed=seed%5, # allowed seed in 0-4 range
            dataset_root=None,
            train_transform=train_transform,
            eval_transform=eval_transform,
        )
        image_size = 224

    # Create Benchmark object with tr, test (and validation) streams
    tr_stream = []
    valid_stream = []    
    for experience in benchmark.train_stream:
        if val_ratio > 0:
            tr_exp_dataset, val_exp_dataset = class_balanced_split(val_ratio, experience)
            tr_stream.append(tr_exp_dataset)
            valid_stream.append(val_exp_dataset)
        else:
            tr_stream.append(experience.dataset)
    
    if num_exps != len(tr_stream):
        print(f'WARNING: Selected number of experiences {num_exps} is different from default CLEAR100 experiences, resetting to {len(tr_stream)} experiences.')

    test_stream = []
    for experience in benchmark.test_stream:
        test_stream.append(experience.dataset)
    
    if val_ratio > 0:
        benchmark = Benchmark(train_stream=tr_stream, test_stream=test_stream, valid_stream=valid_stream)
    else:
        benchmark = Benchmark(train_stream=tr_stream, test_stream=test_stream)

    return benchmark, image_size

def get_iid_dataset(benchmark: Benchmark):
     iid_dataset_tr = ConcatDataset([tr_exp_dataset for tr_exp_dataset in benchmark.train_stream])
     return iid_dataset_tr

def get_downstream_benchmark(downstream_name, dataset_root, seed=42, val_ratio=0.1):
    if downstream_name == 'svhn':
        train_dataset = SVHN(root=dataset_root, split='train', download=True, transform=get_dataset_transforms(downstream_name))
        test_dataset = SVHN(root=dataset_root, split='test', download=True, transform=get_dataset_transforms(downstream_name))
        if val_ratio > 0:
            train_dataset, val_dataset = torch_val_split(val_ratio, train_dataset)
            return Benchmark(train_stream=[train_dataset], test_stream=[test_dataset], valid_stream=[val_dataset])
        else:
            return Benchmark(train_stream=[train_dataset], test_stream=[test_dataset])
        
    elif downstream_name == 'cars':
        train_dataset = StanfordCars(root=dataset_root, split='train', download=False, transform=get_dataset_transforms(downstream_name))
        test_dataset = StanfordCars(root=dataset_root, split='test', download=False, transform=get_dataset_transforms(downstream_name))
        if val_ratio > 0:
            train_dataset, val_dataset = torch_val_split(val_ratio, train_dataset)
            return Benchmark(train_stream=[train_dataset], test_stream=[test_dataset], valid_stream=[val_dataset])
        else:
            return Benchmark(train_stream=[train_dataset], test_stream=[test_dataset])
        

def torch_val_split(val_ratio, dataset):
    if not 0.0 <= val_ratio <= 1.0:
        raise ValueError("validation_size must be a float in [0, 1].")

    num_tot = len(dataset)
    num_val = int(val_ratio * num_tot)

    # Get the labels for each data point in the training set
    try:
        labels = dataset.labels
    except AttributeError:
        labels = [dataset[i][1] for i in range(num_tot)]

    # Perform stratified split to ensure the validation set is class balanced
    train_indices, val_indices = train_test_split(
        np.arange(num_tot),
        test_size=num_val,
        stratify=labels,  # this ensures class balance
        random_state=42
    )
    # Create Subsets for training and validation using the indices
    train_subset = Subset(dataset, train_indices)
    val_subset = Subset(dataset, val_indices)
    return train_subset, val_subset



def class_balanced_split(validation_size, experience):
    # From Avalanche.benchmarks
    """Class-balanced train/validation splits.

    This splitting strategy splits `experience` into two experiences
    (train and validation) of size `validation_size` using a class-balanced
    split. Sample of each class are chosen randomly.

    """
    if not 0.0 <= validation_size <= 1.0:
        raise ValueError("validation_size must be a float in [0, 1].")

    exp_dataset = experience.dataset

    exp_indices = list(range(len(exp_dataset)))
    exp_classes = experience.classes_in_this_experience

    # shuffle exp_indices
    exp_indices = torch.as_tensor(exp_indices)[torch.randperm(len(exp_indices))]
    # shuffle the targets as well
    exp_targets = torch.as_tensor(experience.dataset.targets)[exp_indices]

    train_exp_indices = []
    valid_exp_indices = []
    for cid in exp_classes:  # split indices for each class separately.
        c_indices = exp_indices[exp_targets == cid]
        valid_n_instances = int(validation_size * len(c_indices))
        valid_exp_indices.extend(c_indices[:valid_n_instances])
        train_exp_indices.extend(c_indices[valid_n_instances:])

    if isinstance(exp_dataset, torch.utils.data.Dataset):
        # Use Subset for older versions of Avalanche where AvalancheDataset is a subclass of torch Dataset
        result_train_dataset = Subset(exp_dataset, train_exp_indices)
        result_valid_dataset = Subset(exp_dataset, valid_exp_indices)
    else:
        # Use .subset for newer versions of Avalanche where AvalancheDataset is not a subclass of torch Dataset
        result_train_dataset = exp_dataset.subset(train_exp_indices)
        result_valid_dataset = exp_dataset.subset(valid_exp_indices)

    return result_train_dataset, result_valid_dataset