import torch
from torchvision import transforms

from PIL import ImageFilter, ImageOps, Image
import random
import numpy as np


def get_dataset_transforms(dataset: str):
    """Get corresponding normalization transform for each dataset."""

    if dataset == 'cifar100':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)
        )])
    elif dataset == 'cifar10':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
        )])
    elif dataset == 'tinyimagenet':
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    )])
    elif dataset in ['imagenet100', 'imagenet']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(
            (0.485, 0.456, 0.406), (0.228, 0.224, 0.225)
    )])
    elif dataset in ['clear10', 'clear100']:
        return transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.Normalize(
            (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    )])
    elif dataset == 'svhn':
        return transforms.Compose([
                    transforms.Resize(32),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    elif dataset == 'cars':
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
            (0.4707, 0.4602, 0.4550), (0.2638, 0.2629, 0.2678)
    )]) 
    

    else:
        raise ValueError(f'Base Trandforms for dataset "{dataset}" not supported')


def get_dataset_crop(dataset: str):
    """Get corresponding crop transform for each dataset."""

    if dataset == 'cifar100':
        # return transforms.RandomCrop(32, padding=4),
        return transforms.RandomResizedCrop(32, scale=(0.2, 1.))
    elif dataset == 'cifar10':
        # return transforms.RandomCrop(32, padding=4)
        return transforms.RandomResizedCrop(32, scale=(0.2, 1.))
    elif dataset in ['imagenet100', 'imagenet', 'clear100']:
        # return transforms.RandomCrop(64, padding=8)
        return transforms.RandomResizedCrop(224, scale=(0.2, 1.))
    else:
        raise ValueError("Dataset not supported.")

 
class MultipleCropsTransform:
    """Take N random augmented views of one image."""
    def __init__(self, base_transform, n_crops=20):
        self.base_transform = base_transform
        self.n_crops = n_crops

    def __call__(self, x):
        stacked_views = [] # List of tensors, each contains one view for all samples in x
        for _ in range(self.n_crops):
            view_list = [self.base_transform(sample) for sample in x]
            stacked_views.append(torch.stack(view_list, dim=0))
            
        return stacked_views
        

class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    

class Solarization:
    """Solarization as a callable object."""

    def __call__(self, img: Image) -> Image:
        """Applies solarization to an input image.

        Args:
            img (Image): an image in the PIL.Image format.

        Returns:
            Image: a solarized image.
        """

        return ImageOps.solarize(img)
        
def clamp_transform(image):
    # Clamping operation here
    return torch.clamp(image, min=0, max=1)

def get_transforms_simsiam(dataset: str = 'cifar100'):
    """Returns SimSiam augmentations with dataset specific crop."""

    all_transforms = [
        get_dataset_crop(dataset),
        transforms.RandomApply(
            [transforms.Lambda(clamp_transform),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.4, hue=0.1)]
                                        , p=0.8),
        transforms.RandomGrayscale(p=0.2),
        # transforms.RandomApply([transforms.GaussianBlur([.1, 2.])], p=0.5),
        transforms.RandomHorizontalFlip()
    ]
    
    return all_transforms


def get_transforms_barlow_twins(dataset: str = 'cifar100'):
    """Returns Barlow Twins augmentations with dataset specific crop."""
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(clamp_transform),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            #GaussianBlur(p=1.0),
            #Solarization(p=0.0),
        ]

    return all_transforms

def get_transforms_byol(dataset: str = 'cifar100'):
    """Returns BYOL augmentations with dataset specific crop."""
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply(
                [transforms.Lambda(clamp_transform),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1),
                                        ],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            # GaussianBlur(sigma=[.1, 2.]),
        ]
    return all_transforms

def get_transforms_mae(dataset: str = 'cifar100'):
    """Returns MAE augmentations with dataset specific crop."""
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(),
        ]
    return all_transforms

def get_transforms_emp(dataset: str = 'cifar100'):
    """Returns EMP augmentations with dataset specific crop."""
    normalize = transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])

    if dataset in ['cifar10', 'cifar100']:
        blur_kernel = 5
        crop = transforms.RandomResizedCrop(32,scale=(0.25, 0.25), ratio=(1,1))
        
    elif dataset in ['imagenet', 'imagenet100']:
        blur_kernel = 23 # Same as SwAV
        transforms.RandomResizedCrop(224, scale=(0.25, 0.25),
                                     interpolation=transforms.InterpolationMode.BICUBIC)
        
    
    all_transforms = [   
        crop,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([transforms.GaussianBlur(blur_kernel)], p=0.1),
        transforms.RandomSolarize(threshold=0.5 ,p=0.1), # threshold chosen from PIL solarize implementation
        normalize
    ]
    return all_transforms
 

def get_common_transforms(dataset: str = 'cifar100'):
    "Common transforms for self supervised models for better comparison"
    all_transforms = [
            get_dataset_crop(dataset),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.Lambda(clamp_transform),
                transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ]

    return all_transforms
    

def get_transforms(dataset: str, model: str, n_crops: int = 2):
    """Returns augmentations for self supervised models"""

    if model == "simsiam":
        all_transforms = get_transforms_simsiam(dataset)

    elif model == "barlow_twins":
        all_transforms = get_transforms_barlow_twins(dataset)

    elif model == "byol":
        all_transforms = get_transforms_byol(dataset)

    elif model in ['emp', 'simsiam_cmp', 'byol_cmp']:
        all_transforms = get_transforms_emp(dataset) 

    elif model == "mae":
        all_transforms = get_transforms_mae(dataset)

    elif model == "mae_cmp":
        all_transforms = get_transforms_mae(dataset)

    elif model == "common":
        all_transforms = get_common_transforms(dataset)

    else:
        raise ValueError(f"Model {model} not supported")

    return MultipleCropsTransform(transforms.Compose(all_transforms), n_crops)

def multipatch_transforms(dataset: str, n_crops: int = 2):
    """Returns multipatch augmentations (EMP augmentations)"""

class MultiPatchTransforms(object):
    """Multipatch augmented views generator to be applied on dataset."""
    def __init__(self, num_patch = 2, dataset_name = 'cifar100'):
        self.num_patch = num_patch
        if dataset_name in ['cifar10', 'cifar100']:
            self.crop = transforms.RandomResizedCrop(32,scale=(0.25, 0.25), ratio=(1,1))
        elif dataset_name in ['imagenet', 'imagenet100', 'clear100']:
            self.crop = transforms.RandomResizedCrop(224, scale=(0.25, 0.25), interpolation=transforms.InterpolationMode.BICUBIC)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported in MultiPatchTransforms crop selection.")
        
    def __call__(self, x):
        aug_transform =  transforms.Compose([
            self.crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            GBlur(p=0.1),
            transforms.RandomApply([Solarization()], p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        augmented_x = [aug_transform(x) for i in range(self.num_patch)]
        return augmented_x
    

class MultiPatchMAETransforms(object):
    """Multipatch augmented views generator to be applied on dataset for MAE training."""
    def __init__(self, num_patch = 2, dataset_name = 'cifar100'):
        self.num_patch = num_patch
        if dataset_name in ['cifar10', 'cifar100']:
            self.crop = transforms.RandomCrop(32, padding=4)
        elif dataset_name in ['imagenet', 'imagenet100', 'clear100']:
            self.crop = transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3)
        else:
            raise ValueError(f"Dataset {dataset_name} not supported in MultiPatchMAETransforms crop selection.")
        
    def __call__(self, x):
        aug_transform =  transforms.Compose([
            self.crop,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        augmented_x = [aug_transform(x) for i in range(self.num_patch)]
        return augmented_x
    
class GBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            sigma = np.random.rand() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img

