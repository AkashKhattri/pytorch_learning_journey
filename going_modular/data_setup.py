"""
Contains functionality for creating PyTorch DataLoader's for image classification data.
"""
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = os.cpu_count()

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    batch_size: int,
    num_worksers: int=NUM_WORKERS
):
    """
    Creates taraining a testing DataLoaders

    Takes in a training directory and testing directory path and turns them into PyTorch Datasets and then into PyTorch Dataloaders.

    Args:
        train_dir: Path to training directory.
        test_dir: Path to testing directory.
        transform: torchvision transforms to perform on training and testing data.
        batch_size: Number of samples per batch in each of the DataLoaders.
        num_workers: An integer for number of workers per DataLoader.

    Returns:
        A tuple of (train_dataloader, test_dataloader, class_names)
        Where class_names is a list of the target classes.
        Example usage:
            train_dataloader, test_dataloader, class_names = create_dataloaders(
                                train_dir=pat/to/train_dir,
                                test_dir=path/to/test_dir,
                                batch_size=32.
                                num_workers=4
                )
    """
    #Use ImageFolder to create datasets(s)
    train_data = datasets.ImageFolder(root=train_dir,
                                    transform=transform, # a transform for the data
                                    target_transform=None) # a transform for the label/target

    test_data = datasets.ImageFolder(root=test_dir, transform=transform)

    # Get class names
    class_names = train_data.classes

    # Turn images into Dataloaders
    train_dataloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=num_worksers, pin_memory=True)
    test_dataloader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, num_workers=num_worksers, pin_memory=True)

    train_dataloader, train_dataloader
    return  train_dataloader, test_dataloader, class_names
