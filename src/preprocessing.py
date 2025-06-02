'''
Preprocessing module for ISIC dataset.
'''
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms

class ISICDataset(Dataset):
    """
    Custom dataset for ISIC skin lesion images.
    This dataset reads images and their corresponding labels from a CSV file.
    Args:
        csv_file (str or pd.DataFrame): Path to the CSV file containing image paths and labels.
        transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
        resize (bool, optional): If True, resizes images to a fixed size.
    """
    def __init__(self, csv_file, transform=None, resize=False):
        """
        Initializes the ISICDataset.
        Args:
            csv_file (str or pd.DataFrame): Path to the CSV file containing image paths and labels.
            transform (callable, optional): A function/transform that takes in an PIL image and returns a transformed version.
            resize (bool, optional): If True, resizes images to a fixed size.
        """
        self.data = csv_file
        self.transform = transform
        self.resize = resize

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        Returns:
            int: Number of samples in the dataset.
        """
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Retrieves an item from the dataset.
        Args:
            idx (int): Index of the item to retrieve.
        Returns:
            tuple: (image, label) where image is a PIL image and label is a tensor.
        """
        row = self.data.iloc[idx]
        img = Image.open(row['image_path']).convert("RGB")
        label = torch.tensor(row['label'], dtype=torch.long)
        
        if self.transform:
            img = self.transform(img)
        return img, label

def get_transforms(image_size, horizontal_flip=False, 
                   vertical_flip=False, rotation=False,
                   rotation_degrees=10):
    """
    Creates and returns the transformations for training and validation datasets.
    Args:
        image_size (tuple): Target size for the images (width, height).
        horizontal_flip (bool): If True, applies random horizontal flip.
        vertical_flip (bool): If True, applies random vertical flip.
        rotation (bool): If True, applies random rotation.
        rotation_degrees (int): Degrees for random rotation if rotation is True.
    Returns:
        tuple: (train_transforms, val_transforms) where each is a torchvision.transforms.Compose object.
    """
    train_transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    if horizontal_flip:
        train_transform_list.append(transforms.RandomHorizontalFlip(p=0.5))
    if vertical_flip:
        train_transform_list.append(transforms.RandomVerticalFlip(p=0.5))
    if rotation:
        train_transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))
    
    train_transforms = transforms.Compose(train_transform_list)

    val_transforms = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return train_transforms, val_transforms
