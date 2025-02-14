from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision.datasets import ImageFolder
import os

class AlbumentationsDataset(ImageFolder):
    def __getitem__(self, index: int):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = np.array(sample)
            sample = self.transform(image=sample)["image"]
        if self.target_transform is not None:
            target = self.target_transform(target)

        #{0:'bothcells',1:'healthy',2:'rubbish',3:'unhealthy'}
        if target==3:
            target = 0

        return sample, target

class InferenceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = [os.path.join(root_dir, img) for img in os.listdir(root_dir) if img.endswith(('jpg', 'jpeg', 'png'))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path)
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        if self.transform:
            image = np.array(image)
            image = self.transform(image=image)["image"]

        return image, img_path.split('/')[-1]
