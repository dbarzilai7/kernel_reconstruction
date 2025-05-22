from typing import Any, Callable, Optional, Tuple
from  torchvision.datasets import VisionDataset
import os
import pickle
import numpy as np
from PIL import Image


class CIFAR5M(VisionDataset):
    base_folder = 'CIFAR5M'
    file_list = ["cifar5m_part0.npz", "cifar5m_part1.npz", "cifar5m_part2.npz", "cifar5m_part3.npz", "cifar5m_part4.npz", "cifar5m_part5.npz"]
    root_directory = './data/'

    
    def __init__(
            self,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
        ) -> None:

            super().__init__(self.root_directory, transform=transform, target_transform=target_transform)        
            

        
        
            self.data = []
            self.targets = []

            # now load the picked numpy arrays
            for file_name in self.file_list:
                file_path = os.path.join(self.root, self.base_folder, file_name)
                with open(file_path, "rb") as f:
                    entry = np.load(f, )
                    self.data.append(entry["X"])               
                    self.targets.extend(entry["Y"])
                    
            self.data = np.vstack(self.data).reshape(-1, 32,32,3)
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)