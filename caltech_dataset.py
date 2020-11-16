from torchvision.datasets import VisionDataset

from PIL import Image

import os
import os.path
import sys
    

def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
    

class Caltech(VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        super(Caltech, self).__init__(root, transform=transform, target_transform=target_transform)

        self.split = split # This defines the split you are going to use
                           # (split files are called 'train.txt' and 'test.txt')

        '''
        - Here you should implement the logic for reading the splits files and accessing elements
        - If the RAM size allows it, it is faster to store all data in memory
        - PyTorch Dataset classes use indexes to read elements
        - You should provide a way for the __getitem__ method to access the image-label pair
          through the index
        - Labels should start from 0, so for Caltech you will have lables 0...100 (excluding the background class) 
        '''
        # classes (list): List of the class names sorted alphabetically.
        # class_to_idx (dict): Dict with items (class_name, class_index).
        # samples (list): List of (sample path, class_index) tuples
        # targets (list): The class_index value for each image in the dataset
        classes, class_to_idx = self._find_classes(self.root)
        classes.remove('BACKGROUND_Google')
        class_to_idx.pop('BACKGROUND_Google')
        samples = []
        if split == 'train':
            dataset_path = root + '/../train.txt'
        else:
            dataset_path = root + '/../test.txt'
        fp = open(dataset_path, 'r')
        lines = fp.readlines()
        for line in lines:
            line_dir = line.split('/')[0]
            if line_dir != 'BACKGROUND_Google':
                image = pil_loader(root + '/' + line.replace('\n',''))
                sample = image, class_to_idx[line_dir]
                samples.append(sample)
        self.samples = samples
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.targets = [s[1] for s in samples]
        
    def _find_classes(self, dir):
        '''
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        '''
        classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        classes.sort()
        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
        return classes, class_to_idx
        
    def __getitem__(self, index):
        '''
        __getitem__ should access an element through its index
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        '''
        path, target = self.samples[index]
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target
        image, label = ... # Provide a way to access image and label via index
                           # Image should be a PIL Image
                           # label can be int

        # Applies preprocessing when accessing the image
        if self.transform is not None:
            image = self.transform(image)

        return image, label

    def __len__(self):
        '''
        The __len__ method returns the length of the dataset
        It is mandatory, as this is used by several other components
        '''
        length = len(self.samples) # Provide a way to get the length (number of elements) of the dataset
        return length
        
