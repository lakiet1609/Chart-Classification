import os
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import transforms
import sys
sys.path.insert(0, '../src')

from utils import find_classes

class Custom_Image_Folder(Dataset):
    def __init__(self, target_dir, transform):
        self.paths = list(Path(target_dir).glob('*/*.jpg'))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir=target_dir)

    def load_img(self, idx):
        load_img  = self.paths[idx]
        return Image.open(load_img).convert('RGB')

    def __len__(self):
        return len(self.paths)

    def __getitem__(self,idx):
        image = self.load_img(idx=idx)
        class_name = self.paths[idx].parent.name
        class_idx = self.class_to_idx[class_name]
        if self.transform:
            return self.transform(image), class_idx
        else:
            return image, class_idx


def Load_Data(train_dir: str,
              test_dir: str,
              batch_size: int,
              num_workers: int,
              transform
):

    
    train_data = Custom_Image_Folder(train_dir, transform)
    test_data = Custom_Image_Folder(test_dir, transform)

    class_weights = []
    for root, subdir, files in os.walk(train_dir):
        if len(files) > 0:
            class_weights.append(1/len(files))

    sample_weights = [0] * len(train_data)

    for idx, (data,label) in enumerate(train_data):
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    #Turn into dataloaders
    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=batch_size,
                                  num_workers=num_workers,
                                  sampler=sampler)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=batch_size,
                                 num_workers=num_workers,
                                 shuffle=False)

    return train_dataloader, test_dataloader


