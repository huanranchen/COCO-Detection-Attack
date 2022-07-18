import random
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms


class MyDataset(Dataset):
    def __init__(self, path, mode, augment=False):
        self.mode = mode
        self.files = sorted([os.path.join(path, x) for x in os.listdir(path) if x.endswith('jpg') or x.endswith('png')])
        if augment:
            self.transforms = torchvision.transforms.Compose([
                transforms.RandAugment(1, 5),
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
            ])
        else:
            self.transforms = torchvision.transforms.Compose([
                transforms.Resize((384, 384)),
                transforms.ToTensor(),
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        im = Image.open(fname)
        im = self.transforms(im)
        if im.shape[0] == 1:
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        return im


def get_loader(train_path="/data2/huanghao/COCO/images/train2017/",
               mode='train',
               batch_size=16, num_workers=8,
               pin_memory=True,
               augment=False):
    set = MyDataset(path=train_path, mode=mode, augment=augment)
    train_sampler = torch.utils.data.distributed.DistributedSampler(set)
    train_loader = DataLoader(set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                              sampler=train_sampler)
    return train_loader


if __name__ == '__main__':
    loader = get_loader(train_path='/data2/huanghao/COCO/images/train2017/', mode='train')
    print(loader)
