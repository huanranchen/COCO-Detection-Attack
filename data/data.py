import random
import torch
import torchvision.transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
import torchvision.transforms as transforms
from collections import defaultdict
import json
from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import Dataset


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


class CocoDataset(Dataset):
    """PyTorch dataset for COCO annotations."""

    def __init__(self, root, annFile):
        """Load COCO annotation data."""
        self.data_dir = Path(root)
        self.transforms = torchvision.transforms.Compose([
            transforms.ToTensor(),
        ])

        # load the COCO annotations json
        anno_file_path = annFile
        with open(str(anno_file_path)) as file_obj:
            self.coco_data = json.load(file_obj)
        # put all of the annos into a dict where keys are image IDs to speed up retrieval
        self.image_id_to_annos = defaultdict(list)
        for anno in self.coco_data['annotations']:
            image_id = anno['image_id']
            self.image_id_to_annos[image_id] += [anno]

        self.device = torch.device('cuda')

    def __len__(self):
        return len(self.coco_data['images'])

    def __getitem__(self, index):
        """Return tuple of image and labels as torch tensors."""
        image_data = self.coco_data['images'][index]
        image_id = image_data['id']
        image_path = self.data_dir / image_data['file_name']
        image = Image.open(image_path)

        annos = self.image_id_to_annos[image_id]
        anno_data = {
            'boxes': [],
            'labels': [],
            'area': [],
            'iscrowd': [],
        }
        for anno in annos:
            coco_bbox = anno['bbox']
            left = coco_bbox[0]
            top = coco_bbox[1]
            right = coco_bbox[0] + coco_bbox[2]
            bottom = coco_bbox[1] + coco_bbox[3]
            area = coco_bbox[2] * coco_bbox[3]
            anno_data['boxes'].append([left, top, right, bottom])
            anno_data['labels'].append(anno['category_id'])
            anno_data['area'].append(area)
            anno_data['iscrowd'].append(anno['iscrowd'])

        if len(annos) == 0:
            target = {
                'boxes': torch.randn(0, 4, device=self.device),
                'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64, device=self.device),
                'image_id': torch.tensor([image_id], device=self.device),  # pylint: disable=not-callable (false alarm)
                'area': torch.as_tensor(anno_data['area'], dtype=torch.float32, device=self.device),
                'iscrowd': torch.as_tensor(anno_data['iscrowd'], dtype=torch.int64, device=self.device),
            }
        else:
            target = {
                'boxes': torch.as_tensor(anno_data['boxes'], dtype=torch.float32, device=self.device),
                'labels': torch.as_tensor(anno_data['labels'], dtype=torch.int64, device=self.device),
                'image_id': torch.tensor([image_id], device=self.device),  # pylint: disable=not-callable (false alarm)
                'area': torch.as_tensor(anno_data['area'], dtype=torch.float32, device=self.device),
                'iscrowd': torch.as_tensor(anno_data['iscrowd'], dtype=torch.int64, device=self.device),
            }

        # if self.transforms is not None:
        #     image, target = self.transforms(image, target)
        image = self.transforms(image).cuda()

        return image, target


def get_coco_loader(batch_size=1,
                    num_workers=0,
                    pin_memory=False,
                    ):
    def collate_fn(batch):
        result_x = []
        result_y = []
        for x, y in batch:
            result_x.append(x.unsqueeze(0))
            result_y.append(y)
        return result_x, result_y

    set = CocoDataset(root='/home/nico/data/coco/train2017/',
                      annFile='/home/nico/data/annotations/instances_train2017.json',
                      )
    train_sampler = torch.utils.data.distributed.DistributedSampler(set)
    train_loader = DataLoader(set, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=pin_memory,
                              sampler=train_sampler, collate_fn=collate_fn)
    return train_loader


if __name__ == '__main__':
    loader = get_loader(train_path='/data2/huanghao/COCO/images/train2017/', mode='train')
    print(loader)
