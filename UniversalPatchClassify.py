import os
import torch
from torch import nn
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from utils import tensor2cv2image
import cv2
from matplotlib import pyplot as plt


class Classifier():
    def __init__(self, path='./natural/'):
        self.cnn = nn.Conv2d(3, 7, kernel_size=3, stride=1, padding=1)
        images_names = [i for i in os.listdir(path) if i.endswith('png')]
        if os.path.exists(path + 'dictionary.pth'):
            self.dictionary = torch.load(path + 'dictionary.pth')
        else:
            self.dictionary = {}
            for name in images_names:
                self.dictionary[name] = len(self.dictionary)
            torch.save(self.dictionary, path + 'dictionary.pth')
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # load images into memory
        transform = transforms.ToTensor()
        self.transform = transform
        x, y = [], []
        for name in images_names:
            x.append(transform(Image.open(path + name)))
            y.append(self.dictionary[name])
        self.x = torch.stack(x).to(self.device)
        self.y = torch.tensor(y, device=self.device)

        # load model
        self.path = path
        if os.path.exists(path + 'model.ckpt'):
            self.cnn.load_state_dict(torch.load(path + 'model.ckpt'))

    def train(self, total_epoch=10000,
              lr=1e-2,
              ):
        self.cnn.to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(self.cnn.parameters(), lr=lr, momentum=0.9)
        pbar = tqdm(range(1, total_epoch))
        for epoch in pbar:
            pre = self.cnn(self.x)
            pre = torch.mean(pre, dim=[2, 3])
            loss = criterion(pre, self.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix_str(f'step {epoch}, loss {loss.item()}')

        torch.save(self.cnn.state_dict(), self.path + 'model.ckpt')

    def predict_and_visualize(self, path: str):
        '''
        given a image, output a image use color to represent the most similar patch
        '''
        x = self.transform(Image.open(path)).unsqueeze(0)
        pre = self.cnn(x).squeeze()

        # change the color for visualize
        pre *= (255 / len(self.dictionary))
        pre = tensor2cv2image(pre)
        plt.imshow(pre)


if __name__ == '__main__':
    a = Classifier()
    a.train()
