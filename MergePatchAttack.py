import copy
import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import scale_bbox, get_size_of_bbox, assert_bbox, tensor2cv2image, clamp
from VisualizeDetection import visualizaion
from utils import tensor2cv2image, get_datetime_str
import torch.distributed as dist
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler
import time
import numpy as np
from criterion import GetPatchLoss
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class LinearMergePatchAttack():
    def __init__(self, model: nn.Module,
                 loader: DataLoader):
        self.model = model
        self.loader = loader
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.loss = GetPatchLoss(model, loader)

    @torch.no_grad()
    def draw_landscape(self):
        p1 = torch.load('fasterrcnn.pth').to(self.device).detach()
        p2 = torch.load('ssd.pth').to(self.device).detach()
        p1.requires_grad_(False)
        p2.requires_grad_(False)

        x = np.arange(-2, 2, 0.1)
        coordinate_x, coordinate_y = np.meshgrid(x, x)
        result = []
        for i in range(coordinate_x.shape[0]):
            for j in tqdm(range(coordinate_x.shape[1])):
                x = coordinate_x[i, j]
                y = coordinate_y[i, j]
                now = clamp(x * p1 + y * p2)
                result.append(self.loss(now))

        result = np.array(result)
        result = result.reshape(coordinate_x.shape)
        self.draw_figure(coordinate_x, coordinate_y, result)
        np.save('result.ckpt', result)

    @staticmethod
    def draw_figure(mesh_x, mesh_y, mesh_z):
        figure = plt.figure()
        axes = Axes3D(figure)

        axes.plot_surface(mesh_x, mesh_y, mesh_z, cmap='rainbow')
        plt.show()

        plt.savefig(get_datetime_str() + ".png")

    def linear_merge_patch_and_attack(self,
                                      attack_epoch=1,
                                      attack_step=1000,
                                      lr=5,
                                      aug_image=False,
                                      fp_16=False) -> torch.tensor:
        '''
        use nesterov
        :param x:image
        :param model: detection model, whose output is pytorch detection style
        :return:
        '''
        for s in self.model.modules():
            s.requires_grad_(False)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if aug_image:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip().to(device),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1).to(device),
            ])
        p1 = torch.load('fasterrcnn.pth').to(device).detach()
        p2 = torch.load('ssd.pth').to(device).detach()
        p1.requires_grad_(False)
        p2.requires_grad_(False)
        # optimizer = torch.optim.SGD([adv_x], lr=1e-2)
        criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))
        l1 = torch.tensor(1., requires_grad=True, device=device)
        l2 = torch.tensor(1., requires_grad=True, device=device)
        optimizer = torch.optim.SGD([l1, l2], lr=lr)

        for epoch in range(1, attack_epoch + 1):
            total_loss = 0
            self.loader.sampler.set_epoch(epoch)
            pbar = self.loader
            for step, image in enumerate(pbar):
                with torch.no_grad():
                    image = image.to(device)
                    if aug_image:
                        image = transform(image)
                    predictions = self.model(image)
                    if len(predictions) == 0:
                        continue

                adv_x = clamp(l1 * p1 + l2 * p2)

                # interpolate the patch into images
                for i, pred in enumerate(predictions):
                    scores = pred["scores"]
                    mask = scores > 0.5
                    boxes = pred["boxes"][mask]
                    for now_box_idx in range(boxes.shape[0]):
                        now_box = boxes[now_box_idx]
                        by1, bx1, by2, bx2 = scale_bbox(*tuple(now_box.detach().cpu().numpy().tolist()))
                        if not assert_bbox(bx1, by1, bx2, by2):
                            continue
                        now = F.interpolate(adv_x.unsqueeze(0),
                                            size=get_size_of_bbox(bx1, by1, bx2, by2),
                                            mode='bilinear')
                        try:
                            image[i, :, bx1:bx2, by1:by2] = now
                        except:
                            print(image.shape, now.shape)

                if not fp_16:
                    predictions = self.model(image)
                    if len(predictions) == 0:
                        continue
                    final_scores = []
                    for pred in predictions:
                        scores = pred["scores"]
                        mask = scores > 0.3
                        scores = scores[mask]
                        final_scores.append(scores)
                    if len(final_scores) == 0:
                        continue
                    scores = torch.cat(final_scores, dim=0)
                    loss = criterion(scores)
                    optimizer.zero_grad()
                    loss.backward()

                optimizer.step()
                total_loss += loss.item()
                if step % 5 == 0:
                    print(l1.item(), l2.item(), f'loss={total_loss / (step + 1)}')
                    if step % 100 == 0:
                        torch.save(adv_x.detach(), 'patch.pth')
                        img_x = tensor2cv2image(adv_x.detach().clone())
                        cv2.imwrite('patch.jpg', img_x)
                    if step >= attack_step:
                        return
            print(epoch, total_loss / len(self.loader))

        visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
        time.sleep(2)
        return
