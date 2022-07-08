import os.path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import scale_bbox, get_size_of_bbox, assert_bbox, tensor2cv2image


def attack_detection(x: torch.tensor, model: nn.Module, attack_step=10) -> torch.tensor:
    '''
    Not patch attack. Only detection attack
    :param x:
    :param model: detection model, whose output is pytorch detection style
    :return:
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adv_x = x.clone().to(device)
    adv_x.requires_grad = True
    optimizer = torch.optim.AdamW([adv_x], lr=1e-3)
    criterion = lambda x: F.mse_loss(x, torch.tensor([0.0], device=device))

    for step in range(1, attack_step + 1):
        predictions = model(adv_x)
        # print(predictions)
        if len(predictions) == 0:
            return adv_x.detach()
        pred = predictions[0]
        scores = pred["scores"]
        mask = scores > 0.5
        scores = scores[mask]
        loss = criterion(scores)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(step, loss.item(), scores)

    return adv_x.detach()


def patch_attack_detection(model: nn.Module,
                           loader: DataLoader,
                           attack_epoch=2,
                           attack_step=1000000,
                           patch_size=(3, 64, 64)) -> torch.tensor:
    '''
    Not patch attack. Only detection attack
    :param x:image
    :param model: detection model, whose output is pytorch detection style
    :return:
    '''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    if os.path.exists('patch.pth'):
        adv_x = torch.load('patch.pth')
    else:
        adv_x = torch.clamp(torch.randn(patch_size) / 2 + 1, 0, 1)
    adv_x.requires_grad = True
    optimizer = torch.optim.AdamW([adv_x], lr=1e-3)
    criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))

    for epoch in range(1, attack_epoch + 1):
        total_loss = 0
        pbar = tqdm(loader)
        for step, image in enumerate(pbar):
            image = image.to(device)
            predictions = model(image)
            if len(predictions) == 0:
                continue

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
            # image[:, :, :patch_size[1], :patch_size[2]] = adv_x

            predictions = model(image)

            # from VisualizeDetection import visualizaion
            # from utils import tensor2cv2image
            # visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
            # assert False

            if len(predictions) == 0:
                continue
            final_scores = []
            for pred in predictions:
                scores = pred["scores"]
                mask = scores > 0.5
                scores = scores[mask]
                final_scores.append(scores)
            scores = torch.cat(final_scores, dim=0)
            loss = criterion(scores)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix_str(f'loss={total_loss / (step + 1)}')
                if step >= attack_step:
                    torch.save(adv_x, 'patch.pth')
                    adv_x = tensor2cv2image(adv_x.detach().cpu())
                    cv2.imwrite('patch.jpg', adv_x)
                    return adv_x
        print(epoch, total_loss / len(loader))

    torch.save(adv_x, 'patch.pth')

    adv_x = tensor2cv2image(adv_x)
    cv2.imwrite('patch.jpg', adv_x)
    return adv_x
