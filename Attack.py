import copy
import os.path
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import scale_bbox, get_size_of_bbox, assert_bbox, tensor2cv2image, clamp
from VisualizeDetection import visualizaion
from utils import tensor2cv2image
import torch.distributed as dist
from torchvision import transforms
from torch.cuda.amp import autocast, GradScaler


def reduce_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)
    return rt


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
                           attack_epoch=1000,
                           attack_step=10000,
                           patch_size=(3, 100, 100),
                           m=0.9,
                           use_sign=False,
                           lr=2,
                           aug_image=False,
                           fp_16=False) -> torch.tensor:
    '''
    use nesterov
    :param x:image
    :param model: detection model, whose output is pytorch detection style
    :return:
    '''
    scaler = GradScaler()
    for s in model.modules():
        s.requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if aug_image:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip().to(device),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1).to(device),
        ])
    if os.path.exists('patch.pth'):
        adv_x = torch.load('patch.pth')
    else:
        adv_x = torch.clamp(torch.randn(patch_size) / 2 + 1, 0, 1)
    adv_x.requires_grad = True
    momentum = 0
    # optimizer = torch.optim.SGD([adv_x], lr=1e-2)
    criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))

    for epoch in range(1, attack_epoch + 1):
        total_loss = 0
        pbar = tqdm(loader)
        loader.sampler.set_epoch(epoch)
        for step, image in enumerate(pbar):
            with torch.no_grad():
                image = image.to(device)
                if aug_image:
                    image = transform(image)
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

            if not fp_16:
                predictions = model(image)
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
                loss.backward()
            else:
                raise NotImplementedError
                with autocast():
                    predictions = model(image)
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
                scaler.scale(loss).backward()
                scaler.unscale_(adv_x.grad)

            grad = adv_x.grad.clone()
            adv_x.requires_grad = False
            if use_sign:
                adv_x = clamp(adv_x - lr * grad.sign())
            else:
                momentum = m * momentum - grad
                adv_x += lr * (-grad + m * momentum)
                adv_x = clamp(adv_x)
            adv_x.requires_grad = True
            # optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix_str(f'loss={total_loss / (step + 1)}')
                if step >= attack_step:
                    torch.save(adv_x, 'patch.pth')
                    adv_x = tensor2cv2image(adv_x.detach().cpu())
                    cv2.imwrite('patch.jpg', adv_x)
                    return adv_x
                if step % 1000 == 0:
                    torch.save(adv_x, 'patch.pth')
                    img_x = tensor2cv2image(adv_x.detach().clone())
                    cv2.imwrite('patch.jpg', img_x)
        print(epoch, total_loss / len(loader))

    torch.save(adv_x, 'patch.pth')

    adv_x = tensor2cv2image(adv_x.detach())
    cv2.imwrite('patch.jpg', adv_x)

    visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
    import time
    time.sleep(2)
    return adv_x


def SAM_patch_attack_detection(model: nn.Module,
                               loader: DataLoader,
                               attack_epoch=1000,
                               attack_step=10000,
                               patch_size=(3, 100, 100),
                               m=0.9,
                               use_sign=False,
                               lr=0.5,
                               aug_image=False,
                               fp_16=False) -> torch.tensor:
    '''
    use nesterov
    :param x:image
    :param model: detection model, whose output is pytorch detection style
    :return:
    '''
    for s in model.modules():
        s.requires_grad_(False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if aug_image:
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip().to(device),
            transforms.ColorJitter(0.1, 0.1, 0.1, 0.1).to(device),
        ])
    if os.path.exists('patch.pth'):
        adv_x = torch.load('patch.pth').cuda()
    else:
        adv_x = torch.clamp(torch.randn(patch_size) / 2 + 1, 0, 1).cuda()
    adv_x.requires_grad = True
    momentum = 0
    # optimizer = torch.optim.SGD([adv_x], lr=1e-2)
    criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))

    for epoch in range(1, attack_epoch + 1):
        total_loss = 0
        pbar = tqdm(loader)
        loader.sampler.set_epoch(epoch)
        for step, image in enumerate(pbar):
            # get the ground truth
            with torch.no_grad():
                image = image.to(device)
                if aug_image:
                    image = transform(image)
                predictions = model(image)
                if len(predictions) == 0:
                    continue
            # end

            # interpolate the patch into images
            now_image = copy.deepcopy(image)
            gt = copy.deepcopy(predictions)
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
                        now_image[i, :, bx1:bx2, by1:by2] = now
                    except:
                        print(now_image.shape, now.shape)

            if not fp_16:
                predictions = model(now_image)
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
                loss.backward()
            # end

            # first step of SAM
            grad = adv_x.grad.clone()
            grad = reduce_mean(grad)
            adv_x.requires_grad = False
            adv_x += lr / 50 * grad  # 这里我也不清楚是加grad好还是加grad.sign好
            adv_x.requires_grad = True
            # end

            # the twice
            # interpolate the patch into images
            for i, pred in enumerate(gt):
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

            # second step. normal step
            if not fp_16:
                predictions = model(image)
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
                loss.backward()

            grad = adv_x.grad.clone()
            grad = reduce_mean(grad)
            adv_x.requires_grad = False
            if use_sign:
                adv_x = clamp(adv_x - lr * grad.sign())
            else:
                momentum = m * momentum - grad
                adv_x += lr * (-grad + m * momentum)
                adv_x = clamp(adv_x)
            adv_x.requires_grad = True
            # optimizer.step()
            total_loss += loss.item()
            if step % 10 == 0:
                pbar.set_postfix_str(f'loss={total_loss / (step + 1)}')
                if step >= attack_step:
                    torch.save(adv_x.detach(), 'patch.pth')
                    adv_x = tensor2cv2image(adv_x.detach().cpu())
                    cv2.imwrite('patch.jpg', adv_x)
                    return adv_x
                if step % 1000 == 0:
                    torch.save(adv_x.detach(), 'patch.pth')
                    img_x = tensor2cv2image(adv_x.detach().clone())
                    cv2.imwrite('patch.jpg', img_x)
        print(epoch, total_loss / len(loader))

    torch.save(adv_x, 'patch.pth')

    adv_x = tensor2cv2image(adv_x.detach())
    cv2.imwrite('patch.jpg', adv_x)

    visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
    import time
    time.sleep(2)
    return adv_x


class AttackWithPerturbedNeuralNetwork():
    def __init__(self, model: nn.Module,
                 loader: DataLoader):
        '''
        perturb的时候不确定！！！！！！！！！！！！！！！！！！！！！！！！！
        先实验看看能不能用！！！！！！！！！！！！！！！！！！！！！！！！！！！！
        :param model:
        :param loader:
        '''
        raise NotImplementedError
        self.model = model
        self.loader = loader
        self.original_model = copy.deepcopy(model)

    def patch_attack_detection(self,
                               attack_epoch=1000,
                               attack_step=10000,
                               patch_size=(3, 100, 100)) -> torch.tensor:
        '''
        Not patch attack. Only detection attack
        :param x:image
        :param model: detection model, whose output is pytorch detection style
        :return:
        '''
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if os.path.exists('patch.pth'):
            adv_x = torch.load('patch.pth')
        else:
            adv_x = torch.clamp(torch.randn(patch_size) / 2 + 1, 0, 1)
        adv_x.requires_grad = True
        # optimizer = torch.optim.SGD([adv_x], lr=1e-2)
        criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))

        for epoch in range(1, attack_epoch + 1):
            total_loss = 0
            pbar = tqdm(self.loader)
            self.loader.sampler.set_epoch(epoch)
            for step, image in enumerate(pbar):
                image = image.to(device)
                predictions = self.model(image)
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

                predictions = self.model(image)

                # from VisualizeDetection import visualizaion
                # from utils import tensor2cv2image
                # visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
                # import time
                # time.sleep(2)
                # assert False

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
                loss.backward()
                grad = adv_x.grad.clone()
                adv_x.requires_grad = False
                # adv_x = clamp(adv_x - 0.005 * grad.sign())
                adv_x = clamp(adv_x - 0.05 * grad)
                adv_x.requires_grad = True
                # optimizer.step()
                total_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={total_loss / (step + 1)}')
                    if step >= attack_step:
                        torch.save(adv_x, 'patch.pth')
                        adv_x = tensor2cv2image(adv_x.detach().cpu())
                        cv2.imwrite('patch.jpg', adv_x)
                        return adv_x
                    if step % 100 == 0:
                        torch.save(adv_x, 'patch.pth')
            print(epoch, total_loss / len(self.loader))

        torch.save(adv_x, 'patch.pth')

        adv_x = tensor2cv2image(adv_x.detach())
        cv2.imwrite('patch.jpg', adv_x)

        visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
        import time
        time.sleep(2)
        return adv_x

    @staticmethod
    def add_gaussian_noise(model: nn.Module, scale=1e-6):
        '''
        这样做到底会不会对神经网络有比较大的性能影响？？？？？？？？？？？？？？？？？
        我其实不太清楚？？？？？？？？？？？？？？？？？？？？？？？？
        ？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？
        :param model:
        :param scale:
        :return:
        '''
        for param in model.parameters():
            param.data += scale * torch.randn_like(param.data)
        return model

    def perturb_by_gradient_descent(self):
        pass


class PatchAttackDownsampleByNeuralNetWork():
    def __init__(self, local_rank,
                 model: nn.Module,
                 loader: DataLoader,
                 patch_size=(3, 512, 512), ):
        self.model = model
        self.loader = loader
        self.patch_size = patch_size
        self.device = torch.device('cuda', local_rank)
        for s in self.model.modules():
            s.requires_grad_(False)

        self.downsample_nn = nn.Sequential(
            nn.Conv2d(3, 3, 7, 2, 3),
            nn.ReLU(),
            nn.Conv2d(3, 3, 7, 2, 3),
        ).cuda()

    def patch_attack_detection(self,
                               attack_epoch=1000,
                               attack_step=10000,
                               m=0.9,
                               use_sign=False,
                               lr=0.5,
                               aug_image=False,
                               fp_16=False) -> torch.tensor:
        '''
        use nesterov
        :param x:image
        :param model: detection model, whose output is pytorch detection style
        :return:
        '''
        scaler = GradScaler()
        optimizer = torch.optim.SGD(self.downsample_nn.parameters(), lr=0.1, momentum=0.9, nesterov=True)
        if aug_image:
            transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1, 0.1),
            ])
        if os.path.exists('patch.pth'):
            adv_x = torch.load('patch.pth')
        else:
            adv_x = torch.clamp(torch.randn(self.patch_size) / 2 + 1, 0, 1)
        adv_x.requires_grad = True
        momentum = 0
        # optimizer = torch.optim.SGD([adv_x], lr=1e-2)
        criterion = lambda x: F.mse_loss(x, torch.zeros_like(x))

        for epoch in range(1, attack_epoch + 1):
            total_loss = 0
            pbar = tqdm(self.loader)
            self.loader.sampler.set_epoch(epoch)
            for step, image in enumerate(pbar):
                with torch.no_grad():
                    image = image.to(self.device)
                    if aug_image:
                        image = transform(image)
                    predictions = self.model(image)
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
                        now = F.interpolate(self.downsample_nn(adv_x.unsqueeze(0)),
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

                grad = adv_x.grad.clone()
                adv_x.requires_grad = False
                optimizer.step()
                if use_sign:
                    adv_x = clamp(adv_x - lr * grad.sign())
                else:
                    momentum = m * momentum - grad
                    adv_x += lr * (-grad + m * momentum)
                    adv_x = clamp(adv_x)
                adv_x.requires_grad = True
                # optimizer.step()
                total_loss += loss.item()
                if step % 10 == 0:
                    pbar.set_postfix_str(f'loss={total_loss / (step + 1)}')
                    if step >= attack_step:
                        torch.save(adv_x, 'patch.pth')
                        adv_x = tensor2cv2image(adv_x.detach().cpu())
                        cv2.imwrite('patch.jpg', adv_x)
                        return adv_x
                    if step % 1000 == 0:
                        torch.save(adv_x, 'patch.pth')
                        img_x = tensor2cv2image(adv_x.detach().clone())
                        cv2.imwrite('patch.jpg', img_x)
            print(epoch, total_loss / len(self.loader))

        torch.save(adv_x, 'patch.pth')

        adv_x = tensor2cv2image(adv_x.detach())
        cv2.imwrite('patch.jpg', adv_x)

        visualizaion([predictions[0]], tensor2cv2image(image[0].detach()))
        import time
        time.sleep(2)
        return adv_x
