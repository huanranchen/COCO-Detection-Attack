import torch
import torchvision
from skimage.io import imread
from VisualizeDetection import visualizaion
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn, ssd300_vgg16
from Attack import attack_detection, patch_attack_detection, SAM_patch_attack_detection, \
    AttackWithPerturbedNeuralNetwork
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.distributed as dist
import os
import argparse
from criterion import TestAttackAcc

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

device = torch.device("cuda", local_rank)

from data.data import get_loader


def attack():
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    loader = get_loader(train_path='/home/nico/data/coco/train2017/', batch_size=16)
    # patch_attack_detection(model, loader, attack_epoch=7, attack_step=999999999)
    # SAM_patch_attack_detection(model, loader, attack_epoch=3, attack_step=999999999)
    w = AttackWithPerturbedNeuralNetwork(model, loader)
    # w.patch_attack_detection()
    # w.test_perturb_strength()
    w.adversarial_training_patch()


def draw_2d(dataset_path, model):
    loader = get_loader(dataset_path)
    from Draws.DrawUtils.D2Landscape import D2Landscape
    from criterion import GetPatchLoss
    patch = torch.load('patch.pth').to(device)
    loss = GetPatchLoss(model, loader)
    d = D2Landscape(loss, patch, mode='2D')
    d.synthesize_coordinates()
    d.draw()
    # plt.savefig('landscape.jpg')


def draw_train_test_2d():
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    train_path = "/home/chenziyan/work/data/coco/train/train2017/"
    test_path = "/home/chenziyan/work/data/coco/test/test2017/"
    draw_2d(train_path, model)
    draw_2d(test_path, model)
    plt.savefig('landscape.jpg')


def draw_multi_model_2d():
    train_path = "/home/chenziyan/work/data/coco/train/train2017/"
    model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    draw_2d(train_path, model)

    model = ssd300_vgg16(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    draw_2d(train_path, model)
    plt.legend(['faster rcnn', 'ssd'])
    plt.savefig('landscape.jpg')


def test_accuracy():
    '''
    estimate on test set
    :return:
    '''
    train_path = '/home/nico/data/coco/val2017/'
    # model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
    model = torchvision.models.detection.ssd300_vgg16(pretrained=True).to(device)
    model.eval().to(device)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                      find_unused_parameters=True)
    loader = get_loader(train_path, batch_size=16)
    w = TestAttackAcc(model, loader)
    patch = torch.load('patch.pth').to(device)
    print(w.test_accuracy(patch, total_step=100))


attack()

# image = imread('1.jpg')
# x = image_array2tensor(np.array(image))
# x = attack_detection(x, model, 40)
# pre = model(x)
# image = tensor2cv2image(x)
# visualizaion(pre, image)

# from Draws.DrawUtils.D2Landscape import D2Landscape
# from criterion import get_loss
#
# figure = plt.figure()
# axes = Axes3D(figure)
# wtf = D2Landscape(lambda a: get_loss(a, model), x)
# wtf.synthesize_coordinates()
# wtf.draw(axes=axes)
# plt.show()
# plt.savefig("mypatchlandscape.png")
