import torch
from skimage.io import imread
from VisualizeDetection import visualizaion
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
from Attack import attack_detection, patch_attack_detection
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.distributed as dist
import os

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
FLAGS = parser.parse_args()
local_rank = FLAGS.local_rank
torch.cuda.set_device(local_rank)
dist.init_process_group(backend='nccl')  # nccl是GPU设备上最快、最推荐的后端

device = torch.device("cuda", local_rank)
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
model.eval().to(device)
model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank,
                                                  find_unused_parameters=True)
from data.data import get_loader



def attack():
    loader = get_loader()
    patch_attack_detection(model, loader, )


def draw_2d(dataset_path):
    loader = get_loader(dataset_path)
    from Draws.DrawUtils.D2Landscape import D2Landscape
    from criterion import GetPatchLoss
    patch = torch.load('patch.pth').to(device)
    loss = GetPatchLoss(model, loader)
    d = D2Landscape(loss, patch, mode = '2D')
    d.synthesize_coordinates()
    d.draw()
    # plt.savefig('landscape.jpg')

def draw_train_test_2d():
    train_path = "/home/chenziyan/work/data/coco/train/train2017/"
    test_path = "/home/chenziyan/work/data/coco/test/test2017/"
    draw_2d(train_path)
    draw_2d(test_path)
    plt.savefig('landscape.jpg')

draw_train_test_2d()









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
