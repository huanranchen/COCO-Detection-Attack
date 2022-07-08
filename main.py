from skimage.io import imread
from VisualizeDetection import visualizaion
from torchvision.models.detection import faster_rcnn, fasterrcnn_resnet50_fpn
from Attack import attack_detection, patch_attack_detection
from utils import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = fasterrcnn_resnet50_fpn(pretrained=True).to(device)
from data.data import get_loader

loader = get_loader()
patch_attack_detection(model, loader, )
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
