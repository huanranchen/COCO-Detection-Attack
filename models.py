from torchvision.models.detection import FasterRCNN
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor, get_graph_node_names
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch
from torch import nn
from tqdm import tqdm
import os
from torchvision.models.detection.backbone_utils import BackboneWithFPN, LastLevelMaxPool,_resnet_fpn_extractor,_validate_trainable_layers
from torch import distributed as dist
from torchvision._internally_replaced_utils import load_state_dict_from_url
from backbones import resnet50
from torchvision.models.detection._utils import  overwrite_eps
from torchvision.ops import misc as misc_nn_ops



model_urls = {
    "fasterrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth",
    "fasterrcnn_mobilenet_v3_large_320_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth",
    "fasterrcnn_mobilenet_v3_large_fpn_coco": "https://download.pytorch.org/models/fasterrcnn_mobilenet_v3_large_fpn-fb6a3cc7.pth",
}



def reduce_mean(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.AVG)
    return rt


def faster_rcnn_my_backbone(num_classes=91):
    backbone = torchvision.models.convnext_base(pretrained=False)
    #print(backbone)
    # for i in get_graph_node_names(backbone)[0]:
    #     print(i)
    return_layers = {"features.3.2.add": "0",   # stride 8
                     "features.5.26.add": "1",  # stride 16
                     "features.7.2.add": "2"}  # stride 32
    # 提供给fpn的每个特征层channel
    # in_channels_list = [40, 112, 960]
    new_backbone = create_feature_extractor(backbone, return_layers)
    img = torch.randn(1, 3, 224, 224)
    outputs = new_backbone(img)
    # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    in_channels_list = [256, 512, 1024]
    # --- efficientnet_b0 fpn backbone --- #
    # backbone = torchvision.models.efficientnet_b0(pretrained=True)
    # # print(backbone)
    # return_layers = {"features.3": "0",  # stride 8
    #                  "features.4": "1",  # stride 16
    #                  "features.8": "2"}  # stride 32
    # # 提供给fpn的每个特征层channel
    # in_channels_list = [40, 80, 1280]
    # new_backbone = create_feature_extractor(backbone, return_layers)
    # # img = torch.randn(1, 3, 224, 224)
    # # outputs = new_backbone(img)
    # # [print(f"{k} shape: {v.shape}") for k, v in outputs.items()]
    if not set(return_layers).issubset([name for name, _ in backbone.named_children()]):
        for name, _ in backbone.named_children():
            print(name)
        print(set(return_layers))
        assert False
    backbone_with_fpn = BackboneWithFPN(new_backbone,
                                        return_layers=return_layers,
                                        in_channels_list=in_channels_list,
                                        out_channels=256,
                                        extra_blocks=LastLevelMaxPool())

    anchor_sizes = ((64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                        aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0', '1', '2'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone_with_fpn,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

    return model


def faster_rcnn_resnet50_shakedrop(pretrained=True, progress=True,
                                   num_classes=91, pretrained_backbone=True,
                                   trainable_backbone_layers=None, **kwargs
):
    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 5, 3
    )

    if pretrained:
        # no need to download the backbone if pretrained is set
        pretrained_backbone = False

    backbone = resnet50(pretrained=pretrained_backbone, progress=progress, norm_layer=misc_nn_ops.FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(backbone, trainable_backbone_layers)
    model = FasterRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["fasterrcnn_resnet50_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        overwrite_eps(model, 0.0)
    return model




def training_detectors(loader, model: nn.Module,
                       total_epoch=3,
                       lr=1e-4,
                       weight_decay=1e-4):
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    for epoch in range(1, total_epoch + 1):
        # train
        loader.sampler.set_epoch(epoch)
        pbar = tqdm(loader)
        total_loss = 0
        for step, (x, y) in enumerate(pbar):
            loss = model(x, y)['loss_objectness']
            optimizer.zero_grad()
            loss.backward()
            total_loss += reduce_mean(loss).item()
            optimizer.step()
            if step % 10 == 0:
                pbar.set_description_str(f'loss = {total_loss / (step + 1)}')
                if step % 100 == 0:
                    torch.save(model.state_dict(), 'detector.ckpt')


if __name__ == '__main__':
    # from data import get_coco_loader
    # import argparse
    # import torch.distributed as dist
    # import os
    #
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    # parser.add_argument("--lr", default=1e-3, type=float)
    # FLAGS = parser.parse_args()
    # local_rank = FLAGS.local_rank
    # lr = FLAGS.lr
    # torch.cuda.set_device(local_rank)
    # dist.init_process_group(backend='nccl')
    #
    # device = torch.device("cuda", local_rank)
    # loader = get_coco_loader(batch_size=2)
    # model = faster_rcnn_my_backbone().to(device)
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
    #                                                   output_device=local_rank, find_unused_parameters=True)
    # if os.path.exists('detector.ckpt'):
    #     model.load_state_dict(torch.load('detector.ckpt'))
    #     print('using loaded model')
    #
    # training_detectors(loader, model, total_epoch=3, lr=lr)
    faster_rcnn_resnet50_shakedrop()
