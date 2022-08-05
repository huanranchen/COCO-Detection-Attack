from torchvision.models.detection import FasterRCNN
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.models.detection.rpn import AnchorGenerator
import torch
from torch import nn
from tqdm import tqdm


def faster_rcnn_my_backbone(num_classes=91):
    import torchvision
    from torchvision.models.feature_extraction import create_feature_extractor

    backbone = torchvision.models.convnext_base(pretrained=True)
    backbone = create_feature_extractor(backbone, return_nodes={"features.6": "1"})
    backbone.out_channels = 1024

    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
                                       aspect_ratios=((0.5, 1.0, 2.0),))

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['1'],  # 在哪些特征层上进行RoIAlign pooling
                                                    output_size=[7, 7],  # RoIAlign pooling输出特征矩阵尺寸
                                                    sampling_ratio=2)  # 采样率

    model = FasterRCNN(backbone=backbone,
                       num_classes=num_classes,
                       rpn_anchor_generator=anchor_generator,
                       box_roi_pool=roi_pooler)

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
            loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            total_loss += loss.item()
            optimizer.step()
            if step % 10 == 0:
                pbar.set_description_str(f'loss = {total_loss / step}')

        torch.save(model.state_dict(), 'detector.ckpt')


if __name__ == '__main__':
    from data import get_coco_loader
    import argparse
    import torch.distributed as dist
    import os
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)
    FLAGS = parser.parse_args()
    local_rank = FLAGS.local_rank
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')

    device = torch.device("cuda", local_rank)
    loader = get_coco_loader(batch_size=16)
    model = faster_rcnn_my_backbone().to(device)
    training_detectors(loader, model, total_epoch=3)
